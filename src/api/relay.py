"""
every meaningful event from one configuration's broker, in one stream

the relay is a listener, not a participant. it subscribes to the same subjects
the supervisor and the batch runner already use, stamps each message with its own
arrival time and a sequence number, and hands the envelope to whoever is
connected. it never replies to an rfq and never touches a decision

it also owns the two things the api needs from the bus: the future that an
order's outcome resolves, and the control requests that reset or read a stage
"""

import asyncio
import json
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone

from src.api import settings
from src.common.bid_activity import BID_ACTIVITY_SUBJECT
from src.common.dynamic_state import (
    CONTROL_RESET_SUBJECT, CONTROL_SNAPSHOT_SUBJECT, DYNAMIC_STAGES,
    STATE_EVENT_SUBJECT,
)
from src.common.nats_client import connect_with_retry
from src.supervisor.auction import TRACE_SUBJECT
from src.supervisor.supervisor import STAGE_EVENT_SUBJECT

# rfq.> covers the request and every reply subject under it, in both modes
SUBJECTS = ("rfq.>", "order.logging", "order.verdict.*", STATE_EVENT_SUBJECT,
            TRACE_SUBJECT, STAGE_EVENT_SUBJECT, BID_ACTIVITY_SUBJECT)

# what a late-joining client is caught up with
REPLAY = 400


def _now():
    return datetime.now(timezone.utc).isoformat()


def classify(subject, data):
    """(event type, order id) for one message

    the subject carries the stage and the order for rfq traffic, which is why a
    bid can be attributed without the supervisor annotating it"""
    if subject.startswith("rfq."):
        parts = subject.split(".")
        stage, order_id = parts[1], parts[2]
        if len(parts) == 3:
            # the round context sits in the payload already; hoisting it means a
            # client reads which round it is watching without unpacking the rfq,
            # and a sealed rfq simply has none of it
            extra = {"stage": stage}
            if "round_number" in data:
                extra.update(round_number=data["round_number"],
                             current_best_bid=data.get("current_best_bid"),
                             platform_minimum_decrement=data.get("platform_minimum_decrement"))
            return "rfq", order_id, extra
        # rfq.<stage>.<order>.replies[.r<n>]
        round_number = int(parts[4][1:]) if len(parts) > 4 and parts[4].startswith("r") else None
        return "bid", order_id, {"stage": stage, "round_number": round_number}
    if subject == "order.logging":
        return "order_outcome", data.get("order_id"), {}
    if subject.startswith("order.verdict."):
        return "verdict", data.get("order_id"), {"agent_id": subject.split(".", 2)[2]}
    if subject == STATE_EVENT_SUBJECT:
        return "state_change", data.get("observed_on_order_id") or data.get("order_id"), {}
    if subject == TRACE_SUBJECT:
        return "auction_trace", data.get("order_id"), {}
    if subject == STAGE_EVENT_SUBJECT:
        return data.get("event", "stage"), data.get("order_id"), {}
    if subject == BID_ACTIVITY_SUBJECT:
        return "no_bid", data.get("order_id"), {}
    return "message", data.get("order_id"), {}


class ConfigRelay:
    """one connection to one configuration's broker"""

    def __init__(self, name, sink):
        self.name = name
        self.sink = sink
        self.nc = None
        self._outcomes = {}
        # declines per order, kept only until that order's summary is built
        self._declines = defaultdict(lambda: defaultdict(Counter))

    async def start(self):
        self.nc = await connect_with_retry(settings.nats_url(self.name), f"relay {self.name}")
        for subject in SUBJECTS:
            await self.nc.subscribe(subject, cb=self._on_message)

    async def close(self):
        if self.nc is not None:
            await self.nc.drain()
            self.nc = None

    async def _on_message(self, msg):
        try:
            data = json.loads(msg.data.decode())
        except json.JSONDecodeError:
            return
        kind, order_id, extra = classify(msg.subject, data)
        self.sink(self.name, kind, order_id, data, extra)
        if kind == "no_bid" and order_id:
            self._declines[order_id][data.get("stage")][data.get("reason")] += 1
        if kind == "order_outcome":
            waiter = self._outcomes.get(order_id)
            if waiter is not None and not waiter.done():
                waiter.set_result(data)

    def declines(self, order_id):
        """what every agent that stayed silent on this order said, by stage"""
        return {stage: dict(counts)
                for stage, counts in self._declines.get(order_id, {}).items()}

    def forget_declines(self, order_id):
        self._declines.pop(order_id, None)

    def expect(self, order_id):
        """register interest in an outcome before the order is published"""
        waiter = asyncio.get_running_loop().create_future()
        self._outcomes[order_id] = waiter
        return waiter

    def forget(self, order_id):
        self._outcomes.pop(order_id, None)

    async def publish_order(self, order):
        await self.nc.publish("order.created", json.dumps(order).encode())

    async def control(self, subject, target, payload=None):
        reply = await self.nc.request(
            f"{subject}.{target}", json.dumps(payload or {}).encode(),
            timeout=settings.CONTROL_TIMEOUT_SECONDS)
        return json.loads(reply.data.decode())

    async def reset(self):
        """reseed every stage that holds state, and rewind the supervisor clock

        each target is asked on its own subject: a request gets one reply, and a
        broadcast would leave two of the three unanswered and unchecked"""
        answers = {}
        for target in (*DYNAMIC_STAGES, "supervisor"):
            answers[target] = await self.control(CONTROL_RESET_SUBJECT, target)
        return answers

    async def snapshot(self):
        return {stage: await self.control(CONTROL_SNAPSHOT_SUBJECT, stage)
                for stage in DYNAMIC_STAGES}


class Relay:
    """one relay per configuration, one event stream out of them"""

    def __init__(self, names, log_path=None):
        self.configs = {name: ConfigRelay(name, self._emit) for name in names}
        self.subscribers = set()
        self.recent = deque(maxlen=REPLAY)
        self.seq = 0
        self._log = log_path.open("a", encoding="utf-8") if log_path else None

    async def start(self):
        for relay in self.configs.values():
            await relay.start()

    async def close_connections(self):
        for relay in self.configs.values():
            await relay.close()

    async def rebuild(self, names):
        """point the relay at a different set of configurations

        a mode switch can change which ones are available, since the two modes
        load different fine-tuned ggufs. subscribers and the session log stay"""
        for name in [n for n in self.configs if n not in names]:
            await self.configs.pop(name).close()
        for name in names:
            if name not in self.configs:
                self.configs[name] = ConfigRelay(name, self._emit)

    async def reconnect(self):
        """drop the connections and take them again

        used after a mode switch: the brokers are new processes on the same
        ports. subscribers and the session log are kept, so a client watching
        the stream sees the switch rather than a gap it has to reconnect over"""
        for relay in self.configs.values():
            relay.nc = None
            await relay.start()

    async def close(self):
        await self.close_connections()
        if self._log:
            self._log.close()

    def _emit(self, config, kind, order_id, data, extra):
        self.seq += 1
        envelope = {"seq": self.seq, "type": kind, "config": config,
                    "order_id": order_id, "at": _now(), **extra, "data": data}
        self.publish(envelope)

    def publish(self, envelope):
        """put one envelope on every open stream, and on the session log

        a subscriber that cannot keep up loses its oldest events rather than
        holding the bus up: this is a live view, not a ledger, and the session
        log on disk is the complete record"""
        if "seq" not in envelope:
            self.seq += 1
            envelope = {"seq": self.seq, "at": _now(), **envelope}
        self.recent.append(envelope)
        if self._log:
            self._log.write(json.dumps(envelope) + "\n")
            self._log.flush()
        for queue in self.subscribers:
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            queue.put_nowait(envelope)

    def subscribe(self, maxsize=2000):
        queue = asyncio.Queue(maxsize=maxsize)
        self.subscribers.add(queue)
        return queue

    def unsubscribe(self, queue):
        self.subscribers.discard(queue)

    def replay(self, limit=REPLAY):
        return list(self.recent)[-limit:]
