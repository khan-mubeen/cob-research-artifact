"""
the http and websocket surface the frontend talks to

one websocket carries both directions: a client submits an order on it and reads
the event stream back on the same connection. the same actions exist as plain
http routes, because a curl and a batch script should not have to open a socket

anything that fails during a live submission is printed to this console as it
happens, not only written to a log for afterwards
"""

import asyncio
import json
import os
import traceback
from datetime import datetime
from pathlib import Path

from aiohttp import WSMsgType, web

from src.api import orders as order_source
from src.api import processes, settings
from src.api.pipeline import Busy, Pipeline
from src.api.relay import Relay
from src.common.bidding_modes import REVERSE_AUCTION, SEALED

STATIC_DIR = Path(__file__).resolve().parent / "static"


def announce(message):
    print(f"[api] {datetime.now():%H:%M:%S} {message}", flush=True)


def alarm(message):
    """a failure the operator running the demo has to see, on the console

    the demo is driven from this terminal, so a failure that only reaches a log
    file is a failure nobody notices until afterwards"""
    print(f"\n[api] !! {datetime.now():%H:%M:%S} {message}\n", flush=True)


class Service:
    """everything the routes need, built once at startup"""

    def __init__(self, session):
        self.session = session
        settings.LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.session_log = settings.LOG_DIR / f"session_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
        self.relay = Relay(self.configs, self.session_log)
        self.pipeline = Pipeline(self.relay, self._emit, self._failure)
        # set while the stacks are being restarted into the other bidding mode
        self.switching = False

    @property
    def configs(self):
        # what actually started, which a mode switch can change: the two modes
        # need different fine-tuned ggufs and a machine may hold only one
        return list(self.session.names)

    @property
    def mode(self):
        return self.session.mode

    @property
    def stacks(self):
        return self.session.stacks

    def _emit(self, envelope):
        self.relay.publish(envelope)

    def _failure(self, message):
        alarm(message)
        self.relay.publish({"type": "failure", "message": message})

    async def start(self):
        await self.relay.start()
        self.pipeline.start()
        asyncio.create_task(self._watch_stacks())

    async def close(self):
        await self.pipeline.close()
        await self.relay.close()

    async def _watch_stacks(self):
        """report a dead stack the moment it dies, not at the next order"""
        reported = set()
        while True:
            for name, proc in self.stacks.items():
                if not proc.running() and name not in reported:
                    reported.add(name)
                    self._failure(f"the {name} stack exited with code "
                                  f"{proc.popen.returncode}. last lines:\n{proc.log_tail(15)}")
            await asyncio.sleep(2)

    def stack_health(self):
        return {name: ("running" if proc.running() else "dead")
                for name, proc in self.stacks.items()}

    def resolve_config(self, name):
        if self.switching:
            raise web.HTTPServiceUnavailable(
                reason="the stacks are restarting into the other bidding mode, retry shortly")
        name = name or self.configs[0]
        if name not in self.configs:
            raise web.HTTPBadRequest(reason=f"unknown configuration {name!r}, "
                                            f"running: {', '.join(self.configs)}")
        proc = self.stacks.get(name)
        if proc is not None and not proc.running():
            raise web.HTTPServiceUnavailable(reason=f"the {name} stack is not running")
        return name

    def build_orders(self, payload):
        """the orders one submission asks for, from whichever source it names"""
        if payload.get("path"):
            return order_source.read_file(payload["path"], payload.get("limit")), \
                f"file:{payload['path']}"
        if payload.get("order"):
            return [order_source.validate(payload["order"])], "api:payload"
        if payload.get("order_id"):
            return [order_source.order_by_id(payload["order_id"])], "api:order_id"
        raise order_source.OrderError(
            "a submission needs one of order_id, order or path")

    async def submit(self, payload, wait=False):
        config = self.resolve_config(payload.get("config"))
        batch, source = self.build_orders(payload)
        done = asyncio.get_running_loop().create_future() if wait else None
        job = self.pipeline.enqueue(config, batch, source, done)
        announce(f"queued {job['job_id']} on {config} from {source}: "
                 f"{', '.join(job['order_ids'][:5])}"
                 f"{' ...' if len(job['order_ids']) > 5 else ''}")
        answer = {"job_id": job["job_id"], "config": config, "source": source,
                  "order_ids": job["order_ids"], "queued": self.pipeline.queue.qsize()}
        if done is not None:
            answer["results"] = await done
        return answer

    async def reset(self, names=None):
        """reseed state and prove it, rather than reporting a bare success"""
        out = {}
        for name in names or self.configs:
            name = self.resolve_config(name)
            relay = self.relay.configs[name]
            answers = await relay.reset()
            after = await relay.snapshot()
            out[name] = {"acknowledged": {k: v.get("ok", False) for k, v in answers.items()},
                         "state": after}
            announce(f"reset {name}: " + ", ".join(f"{k}={'ok' if v.get('ok') else 'FAILED'}"
                                                   for k, v in answers.items()))
        self.relay.publish({"type": "state_reset", "data": out})
        return out

    async def switch_mode(self, mode):
        """restart every stack in the other bidding mode

        a busy pipeline is refused rather than queued. queuing would mean tearing
        the stacks down at some later moment the caller cannot see, and orders
        already queued behind the switch would run under a mode nobody chose for
        them. refusing keeps the decision with whoever asked, and the retry is
        one click"""
        if mode not in (SEALED, REVERSE_AUCTION):
            raise order_source.OrderError(
                f"mode must be {SEALED} or {REVERSE_AUCTION}, got {mode!r}")
        if self.switching:
            raise Busy("a mode switch is already running")
        if mode == self.mode:
            return {"mode": self.mode, "switched": False, "reason": "already in this mode"}

        running, queued = self.pipeline.current, self.pipeline.queue.qsize()
        if running or queued:
            what = (f"{running['config']}/{running['order_id']} is still running"
                    if running else f"{queued} order(s) are queued")
            raise Busy(f"{what}, so the mode switch was refused rather than cutting it "
                       "short. retry when the queue is idle")

        self.switching = True
        began = datetime.now()
        self.relay.publish({"type": "mode_switching", "data": {"from": self.mode, "to": mode}})
        announce(f"switching bidding mode {self.mode} -> {mode}, restarting every stack")
        try:
            await self.relay.close_connections()
            await asyncio.to_thread(self.session.restart, mode, announce)
            await self.relay.rebuild(self.configs)
            await self.relay.reconnect()
            processes.record_session(self.session, os.getpid())
        except Exception as exc:
            self.switching = False
            self._failure(f"the mode switch to {mode} failed: {exc}")
            raise
        self.switching = False
        seconds = round((datetime.now() - began).total_seconds(), 1)
        announce(f"bidding mode is now {mode}, took {seconds}s")
        answer = {"mode": mode, "switched": True, "seconds": seconds,
                  "status": self.status()}
        self.relay.publish({"type": "mode_switched", "data": answer})
        return answer

    async def state(self, name):
        relay = self.relay.configs[self.resolve_config(name)]
        return await relay.snapshot()

    def status(self):
        return {
            "mode": self.mode,
            "switching": self.switching,
            "models": settings.model_menu(self.mode),
            "configs": [settings.describe(name, self.mode) for name in self.configs],
            "unavailable": dict(self.session.skipped),
            "stacks": self.stack_health(),
            "queue": self.pipeline.status(),
            "session_log": str(self.session_log),
            "clients": len(self.relay.subscribers),
            "events_seen": self.relay.seq,
        }


# ── routes ────────────────────────────────────────────────────────────────────

async def _payload(request):
    if not request.can_read_body:
        return {}
    try:
        return await request.json()
    except json.JSONDecodeError as exc:
        raise web.HTTPBadRequest(reason=f"body is not valid json: {exc}")


async def post_orders(request):
    service = request.app["service"]
    payload = await _payload(request)
    try:
        return web.json_response(await service.submit(payload, wait=payload.get("wait", False)))
    except order_source.OrderError as exc:
        alarm(f"rejected a submission: {exc}")
        raise web.HTTPBadRequest(reason=str(exc))
    except Busy as exc:
        alarm(f"refused a submission: {exc}")
        raise web.HTTPServiceUnavailable(reason=str(exc))


async def post_reset(request):
    service = request.app["service"]
    payload = await _payload(request)
    names = payload.get("configs") or ([payload["config"]] if payload.get("config") else None)
    try:
        return web.json_response(await service.reset(names))
    except asyncio.TimeoutError:
        alarm("a stack did not answer the reset request")
        raise web.HTTPGatewayTimeout(reason="a stack did not answer the reset request")


async def post_mode(request):
    service = request.app["service"]
    payload = await _payload(request)
    try:
        return web.json_response(await service.switch_mode(payload.get("mode")))
    except order_source.OrderError as exc:
        alarm(f"rejected a mode switch: {exc}")
        raise web.HTTPBadRequest(reason=str(exc))
    except Busy as exc:
        alarm(f"refused a mode switch: {exc}")
        raise web.HTTPConflict(reason=str(exc))
    except processes.StackError as exc:
        raise web.HTTPInternalServerError(reason=str(exc))


async def get_status(request):
    return web.json_response(request.app["service"].status())


async def get_state(request):
    service = request.app["service"]
    return web.json_response(await service.state(request.match_info["config"]))


async def get_agents(request):
    return web.json_response(order_source.rosters())


async def get_index(request):
    return web.FileResponse(STATIC_DIR / "client.html")


async def send(ws, payload, what="message"):
    """send one frame, or say quietly that the client is no longer there

    a client vanishing mid-flight is ordinary: a tab closed, a page reloaded, a
    dev server hot-reloading the component that owns the socket. it is not a
    fault on this side and it does not deserve a traceback. aiohttp's
    ClientConnectionResetError subclasses ConnectionResetError, so this catches
    both that and the plain transport-closed case"""
    if ws.closed:
        return False
    try:
        await ws.send_json(payload)
        return True
    except ConnectionResetError:
        announce(f"client closed before its {what} was sent")
        return False


async def websocket(request):
    """submit on it, stream from it

    one connection both ways, which is why this is a websocket and not an event
    stream plus a post endpoint"""
    service = request.app["service"]
    ws = web.WebSocketResponse(heartbeat=30, max_msg_size=0)
    try:
        await ws.prepare(request)
    except ConnectionResetError:
        announce("client closed during the websocket handshake")
        return ws

    queue = service.relay.subscribe()
    announce(f"client connected, {len(service.relay.subscribers)} open")

    async def pump():
        # stops on the first failed send: nothing reaches a socket that is gone,
        # and raising here would leave an unretrieved task exception behind
        while await send(ws, await queue.get(), "event"):
            pass

    pumping = asyncio.create_task(pump())
    try:
        if await send(ws, {"type": "hello", "data": service.status()}, "hello"):
            for envelope in service.relay.replay(50):
                if not await send(ws, envelope, "replay"):
                    break

        async for msg in ws:
            if msg.type is not WSMsgType.TEXT:
                continue
            try:
                request_payload = json.loads(msg.data)
                action = request_payload.get("action", "submit")
                if action == "submit":
                    await send(ws, {"type": "accepted",
                                    "data": await service.submit(request_payload)}, action)
                elif action == "reset":
                    await send(ws, {"type": "reset_done",
                                    "data": await service.reset(request_payload.get("configs"))},
                               action)
                elif action == "mode":
                    # the switch itself has already happened by the time this
                    # goes out, and every other client hears it on the relay
                    await send(ws, {"type": "mode_result",
                                    "data": await service.switch_mode(
                                        request_payload.get("mode"))}, action)
                elif action == "status":
                    await send(ws, {"type": "status", "data": service.status()}, action)
                else:
                    await send(ws, {"type": "error",
                                    "message": f"unknown action {action!r}"}, "error")
            except (order_source.OrderError, Busy) as exc:
                alarm(f"rejected a websocket submission: {exc}")
                await send(ws, {"type": "error", "message": str(exc)}, "rejection")
            except Exception as exc:
                alarm(f"websocket request failed: {exc!r}\n{traceback.format_exc()}")
                await send(ws, {"type": "error", "message": repr(exc)}, "error report")
    finally:
        pumping.cancel()
        service.relay.unsubscribe(queue)
        announce(f"client disconnected, {len(service.relay.subscribers)} open")
    return ws


@web.middleware
async def allow_any_origin(request, handler):
    """a frontend dev server is a different origin to this api

    everything here binds to localhost and carries no credentials, so the
    permissive header is the whole of what is needed: without it a page served
    from vite on another port cannot call the routes at all. websockets are not
    subject to this, only the http ones are"""
    if request.method == "OPTIONS":
        response = web.Response(status=204)
    else:
        response = await handler(request)
    response.headers.update({
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "content-type",
    })
    return response


def build_app(service):
    app = web.Application(middlewares=[allow_any_origin])
    app["service"] = service
    app.add_routes([
        web.get("/", get_index),
        web.get("/ws", websocket),
        web.get("/api/status", get_status),
        web.get("/api/agents", get_agents),
        web.get("/api/state/{config}", get_state),
        web.post("/api/orders", post_orders),
        web.post("/api/reset", post_reset),
        web.post("/api/mode", post_mode),
        web.static("/static", STATIC_DIR),
        web.options("/api/{tail:.*}", lambda _r: web.Response(status=204)),
    ])
    return app
