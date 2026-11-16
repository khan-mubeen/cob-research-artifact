"""
runtime resource state for the material and tailoring stages

this is the implementation of the "Dynamic Agent State" section of
agent_logic_design.md. until now every order was evaluated against a static
snapshot of the csv rows, so stock and workload never moved as orders were
accepted. state here is seeded once per process from those same rows and then
mutated only in memory, the csv files are never written to

both bidding modes call into this module. nothing mode specific lives in the
state itself, sealed bidding and the reverse auction differ only in how a bid is
priced, not in what a reservation does

the clock is simulated. a batch processes orders in order_created sequence and
"now" advances to the order being processed, carried on the rfq as sim_now.
restock arrival and workload release are evaluated against that walking clock,
never against real elapsed runtime

no locking is used and none is needed. the experiment runner publishes one order
and waits for its order.logging outcome before publishing the next, so exactly
one order touches this state at a time and the clock only moves forward. real
concurrent orders would need that revisited, they are not what this pass runs

state changes only on a reservation acknowledgment, never on a submitted bid,
per the design doc. an agent that bids and loses is untouched

state and the clock live for as long as the process does, so a batch run needs
freshly started agent and supervisor processes. reusing processes across two
batches carries the first batch's stock, reservations and clock into the second

the live demo cannot restart processes between runs, a model reload is minutes,
so DynamicStateService also answers a reset on control.reset.<stage>, which
reseeds from the csv rows the process started with. batch runs never send it and
behave exactly as before
"""

import json
import math
import os
from datetime import datetime, timedelta

from src.common.bidding_formulas import (
    manufacturer_dynamic_check, material_dynamic_check,
)


# ── protocol constants, both sides of the wire read these ─────────────────────

# supervisor -> agent, request/reply. the agent answers ack true or false
RESERVE_SUBJECT = "reserve.request"
# supervisor -> agent, fire and forget. rolls back a hold taken for a chain that
# was then abandoned because a later stage could not honour its own reservation
RELEASE_SUBJECT = "reserve.release"
# agent -> anyone listening, one message per state change, reporting only
STATE_EVENT_SUBJECT = "order.dynamic_state"
# operator -> one stage, request/reply. reseeds that process's state from the csv
# rows it started with and rewinds its clock, so a live demo can be run twice
# without restarting anything. batch runs never send it
CONTROL_RESET_SUBJECT = "control.reset"
# operator -> one stage, request/reply. the stage's current snapshot, which is
# what makes a reset checkable rather than merely acknowledged
CONTROL_SNAPSHOT_SUBJECT = "control.snapshot"

# only these two stages hold state. logistics has no stock or capacity field in
# the schema and deliberately gains none
DYNAMIC_STAGES = ("material", "tailoring")

# trigger A, proactive: at or below this fraction of the starting stock
RESTOCK_THRESHOLD_FRACTION = 0.20

# completion days are compared with a tolerance, they are floats off the wire
_EPSILON = 1e-6

# a published bid rounds estimated_completion_days to two decimals, so the
# recomputed figure is compared at that same precision. without this a promise
# of 0.65 would look broken by a recomputation of 0.6534
_COMPLETION_DECIMALS = 2


def _promise_broken(recomputed, promised):
    return round(recomputed, _COMPLETION_DECIMALS) > round(promised, _COMPLETION_DECIMALS) + _EPSILON


def reserve_subject(stage, agent_id):
    return f"{RESERVE_SUBJECT}.{stage}.{agent_id}"


def release_subject(stage, agent_id):
    return f"{RELEASE_SUBJECT}.{stage}.{agent_id}"


def dynamic_resources_enabled():
    """whether this process runs with dynamic resource state

    off by default, so an unset environment behaves exactly as the static
    snapshot deployment always has"""
    return os.getenv("DYNAMIC_RESOURCES", "0").strip().lower() in ("1", "true", "yes", "on")


def parse_sim_now(value):
    """the simulated clock reading carried on an rfq or reservation request

    None when the message predates dynamic mode, callers treat that as no clock
    advance rather than as an error"""
    if not value:
        return None
    return datetime.fromisoformat(str(value))


def _number(value, default=None):
    """a csv cell as a float, empty cells arrive as nan not None

    an empty material column means the agent does not carry that material at
    all, which is not the same as zero stock"""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return default
    return float(value)


def _event(kind, stage, agent_id, order_id, sim_now, **fields):
    return {
        "event": kind,
        "stage": stage,
        "agent_id": agent_id,
        "order_id": order_id,
        "sim_now": sim_now.isoformat() if isinstance(sim_now, datetime) else sim_now,
        **fields,
    }


class SimClock:
    """the walking clock, shared by every state object in one process

    advance_to never moves backwards. orders arrive in order_created sequence so
    that guard should never fire, it is there so an out-of-sequence message
    cannot rewind a restock that already landed"""

    def __init__(self):
        self.now = None

    def advance_to(self, moment):
        if moment is None:
            return self.now
        if self.now is None or moment > self.now:
            self.now = moment
        return self.now


# ── material ──────────────────────────────────────────────────────────────────

class _MaterialSlot:
    """one agent's state for one material it carries"""

    def __init__(self, agent_id, material, available_kg, restock_amount_kg, restock_time_days):
        self.agent_id = agent_id
        self.material = material
        # available to promise, per the design doc: on-hand minus what is already
        # committed. it goes negative when a bid was won against an in-flight
        # restock, which correctly blocks committing that same stock twice
        self.available_kg = float(available_kg)
        self.starting_kg = float(available_kg)
        self.restock_amount_kg = float(restock_amount_kg)
        self.restock_time_days = float(restock_time_days)
        # at most one restock in flight per agent per material
        self.pending = None
        self.restocks_triggered = 0
        self.restocks_landed = 0

    @property
    def threshold_kg(self):
        return self.starting_kg * RESTOCK_THRESHOLD_FRACTION


class MaterialResourceState:
    """stock for every material agent in one process

    the authoritative numbers live here. the agent profile dicts the bid
    formulas and the slm prompt builder read are a per-decision view, refreshed
    by sync_profile before every decision, which is why both the rule path and
    the slm path see live stock without either of them changing

    there is no stock ceiling. a restock can push a material above where it
    started, the schema carries no maximum and none is invented here"""

    stage = "material"

    def __init__(self, agent_rows, clock=None):
        self.clock = clock or SimClock()
        self._slots = {}
        self._holds = {}
        for agent in agent_rows:
            agent_id = agent["agent_id"]
            for material in str(agent.get("material_types", "")).split("|"):
                material = material.strip()
                if not material:
                    continue
                available = _number(agent.get(f"{material}_available_kg"))
                if available is None:
                    continue
                self._slots[(agent_id, material)] = _MaterialSlot(
                    agent_id, material, available,
                    _number(agent.get(f"{material}_restock_amount_kg"), 0.0),
                    _number(agent.get(f"{material}_restock_time_days"), 0.0),
                )

    # ── views ────────────────────────────────────────────────────────────────

    def slot(self, agent_id, material):
        return self._slots.get((agent_id, material))

    def pending_days(self, slot, now=None):
        """days left on an in-flight restock, or None when nothing is pending

        clamped at zero: a restock already due adds no further wait. the caller
        lands due restocks before this runs, so a negative would mean the clock
        moved without advance_to and the honest answer is still no wait"""
        if slot is None or slot.pending is None:
            return None
        now = self.clock.now if now is None else now
        if now is None:
            return None
        return max(0.0, (slot.pending["eta"] - now).total_seconds() / 86400.0)

    def sync_profile(self, agent):
        """write live stock into the profile dict used for this decision

        the remaining time on an in-flight restock goes in beside the stock, so
        the rule path and the slm prompt both price the wait against the pending
        eta rather than a fresh lead time. it is written on every sync, None when
        nothing is pending, so a landed restock cannot leave a stale value behind"""
        agent_id = agent["agent_id"]
        for (slot_agent, material), slot in self._slots.items():
            if slot_agent == agent_id:
                agent[f"{material}_available_kg"] = slot.available_kg
                agent[f"{material}_restock_pending_days"] = self.pending_days(slot)

    def snapshot(self):
        return {
            f"{agent_id}:{material}": {
                "available_kg": round(slot.available_kg, 3),
                "starting_kg": slot.starting_kg,
                "restock_threshold_kg": round(slot.threshold_kg, 3),
                "restock_pending": slot.pending is not None,
                "restocks_triggered": slot.restocks_triggered,
                "restocks_landed": slot.restocks_landed,
            }
            for (agent_id, material), slot in sorted(self._slots.items())
        }

    # ── clock ────────────────────────────────────────────────────────────────

    def advance_to(self, moment, order_id=None):
        """land every restock whose lead time has elapsed on the walking clock

        a landing re-checks trigger A, because a restock can arrive into a slot
        that is still at or under its threshold once the amount is added"""
        now = self.clock.advance_to(moment)
        events = []
        if now is None:
            return events
        for slot in self._slots.values():
            if slot.pending is None or slot.pending["eta"] > now:
                continue
            before = slot.available_kg
            slot.available_kg += slot.pending["amount"]
            slot.restocks_landed += 1
            landed = slot.pending
            slot.pending = None
            events.append(_event(
                "restock_landed", self.stage, slot.agent_id, landed["order_id"], now,
                material=slot.material,
                triggered_by=landed["trigger"],
                triggered_at=landed["triggered_at"].isoformat(),
                restock_time_days=slot.restock_time_days,
                amount_kg=landed["amount"],
                available_kg_before=round(before, 3),
                available_kg_after=round(slot.available_kg, 3),
                observed_on_order_id=order_id,
            ))
            # a landing can drop straight back under the threshold and re-trigger.
            # file that follow-on against the order being processed, not the one
            # that ordered the landed restock, whose report record is long closed
            events.extend(self._maybe_threshold_restock(
                slot, order_id or landed["order_id"], now))
        return events

    # ── restock triggers ─────────────────────────────────────────────────────

    def _trigger_restock(self, slot, order_id, now, trigger):
        if slot.restock_amount_kg <= 0:
            return []
        if slot.pending is not None:
            # one in flight per agent per material, a second trigger does not stack
            return [_event(
                "restock_skipped_pending", self.stage, slot.agent_id, order_id, now,
                material=slot.material, trigger=trigger,
                pending_eta=slot.pending["eta"].isoformat(),
            )]
        slot.pending = {
            "eta": now + timedelta(days=slot.restock_time_days),
            "amount": slot.restock_amount_kg,
            "trigger": trigger,
            "order_id": order_id,
            "triggered_at": now,
        }
        slot.restocks_triggered += 1
        return [_event(
            "restock_triggered", self.stage, slot.agent_id, order_id, now,
            material=slot.material, trigger=trigger,
            amount_kg=slot.restock_amount_kg,
            restock_time_days=slot.restock_time_days,
            eta=slot.pending["eta"].isoformat(),
            available_kg=round(slot.available_kg, 3),
            restock_threshold_kg=round(slot.threshold_kg, 3),
        )]

    def _maybe_threshold_restock(self, slot, order_id, now):
        if slot.available_kg <= slot.threshold_kg:
            return self._trigger_restock(slot, order_id, now, "threshold")
        return []

    # ── reservation ──────────────────────────────────────────────────────────

    def reserve(self, agent, rfq, promised_completion_days):
        """hold stock for a candidate winner, before any accepted verdict

        the feasibility rule is re-run against live state rather than restated,
        so a reservation can only be honoured on exactly the terms that would
        have produced the bid. return (ack, events)"""
        now = self.clock.advance_to(parse_sim_now(rfq.get("sim_now")))
        agent_id = agent["agent_id"]
        order_id = rfq["order_id"]
        material = rfq["material_type"]
        required_kg = float(rfq["required_kg"])
        slot = self._slots.get((agent_id, material))
        if slot is None:
            return False, [_event("reservation_rejected", self.stage, agent_id, order_id, now,
                                  material=material, reason="material_not_stocked")]

        self.sync_profile(agent)
        reason, completion_days = material_dynamic_check(agent, rfq)
        if reason is not None:
            return False, [_event("reservation_rejected", self.stage, agent_id, order_id, now,
                                  material=material, reason=reason,
                                  required_kg=required_kg,
                                  available_kg=round(slot.available_kg, 3))]
        if _promise_broken(completion_days, promised_completion_days):
            # state moved since the bid: the promise can no longer be met
            return False, [_event("reservation_rejected", self.stage, agent_id, order_id, now,
                                  material=material, reason="promised_completion_no_longer_met",
                                  promised_completion_days=promised_completion_days,
                                  recomputed_completion_days=completion_days,
                                  required_kg=required_kg,
                                  available_kg=round(slot.available_kg, 3))]

        before = slot.available_kg
        slot.available_kg -= required_kg
        # trigger B: the promise leans on the restock lead time, so the restock is
        # ordered the moment the bid is accepted, which is what makes the promised
        # delivery date honourable
        used_restock = completion_days > agent["default_completion_days"] + _EPSILON
        events = [_event(
            "stock_reserved", self.stage, agent_id, order_id, now,
            material=material,
            required_kg=required_kg,
            available_kg_before=round(before, 3),
            available_kg_after=round(slot.available_kg, 3),
            starting_kg=slot.starting_kg,
            restock_threshold_kg=round(slot.threshold_kg, 3),
            completion_days=completion_days,
            used_restock_lead_time=used_restock,
        )]
        triggered_here = False
        if used_restock:
            fired = self._trigger_restock(slot, order_id, now, "commitment")
            events.extend(fired)
            triggered_here = any(e["event"] == "restock_triggered" for e in fired)
        if not triggered_here:
            fired = self._maybe_threshold_restock(slot, order_id, now)
            events.extend(fired)
            triggered_here = any(e["event"] == "restock_triggered" for e in fired)

        self._holds[(order_id, agent_id)] = {
            "material": material, "required_kg": required_kg,
            "triggered_restock": triggered_here,
        }
        self.sync_profile(agent)
        return True, events

    def release(self, agent, order_id):
        """roll a hold back when the chain it belonged to was abandoned

        a restock this hold triggered is cancelled with it, unless it already
        landed, in which case the stock is real and stays"""
        agent_id = agent["agent_id"]
        hold = self._holds.pop((order_id, agent_id), None)
        if hold is None:
            return []
        now = self.clock.now
        slot = self._slots[(agent_id, hold["material"])]
        before = slot.available_kg
        slot.available_kg += hold["required_kg"]
        cancelled = False
        if hold["triggered_restock"] and slot.pending is not None \
                and slot.pending["order_id"] == order_id:
            slot.pending = None
            slot.restocks_triggered -= 1
            cancelled = True
        self.sync_profile(agent)
        return [_event(
            "reservation_released", self.stage, agent_id, order_id, now,
            material=hold["material"],
            released_kg=hold["required_kg"],
            available_kg_before=round(before, 3),
            available_kg_after=round(slot.available_kg, 3),
            restock_cancelled=cancelled,
        )]


# ── tailoring ─────────────────────────────────────────────────────────────────

class _WorkloadSlot:
    """one manufacturer's occupation state

    occupation and release, not depletion and restock: there is no threshold and
    no refill, a reservation simply expires once the promised completion time
    has elapsed on the walking clock"""

    def __init__(self, agent_id, current_workload_units, daily_capacity_units):
        self.agent_id = agent_id
        self.current_units = float(current_workload_units)
        self.starting_units = float(current_workload_units)
        self.daily_capacity_units = float(daily_capacity_units)
        self.holds = []
        self.reservations = 0
        self.releases = 0


class TailoringResourceState:
    """workload for every manufacturer agent in one process"""

    stage = "tailoring"

    def __init__(self, agent_rows, clock=None):
        self.clock = clock or SimClock()
        self._slots = {
            agent["agent_id"]: _WorkloadSlot(
                agent["agent_id"], agent["current_workload_units"],
                agent["daily_capacity_units"],
            )
            for agent in agent_rows
        }
        self._holds = {}

    def slot(self, agent_id):
        return self._slots.get(agent_id)

    def sync_profile(self, agent):
        slot = self._slots.get(agent["agent_id"])
        if slot is not None:
            agent["current_workload_units"] = slot.current_units

    def snapshot(self):
        return {
            agent_id: {
                "current_workload_units": round(slot.current_units, 3),
                "starting_workload_units": slot.starting_units,
                "daily_capacity_units": slot.daily_capacity_units,
                "open_reservations": len(slot.holds),
                "reservations": slot.reservations,
                "releases": slot.releases,
            }
            for agent_id, slot in sorted(self._slots.items())
        }

    def advance_to(self, moment, order_id=None):
        """release every reservation whose promised completion has elapsed"""
        now = self.clock.advance_to(moment)
        events = []
        if now is None:
            return events
        for slot in self._slots.values():
            still_held = []
            for hold in slot.holds:
                if hold["release_at"] > now:
                    still_held.append(hold)
                    continue
                before = slot.current_units
                slot.current_units -= hold["units"]
                slot.releases += 1
                self._holds.pop((hold["order_id"], slot.agent_id), None)
                events.append(_event(
                    "workload_released", self.stage, slot.agent_id, hold["order_id"], now,
                    released_units=hold["units"],
                    completion_days=hold["completion_days"],
                    reserved_at=hold["reserved_at"].isoformat(),
                    current_workload_units_before=round(before, 3),
                    current_workload_units_after=round(slot.current_units, 3),
                    observed_on_order_id=order_id,
                ))
            slot.holds = still_held
        return events

    def reserve(self, agent, rfq, promised_completion_days):
        """occupy capacity for a candidate winner, before any accepted verdict"""
        now = self.clock.advance_to(parse_sim_now(rfq.get("sim_now")))
        agent_id = agent["agent_id"]
        order_id = rfq["order_id"]
        units = float(rfq["quantity"])
        slot = self._slots.get(agent_id)
        if slot is None:
            return False, [_event("reservation_rejected", self.stage, agent_id, order_id, now,
                                  reason="agent_not_known")]

        self.sync_profile(agent)
        reason, completion_days = manufacturer_dynamic_check(agent, rfq)
        if reason is not None:
            return False, [_event("reservation_rejected", self.stage, agent_id, order_id, now,
                                  reason=reason, quantity=units,
                                  current_workload_units=round(slot.current_units, 3))]
        if _promise_broken(completion_days, promised_completion_days):
            return False, [_event("reservation_rejected", self.stage, agent_id, order_id, now,
                                  reason="promised_completion_no_longer_met",
                                  promised_completion_days=promised_completion_days,
                                  recomputed_completion_days=completion_days,
                                  quantity=units,
                                  current_workload_units=round(slot.current_units, 3))]

        before = slot.current_units
        slot.current_units += units
        slot.reservations += 1
        # self-release is scheduled against the accepted completion time, the
        # figure the agent itself promised for this order
        hold = {
            "order_id": order_id, "units": units,
            "completion_days": promised_completion_days,
            "reserved_at": now,
            "release_at": now + timedelta(days=float(promised_completion_days)),
        }
        slot.holds.append(hold)
        self._holds[(order_id, agent_id)] = hold
        self.sync_profile(agent)
        return True, [_event(
            "workload_reserved", self.stage, agent_id, order_id, now,
            reserved_units=units,
            current_workload_units_before=round(before, 3),
            current_workload_units_after=round(slot.current_units, 3),
            starting_workload_units=slot.starting_units,
            daily_capacity_units=slot.daily_capacity_units,
            completion_days=promised_completion_days,
            release_at=hold["release_at"].isoformat(),
        )]

    def release(self, agent, order_id):
        agent_id = agent["agent_id"]
        hold = self._holds.pop((order_id, agent_id), None)
        if hold is None:
            return []
        slot = self._slots[agent_id]
        if hold in slot.holds:
            slot.holds.remove(hold)
        before = slot.current_units
        slot.current_units -= hold["units"]
        self.sync_profile(agent)
        return [_event(
            "reservation_released", self.stage, agent_id, order_id, self.clock.now,
            released_units=hold["units"],
            current_workload_units_before=round(before, 3),
            current_workload_units_after=round(slot.current_units, 3),
        )]


STAGE_STATE = {
    "material": MaterialResourceState,
    "tailoring": TailoringResourceState,
    "manufacturer": TailoringResourceState,
}


def build_state(stage, agent_rows, clock=None):
    """the state object for one stage, dispatched by stage name

    an unknown stage raises rather than returning nothing, a stage that silently
    kept no state would look like it was running dynamically when it was not"""
    factory = STAGE_STATE.get(stage)
    if factory is None:
        raise ValueError(f"build_state: no dynamic state for stage {stage!r}")
    return factory(agent_rows, clock=clock)


# ── agent-side wiring ─────────────────────────────────────────────────────────

class DynamicStateService:
    """the agent process's endpoint for its own resource state

    one instance per stage process. it owns the state object, advances the
    walking clock when an rfq arrives, answers the supervisor's reservation
    requests, and publishes every state change on one feed for reporting

    the bid path is untouched: agents keep calling decide_*_bid() and
    slm_decide_*_bid() exactly as before, they simply read a profile dict that
    now carries live numbers"""

    def __init__(self, nc, stage, state, agents):
        self.nc = nc
        self.stage = stage
        self.state = state
        self.agents = {agent["agent_id"]: agent for agent in agents}
        # the csv rows as they were read, kept so a reset can put both the state
        # object and the profile dicts back exactly where they started
        self._seed = {agent["agent_id"]: dict(agent) for agent in agents}

    async def start(self):
        for agent_id in self.agents:
            await self.nc.subscribe(reserve_subject(self.stage, agent_id),
                                    cb=self._on_reserve)
            await self.nc.subscribe(release_subject(self.stage, agent_id),
                                    cb=self._on_release)
        await self.nc.subscribe(f"{CONTROL_RESET_SUBJECT}.{self.stage}", cb=self._on_reset)
        await self.nc.subscribe(f"{CONTROL_SNAPSHOT_SUBJECT}.{self.stage}", cb=self._on_snapshot)
        print(f"[{self.stage}] dynamic resources active for "
              f"{len(self.agents)} agent(s), reservation endpoint ready")

    def reset(self):
        """reseed from the csv rows this process started with

        the profile dicts are refilled in place rather than replaced: the agent
        callbacks closed over them at subscribe time and would otherwise keep
        reading the depleted ones. the state object and the clock are rebuilt, so
        holds, in-flight restocks and the reading of "now" all go with them"""
        for agent_id, agent in self.agents.items():
            agent.clear()
            agent.update(self._seed[agent_id])
        self.state = build_state(self.stage, list(self.agents.values()), clock=SimClock())
        return self.state.snapshot()

    async def _on_reset(self, msg):
        snapshot = self.reset()
        print(f"[{self.stage}] state reset to the csv baseline, clock cleared")
        await msg.respond(json.dumps({"ok": True, "stage": self.stage,
                                      "snapshot": snapshot}).encode())

    async def _on_snapshot(self, msg):
        await msg.respond(json.dumps({
            "ok": True,
            "stage": self.stage,
            "sim_now": self.state.clock.now.isoformat() if self.state.clock.now else None,
            "snapshot": self.state.snapshot(),
        }).encode())

    async def on_rfq(self, agent, rfq):
        """advance the clock to this order and refresh the agent's view

        every restock that came due and every workload reservation that expired
        between the previous order and this one lands here, before any bid is
        calculated for it"""
        events = self.state.advance_to(parse_sim_now(rfq.get("sim_now")),
                                       order_id=rfq.get("order_id"))
        await self._publish(events)
        self.state.sync_profile(agent)

    async def _on_reserve(self, msg):
        request = json.loads(msg.data.decode())
        agent = self.agents.get(request["agent_id"])
        if agent is None:
            await msg.respond(json.dumps({"ack": False, "reason": "unknown_agent"}).encode())
            return
        ack, events = self.state.reserve(agent, request["rfq"],
                                         request["promised_completion_days"])
        await self._publish(events)
        reason = next((e.get("reason") for e in events
                       if e["event"] == "reservation_rejected"), None)
        print(f"[{agent['agent_id']}] reservation {'acked' if ack else 'refused'} on "
              f"{request['rfq']['order_id']}" + (f": {reason}" if reason else ""))
        await msg.respond(json.dumps({
            "ack": ack,
            "stage": self.stage,
            "agent_id": request["agent_id"],
            "order_id": request["rfq"]["order_id"],
            "reason": reason,
        }).encode())

    async def _on_release(self, msg):
        request = json.loads(msg.data.decode())
        agent = self.agents.get(request["agent_id"])
        if agent is None:
            return
        await self._publish(self.state.release(agent, request["order_id"]))

    async def _publish(self, events):
        for event in events:
            await self.nc.publish(STATE_EVENT_SUBJECT, json.dumps(event).encode())
