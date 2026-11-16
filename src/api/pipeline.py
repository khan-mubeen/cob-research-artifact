"""
the one path an order takes, whoever asked for it

submit_order() publishes one order on one configuration's broker and waits for
that order's own order.logging outcome. a file of orders is a loop over it and a
posted order is one call to it, so the two sources cannot drift apart: there is
only one implementation to drift

orders run one at a time, in submission order, exactly as the batch runner
sequences them. dynamic state is in-process memory with no locking, and the
walking clock only moves forward, so two orders in flight at once against one
stack would corrupt both
"""

import asyncio
import time
from datetime import datetime, timezone

from src.api import settings
from src.api.infeasibility import summarize


class Busy(RuntimeError):
    """the queue is full, said out loud rather than by dropping the order"""


async def submit_order(relay, order, timeout=None):
    """publish one order, wait for its outcome, return what happened

    the only function that publishes order.created. it returns the outcome the
    supervisor logged, or a timeout record naming the order, never an exception
    for an order that simply did not finish"""
    order_id = order["order_id"]
    waiter = relay.expect(order_id)
    started = time.perf_counter()
    try:
        await relay.publish_order(order)
        outcome = await asyncio.wait_for(
            waiter, timeout=timeout or settings.ORDER_TIMEOUT_SECONDS)
        status = outcome.get("status", "unknown")
    except asyncio.TimeoutError:
        outcome = {"order_id": order_id, "status": "timeout",
                   "reason": f"no outcome within {timeout or settings.ORDER_TIMEOUT_SECONDS}s"}
        status = "timeout"
    finally:
        relay.forget(order_id)

    # the declines are already on the wire, this only reads them back out
    summary = summarize(outcome, relay.declines(order_id))
    relay.forget_declines(order_id)

    result = {
        "order_id": order_id,
        "config": relay.name,
        "status": status,
        "outcome": outcome,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    if summary:
        result["summary"] = summary
    return result


class Pipeline:
    """the submission queue: one worker, one order at a time, in order

    a job is one order or one file of them. a file job runs its orders through
    submit_order one by one without releasing the worker, so a file and a live
    submission cannot interleave halfway through a chain"""

    def __init__(self, relay, on_event, on_failure, max_queued=32):
        self.relay = relay
        self.on_event = on_event
        self.on_failure = on_failure
        self.queue = asyncio.Queue(maxsize=max_queued)
        self.current = None
        self.completed = 0
        self._worker = None
        self._counter = 0

    def start(self):
        self._worker = asyncio.create_task(self._run())

    async def close(self):
        if self._worker:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass

    def status(self):
        return {
            "queued": self.queue.qsize(),
            "capacity": self.queue.maxsize,
            "running": self.current,
            "completed_jobs": self.completed,
        }

    def enqueue(self, config, orders, source, done=None):
        """accept a job or refuse it loudly, never silently drop one"""
        self._counter += 1
        job = {"job_id": f"job-{self._counter:04d}", "config": config, "source": source,
               "orders": orders, "order_ids": [o["order_id"] for o in orders],
               "done": done}
        try:
            self.queue.put_nowait(job)
        except asyncio.QueueFull:
            raise Busy(f"{self.queue.maxsize} orders are already queued, "
                       "this submission was refused rather than dropped")
        self.on_event({"type": "order_queued", "config": config,
                       "job_id": job["job_id"], "source": source,
                       "order_ids": job["order_ids"],
                       "position": self.queue.qsize()})
        return job

    async def _run(self):
        while True:
            job = await self.queue.get()
            results = []
            try:
                for order in job["orders"]:
                    self.current = {"job_id": job["job_id"], "config": job["config"],
                                    "order_id": order["order_id"]}
                    self.on_event({"type": "order_started", "config": job["config"],
                                   "job_id": job["job_id"], "order_id": order["order_id"],
                                   "source": job["source"], "data": order})
                    result = await submit_order(self.relay.configs[job["config"]], order)
                    results.append(result)
                    self.on_event({"type": "order_finished", "config": job["config"],
                                   "job_id": job["job_id"], "order_id": order["order_id"],
                                   "data": result})
                    if result["status"] not in ("completed", "infeasible"):
                        self.on_failure(
                            f"{job['config']}/{order['order_id']} ended as "
                            f"{result['status']}: {result['outcome'].get('reason', '')}")
            except Exception as exc:
                self.on_failure(f"{job['config']}/{job['job_id']} failed: {exc!r}")
                results.append({"job_id": job["job_id"], "status": "error", "error": repr(exc)})
            finally:
                self.current = None
                self.completed += 1
                self.queue.task_done()
                if job["done"] is not None and not job["done"].done():
                    job["done"].set_result(results)
