"""
the operator entrypoint for the live demo

  python -m src.api up                          brokers, stacks and the server
  python -m src.api up --configs rule           one configuration, no gpu
  python -m src.api submit ORD-15478            an order id from orders.csv
  python -m src.api submit ORD-15478 --config granite-tuned
  python -m src.api submit --file data/demo/restock_sequence.json
  python -m src.api mode reverse_auction        switch mode, restarts the stacks
  python -m src.api reset                       state back to the csv baseline
  python -m src.api status                      what is running
  python -m src.api down                        stop anything left running

up runs in the foreground and stops everything it started on ctrl-c. the other
subcommands talk to a running server over http, so they can be run from a second
terminal while the demo is going
"""

import argparse
import asyncio
import json
import os
import signal
import sys
from datetime import datetime

import aiohttp
from aiohttp import web

from src.api import processes, settings
from src.api.server import Service, build_app, announce


def base_url():
    return f"http://{settings.SERVER_HOST}:{settings.SERVER_PORT}"


async def call(method, path, payload=None):
    async with aiohttp.ClientSession() as session:
        try:
            async with session.request(method, base_url() + path, json=payload,
                                       timeout=aiohttp.ClientTimeout(total=None)) as response:
                body = await response.text()
                if response.status >= 400:
                    print(f"{response.status} {response.reason}: {body}", file=sys.stderr)
                    return 1
                print(json.dumps(json.loads(body), indent=2))
                return 0
        except aiohttp.ClientConnectorError:
            print(f"nothing is listening on {base_url()}, start it with "
                  "`python -m src.api up`", file=sys.stderr)
            return 1


# ── up ────────────────────────────────────────────────────────────────────────

def start_everything(names, mode, force):
    """brokers and stacks, in order, failing loudly and cleaning up after itself"""
    running = processes.leftovers()
    if running and not force:
        listing = ", ".join(f"{n} (pid {e['pid']})" for n, e in running.items())
        raise processes.StackError(
            f"a previous session is still running: {listing}. stop it with "
            "`python -m src.api down`, or start with --force to replace it")
    if force and running:
        for line in processes.kill_leftovers():
            announce(f"stopped leftover {line}")

    if processes.port_open(settings.SERVER_PORT):
        raise processes.StackError(
            f"port {settings.SERVER_PORT} is taken, an api is probably still up. "
            "stop it with `python -m src.api down`, or set DEMO_PORT")

    session = processes.Session(names, mode).start(announce)
    # the api process goes in the pid file too, so `down` from another terminal
    # stops the whole session rather than leaving the server holding its port
    processes.record_session(session, os.getpid())
    return session


async def serve(session):
    service = Service(session)
    await service.start()
    app = build_app(service)

    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, settings.SERVER_HOST, settings.SERVER_PORT)
    await site.start()

    announce(f"api on {base_url()}, test client at {base_url()}/")
    announce(f"session log {service.session_log}")
    announce("ready for orders, ctrl-c to stop")

    stop = asyncio.get_running_loop().create_future()
    try:
        signal.signal(signal.SIGINT, lambda *_: stop.done() or stop.set_result(True))
    except ValueError:
        pass
    try:
        await stop
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        announce("shutting down")
        await service.close()
        await runner.cleanup()


def command_up(args):
    names = [n.strip() for n in args.configs.split(",") if n.strip()]
    unknown = [n for n in names if n not in settings.CONFIG_NAMES]
    if unknown:
        raise SystemExit(f"unknown configuration(s): {', '.join(unknown)}. "
                         f"choose from {', '.join(settings.CONFIG_NAMES)}")
    mode = settings.bidding_mode()

    announce(f"bidding mode {mode}, configurations: {', '.join(names)}")
    session = start_everything(names, mode, args.force)
    try:
        asyncio.run(serve(session))
    except KeyboardInterrupt:
        pass
    finally:
        session.stop()
        if settings.PID_FILE.is_file():
            settings.PID_FILE.unlink()
        announce("everything stopped")
    return 0


def command_down(_args):
    stopped = processes.kill_leftovers()
    print("stopped: " + (", ".join(stopped) if stopped else "nothing was running"))
    return 0


def command_submit(args):
    payload = {"config": args.config, "wait": not args.no_wait}
    if args.file:
        payload["path"] = args.file
        if args.limit:
            payload["limit"] = args.limit
    elif args.order_id:
        payload["order_id"] = args.order_id
    else:
        raise SystemExit("give an order id or --file")
    return asyncio.run(call("POST", "/api/orders", payload))


def command_mode(args):
    return asyncio.run(call("POST", "/api/mode", {"mode": args.mode}))


def command_reset(args):
    payload = {"config": args.config} if args.config else {}
    return asyncio.run(call("POST", "/api/reset", payload))


def command_status(args):
    if args.config:
        return asyncio.run(call("GET", f"/api/state/{args.config}"))
    return asyncio.run(call("GET", "/api/status"))


def main():
    parser = argparse.ArgumentParser(prog="python -m src.api", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command")

    up = sub.add_parser("up", help="start the brokers, the stacks and the api")
    up.add_argument("--configs", default=",".join(settings.CONFIG_NAMES),
                    help="which configurations to run, comma separated")
    up.add_argument("--force", action="store_true",
                    help="replace a session that is still running")
    up.set_defaults(func=command_up)

    submit = sub.add_parser("submit", help="submit an order id, or a file of orders")
    submit.add_argument("order_id", nargs="?")
    submit.add_argument("--file", help="json or jsonl file of orders")
    submit.add_argument("--limit", type=int, help="only the first N orders of the file")
    submit.add_argument("--config", default="rule",
                        help=f"which configuration answers it: {', '.join(settings.CONFIG_NAMES)}")
    submit.add_argument("--no-wait", action="store_true",
                        help="return as soon as it is queued")
    submit.set_defaults(func=command_submit)

    mode = sub.add_parser("mode", help="switch bidding mode, restarting every stack")
    mode.add_argument("mode", choices=("sealed", "reverse_auction"))
    mode.set_defaults(func=command_mode)

    reset = sub.add_parser("reset", help="reseed agent state from the csv baseline")
    reset.add_argument("--config", help="one configuration, default all of them")
    reset.set_defaults(func=command_reset)

    status = sub.add_parser("status", help="what is running, or one configuration's state")
    status.add_argument("--config", help="show this configuration's stock and workload")
    status.set_defaults(func=command_status)

    down = sub.add_parser("down", help="stop a session left running")
    down.set_defaults(func=command_down)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 1
    try:
        return args.func(args)
    except processes.StackError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
