"""run one dynamic-resources case end to end and write a status note

a dynamic run cannot be resumed. stock, workload and the sim clock live only in
the agent processes' memory, so restarting mid-batch and skipping the recorded
orders would evaluate the rest against freshly seeded stock and a reset clock.
the numbers would be wrong rather than obviously broken. recovery is always a
clean re-run from order one.

nothing is restarted automatically. on any failure this writes what died, where,
and the exact command to run, then stops. a heartbeat line at the top of the
status file is refreshed while a batch is running, so a machine that dies mid-run
leaves a timestamp saying how far it got.

usage:
  python run_case.py <case-name> <mode> <slm 0|1>
"""

import os
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
# logs and the bundled broker stay outside the repo
SCRATCH = Path(os.getenv("DYNAMIC_RUN_SCRATCH", HERE / "_logs"))
PY = REPO / ".venv" / "Scripts" / "python.exe"
STATUS = REPO / "reports" / "bidding" / "dynamic_resources_run_status.md"
ORDERS = int(os.getenv("DYNAMIC_RUN_ORDERS", "300"))
LIVE_START, LIVE_END = "<!-- LIVE:START -->", "<!-- LIVE:END -->"

SERVICES = [
    ("material", "src/agents/material/agent.py"),
    ("manufacturer", "src/agents/manufacturer/agent.py"),
    ("logistics", "src/agents/logistics/agent.py"),
    ("supervisor", "src/supervisor/supervisor.py"),
]

NEXT = {
    "case3_slm_sealed": "case 4, SLM reverse auction",
    "case4_slm_auction": "all four cases done, charts and final report",
}


SCRATCH.mkdir(parents=True, exist_ok=True)


def now():
    return datetime.now().strftime("%H:%M:%S")


def log(msg):
    print(f"[{now()}] {msg}", flush=True)


def append_status(title, body):
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%a %d %b %H:%M")
    with STATUS.open("a", encoding="utf-8") as fh:
        fh.write(f"\n## {title}\n\n_{stamp}_\n\n{body.strip()}\n")


def set_live(text):
    """rewrite the heartbeat block in place, leaving the milestones untouched"""
    block = f"{LIVE_START}\n{text.strip()}\n{LIVE_END}"
    if not STATUS.is_file():
        STATUS.write_text(block + "\n", encoding="utf-8")
        return
    current = STATUS.read_text(encoding="utf-8")
    if LIVE_START in current:
        current = re.sub(re.escape(LIVE_START) + r".*?" + re.escape(LIVE_END),
                         lambda _: block, current, count=1, flags=re.S)
    else:
        lines = current.split("\n")
        cut = 1 if lines and lines[0].startswith("# ") else 0
        current = "\n".join(lines[:cut] + ["", block] + lines[cut:])
    STATUS.write_text(current, encoding="utf-8")


class Heartbeat(threading.Thread):
    """refresh the live block while a batch runs, so a dead machine leaves a mark"""

    def __init__(self, case, run_path, eta_minutes, period=120):
        super().__init__(daemon=True)
        self.case, self.run_path, self.period = case, run_path, period
        self.started = datetime.now()
        self.eta = self.started + timedelta(minutes=eta_minutes)
        self._stop = threading.Event()

    def done(self):
        p = self.run_path()
        if p is None or not p.is_file():
            return 0, None
        try:
            return sum(1 for _ in p.open(encoding="utf-8")), p
        except OSError:
            return 0, p

    def tick(self):
        n, p = self.done()
        name = p.relative_to(REPO).as_posix() if p else "not created yet"
        set_live(
            f"**RUNNING: {self.case}** \n"
            f"started {self.started:%a %d %b %H:%M}, "
            f"last alive **{datetime.now():%a %d %b %H:%M}**, "
            f"expected finish about {self.eta:%H:%M}\n\n"
            f"progress {n} of {ORDERS} orders, file `{name}`\n\n"
            f"If 'last alive' is far in the past and no milestone section was added "
            f"below, the machine or the run died at about that time. Nothing restarts "
            f"itself; see the crash-safety note for what to run.")

    def run(self):
        while not self._stop.wait(self.period):
            try:
                self.tick()
            except Exception:
                pass

    def stop(self):
        self._stop.set()


def env_for(mode, use_slm):
    env = dict(os.environ)
    env.update({
        "NATS_URL": "nats://localhost:4222",
        "BIDDING_MODE": mode,
        "DYNAMIC_RESOURCES": "1",
        "PYTHONPATH": str(REPO),
        "PYTHONUNBUFFERED": "1",
    })
    if use_slm:
        # slm decisions serialise through one llama context per stage process.
        # measured worst case is 9 calls in logistics at ~0.94 s, so the 10 s
        # defaults stand. the 2 s rule-based override must not leak in here
        env["USE_SLM"] = "1"
        env.pop("BIDDING_WINDOW_SECONDS", None)
        env.pop("AUCTION_ROUND_WINDOW_SECONDS", None)
        env.setdefault("SLM_N_GPU_LAYERS", "-1")
        # llama_cpp resolves cudart/cublas through PATH, not add_dll_directory,
        # so the bundled cuda runtime has to be on it before the import
        cuda = REPO / ".venv" / "Lib" / "site-packages" / "nvidia" / "cu13" / "bin" / "x86_64"
        if not cuda.is_dir():
            raise SystemExit(f"cuda runtime dir missing: {cuda}")
        env["PATH"] = f"{cuda};{env.get('PATH', '')}"
        # the supervisor silently caps slm runs at 200 orders by default, which
        # times out every order past it. this must cover the whole batch
        env["SLM_ORDER_SAMPLE_SIZE"] = str(ORDERS)
        env["SLM_MODEL_SLUG"] = os.getenv("SLM_MODEL_SLUG", "")
        gguf = env.get("SLM_MODEL_PATH")
        if not gguf or not Path(gguf).is_file():
            raise SystemExit(f"USE_SLM set but SLM_MODEL_PATH missing or not a file: {gguf!r}")
    else:
        env.pop("USE_SLM", None)
        env["BIDDING_WINDOW_SECONDS"] = "2"
        env["AUCTION_ROUND_WINDOW_SECONDS"] = "2"
    return env


def broker_up():
    import socket
    s = socket.socket()
    s.settimeout(1)
    return s.connect_ex(("127.0.0.1", 4222)) == 0


def nats_binary():
    """the broker executable, on PATH or named by NATS_SERVER_BIN"""
    named = os.getenv("NATS_SERVER_BIN")
    if named and Path(named).is_file():
        return Path(named)
    found = shutil.which("nats-server")
    if found:
        return Path(found)
    raise RuntimeError("no nats-server found, set NATS_SERVER_BIN")


def start_broker():
    if broker_up():
        return
    log("starting nats broker")
    subprocess.Popen([str(nats_binary()), "-p", "4222"],
                     stdout=(SCRATCH / "nats_overnight.log").open("ab"),
                     stderr=subprocess.STDOUT)
    for _ in range(40):
        if broker_up():
            return
        time.sleep(0.25)
    raise RuntimeError("nats broker did not come up on 4222")


def start_services(case, env, ready_timeout):
    """fresh processes for every run, state does not reset itself"""
    procs = []
    for name, script in SERVICES:
        out = (SCRATCH / f"{case}_{name}.log").open("wb")
        procs.append((name, subprocess.Popen([str(PY), "-u", str(REPO / script)],
                                             cwd=REPO, env=env, stdout=out,
                                             stderr=subprocess.STDOUT), out))
    want = {"material": "material stage ready", "manufacturer": "manufacturer stage ready",
            "logistics": "logistics stage ready", "supervisor": "waiting for orders"}
    deadline, ready = time.time() + ready_timeout, set()
    while time.time() < deadline and len(ready) < len(want):
        for name, proc, _ in procs:
            if name in ready:
                continue
            if proc.poll() is not None:
                raise RuntimeError(f"{name} exited early with code {proc.returncode}, "
                                   f"see {case}_{name}.log")
            text = (SCRATCH / f"{case}_{name}.log").read_text(encoding="utf-8", errors="ignore")
            if want[name] in text:
                ready.add(name)
        time.sleep(1)
    if len(ready) < len(want):
        raise RuntimeError(f"services not ready in {ready_timeout}s, missing "
                           f"{sorted(set(want) - ready)}")
    log(f"stack ready: {', '.join(sorted(ready))}")
    return procs


def stop(procs):
    for _, proc, _ in procs:
        if proc.poll() is None:
            proc.terminate()
    for _, proc, out in procs:
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
        out.close()


def newest_run(mode, prefix):
    d = REPO / "reports" / "bidding" / mode / "dynamic_resources"
    runs = sorted(d.glob(f"{prefix}run_*.jsonl"))
    return runs[-1] if runs else None


def validate(path):
    r = subprocess.run([str(PY), str(HERE / "check_run.py"), str(path)],
                       cwd=REPO, capture_output=True, text=True)
    return r.returncode == 0, r.stdout.strip()


def restart_command(case, mode, use_slm):
    slm = "1" if use_slm else "0"
    return (f"cd . && "
            f".venv\\Scripts\\python.exe "
            f"\"{HERE / 'run_case.py'}\" {case} {mode} {slm}")


def run_case(case, mode, use_slm, eta_minutes):
    env = env_for(mode, use_slm)
    slug = os.getenv("SLM_MODEL_SLUG", "").strip()
    prefix = (f"slm_{slug}_" if slug else "slm_") if use_slm else "rule_"
    ready_timeout = 900 if use_slm else 120     # slm processes load a model first
    started = time.time()
    procs, beat = [], None
    try:
        start_broker()
        procs = start_services(case, env, ready_timeout)
        beat = Heartbeat(case, lambda: newest_run(mode, prefix), eta_minutes)
        beat.tick()
        beat.start()
        with (SCRATCH / f"{case}_batch.log").open("wb") as out:
            rc = subprocess.run(
                [str(PY), "-u", str(REPO / "src/experiments/run_bidding_experiment.py"),
                 "--holdout", "--orders", str(ORDERS), "--quiet"],
                cwd=REPO, env=env, stdout=out, stderr=subprocess.STDOUT).returncode
        if rc != 0:
            raise RuntimeError(f"batch runner exited {rc}, see {case}_batch.log")
        path = newest_run(mode, prefix)
        if path is None:
            raise RuntimeError("no run file was produced")
        n = sum(1 for _ in path.open(encoding="utf-8"))
        if n != ORDERS:
            raise RuntimeError(f"run file holds {n} records, expected {ORDERS}")
        ok, report = validate(path)
        mins = (time.time() - started) / 60
        append_status(
            f"{case} COMPLETE" + ("" if ok else ", INVARIANTS FAILED"),
            f"Run file `{path.relative_to(REPO).as_posix()}`\n\n"
            f"Took {mins:.0f} min.\n\n```\n{report}\n```\n\n"
            f"**Next:** {NEXT.get(case, 'nothing queued')}\n\n"
            f"**Needs a decision from you:** "
            f"{'no' if ok else 'yes, the invariant check above failed'}")
        set_live(f"**Idle.** {case} finished {datetime.now():%a %d %b %H:%M}. "
                 f"Next: {NEXT.get(case, 'nothing queued')}.")
        log(f"{case} done in {mins:.0f} min, invariants {'ok' if ok else 'FAILED'}")
        return ok
    except Exception as exc:
        mins = (time.time() - started) / 60
        path = newest_run(mode, prefix)
        got = 0
        if path and path.is_file():
            got = sum(1 for _ in path.open(encoding="utf-8"))
        log(f"{case} FAILED after {mins:.0f} min: {exc}")
        append_status(
            f"{case} FAILED, STOPPED",
            f"`{exc}`\n\nDied after {mins:.0f} min at about {got} of {ORDERS} orders.\n\n"
            f"Nothing was restarted. A dynamic run cannot be resumed, so recovery is a "
            f"clean re-run from order one; the partial file "
            f"`{path.relative_to(REPO).as_posix() if path else '(none)'}` is incomplete "
            f"and should be deleted first.\n\n"
            f"To restart by hand:\n\n```\n{restart_command(case, mode, use_slm)}\n```\n\n"
            f"**Next:** nothing runs until this is restarted.\n\n"
            f"**Needs a decision from you:** yes.")
        set_live(f"**STOPPED.** {case} failed {datetime.now():%a %d %b %H:%M} "
                 f"at about {got} of {ORDERS} orders. Nothing is running. "
                 f"See the failure section below.")
        return False
    finally:
        if beat:
            beat.stop()
        stop(procs)


if __name__ == "__main__":
    name, mode, slm = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
    eta = int(sys.argv[4]) if len(sys.argv) > 4 else 180
    sys.exit(0 if run_case(name, mode, slm, eta) else 1)
