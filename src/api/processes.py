"""
bring the brokers and the configuration stacks up, and notice when one dies

crash safety here means three things: starting twice does not silently produce
two stacks bidding into one broker, a stack that dies is reported the moment it
happens rather than at the next order, and shutting down leaves nothing behind.
the pid file records what was started so a later run can tell an orphan from a
free port
"""

import json
import os
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from config import PROJECT_ROOT
from src.api import nats_server, settings

READY_LINE = "ready"
# a model-backed stack loads and warms a gguf before it subscribes
READY_TIMEOUT_RULE = 90
READY_TIMEOUT_MODEL = 600


class StackError(RuntimeError):
    pass


def nats_binary(announce=print):
    """the broker executable, fetched once if this machine has never had it

    NATS_SERVER_BIN and a system install still win, see src/api/nats_server.py"""
    try:
        return nats_server.resolve(announce)
    except nats_server.BrokerBinaryError as exc:
        raise StackError(str(exc)) from exc


def port_open(port, host="127.0.0.1", timeout=0.4):
    with socket.socket() as s:
        s.settimeout(timeout)
        return s.connect_ex((host, port)) == 0


def alive(pid):
    if pid is None:
        return False
    try:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                             capture_output=True, text=True).stdout
    except OSError:
        return False
    return str(pid) in out


def read_pid_file():
    if not settings.PID_FILE.is_file():
        return {}
    try:
        return json.loads(settings.PID_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def write_pid_file(record):
    settings.PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    settings.PID_FILE.write_text(json.dumps(record, indent=2), encoding="utf-8")


def record_session(session, api_pid):
    """what is running right now, so `down` from another terminal stops all of it

    rewritten after a mode switch as well as at startup, the pids change"""
    processes = {p.name: {"pid": p.pid, "log": str(p.log_path)} for p in session.processes}
    processes["api"] = {"pid": api_pid, "log": None}
    write_pid_file({
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "mode": session.mode,
        "processes": processes,
    })


def leftovers():
    """pids from an earlier session that are still running"""
    return {name: entry for name, entry in read_pid_file().get("processes", {}).items()
            if alive(entry.get("pid"))}


def stack_env(name, mode):
    """the environment one stack process runs under

    the same variables the batch runner sets, so a demo stack and a batch stack
    are configured identically apart from the broker port"""
    env = dict(os.environ)
    env.update({
        "PYTHONPATH": str(PROJECT_ROOT),
        "PYTHONUNBUFFERED": "1",
        "NATS_URL": settings.nats_url(name),
        "BIDDING_MODE": mode,
        "DYNAMIC_RESOURCES": "1",
        "BIDDING_WINDOW_SECONDS": str(settings.window_seconds(name)),
        "AUCTION_ROUND_WINDOW_SECONDS": str(settings.window_seconds(name)),
    })
    model = settings.model_path(name, mode)
    if model is None:
        env.pop("USE_SLM", None)
        return env

    if not model.is_file():
        # unreachable in normal flow, Session.start checks before spawning
        raise StackError(f"{name}: model file missing at {model}")
    if not settings.CUDA_RUNTIME_DIR.is_dir():
        raise StackError(f"cuda runtime dir missing: {settings.CUDA_RUNTIME_DIR}")
    env.update({
        "USE_SLM": "1",
        "SLM_MODEL_PATH": str(model),
        "SLM_N_GPU_LAYERS": os.getenv("SLM_N_GPU_LAYERS", "-1"),
        # the supervisor stops taking orders after this many in one run. a demo
        # session is open ended, so it must not quietly cap out mid-demo
        "SLM_ORDER_SAMPLE_SIZE": os.getenv("SLM_ORDER_SAMPLE_SIZE", "100000"),
        "PATH": f"{settings.CUDA_RUNTIME_DIR};{env.get('PATH', '')}",
    })
    return env


class Process:
    """one child, its log file, and whether it is still up"""

    def __init__(self, name, popen, log_path):
        self.name = name
        self.popen = popen
        self.log_path = log_path
        self.started_at = datetime.now()

    @property
    def pid(self):
        return self.popen.pid

    def running(self):
        return self.popen.poll() is None

    def log_tail(self, lines=25):
        if not self.log_path.is_file():
            return ""
        text = self.log_path.read_text(encoding="utf-8", errors="replace")
        return "\n".join(text.splitlines()[-lines:])

    def stop(self, timeout=20):
        if not self.running():
            return
        try:
            self.popen.terminate()
            self.popen.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, OSError):
            self.popen.kill()


def _spawn(name, cmd, env=None):
    settings.LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = settings.LOG_DIR / f"{name}.log"
    handle = log_path.open("wb")
    popen = subprocess.Popen([str(c) for c in cmd], cwd=PROJECT_ROOT, env=env,
                             stdout=handle, stderr=subprocess.STDOUT)
    return Process(name, popen, log_path)


def start_broker(name, binary):
    port = settings.PORTS[name]
    if port_open(port):
        raise StackError(
            f"port {port} for the {name} configuration is already in use. "
            f"run `python -m src.api down` first, or stop whatever holds it")
    proc = _spawn(f"broker_{name}", [binary, "-p", str(port), "-a", "127.0.0.1"])
    for _ in range(60):
        if port_open(port):
            return proc
        if not proc.running():
            raise StackError(f"broker for {name} exited: {proc.log_tail()}")
        time.sleep(0.25)
    proc.stop()
    raise StackError(f"broker for {name} did not open port {port}")


def start_stack(name, mode):
    env = stack_env(name, mode)
    proc = _spawn(f"stack_{name}", [sys.executable, "-u", "-m", "src.api.stack", name], env)
    return proc


def wait_ready(proc, name, on_progress=None):
    """block until the stack prints its ready line, or fail loudly

    a model-backed stack spends most of this loading and warming the gguf"""
    budget = READY_TIMEOUT_RULE if settings.model_of(name) is None else READY_TIMEOUT_MODEL
    deadline = time.time() + budget
    marker = f"[stack {name}] {READY_LINE}"
    while time.time() < deadline:
        if not proc.running():
            raise StackError(f"{name} stack exited with code {proc.popen.returncode}:\n"
                             f"{proc.log_tail()}")
        if marker in proc.log_tail(lines=400):
            return
        if on_progress:
            on_progress(name, proc)
        time.sleep(1)
    proc.stop()
    raise StackError(f"{name} stack was not ready in {budget}s:\n{proc.log_tail()}")


def stop_all(processes):
    for proc in reversed(list(processes)):
        proc.stop()


class Session:
    """the brokers and stacks of one run, and the mode they run in

    a bidding mode is fixed at process start: the supervisor and every agent
    read BIDDING_MODE once, and the fine-tuned gguf differs between the two.
    switching therefore means starting the whole set again, which is what
    restart() does. it lives here rather than in the server so the cli and the
    api perform exactly the same steps"""

    def __init__(self, names, mode):
        # what was asked for, which is not always what can run: a configuration
        # whose gguf is not on this machine is skipped, and skipping is decided
        # again on every start because the mode chooses which gguf is needed
        self.requested = list(names)
        self.names = []
        self.skipped = {}
        self.mode = mode
        self.processes = []
        self.stacks = {}

    def start(self, announce=lambda _m: None):
        self.names, self.skipped = settings.available(self.requested, self.mode)
        for name, reason in self.skipped.items():
            announce(f"{name} unavailable, {reason}")
        if not [n for n in self.names if settings.model_of(n)]:
            announce("no model-backed configuration is available on this machine, "
                     "starting rule-based only")

        # resolved once for the whole session, so five brokers do not each go
        # looking, and a first run downloads the binary exactly once
        binary = nats_binary(announce)

        started = []
        try:
            for name in self.names:
                started.append(start_broker(name, binary))
                announce(f"broker for {name} up on port {settings.PORTS[name]}")

            stacks = {}
            for name in self.names:
                model = settings.model_path(name, self.mode)
                announce(f"starting the {name} stack"
                         + (f", loading {model.name}" if model else ""))
                proc = start_stack(name, self.mode)
                started.append(proc)
                stacks[name] = proc

            for name, proc in stacks.items():
                began = time.time()
                wait_ready(proc, name)
                announce(f"{name} stack ready in {time.time() - began:.0f}s (pid {proc.pid})")

            self.processes, self.stacks = started, stacks
            return self
        except Exception:
            stop_all(started)
            raise

    def stop(self):
        stop_all(self.processes)
        self.processes, self.stacks = [], {}

    def restart(self, mode, announce=lambda _m: None):
        """stop everything and bring it back up in the other mode

        the brokers go too. they would survive a mode change, but leaving them
        up means a stack that fails to restart leaves a broker with no agents
        on it, which looks like a working configuration that answers nothing"""
        announce(f"stopping the {self.mode} session")
        self.stop()
        self.mode = mode
        return self.start(announce)


def kill_leftovers():
    """stop anything a previous session left running, by pid"""
    stopped = []
    for name, entry in leftovers().items():
        pid = entry["pid"]
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
        stopped.append(f"{name} (pid {pid})")
    if settings.PID_FILE.is_file():
        settings.PID_FILE.unlink()
    return stopped
