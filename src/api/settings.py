"""
the demo configurations and where everything lives

one configuration is one broker plus one stack process. they are kept apart by
port rather than by subject prefix: the bidding code names its subjects as
constants, and separate brokers isolate the stacks without touching any of them.
routing an order is then a choice of connection, not a reload

a stack process runs the supervisor and all three stage agents together against
one shared model. the batch deployment gives every service its own process and
therefore its own copy of the model, which is four copies per configuration and
does not leave room for a second architecture on a 24 GB card
"""

import os
from pathlib import Path

from config import PROJECT_ROOT, DATA_VERSION
from src.common.bidding_modes import REVERSE_AUCTION, SEALED, bidding_mode

DATA_DIR = PROJECT_ROOT / "data" / "synthetic" / DATA_VERSION
# logs, pid file and session event logs, generated rather than authored
RUN_DIR = Path(os.getenv("DEMO_RUN_DIR", PROJECT_ROOT / ".run"))
LOG_DIR = RUN_DIR / "logs"
PID_FILE = RUN_DIR / "stacks.json"

# llama_cpp resolves cudart and cublas through PATH, not add_dll_directory, so
# the bundled cuda runtime has to be on it before llama_cpp is imported
CUDA_RUNTIME_DIR = PROJECT_ROOT / ".venv" / "Lib" / "site-packages" / "nvidia" / "cu13" / "bin" / "x86_64"

# which orders to run live and what each one shows: data/demo/DEMO_ORDERS.md

MODELS_DIR = PROJECT_ROOT / "models"

# the six architectures from the overnight run, in the order the ui offers them.
# base is the untrained download, the two fine-tunes are per mode: sealed asks a
# model to derive a price, the auction asks it to apply the round rules, and one
# adapter cannot do both, that mix-up is what shipped a 100x mispriced batch
# once already. a null base means no untrained gguf was ever built for that
# architecture, only a bf16 checkpoint
MODELS = {
    "qwen": {
        "label": "Qwen3.5-4B",
        "base": MODELS_DIR / "Qwen3.5-4B-Q8_0.gguf",
        SEALED: MODELS_DIR / "epoch-2/deployment/qwen-sealed/qwen-sealed-nomtp-matched-Q8_0.gguf",
        REVERSE_AUCTION: MODELS_DIR / "reverse_auction/epoch-1/v4/deployment/qwen-auction/qwen-auction-nomtp-matched-Q8_0.gguf",
        "window": 18.0,
    },
    "granite": {
        "label": "Granite 4.1-3B",
        "base": MODELS_DIR / "feasibility/smoke-artifacts/granite41-3b/granite41-3b-Q8_0.gguf",
        SEALED: MODELS_DIR / "epoch-2/deployment/granite-sealed/granite-sealed-nomtp-matched-Q8_0.gguf",
        REVERSE_AUCTION: MODELS_DIR / "reverse_auction/epoch-1/v4/deployment/granite-auction/granite-auction-nomtp-matched-Q8_0.gguf",
        "window": 15.0,
    },
    "phi": {
        "label": "Phi-4-mini",
        "base": MODELS_DIR / "feasibility/smoke-artifacts/phi4mini/phi4mini-Q8_0.gguf",
        SEALED: MODELS_DIR / "epoch-2/deployment/phi-sealed/phi-sealed-nomtp-matched-Q8_0.gguf",
        REVERSE_AUCTION: MODELS_DIR / "reverse_auction/epoch-1/v4/deployment/phi-auction/phi-auction-nomtp-matched-Q8_0.gguf",
        "window": 18.0,
    },
    "ministral": {
        "label": "Ministral-3B",
        "base": None,
        SEALED: MODELS_DIR / "epoch-2/deployment/ministral-sealed/ministral-sealed-nomtp-matched-Q8_0.gguf",
        REVERSE_AUCTION: MODELS_DIR / "reverse_auction/epoch-1/v4/deployment/ministral-auction/ministral-auction-nomtp-matched-Q8_0.gguf",
        "window": 18.0,
    },
    "gemma": {
        "label": "Gemma 4-E2B",
        "base": MODELS_DIR / "feasibility/smoke-artifacts/gemma4-e2b/gemma4-e2b-Q8_0.gguf",
        SEALED: MODELS_DIR / "epoch-2/deployment/gemma-sealed/gemma-sealed-nomtp-matched-Q8_0.gguf",
        REVERSE_AUCTION: MODELS_DIR / "reverse_auction/epoch-1/v4/deployment/gemma-auction/gemma-auction-nomtp-matched-Q8_0.gguf",
        "window": 18.0,
    },
    "smollm3": {
        "label": "SmolLM3-3B",
        "base": None,
        SEALED: MODELS_DIR / "epoch-2/deployment/smollm3-sealed/smollm3-sealed-nomtp-matched-Q8_0.gguf",
        REVERSE_AUCTION: MODELS_DIR / "reverse_auction/epoch-1/v4/deployment/smollm3-auction/smollm3-auction-nomtp-matched-Q8_0.gguf",
        "window": 18.0,
    },
}

# every architecture is offered in the ui, only the live ones are selectable
MODEL_ORDER = ("qwen", "granite", "phi", "ministral", "gemma", "smollm3")

PRIMARY_MODEL = "qwen"

# the second architecture that runs beside qwen. swap the line to change it. the
# commented candidates are in the order their 150-order results rank them;
# granite passed all four gates and both pairs fit the card together, measured.
# ministral and smollm3 have no untrained base gguf on disk, so they cannot fill
# a base configuration without a build, which is out of scope
SECOND_MODEL = "granite"
# SECOND_MODEL = "phi"
# SECOND_MODEL = "gemma"
# SECOND_MODEL = None          # qwen only

ACTIVE_MODELS = tuple(m for m in (PRIMARY_MODEL, SECOND_MODEL) if m)

# rule first, then base and fine-tuned for each live architecture
CONFIG_NAMES = ("rule",) + tuple(f"{model}-{variant}"
                                 for model in ACTIVE_MODELS
                                 for variant in ("base", "tuned"))

# own ports, deliberately not 4222: a broker left over from a batch run would
# otherwise be adopted here, with its agent processes still subscribed
PORTS = {name: 4232 + i for i, name in enumerate(CONFIG_NAMES)}

SERVER_HOST = os.getenv("DEMO_HOST", "127.0.0.1")
SERVER_PORT = int(os.getenv("DEMO_PORT", "8000"))

# how long one order may take before it is reported as a failure. an slm auction
# order is 9 windows plus inference, so this is generous on purpose
ORDER_TIMEOUT_SECONDS = float(os.getenv("DEMO_ORDER_TIMEOUT", "300"))

# a reset or a snapshot is answered from memory, this only has to cover the trip
CONTROL_TIMEOUT_SECONDS = float(os.getenv("DEMO_CONTROL_TIMEOUT", "10"))

# measured, not assumed. a window has to outlast the slowest real reply, and the
# two kinds of agent are three orders of magnitude apart:
#
#   rule    slowest reply 4.4 ms over 109 bids, decisions are arithmetic
#   slm     one inference per eligible agent, serialised through one context.
#           qwen 1.3-1.6 s each, granite 1.0-1.2 s. the widest stage is
#           logistics with 12 agents on the roster, so the window is that count
#           times the rate: qwen 18 s, granite 15 s
#
# the same value serves both modes: a round is a bidding window like any other,
# and reusing one number is what stops BIDDING_WINDOW_SECONDS and
# AUCTION_ROUND_WINDOW_SECONDS drifting apart again
#
# DEMO_WINDOW_<CONFIG> overrides one, which is how the timing pass runs with the
# window wide open. see src/api/timing.py
# 2 s rather than the 1 s the measurement allows: rule-based agents answer in
# milliseconds whatever the window is, so this costs nothing in latency and gives
# a live audience time to read a stage before the next one opens. pacing, not
# performance. only the demo stacks read it, see stack_env in processes.py
RULE_WINDOW = 2.0


def model_of(name):
    """the architecture a configuration runs, None for the rule-based one"""
    return None if name == "rule" else name.split("-")[0]


def variant_of(name):
    return None if name == "rule" else name.split("-")[1]


def _windows():
    out = {}
    for name in CONFIG_NAMES:
        model = model_of(name)
        default = RULE_WINDOW if model is None else MODELS[model]["window"]
        override = os.getenv(f"DEMO_WINDOW_{name.upper().replace('-', '_')}", "").strip()
        out[name] = float(override) if override else default
    return out


WINDOWS = _windows()

CONFIG_LABELS = {"rule": "rule-based, no model"}
for _model in ACTIVE_MODELS:
    CONFIG_LABELS[f"{_model}-base"] = f"base {MODELS[_model]['label']}, untrained"
    CONFIG_LABELS[f"{_model}-tuned"] = f"fine-tuned {MODELS[_model]['label']}"


def model_path(name, mode):
    """the gguf a configuration loads, or None when it runs rule-based"""
    model = model_of(name)
    if model is None:
        return None
    spec = MODELS[model]
    return spec["base"] if variant_of(name) == "base" else spec[mode]


def window_seconds(name):
    # one window per configuration, the mode does not change it
    return WINDOWS[name]


def nats_url(name):
    return f"nats://127.0.0.1:{PORTS[name]}"


def describe(name, mode):
    path = model_path(name, mode)
    return {
        "config": name,
        "label": CONFIG_LABELS[name],
        "model": model_of(name),
        "variant": variant_of(name),
        "mode": mode,
        "port": PORTS[name],
        "gguf": path.name if path else None,
        "window_seconds": window_seconds(name),
    }


def unavailable_reason(name, mode):
    """why a configuration cannot run, or None when it can

    a plain file-existence check on the one gguf that configuration loads. it is
    asked before anything is spawned, so a stack whose model is not on the disk
    is skipped rather than started and left to fail on load. every architecture
    is asked the same question, none is assumed to be present"""
    model = model_of(name)
    if model is None:
        return None                     # rule-based loads nothing
    path = model_path(name, mode)
    if path is None:
        return f"no untrained gguf was ever built for {MODELS[model]['label']}"
    if not path.is_file():
        return f"gguf not found at {path}"
    return None


def available(names, mode):
    """(what can run, {what cannot: why}) for the configurations asked for"""
    ready, skipped = [], {}
    for name in names:
        reason = unavailable_reason(name, mode)
        if reason is None:
            ready.append(name)
        else:
            skipped[name] = reason
    return ready, skipped


def model_menu(mode):
    """every architecture, and whether each variant can run, for the dropdown

    an architecture that is not the configured second model, and one whose files
    are not on this machine, are both simply not selectable. the reason says
    which it is rather than leaving a greyed-out row unexplained"""
    menu = []
    for model in MODEL_ORDER:
        entry = {"model": model, "label": MODELS[model]["label"], "variants": {}}
        if model not in ACTIVE_MODELS:
            entry["active"] = False
            entry["reason"] = "not the configured second model"
            menu.append(entry)
            continue
        for variant in ("base", "tuned"):
            reason = unavailable_reason(f"{model}-{variant}", mode)
            entry["variants"][variant] = {"available": reason is None, "reason": reason}
        entry["active"] = any(v["available"] for v in entry["variants"].values())
        entry["reason"] = None if entry["active"] else "no gguf on this machine"
        menu.append(entry)
    return menu
