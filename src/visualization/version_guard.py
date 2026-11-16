"""stop a v4-only diagnostic from silently reporting v4 under a v5 environment

config.DATA_VERSION is pinned to "v4" deliberately and is not read from the environment, so
bid_competition_diagnostics.py and slm_competition_diagnostics.py load data/synthetic/v4 whatever
COB_DATASET_VERSION says. before this guard, `COB_DATASET_VERSION=v5 make bid_check` printed
"version v4" and produced a plausible-looking report: v4 also holds 50,000 orders and the same agent
ids, so the mistake did not show in the output. the pre-freeze pass caught it by reading the code,
which is not a safety net anyone should rely on twice

what this does NOT do: it never changes what these targets do when no environment variable is set,
which is every existing v4 invocation, and it does not touch config.DATA_VERSION
"""

import importlib
import os

ENV_VAR = "COB_DATASET_VERSION"


def requested_version(environ=None):
    """the dataset version the environment asks for, or None when nothing is set"""
    raw = (environ or os.environ).get(ENV_VAR, "").strip()
    return raw or None


def mismatch(environ=None):
    """(requested, pinned) when the environment asks for data this script cannot read, else None"""
    from config import DATA_VERSION
    requested = requested_version(environ)
    if requested and requested != DATA_VERSION:
        return requested, DATA_VERSION
    return None


def route_or_refuse(target, pair, route="", equivalent=""):
    """route to the version's own diagnostic, or exit naming the command to run instead

    routing is only offered where a real equivalent exists. where it does not, refusing loudly is
    the whole point: a silent v4 report under a v5 environment is the failure being prevented"""
    requested, pinned = pair
    if route and requested == "v5":
        print(f"{target}: {ENV_VAR}={requested} is set and this diagnostic reads "
              f"data/synthetic/{pinned}, so it is routing to {route}", flush=True)
        return importlib.import_module(route).main()
    raise SystemExit(
        f"{target} cannot run under {ENV_VAR}={requested}.\n"
        f"  it reads config.DATA_VERSION ({pinned}) and would report {pinned} data under a "
        f"{requested} environment.\n"
        f"  run the {requested} equivalent instead:\n"
        f"    {equivalent}\n"
        f"  or unset {ENV_VAR} to run the {pinned} diagnostic deliberately.")
