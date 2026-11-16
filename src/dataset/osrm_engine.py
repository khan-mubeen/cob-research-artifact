"""find the pinned OSRM engine, routing profile and OSM extract, and fetch each
once if this machine has never had it

this mirrors src/api/nats_server.py on purpose: a pinned version, a download into
the gitignored tools/ directory, a checksum before anything is kept, a run of the
binary to prove it works, and a stamp file so a repeat start is a local file
check. none of it is committed. the extract alone is 700 MB and the binaries are
platform specific

three things are pinned, because a distance matrix is only reproducible when all
three are:
  the engine   OSRM v26.9.0, the win32-x64 node binding release asset, which
               ships osrm-extract, osrm-contract and osrm-routed as executables
  the profile  car.lua and its lib/*.lua at the same git tag
  the map      Geofabrik's finland-260101.osm.pbf, the 1 January snapshot that
               Geofabrik keeps long term, checked against its published md5 and
               the sha-256 recorded when it was first retrieved

only the windows engine build is pinned, because it is the only one that has
been run. another platform raises with the manual route instead of guessing a
hash. osmium, used to read coordinates out of the same extract, is installed into
tools/osrm/pylib from a hash-pinned wheel and never into the project venv
"""

import contextlib
import hashlib
import json
import os
import platform
import shutil
import ssl
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from config import PROJECT_ROOT

VERSION = "v26.9.0"
RELEASE = f"https://github.com/Project-OSRM/osrm-backend/releases/download/{VERSION}"

# platform -> (release asset, sha-256 recorded when it was first retrieved). the
# release publishes no checksum file, so the recorded hash is the pin
ENGINE_ASSETS = {
    "windows": ("node_osrm-v26.9.0-8-win32-x64-Release.tar.gz",
                "6a6ad04a12b741414b7f3892f1e7c20b5e8bc89a8b4ff7347f2f6104c3e0bc64"),
}
ENGINE_BINARIES = ("osrm-extract", "osrm-contract", "osrm-routed")

PROFILE_BASE = f"https://raw.githubusercontent.com/Project-OSRM/osrm-backend/{VERSION}/profiles/"
PROFILE_FILES = {
    "car.lua": "1b03f79784b94cb7d71b89c2eb62410a17e96d92a8e5f407f3b2dd244a179818",
    "lib/access.lua": "ca4d1e49df9edbcab19a1fbda9d49f9caae1318469fa46130765f19076f7a4d0",
    "lib/destination.lua": "3f28e1a7e6d06352146d6bc213bb6f0536f65dec57a1794f3f34285bbc90057b",
    "lib/guidance.lua": "e9ba9926d6bb687c10377d500cb957a45a534606924ea0ac3248e2ee1c78c0dc",
    "lib/maxspeed.lua": "1a0503cbd4c748d47eef5c606cc13d216aac7363232f7aad8bd76559c8e9e5c8",
    "lib/measure.lua": "b7035a033bc36c65bbeda100e3832b1160cbfdee82f01d8548e5e6d396d7213f",
    "lib/obstacles.lua": "2b6ba1489e1ea86838a7fb4ae69e793708dce602843ba7f255abd9a0b50b3238",
    "lib/pprint.lua": "c00090d5c89bde6647e93a5c79f67546bea67d0d5d9c105b838dcabf88ec8aba",
    "lib/profile_debugger.lua": "8e2e9f4769a7a3d0ad939ca3781860dfafc0bfb492a7cfbd486f62a056d1f23b",
    "lib/relations.lua": "9a099167d9d988f8873e0852f7ff101170f06ca6e43acc5f0d4fb117ca5b2a24",
    "lib/sequence.lua": "ac44c891d08f372339033d76d18e2b4a007108d046eb1b8f99d591ffbc8a8e24",
    "lib/set.lua": "b15c73127b0e1866ccde4c6a7790793b0b48fc8a9aa692cf924514f50caf4bc3",
    "lib/tags.lua": "b381372b9922657c4db41cc84cee73fb8e195afb6e9eaf1fcd8fcdb8f53d22fa",
    "lib/traffic_signal.lua": "dcb87352dbb9354b3bcc4c5705a589395090a42da720c7a4b892c22d02d91447",
    "lib/utils.lua": "1be04b57840cf3fb40944842c2a77cd186c4d2bdf290f7b0f190d44f15fb4a08",
    "lib/way_handlers.lua": "3b8cffbd4c38672403547914cb7cd5aff5967129eea216f4bae428a13b0cbe65",
}

EXTRACT = {
    "file": "finland-260101.osm.pbf",
    "source_url": "https://download.geofabrik.de/europe/finland-260101.osm.pbf",
    "md5_url": "https://download.geofabrik.de/europe/finland-260101.osm.pbf.md5",
    "published_md5": "349396cf757484f3f1f9fad60a7f0630",
    "sha256": "fbd7881ea05656b667404a2ca7d058289666a16efc4275890cbd799e85e050c2",
    "bytes": 700378696,
    "osm_data_timestamp": "2026-01-01T21:21:30Z",
    "provider": "Geofabrik GmbH, extract of OpenStreetMap data",
    "licence_note": "© OpenStreetMap contributors, Open Database License (ODbL) 1.0",
}

OSMIUM_WHEEL = {
    "requirement": "osmium==4.3.1",
    "file": "osmium-4.3.1-cp311-cp311-win_amd64.whl",
    "sha256": "fb6e1cc2980cbdf19f8d8723a096b43a1e30bafe7806ad82b174ba007e076fce",
}

# downloaded, never authored, never committed
TOOLS_DIR = Path(os.getenv("OSRM_TOOLS_DIR", PROJECT_ROOT / "tools" / "osrm"))
ENGINE_DIR = TOOLS_DIR / "engine"
PROFILE_DIR = TOOLS_DIR / "profiles"
DATA_DIR = TOOLS_DIR / "data"
PYLIB_DIR = TOOLS_DIR / "pylib"
STAMP = "osrm.verified.json"
GRAPH_STAMP = "graph.verified.json"
ALGORITHM = "ch"

USER_AGENT = "chain-of-bid v5 route distance (academic research)"

MANUAL_FALLBACK = (
    f"download OSRM {VERSION} from {RELEASE}, the profiles from {PROFILE_BASE} and "
    f"{EXTRACT['source_url']} yourself, then place them under {TOOLS_DIR}")


class OsrmToolError(RuntimeError):
    """a pinned tool or input is missing, wrong, or failed to run"""


def _ssl_context():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url, dst, expected_sha256=None, expected_md5=None, announce=print):
    """stream to a .part file, check it, and only then give it its real name

    an interrupted or mismatched download leaves nothing that looks finished"""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    partial = dst.with_name(dst.name + ".part")
    partial.unlink(missing_ok=True)
    announce(f"  fetch    {url}")
    sha, md5 = hashlib.sha256(), hashlib.md5()
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=600, context=_ssl_context()) as response, \
                open(partial, "wb") as out:
            for chunk in iter(lambda: response.read(1 << 20), b""):
                out.write(chunk)
                sha.update(chunk)
                md5.update(chunk)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        partial.unlink(missing_ok=True)
        raise OsrmToolError(f"could not retrieve {url}: {exc}. {MANUAL_FALLBACK}") from exc

    if expected_sha256 and sha.hexdigest() != expected_sha256:
        partial.unlink(missing_ok=True)
        raise OsrmToolError(
            f"{url} downloaded with sha-256 {sha.hexdigest()}, the pin is "
            f"{expected_sha256}; nothing was kept")
    if expected_md5 and md5.hexdigest() != expected_md5:
        partial.unlink(missing_ok=True)
        raise OsrmToolError(
            f"{url} downloaded with md5 {md5.hexdigest()}, the provider publishes "
            f"{expected_md5}; nothing was kept")
    partial.replace(dst)
    return sha.hexdigest()


def _read_json(path):
    path = Path(path)
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


# ── engine ───────────────────────────────────────────────────────────────────

def _platform_key():
    if platform.system() == "Windows":
        return "windows"
    raise OsrmToolError(
        f"only the windows OSRM {VERSION} build is pinned and has been run; "
        f"{platform.system()!r} is not. {MANUAL_FALLBACK}")


def engine_binary(name):
    suffix = ".exe" if platform.system() == "Windows" else ""
    return ENGINE_DIR / "binding_napi_v8" / f"{name}{suffix}"


def _reported_version(binary):
    try:
        out = subprocess.run([str(binary), "--version"], capture_output=True,
                             text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    reported = (out.stdout + out.stderr).strip()
    return reported if VERSION.lstrip("v") in reported else None


def resolve_engine(announce=print):
    """the verified engine, extracting or downloading it only when needed"""
    stamp = _read_json(ENGINE_DIR / STAMP)
    if stamp and all(engine_binary(n).is_file() for n in ENGINE_BINARIES):
        return stamp

    asset, pinned = ENGINE_ASSETS[_platform_key()]
    archive = TOOLS_DIR / asset
    if not archive.is_file() or sha256_of(archive) != pinned:
        _download(f"{RELEASE}/{asset}", archive, expected_sha256=pinned, announce=announce)

    shutil.rmtree(ENGINE_DIR, ignore_errors=True)
    with tarfile.open(archive) as tf:
        tf.extractall(ENGINE_DIR, filter="data")

    missing = [n for n in ENGINE_BINARIES if not engine_binary(n).is_file()]
    if missing:
        raise OsrmToolError(f"{asset} holds no {missing}. {MANUAL_FALLBACK}")
    reported = _reported_version(engine_binary("osrm-extract"))
    if reported is None:
        shutil.rmtree(ENGINE_DIR, ignore_errors=True)
        raise OsrmToolError(
            f"osrm-extract from {asset} did not run and report {VERSION}; the engine "
            f"was deleted rather than kept. {MANUAL_FALLBACK}")

    info = {"version": VERSION, "asset": asset, "asset_sha256": pinned,
            "reported_version": reported}
    _write_json(ENGINE_DIR / STAMP, info)
    announce(f"  engine   OSRM {reported} ready in {ENGINE_DIR}")
    return info


# ── profile ──────────────────────────────────────────────────────────────────

def resolve_profile(announce=print):
    """car.lua and every lib file it can require, each checked against its pin"""
    for rel, pinned in PROFILE_FILES.items():
        path = PROFILE_DIR / rel
        if not path.is_file() or sha256_of(path) != pinned:
            _download(PROFILE_BASE + rel, path, expected_sha256=pinned, announce=announce)
    return {"profile": "car.lua", "git_tag": VERSION, "source": PROFILE_BASE,
            "sha256_by_file": dict(PROFILE_FILES)}


def profile_path():
    return PROFILE_DIR / "car.lua"


# ── extract ──────────────────────────────────────────────────────────────────

def extract_path(data_dir=None):
    return Path(data_dir or DATA_DIR) / EXTRACT["file"]


def resolve_extract(announce=print):
    """the pinned OSM extract, with a sidecar recording when it was retrieved

    the sidecar lets a repeat run skip rehashing 700 MB. a file that is already
    here and matches the pin but has no sidecar gets one dated from the file's
    own modification time, and the sidecar says that is what the date means"""
    path = extract_path()
    sidecar_path = path.with_name(path.name + ".retrieved.json")
    sidecar = _read_json(sidecar_path)
    if (path.is_file() and sidecar and sidecar.get("sha256") == EXTRACT["sha256"]
            and path.stat().st_size == EXTRACT["bytes"]):
        return sidecar

    if path.is_file() and sha256_of(path) == EXTRACT["sha256"]:
        modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        basis = ("the modification date of a copy already on disk whose sha-256 "
                 "matched the pin when this sidecar was written")
    else:
        _download(EXTRACT["source_url"], path, expected_sha256=EXTRACT["sha256"],
                  expected_md5=EXTRACT["published_md5"], announce=announce)
        modified = datetime.now(timezone.utc)
        basis = "the date this copy was downloaded and checked against both pins"

    sidecar = {**EXTRACT, "retrieval_date": modified.strftime("%Y-%m-%d"),
               "retrieval_date_basis": basis}
    _write_json(sidecar_path, sidecar)
    return sidecar


# ── osmium, outside the project venv ─────────────────────────────────────────

def resolve_pylib(announce=print):
    """make the pinned osmium importable from tools/osrm/pylib"""
    if not (PYLIB_DIR / "osmium").is_dir():
        announce(f"  install  {OSMIUM_WHEEL['requirement']} into {PYLIB_DIR}")
        with tempfile.TemporaryDirectory() as tmp:
            requirements = Path(tmp) / "requirements.txt"
            requirements.write_text(
                f"{OSMIUM_WHEEL['requirement']} --hash=sha256:{OSMIUM_WHEEL['sha256']}\n",
                encoding="utf-8")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--quiet", "--no-deps",
                 "--require-hashes", "--only-binary=:all:", "--target", str(PYLIB_DIR),
                 "-r", str(requirements)],
                capture_output=True, text=True)
        if result.returncode != 0:
            raise OsrmToolError(
                f"could not install {OSMIUM_WHEEL['requirement']} into {PYLIB_DIR}: "
                f"{(result.stdout + result.stderr).strip()[-800:]}")
    if str(PYLIB_DIR) not in sys.path:
        sys.path.insert(0, str(PYLIB_DIR))
    import osmium
    return {"package": "osmium", "version": getattr(osmium, "__version__", "4.3.1"),
            "wheel": OSMIUM_WHEEL["file"], "wheel_sha256": OSMIUM_WHEEL["sha256"]}


# ── graph ────────────────────────────────────────────────────────────────────

def _run_logged(command, log_path):
    with open(log_path, "w", encoding="utf-8", newline="\n") as log:
        result = subprocess.run([str(c) for c in command], stdout=log,
                                stderr=subprocess.STDOUT)
    if result.returncode != 0:
        tail = Path(log_path).read_text(encoding="utf-8", errors="replace")[-1500:]
        raise OsrmToolError(
            f"{Path(command[0]).name} exited {result.returncode}, see {log_path}:\n{tail}")


def graph_inputs():
    """what a built graph must have been built from to be reused"""
    return {"osrm_version": VERSION,
            "engine_asset_sha256": ENGINE_ASSETS[_platform_key()][1],
            "extract_sha256": EXTRACT["sha256"],
            "profile_sha256_by_file": dict(PROFILE_FILES),
            "algorithm": ALGORITHM}


def prepare_graph(data_dir=None, announce=print):
    """osrm-extract then osrm-contract, skipped when a matching graph exists

    data_dir other than the default builds a second, independent graph from the
    same pinned inputs, which is how the determinism check proves a rebuild
    produces the same matrix"""
    data_dir = Path(data_dir or DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)
    resolve_engine(announce)
    resolve_profile(announce)
    resolve_extract(announce)

    pbf = extract_path(data_dir)
    if data_dir != DATA_DIR and not pbf.is_file():
        try:
            os.link(extract_path(), pbf)
        except OSError:
            shutil.copy2(extract_path(), pbf)

    base = data_dir / EXTRACT["file"].replace(".osm.pbf", ".osrm")
    expected = graph_inputs()
    if _read_json(data_dir / GRAPH_STAMP) == expected and Path(f"{base}.hsgr").is_file():
        return base

    started = time.monotonic()
    announce(f"  extract  {pbf.name} with car.lua, this takes a minute or two")
    _run_logged([engine_binary("osrm-extract"), "-p", profile_path(), pbf],
                data_dir / "osrm-extract.log")
    announce("  contract building the contraction hierarchy")
    _run_logged([engine_binary("osrm-contract"), base], data_dir / "osrm-contract.log")
    _write_json(data_dir / GRAPH_STAMP, expected)
    announce(f"  graph    ready at {base} after {time.monotonic() - started:.0f} s")
    return base


# ── routing server, localhost only ───────────────────────────────────────────

def query_json(url, timeout=120):
    """a GET against the local router, ignoring any proxy the environment sets"""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=timeout) as response:
        return json.loads(response.read())


@contextlib.contextmanager
def routed(base, port=5055, announce=print, timeout=300):
    """osrm-routed on 127.0.0.1 for the duration of the block, then stopped"""
    log_path = Path(base).parent / f"osrm-routed-{port}.log"
    log = open(log_path, "w", encoding="utf-8", newline="\n")
    process = subprocess.Popen(
        [str(engine_binary("osrm-routed")), "--algorithm", ALGORITHM, "--ip", "127.0.0.1",
         "--port", str(port), str(base)],
        stdout=log, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + timeout
        while True:
            if process.poll() is not None:
                raise OsrmToolError(f"osrm-routed exited {process.returncode}, see {log_path}")
            try:
                query_json(f"{url}/nearest/v1/driving/24.9435408,60.1666204", timeout=5)
                break
            except (OSError, ValueError):
                if time.monotonic() > deadline:
                    raise OsrmToolError(f"osrm-routed did not answer within {timeout} s")
                time.sleep(1)
        announce(f"  routed   listening on {url}")
        yield url
    finally:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
        log.close()


def main():
    try:
        engine = resolve_engine()
        profile = resolve_profile()
        extract = resolve_extract()
        base = prepare_graph()
    except OsrmToolError as exc:
        print(f"osrm tools incomplete: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"engine": engine, "profile": profile["git_tag"],
                      "extract": extract["file"], "graph": str(base)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
