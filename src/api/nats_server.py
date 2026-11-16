"""
find the broker binary, and fetch it once if this machine has never had it

resolution order, first hit wins:
  NATS_SERVER_BIN     an explicit path, always takes priority
  PATH                a system install
  tools/              the copy this module downloads
  the release         downloaded, checksummed and unpacked into tools/

the version is pinned rather than tracking latest: every run in this project so
far was verified against v2.14.5 and a broker that changes under the experiments
is a variable nobody asked for. the binary itself is never committed, it is
platform specific and the repository does not track binary dependencies
"""

import hashlib
import os
import platform
import shutil
import stat
import struct
import tarfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from config import PROJECT_ROOT

VERSION = "v2.14.5"
RELEASE = f"https://github.com/nats-io/nats-server/releases/download/{VERSION}"

# downloaded, never authored, never committed
TOOLS_DIR = Path(os.getenv("DEMO_TOOLS_DIR", PROJECT_ROOT / "tools"))

# written next to the binary once it has been checksummed and run. its presence
# is what makes a repeat start a local file check with no network and no exec
STAMP = "nats-server.verified"

MANUAL_FALLBACK = (
    f"download nats-server {VERSION} yourself from {RELEASE}, then either put it "
    "on PATH or point NATS_SERVER_BIN at it")


class BrokerBinaryError(RuntimeError):
    """the broker binary is not here and could not be fetched"""


ARCHES = {"amd64": "amd64", "x86_64": "amd64", "x64": "amd64",
          "arm64": "arm64", "aarch64": "arm64"}


def _machine():
    """this machine's architecture, from whichever source actually answers

    platform.machine() reads PROCESSOR_ARCHITECTURE on windows and comes back
    empty under some shells, make's own included, which is how this was found.
    the environment is asked next, then the pointer size, because failing to
    name a machine we are running on is worse than inferring the common case"""
    for value in (platform.machine(),
                  os.environ.get("PROCESSOR_ARCHITEW6432", ""),
                  os.environ.get("PROCESSOR_ARCHITECTURE", "")):
        arch = ARCHES.get(value.strip().lower())
        if arch:
            return arch
    # nothing named it. 64-bit intel is the only thing this project has run on,
    # and a wrong guess cannot slip through: the binary is checksummed and then
    # made to report its version before it is kept
    return "amd64" if struct.calcsize("P") == 8 else None


def _platform():
    """(asset name, name inside the archive) for this machine

    raises rather than guessing at the os, which no fallback can infer"""
    system = platform.system().lower()
    arch = _machine()
    if arch is None:
        raise BrokerBinaryError(
            f"no nats-server build is known for {platform.machine()!r} "
            f"32-bit, {MANUAL_FALLBACK}")
    if system == "windows":
        return f"nats-server-{VERSION}-windows-{arch}.zip", "nats-server.exe"
    if system in ("linux", "darwin"):
        return f"nats-server-{VERSION}-{system}-{arch}.tar.gz", "nats-server"
    raise BrokerBinaryError(
        f"no nats-server build is known for {platform.system()!r}, {MANUAL_FALLBACK}")


def local_binary():
    """the downloaded copy, if it is here and was verified"""
    _, name = _platform()
    binary = TOOLS_DIR / name
    if binary.is_file() and (TOOLS_DIR / STAMP).is_file():
        return binary
    return None


def _fetch(url, timeout=60):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.read()
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise BrokerBinaryError(
            f"could not reach {url}: {exc}. {MANUAL_FALLBACK}") from exc


def _published_sha256(asset):
    """the checksum the release publishes for one asset, or None if unlisted"""
    for line in _fetch(f"{RELEASE}/SHA256SUMS", timeout=30).decode().splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1] == asset:
            return parts[0]
    return None


def _extract(archive, member, into):
    """pull the one binary out of the release archive"""
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            name = next((n for n in zf.namelist() if n.endswith(member)), None)
            if name is None:
                raise BrokerBinaryError(f"{archive.name} holds no {member}")
            with zf.open(name) as src, into.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    else:
        with tarfile.open(archive) as tf:
            name = next((n for n in tf.getnames() if n.endswith(member)), None)
            if name is None:
                raise BrokerBinaryError(f"{archive.name} holds no {member}")
            src = tf.extractfile(name)
            with into.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    # a tar or zip member carries its own mode, which the copy above does not,
    # and a binary without the execute bit fails in a way that reads like
    # anything but a permissions problem
    into.chmod(into.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _runs(binary):
    """does it actually execute and report the version we asked for"""
    import subprocess
    try:
        out = subprocess.run([str(binary), "--version"], capture_output=True,
                             text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    reported = (out.stdout + out.stderr).strip()
    return reported if VERSION.lstrip("v") in reported else None


def download(announce=print):
    """fetch, checksum and unpack the pinned release into tools/

    the archive lands on a .part file and is only renamed once its checksum
    matches, so an interrupted download leaves nothing that looks finished"""
    asset, member = _platform()
    TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    binary = TOOLS_DIR / member
    archive = TOOLS_DIR / asset
    partial = archive.with_suffix(archive.suffix + ".part")

    # anything left from an interrupted attempt goes before this one starts
    for leftover in (partial, archive, TOOLS_DIR / STAMP):
        leftover.unlink(missing_ok=True)

    announce(f"nats-server {VERSION} is not on this machine, fetching {asset}")
    expected = _published_sha256(asset)
    payload = _fetch(f"{RELEASE}/{asset}")
    partial.write_bytes(payload)

    got = hashlib.sha256(payload).hexdigest()
    if expected is None:
        announce(f"the release lists no checksum for {asset}, "
                 "falling back to the version check alone")
    elif got != expected:
        partial.unlink(missing_ok=True)
        raise BrokerBinaryError(
            f"{asset} downloaded but its sha256 is {got[:16]}… where the release "
            f"publishes {expected[:16]}…, the file is incomplete or wrong. "
            f"nothing was kept. {MANUAL_FALLBACK}")

    partial.rename(archive)
    _extract(archive, member, binary)
    archive.unlink(missing_ok=True)

    reported = _runs(binary)
    if reported is None:
        binary.unlink(missing_ok=True)
        raise BrokerBinaryError(
            f"the downloaded binary did not run and report {VERSION}, it has been "
            f"deleted rather than kept. {MANUAL_FALLBACK}")

    (TOOLS_DIR / STAMP).write_text(reported + "\n", encoding="utf-8")
    announce(f"nats-server ready at {binary} ({reported})")
    return binary


def resolve(announce=print):
    """the broker binary, downloading it once if there is no other copy

    an explicit NATS_SERVER_BIN and a system install both win over the download,
    so a machine that is already set up keeps working exactly as it did"""
    named = os.getenv("NATS_SERVER_BIN")
    if named and Path(named).is_file():
        return Path(named)
    found = shutil.which("nats-server")
    if found:
        return Path(found)
    existing = local_binary()
    if existing:
        return existing
    return download(announce)
