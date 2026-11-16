"""the accelerator backend and the execution profile: two independent axes, one shared training path

  --backend cuda | rocm            what the accelerator is
  --profile local | lumi | roihu   where the job runs

these are orthogonal, never a fixed pairing. the real matrix already has four live cells:

  cuda + local   this machine today, every existing command, unchanged
  rocm + lumi    the near-term target, MI250X gfx90a
  cuda + roihu   a future cluster, the same experiment logic
  rocm + local   an AMD workstation, exactly the isolated question backend_smoke.py was built for

what belongs to a backend: the optimizer that is actually available, the attention implementation,
compiled-kernel cache locations and the environment variables the runtime reads. what belongs to a
profile: filesystem roots, hugging face caches, scheduler resources and wall times, and the module
or container setup a site needs. neither knows about the other, and neither duplicates the training
implementation: src/experiments/finetune_lora.py stays the one training path and is called with
ordinary flags

no site path is hardcoded here. a profile that runs off this machine resolves its roots from an
environment variable (COB_PROFILE_ROOT, or the site's own SLURM/scratch variable), because the
correct absolute paths on LUMI and Roihu cannot be verified from here. the training code never
reads HF_HOME or a model directory of its own: it reads what the resolved profile hands it

nothing in this module imports torch at import time, so it stays usable in a scheduler script, a
test, or a login node with no accelerator
"""

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from config import PROJECT_ROOT

BACKEND_CHOICES = ("cuda", "rocm")

# the LUMI venv's discovered-good package set, recorded for the same reason model_registry pins a
# revision: these versions are the result of real debugging on the machine, not a fresh resolve.
# peft 0.21.0 in particular is load-bearing -- the container's own 0.17.1 imports HybridCache, a
# symbol transformers 5.x removed, and unsloth cannot be imported with it. torch and bitsandbytes
# come from the container image and must never be replaced by a pip wheel: the container builds
# carry the gfx90a kernels that a PyPI wheel does not. this is a record, not a constraint the code
# enforces; rebuilding the venv is a manual act and this is what it has to reproduce
LUMI_VENV_PINS = {
    "peft": "0.21.0", "transformers": "5.17.0", "trl": "1.13.0",
    "unsloth": "2026.9.7", "unsloth_zoo": "2026.9.6",
    "torch": "2.7.1+rocm6.2.4 (container, do not replace)",
    "bitsandbytes": "0.43.3.dev0 (container, do not replace)",
}
PROFILE_CHOICES = ("local", "lumi", "roihu")


class ExecutionError(SystemExit):
    """a launch that cannot be resolved; carries what is missing and how to give it"""


@dataclass(frozen=True)
class Backend:
    name: str
    default_optim: str
    fallback_optim: str
    attention: str
    env: dict
    notes: str


@dataclass(frozen=True)
class Profile:
    name: str
    scheduler: str
    root_env: str
    default_root: str
    partitions: dict
    gpus_per_node: int
    account_env: str
    setup: tuple
    env: dict
    notes: str
    root_required: bool = True
    # shell lines a generated job runs before the training command, for settings that can only be
    # resolved on the compute node. they are emitted verbatim, so a value stays a runtime
    # expression instead of being frozen into the script at generation time. empty for every
    # profile that needs nothing, which is how local and roihu stay exactly as they were
    runtime_setup: tuple = ()
    # a site whose training runs inside a container names the command that enters it and the lines
    # to run once inside, before the training command. the generator wraps the command in these
    # rather than knowing which site needs a container, so a profile that leaves them empty emits
    # exactly the bare command it always emitted
    container_exec: str = ""
    container_prologue: tuple = ()
    # which base layout this site's generated jobs read their weights from. a site that has
    # materialized models/base/<model> says so here once, instead of every operator remembering a
    # flag. empty means the registry default, which is what a site with no base tree needs
    base_layout: str = ""


# ── backends ────────────────────────────────────────────────────────────────

BACKENDS = {
    "cuda": Backend(
        name="cuda",
        default_optim="adamw_8bit",
        fallback_optim="adamw_torch",
        attention="sdpa",
        env={},
        notes=("every completed run in this project; bitsandbytes 8-bit adamw is available and "
               "unsloth selects sdpa or xformers by capability"),
    ),
    "rocm": Backend(
        name="rocm",
        # bitsandbytes on gfx90a is unconfirmed (reports/v5/lumi, the LUMI audit), so the ROCm
        # default is the optimizer that needs no bitsandbytes kernel. adamw_8bit stays reachable
        # with an explicit --optim once a LUMI smoke run proves it
        default_optim="adamw_torch",
        fallback_optim="adamw_8bit",
        attention="flash_attention_2",
        env={"PYTORCH_ROCM_ARCH": "gfx90a"},
        notes=("torch exposes ROCm through the same torch.cuda namespace; bitsandbytes is still "
               "imported by unsloth even when its optimizer is unused, which is the open risk"),
    ),
}


# ── profiles ────────────────────────────────────────────────────────────────

PROFILES = {
    "local": Profile(
        name="local",
        scheduler="none",
        root_env="COB_PROFILE_ROOT",
        default_root=str(PROJECT_ROOT),
        partitions={},
        gpus_per_node=1,
        account_env="",
        setup=(),
        env={},
        notes="this machine; the project root is the root and the hugging face cache stays the user default",
        root_required=False,
    ),
    "lumi": Profile(
        name="lumi",
        scheduler="slurm",
        root_env="COB_PROFILE_ROOT",
        # LUMI's real scratch path is site-specific and cannot be verified from here, so it is
        # taken from the environment rather than guessed
        default_root="",
        partitions={"dev-g": {"max_hours": 3, "note": "development and short smoke jobs"},
                    "small-g": {"max_hours": 72, "note": "up to 3 days, part of a node"},
                    "standard-g": {"max_hours": 48, "note": "up to 2 days, whole nodes"}},
        gpus_per_node=8,
        account_env="COB_SLURM_ACCOUNT",
        # the real container module, which REPLACES singularity-AI-bindings rather than adding to
        # it: the PyTorch module declares conflict('singularity-AI-bindings'), so loading both is
        # refused. the load order matters and is not cosmetic -- $SIF is populated by
        # create_container_vars(), a LUMI stack function, so the stack has to be loaded before the
        # container module or $SIF silently stays empty and the job runs on the host python
        setup=('# lmod reaches a batch job as an exported shell function through sbatch',
               '# --export=ALL. source it explicitly if that did not happen, because the failure',
               '# otherwise is "module: command not found" several lines later',
               'if ! type module >/dev/null 2>&1; then',
               '  source "${LMOD_PKG:-/opt/cray/pe/lmod/lmod}/init/bash"',
               'fi',
               '# sbatch propagates the submitting shell\'s modules, and the PyTorch container',
               '# module declares conflict(\'singularity-AI-bindings\'). a submitter who happened to',
               '# have the bindings loaded therefore got "Cannot load module ... because these',
               '# module(s) are loaded: singularity-AI-bindings", an empty $SIF and an exit 2.',
               '# purging first makes the job independent of whatever state it was submitted from.',
               '# plain purge, not --force: it clears the bindings while leaving LUMI\'s sticky',
               '# modules in place, and --force would strip those too for no benefit here',
               'module purge',
               'export EBU_USER_PREFIX="${COB_EB_USER_PREFIX:-/project//anonymous/home/EasyBuild}"',
               'module load LUMI/25.03 partition/container',
               'module use "${EBU_USER_PREFIX}/modules/container"',
               'module load PyTorch/2.7.1-rocm-6.2.4-python-3.12-singularity-20250827',
               'if [ -z "${SIF:-}" ]; then',
               '  echo "SIF is empty: the LUMI stack must be loaded before the container module" >&2',
               '  exit 2',
               'fi'),
        # HF_HOME and HF_HUB_CACHE are already emitted by job_environment() and resolve under the
        # profile root, so only the third one is new: HF_HUB_CACHE does NOT move the xet/CAS chunk
        # cache, which defaults into $HOME and filled a 22G home quota on this machine once
        env={"HSA_OVERRIDE_GFX_VERSION": "", "MIOPEN_USER_DB_PATH": "{root}/.miopen",
             "MIOPEN_CUSTOM_CACHE_DIR": "{root}/.miopen",
             "HF_XET_CACHE": "{root}/hf-cache/xet",
             "PIP_CACHE_DIR": "{root}/pip-cache",
             "GPU_ARCHS": "gfx90a"},
        # the compiled-kernel caches, on node-local storage, resolved on the compute node
        #
        # three real variables, each read by a package in this stack and each verified in its
        # source rather than assumed:
        #   UNSLOTH_COMPILE_LOCATION   unsloth_zoo/compiler.py, the generated patched sources.
        #                              its default is the RELATIVE path "unsloth_compiled_cache",
        #                              so unset it lands in whatever the job's working directory is
        #   TORCHINDUCTOR_CACHE_DIR    torch/_inductor, the fx graph, autograd and autotune caches
        #   TRITON_CACHE_DIR           triton/knobs.py, the compiled kernel cache
        #
        # all three are loose many-small-file trees rebuilt per environment. LUMI's storage
        # guidance is explicit that many small files strain Lustre on scratch and on flash alike,
        # so none of them belongs under the profile root. TMPDIR on a compute node is node-local
        # and memory-backed, and it is read HERE rather than substituted at generation time: a
        # login node's temporary directory is not the one the job will have
        runtime_setup=(
            'export COB_JOB_CACHE="${TMPDIR:-/tmp}/cob-${SLURM_JOB_ID:-$$}"',
            'mkdir -p "${COB_JOB_CACHE}/unsloth" "${COB_JOB_CACHE}/inductor" \\',
            '         "${COB_JOB_CACHE}/triton" "${COB_JOB_CACHE}/aiter" "${COB_JOB_CACHE}/bin"',
            'export UNSLOTH_COMPILE_LOCATION="${COB_JOB_CACHE}/unsloth"',
            'export TORCHINDUCTOR_CACHE_DIR="${COB_JOB_CACHE}/inductor"',
            'export TRITON_CACHE_DIR="${COB_JOB_CACHE}/triton"',
            '# aiter jit-builds kernels at import and defaults its build tree to $HOME/.aiter',
            'export AITER_JIT_DIR="${COB_JOB_CACHE}/aiter"',
            '# node-local and memory-backed: it counts against this job\'s allocation, so it is'
            ' released when the job ends',
            'trap \'rm -rf "${COB_JOB_CACHE}"\' EXIT',
        ),
        # training runs inside the container, not merely with its module loaded. the venv was built
        # from the container's own python 3.12 and is only valid in there
        container_exec='singularity exec "$SIF" bash -s',
        container_prologue=(
            'set -euo pipefail',
            '# singularity resets PATH, so anything prepended on the host is gone by here. aiter',
            '# locates a compiler by shelling out to `which c++` literally, which HIP_CLANG_PATH',
            '# does not satisfy; the container has no c++/g++/gcc/cc on PATH but does ship a real',
            '# compiler as ROCm clang. without this the job dies inside accelerate\'s deepspeed',
            '# import with a FileNotFoundError deep in aiter\'s build tree, naming no compiler',
            'export HIP_CLANG_PATH="${ROCM_PATH:-/opt/rocm}/llvm/bin"',
            'ln -sf "${HIP_CLANG_PATH}/clang++" "${COB_JOB_CACHE}/bin/c++"',
            'ln -sf "${HIP_CLANG_PATH}/clang++" "${COB_JOB_CACHE}/bin/g++"',
            'ln -sf "${HIP_CLANG_PATH}/clang" "${COB_JOB_CACHE}/bin/cc"',
            'ln -sf "${HIP_CLANG_PATH}/clang" "${COB_JOB_CACHE}/bin/gcc"',
            'export PATH="${COB_JOB_CACHE}/bin:${PATH}"',
            'source "${COB_PROFILE_ROOT}/unsloth-env/bin/activate"',
            '# fail here rather than after minutes of setup. it runs in this same container',
            '# process, so the aiter build it triggers is the one the training run then reuses',
            'python -c "import torch, unsloth" || {',
            '  echo "venv at ${COB_PROFILE_ROOT}/unsloth-env cannot import torch+unsloth" >&2',
            '  exit 91',
            '}',
        ),
        # every base is materialized under models/base/<model> on this site, so jobs read weights
        # from there. a directory carries no hub revision, so --revision drops out with it
        base_layout="profile",
        notes="AMD MI250X (gfx90a), 8 GCDs per node, Slurm; every path resolves under COB_PROFILE_ROOT",
    ),
    "roihu": Profile(
        name="roihu",
        scheduler="slurm",
        root_env="COB_PROFILE_ROOT",
        default_root="",
        partitions={"gpu": {"max_hours": 36, "note": "placeholder until the real partitions are known"}},
        gpus_per_node=4,
        account_env="COB_SLURM_ACCOUNT",
        setup=("# module load <site python/cuda modules>",),
        env={},
        notes=("stub profile, written and exercised by the tests but never run: it exists to prove a "
               "cuda + roihu launch goes through the same shared path with no training-code change. "
               "its partitions and setup lines are placeholders and must be filled from Roihu's own "
               "documentation before use"),
    ),
}


# ── resolution ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Execution:
    backend: Backend
    profile: Profile
    root: Path
    hf_home: Path
    hf_hub_cache: Path
    models_dir: Path
    reports_dir: Path

    # ── the storage hierarchy, rooted at this profile ───────────────────────
    #
    # one place each for the pinned bases, the training runs, the selected pointers and the
    # exports, so a site is described by its root and nothing else. these are derived from
    # models_dir and add no new configuration: a profile that never asks for them is unchanged

    @property
    def base_dir(self):
        """models/base/ under this profile: the six self-contained pinned training bases"""
        return self.models_dir / "base"

    @property
    def training_dir(self):
        """models/training/ under this profile: <mode>/<model>/<run>/ per training attempt"""
        return self.models_dir / "training"

    @property
    def selected_dir(self):
        """models/selected/ under this profile: <mode>/<model>/ pointer to the chosen adapter"""
        return self.models_dir / "selected"

    @property
    def exports_dir(self):
        """models/exports/ under this profile: merged_hf/ and gguf/ named from the selection"""
        return self.models_dir / "exports"

    def base_model_dir(self, model):
        return self.base_dir / model

    def environment(self):
        """the environment a job of this shape needs, profile first then backend

        HF_HOME and HF_HUB_CACHE always come from the profile, never from a constant in the
        training code"""
        out = {"HF_HOME": str(self.hf_home), "HF_HUB_CACHE": str(self.hf_hub_cache)}
        for key, value in self.profile.env.items():
            if value:
                out[key] = value.replace("{root}", str(self.root))
        out.update({k: v for k, v in self.backend.env.items() if v})
        return out

    def job_environment(self, root_expr):
        """the same settings for a generated job script, rooted at a shell expression

        a script must not carry the absolute path of the machine it was generated on: the root is
        whatever COB_PROFILE_ROOT holds at submit time. separators stay posix here because the
        target is a linux cluster even when the script is written on windows"""
        out = {}
        if self.profile.name != "local":
            out["HF_HOME"] = f"{root_expr}/hf-cache"
            out["HF_HUB_CACHE"] = f"{root_expr}/hf-cache/hub"
        for key, value in self.profile.env.items():
            if value:
                out[key] = value.replace("{root}", root_expr)
        out.update({k: v for k, v in self.backend.env.items() if v})
        return out

    def job_paths(self, root_expr):
        """the same hierarchy for a generated job script, rooted at a shell expression

        a script must not carry the absolute path of the machine it was generated on, so these are
        strings built from root_expr rather than the resolved Path fields above. separators stay
        posix because the target is a linux cluster even when the script is written elsewhere"""
        root = str(root_expr).rstrip("/")
        return {"models": f"{root}/models", "base": f"{root}/models/base",
                "training": f"{root}/models/training", "selected": f"{root}/models/selected",
                "exports": f"{root}/models/exports"}

    def job_training_dir(self, root_expr, mode, model, run):
        """<root>/models/training/<mode>/<model>/<run>, the run root one attempt writes into

        the run segment is a real directory so the training tree lines up with the selection record
        that groups three validation results by the same run string"""
        return f"{self.job_paths(root_expr)['training']}/{mode}/{model}/{run}"

    def as_dict(self):
        return {"backend": self.backend.name, "profile": self.profile.name, "root": str(self.root),
                "hf_home": str(self.hf_home), "hf_hub_cache": str(self.hf_hub_cache),
                "models_dir": str(self.models_dir), "base_dir": str(self.base_dir),
                "training_dir": str(self.training_dir), "default_optim": self.backend.default_optim}


def resolve(backend, profile, root=None, environ=None):
    """one (backend, profile) pair into concrete settings and paths

    any backend may pair with any profile: nothing here rejects rocm+local or cuda+roihu"""
    environ = os.environ if environ is None else environ
    if backend not in BACKENDS:
        raise ExecutionError(f"unknown backend {backend!r}, expected one of {', '.join(BACKEND_CHOICES)}")
    if profile not in PROFILES:
        raise ExecutionError(f"unknown profile {profile!r}, expected one of {', '.join(PROFILE_CHOICES)}")
    spec = PROFILES[profile]
    chosen = root or environ.get(spec.root_env) or spec.default_root
    if not chosen:
        raise ExecutionError(
            f"profile {profile!r} has no root: set {spec.root_env} to the directory that holds this "
            f"profile's caches, checkpoints and reports, or pass --profile-root. it is not guessed "
            f"here because the correct path is site-specific")
    base = Path(chosen)
    if spec.name == "local":
        # report the cache this machine's tools already use, do not invent one: redirecting HF_HOME
        # here would point every existing command at an empty cache and force re-downloads
        home = environ.get("HF_HOME")
        cache = Path(home) if home else Path.home() / ".cache" / "huggingface"
        hub = Path(environ.get("HF_HUB_CACHE") or (cache / "hub"))
    else:
        cache = base / "hf-cache"
        hub = cache / "hub"
    return Execution(
        backend=BACKENDS[backend], profile=spec, root=base,
        hf_home=cache, hf_hub_cache=hub,
        models_dir=base / "models", reports_dir=base / "reports",
    )


def require(backend, profile, where, root=None):
    """fail early and say exactly what is missing

    this applies only to entry points introduced for a scheduler: everything that already runs on
    this machine keeps working with no new flags"""
    missing = [name for name, value in (("--backend", backend), ("--profile", profile)) if not value]
    if missing:
        raise ExecutionError(
            f"{where}: {' and '.join(missing)} must be given explicitly.\n"
            f"  --backend {' | '.join(BACKEND_CHOICES)}\n"
            f"  --profile {' | '.join(PROFILE_CHOICES)}\n"
            f"a real training launch never guesses its hardware; only the diagnostic probe in "
            f"src/experiments/v5/backend_smoke.py detects a backend by itself")
    return resolve(backend, profile, root=root)


def add_arguments(parser):
    """--backend and --profile with no default, so an omission is visible rather than assumed"""
    parser.add_argument("--backend", choices=BACKEND_CHOICES, default=None,
                        help="accelerator backend; required, never detected for a real launch")
    parser.add_argument("--profile", choices=PROFILE_CHOICES, default=None,
                        help="execution profile: where the job runs and which paths it uses")
    parser.add_argument("--profile-root", default="",
                        help=f"override the profile root instead of the profile's root environment variable")
    return parser


# ── hardware identity, for provenance ───────────────────────────────────────

def hardware_id(environ=None):
    """a short identifier for the machine a measurement was taken on

    every timing record carries this, so a LUMI number and a number from this machine can never sit
    in one column without the difference being visible. torch is imported lazily and its absence is
    reported rather than raised"""
    environ = os.environ if environ is None else environ
    info = {"host": environ.get("SLURMD_NODENAME") or environ.get("COMPUTERNAME") or environ.get("HOSTNAME") or "unknown",
            "slurm_job_id": environ.get("SLURM_JOB_ID")}
    try:
        import torch
        info["torch"] = torch.__version__
        info["torch_backend"] = "rocm" if getattr(torch.version, "hip", None) else ("cuda" if torch.version.cuda else "cpu")
        info["torch_backend_version"] = getattr(torch.version, "hip", None) or torch.version.cuda
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info.update({"device": props.name, "architecture": getattr(props, "gcnArchName", None),
                         "device_count": torch.cuda.device_count(),
                         "memory_gib": round(props.total_memory / 1024 ** 3, 2)})
        else:
            info["device"] = "none"
    except Exception as exc:
        info["torch"] = f"unavailable: {type(exc).__name__}"
        info["device"] = "unknown"
    return info


NO_GIT_ENV = "COB_NO_GIT"


def git_commit():
    """the commit a result was produced at, or a marker when git cannot answer

    with COB_NO_GIT=1 no git process is started at all and the answer is "unknown", the documented value
    for a commit git did not supply; the code that produced the result is then identified by file
    hashes instead (reports/v5/evaluation/provenance/code_sha256.csv)"""
    if os.environ.get(NO_GIT_ENV) == "1":
        return "unknown"
    git = shutil.which("git")
    if git is None:
        return "git-not-available"
    try:
        out = subprocess.run([git, "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
                             text=True, timeout=30)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def worktree_entries():
    """every uncommitted or untracked entry, exactly as git porcelain reports them

    returns None when git cannot answer at all, which is not the same as a clean tree and must not
    be read as one. dirty_worktree() answers its yes/no question from this same list, so there is
    one git call here rather than two that could drift apart. with COB_NO_GIT=1 no git process is
    started and the answer is None (unknown)"""
    if os.environ.get(NO_GIT_ENV) == "1":
        return None
    git = shutil.which("git")
    if git is None:
        return None
    try:
        out = subprocess.run([git, "status", "--porcelain"], cwd=PROJECT_ROOT, capture_output=True,
                             text=True, timeout=30)
        return [line.strip() for line in out.stdout.splitlines() if line.strip()]
    except Exception:
        return None


def dirty_worktree():
    entries = worktree_entries()
    return None if entries is None else bool(entries)
