"""the six architectures and the exact settings each one trains with

every field here is read from the frozen protocol, not chosen now:
  checkpoints and revisions   reports/epoch-2/production/protocol/protocol_epoch2_<model>.json
  loss-masking markers        reports/epoch-*/production/gates/<model>_template_alignment.json
  fused LoRA names for phi    the same protocol file, "functionally equivalent to the direct set"
  smollm3's template kwargs   its non-thinking mode is template-controlled, so training renders it

phi and smollm3 train from a local base copy with a pad token set from an existing reserved token,
because both ship pad_token == eos_token, which unsloth rejects. those two are local directories,
so a profile that downloads models cannot fetch them from the hub by name: the download step
reports them as local and leaves them alone

the windows come from the measured v5 token lengths (reports/v5/phase6_token_lengths.json) through
src/experiments/v5/training_data.py, so this module never restates a number that is measured
elsewhere
"""

import os
from pathlib import Path

from config import PROJECT_ROOT
from src.experiments.v5.training_data import V5_WINDOWS

LOCAL_BASE = PROJECT_ROOT / "models" / "base-checkpoints" / "hf"

# the qwen-style markers finetune_lora.py already defaults to
_IM_START = ("<|im_start|>user\n", "<|im_start|>assistant\n")

MODELS = {
    "qwen": {
        "checkpoint": "Qwen/Qwen3.5-4B",
        "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "local": False,
        "instruction_part": _IM_START[0],
        "response_part": _IM_START[1],
        "lora_targets": "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        "template_kwargs": "",
        "adapter_stem": "qwen3.5-4b",
        "notes": "multimodal; unsloth targets the language tower only",
    },
    "granite": {
        "checkpoint": "ibm-granite/granite-4.1-3b",
        "revision": "c0650403e44e78ec0262dab1c90914c65b196c4e",
        "local": False,
        "instruction_part": "<|start_of_role|>user<|end_of_role|>",
        "response_part": "<|start_of_role|>assistant<|end_of_role|>",
        "lora_targets": "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        "template_kwargs": "",
        "adapter_stem": "granite-4.1-3b",
        "notes": "text-only; the one model the plain peft path can also train",
    },
    "phi": {
        "checkpoint": str(LOCAL_BASE / "phi4mini-v2-base"),
        "revision": "cfbefacb99257ffa30c83adab238a50856ac3083",
        "local": True,
        "hub_source": "microsoft/Phi-4-mini-instruct",
        "instruction_part": "<|user|>",
        "response_part": "<|assistant|>",
        # phi3 fuses q/k/v and gate/up, so the fused names are the functional equivalent
        "lora_targets": "qkv_proj,o_proj,gate_up_proj,down_proj",
        "template_kwargs": "",
        "adapter_stem": "phi4mini",
        "notes": "local base copy with pad_token <|endofprompt|>; reserved memory exceeded physical on this machine",
    },
    "ministral": {
        "checkpoint": "mistralai/Ministral-3-3B-Instruct-2512-BF16",
        "revision": "b6d637bef2393152b3da2b2fde72eecdee30557e",
        "local": False,
        "instruction_part": "[INST]",
        "response_part": "[/INST]",
        "lora_targets": "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        "template_kwargs": "",
        "adapter_stem": "ministral3",
        "notes": "BF16 repo, not the default FP8 one; vision tower shares projection names and stays frozen",
    },
    "gemma": {
        "checkpoint": "google/gemma-4-E2B-it",
        "revision": "179516f0c449474fdc46f08f30ead5b11e178497",
        "local": False,
        "instruction_part": "<|turn>user",
        "response_part": "<|turn>model",
        "lora_targets": "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        "template_kwargs": "",
        "adapter_stem": "gemma4-e2b",
        "notes": "unsloth only: plain peft cannot wrap its clippable linears, so it has no fallback path",
    },
    "smollm3": {
        "checkpoint": str(LOCAL_BASE / "smollm3-v2-base"),
        "revision": "a07cc9a04f16550a088caea529712d1d335b0ac1",
        "local": True,
        "hub_source": "HuggingFaceTB/SmolLM3-3B",
        "instruction_part": _IM_START[0],
        "response_part": _IM_START[1],
        "lora_targets": "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        "template_kwargs": '{"enable_thinking": false}',
        "adapter_stem": "smollm3",
        "notes": "local base copy with pad_token <|reserved_special_token_2|>; thinking mode is template-controlled",
    },
}

# the frozen hyperparameters every model shares, from protocol_v2 and the epoch-2 protocol files
FROZEN = {
    "seed": 42,
    "lora_rank": 16,
    "lora_alpha": 16,
    "lora_dropout": 0,
    "learning_rate": 2e-4,
    "warmup_steps": 5,
    "weight_decay": 0.01,
    "lr_scheduler": "linear",
    "per_device_batch_size": 8,
    "grad_accum": 2,
    "effective_batch": 16,
    "save_steps": 250,
    "precision": "bf16, unsloth gradient checkpointing",
}

MODES = {
    "sealed": {"dataset": "v5_sealed", "window": V5_WINDOWS["v5_sealed"]},
    "reverse_auction": {"dataset": "v5_reverse_auction", "window": V5_WINDOWS["v5_reverse_auction"]},
    "supervisor": {"dataset": "v5_supervisor", "window": V5_WINDOWS["v5_supervisor"]},
}


def model(name):
    try:
        return MODELS[name]
    except KeyError:
        raise ValueError(f"unknown model {name!r}, expected one of {', '.join(sorted(MODELS))}") from None


def checkpoint_for(name, root_expr=None):
    """where this model's weights are, optionally rooted at another site

    the two local copies live under models/base-checkpoints/hf/ on this machine; a generated job
    script for another site must name that site's own copy instead of this machine's absolute path,
    which is what root_expr is for"""
    spec = model(name)
    if spec["local"] and root_expr:
        return f"{root_expr}/models/base-checkpoints/hf/{Path(spec['checkpoint']).name}"
    return spec["checkpoint"]


# ── the profile-rooted base tree ────────────────────────────────────────────
#
# models/base/<model>/ is the storage hierarchy's one place for all six pinned training bases: a
# self-contained directory per architecture, hub snapshot or derived local copy alike. it is a
# third, separate field, not a replacement: spec["checkpoint"] stays exactly what it was, the hub
# source and revision stay the pins, and nothing here decides that a run must use this layout.
# choosing it is one explicit flag, so a job generated before the tree is materialized still names
# the repository it always named

BASE_DIRNAME = "base"
BASE_LAYOUT_CHOICES = ("registry", "profile")
BASE_LAYOUT_DEFAULT = "registry"


def base_dir(name, root_expr=None):
    """this model's directory in the profile-rooted base tree

    root_expr is a site root, either a real path or a shell expression a generated script expands
    at submit time. separators stay posix: the target is a linux cluster even when the generator
    runs elsewhere"""
    model(name)
    root = str(root_expr) if root_expr else str(PROJECT_ROOT)
    return f"{root.rstrip('/')}/models/{BASE_DIRNAME}/{name}"


def base_source(name):
    """where this model's base directory has to come from, for a downloader or an operator

    a hub model names its repository and revision, a derived local copy names the upstream it was
    derived from and the directory the pad-token fix already lives in. one shape for both, so a
    caller never branches on spec["local"] again"""
    spec = model(name)
    if spec["local"]:
        return {"kind": "local_derived_copy", "hub_source": spec.get("hub_source"),
                "revision": spec["revision"], "derive_from": spec["checkpoint"]}
    return {"kind": "hub_revision", "hub_source": spec["checkpoint"],
            "revision": spec["revision"], "derive_from": None}


def resolve_base_root(root=None, environ=None):
    """the root the base tree hangs under, for a command with no execution profile of its own

    evaluate.py and inference_profile.py name one model, not a site, so they read the same
    COB_PROFILE_ROOT a generated job would expand and fall back to this checkout. a profile that
    has already been resolved should pass its own root instead of calling this"""
    environ = os.environ if environ is None else environ
    return root or environ.get("COB_PROFILE_ROOT") or str(PROJECT_ROOT)


def training_path(name, root_expr=None, layout=BASE_LAYOUT_DEFAULT):
    """the weights path for one model under the requested base layout

      registry   spec["checkpoint"] as it always was: a hub id, or a local derived directory
      profile    the profile-rooted models/base/<model> directory

    registry is the default everywhere, so every existing caller and every already generated job
    resolves to exactly the string it resolved to before"""
    if layout not in BASE_LAYOUT_CHOICES:
        raise ValueError(f"unknown base layout {layout!r}, expected one of {', '.join(BASE_LAYOUT_CHOICES)}")
    if layout == "profile":
        return base_dir(name, root_expr)
    return checkpoint_for(name, root_expr)


def pins_revision(name, layout=BASE_LAYOUT_DEFAULT):
    """does a run against this layout still pass --revision

    a hub id is pinned by revision; a directory on disk is not, which is why the two local copies
    never carried one. the base tree is directories for all six, so it pins none of them: the
    revision that produced it is recorded in the freeze record instead"""
    return layout == "registry" and not model(name)["local"]


def train_flags(name, mode, root_expr=None, base_layout=BASE_LAYOUT_DEFAULT):
    """the finetune_lora.py flags this model and mode need, protocol values only

    returned as a list so a launcher can print it verbatim into a job script and a reader can check
    it against the protocol line by line. base_layout picks which base path the run reads and
    defaults to the registry one, so an omitted argument renders the identical flag list"""
    spec = model(name)
    settings = MODES[mode]
    flags = ["--dataset", settings["dataset"],
             "--max-seq-length", str(settings["window"]),
             "--batch-size", str(FROZEN["per_device_batch_size"]),
             "--grad-accum", str(FROZEN["grad_accum"]),
             "--base-model", training_path(name, root_expr, layout=base_layout),
             "--instruction-part", spec["instruction_part"],
             "--response-part", spec["response_part"],
             "--lora-targets", spec["lora_targets"],
             "--save-steps", str(FROZEN["save_steps"])]
    if spec["revision"] and pins_revision(name, base_layout):
        flags += ["--revision", spec["revision"]]
    if spec["template_kwargs"]:
        flags += ["--template-kwargs", spec["template_kwargs"]]
    return flags
