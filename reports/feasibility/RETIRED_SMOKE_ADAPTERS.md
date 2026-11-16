# Retired smoke artifacts — what they proved, and their retention status

Smoke artifacts are not scratch files. Each one is cited as evidence that a specific
gate passed **before** real Protocol-v2 training began. This file preserves that
evidence in text so the finding stays checkable after the weights themselves are gone
or while they are held pending review.

Every artifact named here was SHA-256 hashed and enumerated **before** any removal.
The complete per-file record (path, byte size, SHA-256) is in
`reports/feasibility/retired_smoke_hashes.json`. Relocation of every file is recorded
in `reports/relocation_manifest.jsonl`.

---

## Part 1 — LoRA smoke adapters (weights REMOVED)

These three adapters each proved one thing: that LoRA could be attached to the
architecture and that gradients actually flowed. Their scientific content is the
trainable-parameter count and the loss trajectory, both reproduced below and both
independently preserved in the retained probe records
(`reports/feasibility/probe_<model>.json`). The weights themselves were superseded
by the real Protocol-v2 adapter for each architecture and have been deleted.

### `adapter_granite41-3b/` — removed

Proved that LoRA attaches cleanly to `ibm-granite/granite-4.1-3b` using the strict
Qwen target-name set with no adaptation at all: **31,129,600 trainable parameters
(0.907% of total)**. The eight-step gradient smoke moved the loss monotonically from
**5.411 to 2.034**, confirming gradients flowed to the adapted projections. Granite was
the only candidate whose module names matched the reference set exactly. Superseded by
`models/epoch-1/production/granite-4.1-3b-bidding-lora-v2/`; the 3 files (118.8 MiB)
were hash-recorded and removed.

### `adapter_phi4mini/` — removed

Proved that LoRA attaches to `microsoft/Phi-4-mini-instruct` **only via functionally
equivalent fused target names** (`qkv_proj`, `gate_up_proj`), yielding **23,068,672
trainable parameters (0.598%)**. This was the decisive measurement behind the disclosed
Phi adaptation: the strict Qwen names would have silently trained just 8.9 M parameters
instead. Loss moved across the smoke run, confirming gradient flow. Superseded by
`models/epoch-1/production/phi4mini-bidding-lora-v2/`; the 3 files (88.0 MiB) were
hash-recorded and removed.

### `adapter_lfm25-8b-a1b/` — removed

Proved the negative result that de-scoped LFM2.5-8B-A1B. LoRA attached and the loss
moved (3.067 → 2.341), but reached only **5,701,632 trainable parameters — 0.067% of
the model**, because 91.5% of LFM's parameters sit in fused 3-D expert tensors
(`Lfm2MoeExperts.gate_up_proj` / `down_proj`) that standard `nn.Linear` LoRA cannot
address. This measurement, not an assumption, is why LFM2.5 was judged unable to
participate in a fair Protocol-v2 comparison. There is no superseding adapter because
the model was de-scoped; the 3 files (21.8 MiB) were hash-recorded and removed.

### `models/feasibility/smoke-artifacts/granite41-3b-adapter-smoke/` — removed

The eight-step Granite training smoke run (12 files, 186.6 MiB), originally at
`models/finetune/granite-4.1-3b-bidding-lora-smoke/`. Plan §6.6 cleared it against all
three deletion tests: no script reference, no active target, isolated subtree. Its gate
evidence survives in the retained Granite launch-gate and probe records. Hash-recorded
and removed.

---

## Part 2 — Template-alignment merges and GGUFs (RETAINED, delete-hold)

**These are not deleted.** Plan §6.2–§6.4 marks them DELETE-BLOCKED because the
production launch gates cite these exact files as what template alignment was verified
against. They were isolated into `models/feasibility/smoke-artifacts/<model>/` and the
deletion decision is deliberately left to a separate, later task once this retention
note can be reviewed.

What the template-alignment check compared, in each case: the **training-time render**
produced by `tokenizer.apply_chat_template` on a real bidding example against the
**inference-time render** produced by the chat template embedded in that model's
converted GGUF. It asserts seven properties — that the generation prompt is an exact
prefix of the training text, that the assistant content follows that prefix, that the
GGUF-embedded template renders identically to the HF template, and that the
instruction/response markers each occur exactly once. This is the check that prevents
training a model on a different assistant-turn boundary than deployment uses.

| Isolated artifact | Gate that cites it | Result |
|---|---|---|
| `smoke-artifacts/granite41-3b/` (`merged/`, bf16, Q8_0) | `granite_template_alignment.json` → cited by `granite_launch_gates.json` | **PASS**, 7/7 checks |
| `smoke-artifacts/phi4mini/` (`merged/`, bf16, Q8_0) | `phi_template_alignment.json` | **PASS**, 7/7 checks |
| `smoke-artifacts/gemma4-e2b/` (`merged/`, bf16, Q8_0) | `gemma_template_alignment.json` | **PASS**, 7/7 (GGUF template text differs but renders identically) |

Before any future deletion, re-run the plan's three-part test.

**Reference status updated 2026-07-25/26.** The two script references named here have been
resolved, and a third, stronger reference has appeared:

- `template_check.py` **no longer names any artifact.** Its CLI example carried the live
  Granite Q8 path, which made a documentation string into an executable dependency
  (`--gguf` is required and is read at run time). The example now uses placeholders, and
  the docstring records that the check reads only the GGUF's embedded
  `tokenizer.chat_template` — so a fresh conversion of the base checkpoint satisfies it
  just as well as an archived one. **No specific archived artifact is required.**
- `gemma_unsloth_smoke.py` **no longer hardcodes an absolute path.** Its `MERGED` target
  was `.\models\feasibility\gemma4-e2b-merged` — both absolute and stale
  after the reorganization. It now derives from the repo root, defaults to the retained
  merge's real location, and is overridable via `GEMMA_MERGE_DIR`.
- **New:** the three `*_template_alignment.json` gate records now point *into*
  `models/feasibility/smoke-artifacts/`. Their `gguf` fields were stale pre-reorganization
  paths and were repaired on 2026-07-25; the original strings are preserved alongside as
  `gguf_at_check_time`. These artifacts are therefore now **live evidence targets of
  passing launch gates**, which is a stronger reason to retain them than the script
  references that previously blocked deletion.

Clearing those two script references does **not** authorise deletion and does not decide
anything. It only means the blockers recorded above are no longer the reason to keep
these artifacts. Deletion remains a separate, explicit decision.

---

## Not smoke artifacts — never treat as such

Qwen r1 / r2 / r3, Granite v1, and the quarantined broken Qwen conversion are completed
experiments and a documented failure. They are cited comparison evidence, not temporary
files, regardless of age. See `models/epoch-1/ablation/` and
`models/epoch-1/deployment/quarantine/`.
