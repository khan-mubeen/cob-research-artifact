"""
aggregate all feasibility evidence into the machine-readable matrix.

sources: probe_*.json (stages B-D, executed), convert_*.json (stage E,
executed), plus the stage A documentation table pinned inline with sources.
qwen's B-D row cites the executed fine-tuning runs rather than the probe
script, its stage E row comes from convert_qwen35-r1.json.

outputs: reports/feasibility/feasibility_matrix.json / .csv
"""

import csv
import json
from datetime import datetime
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parents[3] / "reports" / "feasibility"

STAGE_A = {
    "qwen35-4b": {
        "checkpoint": "Qwen/Qwen3.5-4B", "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "instruct": True, "params_total": 4_560_499_200, "params_active": None,
        "params_note": "multimodal checkpoint, language tower trained; 4.56B stored",
        "arch": "Qwen3_5ForConditionalGeneration (hybrid linear-attention, multimodal)",
        "trust_remote_code": False, "gated": False, "access_verified": "downloaded + trained (July runs)",
        "license": "Apache-2.0",
        "thinking": "template kwarg enable_thinking=False, verified matches gguf render",
    },
    "phi4mini": {
        "checkpoint": "microsoft/Phi-4-mini-instruct", "revision": "cfbefacb99257ffa30c83adab238a50856ac3083",
        "instruct": True, "params_total": 3_836_021_760, "params_active": None, "params_note": "dense",
        "arch": "Phi3ForCausalLM (dense, fused qkv/gate_up projections)",
        "trust_remote_code": False, "gated": False, "access_verified": "downloaded 7.17 GiB",
        "license": "MIT",
        "thinking": "not a reasoning model, no controls needed (verified: all kwargs ignored)",
    },
    "granite41-3b": {
        "checkpoint": "ibm-granite/granite-4.1-3b", "revision": "c0650403e44e78ec0262dab1c90914c65b196c4e",
        "instruct": True, "params_total": 3_400_000_000, "params_active": None, "params_note": "dense, card-stated 3.4B",
        "arch": "GraniteForCausalLM (dense)",
        "trust_remote_code": False, "gated": False, "access_verified": "downloaded 6.35 GiB",
        "license": "Apache-2.0",
        "thinking": "no reasoning modes (model card)",
    },
    "nemotron3-nano": {
        "checkpoint": "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16", "revision": "dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f",
        "instruct": True, "params_total": 3_973_556_832, "params_active": None,
        "params_note": "hybrid mamba2+attention (nemotron_h)",
        "arch": "NemotronHForCausalLM (custom modeling files in repo)",
        "trust_remote_code": "required but still fails (mamba-ssm)", "gated": False,
        "access_verified": "downloaded 7.42 GiB",
        "license": "NVIDIA Open Model License (repo LICENSE file is 0 bytes; terms on nvidia.com)",
        "thinking": "reasoning on/off via <think> in template + reasoning parser (docs, NOT executed here)",
    },
    "gemma4-e2b": {
        "checkpoint": "google/gemma-4-E2B-it", "revision": "179516f0c449474fdc46f08f30ead5b11e178497",
        "instruct": True, "params_total": 5_123_178_051, "params_active": 2_300_000_000,
        "params_note": "stored 5.1B vs EFFECTIVE 2.3B (per-layer embeddings), multimodal any-to-any",
        "arch": "Gemma4ForConditionalGeneration",
        "trust_remote_code": False, "gated": False, "access_verified": "downloaded 9.57 GiB",
        "license": "Apache-2.0 (gemma 4 dropped the gemma terms gate, verified by ungated download)",
        "thinking": "thinking enabled by <|think|> token in system prompt, disabled by omitting it (docs)",
    },
    "lfm25-8b-a1b": {
        "checkpoint": "LiquidAI/LFM2.5-8B-A1B", "revision": "5673e0de372b64331504de73bbbc33b0dde71903",
        "instruct": True, "params_total": 8_467_856_832, "params_active": 1_500_000_000,
        "params_note": "MoE 32 experts / 4 active, 8.3B total / 1.5B active (card)",
        "arch": "Lfm2MoeForCausalLM",
        "trust_remote_code": False, "gated": False, "access_verified": "downloaded 15.79 GiB",
        "license": "LFM Open License v1.0 ($10M revenue threshold on commercial use; research permitted)",
        "thinking": "reasoning ALWAYS ON, no official disable (card + verified: all kwargs ignored, "
                    "<think> consumed full budget in probe)",
    },
}

# executed qwen references from the july training/eval runs (same gpu)
QWEN_EXECUTED = {
    "B_json_ok": "3/3 equivalent (829/829 distinct real generations, slm_check)",
    "C_trainable_params": 21_233_664, "C_step_s": 6.6, "C_peak_vram_gib": "10.3 (allocated, fits natively)",
    "C_exact_config": True,
}


def load(pattern):
    files = sorted(OUT_DIR.glob(pattern))
    return json.load(open(files[-1], encoding="utf-8")) if files else None


def main():
    rows = {}
    for tag in STAGE_A:
        rows[tag] = {"stage_a": STAGE_A[tag]}

    for tag in ("phi4mini", "granite41-3b", "nemotron3-nano", "gemma4-e2b", "lfm25-8b-a1b"):
        probe = load(f"probe_{tag}.json")
        if probe:
            rows[tag]["stages_bcd"] = {
                name: {k: v for k, v in step.items()
                       if k in ("status", "error", "trust_remote_code", "load_seconds",
                                "load_vram_gib", "json_ok_count", "trainable_params",
                                "strict_trainable_params", "batch_size_used", "exact_qwen_config",
                                "median_step_s", "peak_train_vram_gib", "loss_changed",
                                "merge_ok", "oom_note", "thinking_controls")}
                for name, step in probe["steps"].items()}
    gemma_unsloth = load("probe_gemma4-e2b-unsloth.json")
    if gemma_unsloth:
        rows["gemma4-e2b"]["unsloth_smoke"] = gemma_unsloth

    for tag, key in (("qwen35-r1", "qwen35-4b"), ("granite41-3b", "granite41-3b"),
                     ("phi4mini", "phi4mini"), ("gemma4-e2b", "gemma4-e2b")):
        conv = load(f"convert_{tag}.json")
        if conv:
            rows[key]["stage_e"] = conv
    rows["qwen35-4b"]["executed_reference"] = QWEN_EXECUTED

    payload = {"generated_at": datetime.now().isoformat(timespec="seconds"),
               "bidding_window_seconds_current": 10,
               "runtime": "llama-cpp-python 0.3.34 cu132 (project runtime)",
               "convert_tooling": "llama.cpp@571d0d540df04f25298d0e159e520d9fc62ed121 + b10068 quantize",
               "models": rows}
    with open(OUT_DIR / "feasibility_matrix.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    fields = ["model", "revision", "license", "gated", "trust_remote_code", "params_total",
              "params_active", "loads", "json_smoke", "lora_attach", "gradient_smoke",
              "exact_qwen_config", "median_step_s", "merge", "gguf_convert", "runtime_load",
              "runtime_json", "steady_latency_s"]
    with open(OUT_DIR / "feasibility_matrix.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for tag, row in rows.items():
            a = row["stage_a"]
            bcd = row.get("stages_bcd", {})
            e = row.get("stage_e", {})
            rt = e.get("runtime_test", {})
            uns = row.get("unsloth_smoke", {})

            def st(step, key="status"):
                return bcd.get(step, {}).get(key, "")

            grad = st("C2_gradient_smoke") or ("pass(unsloth)" if uns.get("loss_changed") else "")
            w.writerow({
                "model": a["checkpoint"], "revision": a["revision"][:8], "license": a["license"],
                "gated": a["gated"], "trust_remote_code": a["trust_remote_code"],
                "params_total": a["params_total"], "params_active": a["params_active"],
                "loads": "executed" if tag == "qwen35-4b" else st("B2_model_load"),
                "json_smoke": QWEN_EXECUTED["B_json_ok"] if tag == "qwen35-4b"
                              else f"{st('B4_json_inference', 'json_ok_count')}/3",
                "lora_attach": "executed" if tag == "qwen35-4b" else
                               (st("C1_lora_attach") if st("C1_lora_attach") == "pass"
                                else ("pass(unsloth-only)" if uns else st("C1_lora_attach"))),
                "gradient_smoke": "executed" if tag == "qwen35-4b" else grad,
                "exact_qwen_config": True if tag == "qwen35-4b" else
                                     bcd.get("C2_gradient_smoke", {}).get("exact_qwen_config",
                                                                          uns.get("exact_qwen_config", "")),
                "median_step_s": QWEN_EXECUTED["C_step_s"] if tag == "qwen35-4b" else
                                 bcd.get("C2_gradient_smoke", {}).get("median_step_s",
                                                                      uns.get("median_step_s", "")),
                "merge": "executed" if tag == "qwen35-4b" and e else st("C3_adapter_save_reload_merge") or
                         ("pass" if uns.get("merged_saved_to") else ""),
                "gguf_convert": ("pass" if e.get("convert_rc") == 0 else
                                 ("fail" if e.get("convert_rc") is not None else "not_run")),
                "runtime_load": ("pass" if e.get("runtime_load_rc") == 0 else
                                 ("fail" if e.get("runtime_load_rc") is not None else "not_run")),
                "runtime_json": rt.get("json_ok", ""),
                "steady_latency_s": rt.get("median_steady_s", ""),
            })
    print("written feasibility_matrix.json / .csv")


if __name__ == "__main__":
    main()
