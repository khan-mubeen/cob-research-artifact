"""gemma4 C2-equivalent gradient smoke using the qwen training stack

plain peft 0.19.1 cannot wrap gemma4's clippable linears, as recorded in
probe_gemma4-e2b.json; unsloth supports them. the merged bf16 output feeds the
stage E conversion test."""

import json
import os
import statistics
import time

from unsloth import FastLanguageModel

import torch
from datasets import Dataset
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer

REPO = Path(__file__).resolve().parents[3]
OUT = str(REPO / "reports" / "feasibility" / "probe_gemma4-e2b-unsloth.json")
# defaults to the retained bf16 merge produced by this smoke; set
# GEMMA_MERGE_DIR to preserve it and write a fresh merge elsewhere
MERGED = os.getenv("GEMMA_MERGE_DIR", str(
    REPO / "models" / "feasibility" / "smoke-artifacts" / "gemma4-e2b" / "merged"))
DATA = str(REPO / "data" / "finetune" / "v4")

result = {"tag": "gemma4-e2b-unsloth", "path": "unsloth (same stack as qwen training)"}

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="google/gemma-4-E2B-it", max_seq_length=1024,
    load_in_4bit=False, dtype=None)
model = FastLanguageModel.get_peft_model(
    model, r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16, lora_dropout=0, bias="none",
    use_gradient_checkpointing="unsloth", random_state=42)
result["trainable_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)

records = []
for stage in ("material", "tailoring", "logistics"):
    with open(rf"{DATA}\{stage}_train.jsonl", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 17:
                break
            records.append(json.loads(line))
records = records[:48]
texts = [tokenizer.apply_chat_template(
    [{"role": "system", "content": r["system"]},
     {"role": "user", "content": r["user"]},
     {"role": "assistant", "content": r["completion"]}],
    tokenize=False, add_generation_prompt=False) for r in records]

torch.cuda.reset_peak_memory_stats()
times, last = [], [None]


class T(TrainerCallback):
    def on_step_end(self, a, s, c, **k):
        now = time.perf_counter()
        if last[0] is not None:
            times.append(now - last[0])
        last[0] = now


trainer = SFTTrainer(
    model=model, tokenizer=tokenizer,
    train_dataset=Dataset.from_list([{"text": t} for t in texts]),
    args=SFTConfig(output_dir=str(Path(os.getenv("SMOKE_OUTPUT_DIR",
                                       str(REPO / "reports" / "feasibility" / "smoke_gemma4_unsloth")))),
                   per_device_train_batch_size=8, gradient_accumulation_steps=2,
                   max_steps=8, learning_rate=2e-4, warmup_steps=2, logging_steps=1,
                   optim="adamw_8bit", lr_scheduler_type="linear", seed=42,
                   max_seq_length=1024, dataset_text_field="text", report_to="none",
                   save_strategy="no"))
trainer.add_callback(T())
trainer.train()
losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
result.update({
    "losses": losses, "loss_changed": len(set(losses)) > 1,
    "median_step_s": round(statistics.median(times), 2) if times else None,
    "peak_train_vram_reserved_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 2),
    "batch_size_used": 8, "exact_qwen_config": True,
})

model.save_pretrained_merged(MERGED, tokenizer, save_method="merged_16bit")
result["merged_saved_to"] = MERGED

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)
print(json.dumps(result, indent=2))
