"""launch gate: chat-template alignment between training render and gguf
inference render for a candidate model, the same check qwen got before its
round-one training. run before training.

three-way comparison on a real bidding example:
1. training text: apply_chat_template(system+user+assistant) - what the
   trainer masks and supervises
2. generation prompt: apply_chat_template(system+user, add_generation_prompt)
   - must be an exact prefix of (1) up to the assistant content
3. the chat template embedded in the converted gguf - llama-cpp-python's
   jinja formatter uses exactly this string, so it must render (2) identically

also asserts the train_on_responses_only markers occur exactly once per turn
in the training text, so the loss mask lands where intended.

--gguf may be any conversion of the candidate that carries the chat template
the deployment runtime will use; the check reads only the embedded
tokenizer.chat_template field, so a fresh conversion of the base checkpoint is
as valid as an archived one. no specific archived artifact is required.

usage (.venv-feas python), placeholders in <>:
  python src/experiments/cross_model/template_check.py --model <hf-repo-or-local-path> \
      --revision <commit-sha> \
      --gguf <path/to/candidate.gguf> \
      --instruction-part "<|start_of_role|>user<|end_of_role|>" \
      --response-part "<|start_of_role|>assistant<|end_of_role|>" \
      --out <reports/.../gates/NAME_template_alignment.json>

the instruction and response parts above are granite-4.1 role markers; each
template family uses its own markers.
"""

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(os.getenv("LLAMA_TOOLS_ROOT", r"tools")) /
                       "llama.cpp-e3546c79" / "gguf-py"))

from gguf import GGUFReader  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--instruction-part", required=True)
    parser.add_argument("--response-part", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--template-kwargs", default="",
                        help='JSON kwargs for apply_chat_template on BOTH renders, e.g. '
                             '\'{"enable_thinking": false}\'. REQUIRED for reasoning-capable '
                             'templates (qwen, smollm3, ...): without it the generation prompt '
                             'renders in THINKING mode (open <think>) while the training text '
                             'renders the closed empty block, so the two cannot match and the gate '
                             'reports a FAIL that is an artefact of the invocation, not a real '
                             'misalignment. The script detects this case and says so.')
    parser.add_argument("--allow-think-suffix", action="store_true",
                        help="accept a pre-closed empty <think> block between response_part and "
                             "assistant content (the qwen/smollm3 non-thinking pattern); the "
                             "exact-prefix property is still enforced")
    args = parser.parse_args()
    tk = json.loads(args.template_kwargs) if args.template_kwargs else {}

    with open(REPO / "data" / "finetune" / "v4" / "material_train.jsonl",
              encoding="utf-8") as f:
        rec = json.loads(f.readline())
    messages = [{"role": "system", "content": rec["system"]},
                {"role": "user", "content": rec["user"]}]
    full = messages + [{"role": "assistant", "content": rec["completion"]}]

    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)

    # a reasoning-capable template branches on enable_thinking. with the flag
    # unset it defaults to thinking mode for a generation prompt but emits the
    # closed empty block for a completed assistant turn, so the prefix check
    # cannot pass however well-aligned the model actually is. warn loudly rather
    # than silently reporting a FAIL the operator will read as a real defect.
    thinking_capable = "enable_thinking" in (tok.chat_template or "")
    thinking_kwarg_missing = thinking_capable and "enable_thinking" not in tk
    if thinking_kwarg_missing:
        print("WARNING: this template branches on `enable_thinking` but no --template-kwargs was "
              "given.\n         The generation prompt will render in THINKING mode while the "
              "training text\n         renders the closed block, which fails the prefix check for "
              "reasons that have\n         nothing to do with alignment. Re-run with "
              "--template-kwargs '{\"enable_thinking\": false}'\n         if the deployed path uses "
              "non-thinking decoding (all bidding evals do).\n", flush=True)

    train_text = tok.apply_chat_template(full, tokenize=False, add_generation_prompt=False, **tk)
    gen_prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **tk)

    r = GGUFReader(args.gguf)
    gguf_template = r.fields["tokenizer.chat_template"].contents()
    gguf_gen_prompt = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, chat_template=gguf_template, **tk)

    checks = {
        "gen_prompt_is_prefix_of_train_text": train_text.startswith(gen_prompt),
        "assistant_content_follows_prefix": train_text[len(gen_prompt):].startswith(
            rec["completion"][:40]),
        "gguf_template_equals_hf_template": gguf_template == tok.chat_template,
        "gguf_render_equals_hf_render": gguf_gen_prompt == gen_prompt,
        "instruction_part_count_in_train_text": train_text.count(args.instruction_part),
        "response_part_count_in_train_text": train_text.count(args.response_part),
        "response_part_ends_gen_prompt": gen_prompt.endswith(args.response_part)
                                         or gen_prompt.rstrip("\n").endswith(args.response_part)
                                         or (args.allow_think_suffix and args.response_part in gen_prompt
                                             and set(gen_prompt.split(args.response_part, 1)[1]
                                                     .replace("<think>", "").replace("</think>", "")) <= {"\n", " "}),
    }
    passed = (checks["gen_prompt_is_prefix_of_train_text"]
              and checks["assistant_content_follows_prefix"]
              and checks["gguf_render_equals_hf_render"]
              and checks["instruction_part_count_in_train_text"] == 1
              and checks["response_part_count_in_train_text"] == 1
              and checks["response_part_ends_gen_prompt"])

    result = {
        "model": args.model, "revision": args.revision, "gguf": args.gguf,
        "instruction_part": args.instruction_part, "response_part": args.response_part,
        "template_kwargs": tk, "thinking_capable_template": thinking_capable,
        "checks": checks, "gate": "PASS" if passed else "FAIL",
        "gen_prompt_tail": gen_prompt[-120:],
        "train_text_around_boundary": train_text[len(gen_prompt) - 60:len(gen_prompt) + 60],
    }
    if not passed and thinking_kwarg_missing:
        result["likely_cause"] = (
            "PROBABLE FALSE FAILURE: this template branches on `enable_thinking` and no "
            "--template-kwargs was supplied, so the generation prompt rendered in thinking mode "
            "while the training text rendered the closed block. Re-run with "
            "'{\"enable_thinking\": false}' before treating this as a real misalignment.")
        print("\n" + result["likely_cause"] + "\n", flush=True)
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
