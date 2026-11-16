"""merge a probe smoke adapter into its base checkpoint and save the result,
input for the stage E conversion test. run with the .venv-feas python."""

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    base = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, dtype=torch.bfloat16,
        trust_remote_code=args.trust_remote_code)
    merged = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()
    merged.save_pretrained(args.out)
    AutoTokenizer.from_pretrained(args.model, revision=args.revision).save_pretrained(args.out)
    print(f"merged -> {args.out}")


if __name__ == "__main__":
    main()
