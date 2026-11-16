"""rewrite a merged vision-language checkpoint as its text tower alone

llama.cpp's mistral3 handler registers `Ministral3ForCausalLM` and dispatches to a
LlamaModel subclass, so it wants a plain text-only checkpoint. what transformers
writes for `Mistral3ForConditionalGeneration` is triple nested,
`model.language_model.model.layers.0...`, alongside 218 vision tensors, and the
converter cannot map it:

    ValueError: Can not map tensor 'model.model.embed_tokens.weight'

this keeps the language tower, strips the wrapper level so keys land in the
`model.layers.0...` shape a llama-style loader expects, drops the vision tower and
projector, and writes a config from the base's text_config. the rope parameters
the converter asserts on (rope_type yarn, mscale_all_dim, llama_4_scaling_beta)
are carried across unchanged.

usage:
  python -m src.experiments.dynamic_resources.flatten_text_tower <merged_dir> <out_dir> [base_dir]
"""

import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

# everything under these belongs to the vision path and has no place in a text gguf
DROP_PREFIXES = ("model.vision_tower.", "vision_tower.", "model.multi_modal_projector.",
                 "multi_modal_projector.")
# tokenizer artifacts the converter reads
CARRY = ("tokenizer.json", "tokenizer_config.json", "tokenizer.model", "vocab.json",
         "merges.txt", "added_tokens.json", "special_tokens_map.json",
         "chat_template.jinja", "generation_config.json")


def flatten_key(key):
    """wrapper key -> the text-only key a llama-style converter maps

    model.language_model.model.layers.0.x  ->  model.layers.0.x
    model.language_model.lm_head.weight    ->  lm_head.weight"""
    for prefix in ("model.language_model.", "language_model."):
        if key.startswith(prefix):
            rest = key[len(prefix):]
            if rest.startswith("model."):
                return "model." + rest[len("model."):]
            if rest.startswith("lm_head"):
                return rest
            return "model." + rest
    if key.startswith("lm_head"):
        return key
    return None


def carry_from_base(base, out):
    """take tokenizer artifacts from the base, not the merged copy

    a lora merge never touches the tokenizer, and transformers 5.5
    save_pretrained mangles it on the way out: it wrote added_tokens_decoder as
    null where the base has 1000 entries, and LlamaTokenizer then calls
    add_tokens(None) and raises
    `TypeError: Input must be a List[Union[str, AddedToken]]`.
    copying the base's files verbatim sidesteps the whole round trip"""
    if base is None or not base.is_dir():
        return
    taken = []
    for name in CARRY + ("tekken.json",):
        src = base / name
        if src.is_file():
            shutil.copy2(src, out / name)
            taken.append(name)
    if taken:
        print(f"tokenizer taken from base: {', '.join(taken)}")


def main(merged, out, base=None):
    merged, out = Path(merged), Path(out)
    base = Path(base) if base else None
    out.mkdir(parents=True, exist_ok=True)

    tensors, dropped, unmapped = {}, 0, []
    for shard in sorted(merged.glob("*.safetensors")):
        for key, value in load_file(str(shard)).items():
            if key.startswith(DROP_PREFIXES):
                dropped += 1
                continue
            flat = flatten_key(key)
            if flat is None:
                unmapped.append(key)
                continue
            tensors[flat] = value
    if unmapped:
        print(f"unmapped keys ({len(unmapped)}), first few: {unmapped[:5]}")
    print(f"kept {len(tensors)} text tensors, dropped {dropped} vision/projector")

    cfg = json.loads((merged / "config.json").read_text(encoding="utf-8"))
    text = cfg.get("text_config", cfg)
    text = dict(text)
    text["architectures"] = ["Ministral3ForCausalLM"]
    text.setdefault("model_type", "ministral3")
    for inherited in ("dtype", "torch_dtype", "transformers_version"):
        if inherited in cfg and inherited not in text:
            text[inherited] = cfg[inherited]
    # a tied lm_head is normal here; record it so the converter does not expect one
    if not any(k.startswith("lm_head") for k in tensors):
        text["tie_word_embeddings"] = True
        print("no lm_head tensor, marking tie_word_embeddings")
    (out / "config.json").write_text(json.dumps(text, indent=2), encoding="utf-8")

    save_file(tensors, str(out / "model.safetensors"), metadata={"format": "pt"})
    for name in CARRY:
        src = merged / name
        if src.is_file():
            shutil.copy2(src, out / name)
    carry_from_base(base, out)
    print(f"wrote {out}")
    print(f"  layers {text.get('num_hidden_layers')}  rope "
          f"{(text.get('rope_parameters') or {}).get('rope_type')}")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
