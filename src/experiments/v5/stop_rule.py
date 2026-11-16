"""the v5 evaluation generation stop rule: also stop on the chat template's end-of-turn token

why it exists: the qwen reverse_auction adapters end every answer with <|im_end|> (248046), the
token the chat template writes after an assistant turn and the one in their training labels, but
the checkpoint's generation config stops only on <|endoftext|> (248044). generation therefore ran
past the answer, role-played a new turn and repeated the JSON until the 200-token cap, and the
evaluator's parser (first "{" to last "}") rejected the doubled object. see
prompt-reports/qwen_reverse_auction_stop_diagnosis_20261001.md for the diagnosis

the rule: the end-of-turn id is read from the chat template itself (the first special token the
template writes after an assistant message), and it is added to the generation stop ids only when
the model's generation config does not already stop on it. when it is already a stop id, or the
cell is outside SCOPE, stop_ids() returns None and the caller passes nothing to generate(), so the
generation call is byte-for-byte the call made before this rule existed

used by the v5 paths only (evaluate.py, evaluate_sharded.py, evaluate_descriptive.py,
inference_profile.py). v4 callers of eval_finetune.generate_all and eval_reverse_auction.
generate_all never pass stop ids, so v4 generation is unchanged
"""

import re

# None would apply the rule to every cell; a set of (model, mode) pairs restricts it. set from the phase 3
# harmlessness check (validation, 300 rows per cell, selected epoch, plus the qwen base arm per mode):
#   15 non-qwen cells   rule adds no stop id (end-of-turn already a stop id): outputs identical
#   qwen sealed, qwen supervisor (selected epoch and base), qwen reverse_auction base
#                       300/300 outputs differ, in every stored example ONLY by a trailing newline: without
#                       the rule the model writes <|im_end|>\n and stops on <|endoftext|>; with it, generation
#                       stops at <|im_end|> before the newline. the json and decisions are unchanged
# the task requires identical outputs outside the cell being fixed, so the rule is restricted to qwen
# reverse_auction, the cell that is fully re-evaluated with it (base arm included). every other cell keeps
# the exact generation its one-shot results were produced with
SCOPE = frozenset({("qwen", "reverse_auction")})

CANONICAL_RUN = re.compile(r"^(?P<model>[a-z0-9]+)-(?P<mode>sealed|reverse_auction|supervisor)-v5-r01$")
_MARKER = "ZQXJ7"


def text_tokenizer(tokenizer):
    return getattr(tokenizer, "tokenizer", tokenizer)


def end_of_turn(tokenizer):
    """(id, token) of the special token the chat template writes right after an assistant message"""
    text_tok = text_tokenizer(tokenizer)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": _MARKER}],
        tokenize=False, enable_thinking=False)
    tail = rendered[rendered.index(_MARKER) + len(_MARKER):]
    special = set(text_tok.all_special_ids)
    special |= {i for i, t in getattr(text_tok, "added_tokens_decoder", {}).items() if getattr(t, "special", False)}
    for i in text_tok(tail, add_special_tokens=False)["input_ids"]:
        if i in special:
            return int(i), text_tok.convert_ids_to_tokens(int(i))
    return None, None


def generation_stop_ids(model):
    eos = model.generation_config.eos_token_id
    if eos is None:
        return []
    return [int(eos)] if isinstance(eos, int) else [int(e) for e in eos]


def cell_of(run, model_name=""):
    m = CANONICAL_RUN.match(run or "")
    return (m["model"], m["mode"]) if m else (model_name or None, None)


def stop_ids(model, tokenizer, model_name=None, mode=None):
    """(ids to pass as eos_token_id, or None to pass nothing; the record of what was decided)"""
    before = generation_stop_ids(model)
    eot, token = end_of_turn(tokenizer)
    in_scope = SCOPE is None or (model_name, mode) in SCOPE
    add = eot is not None and eot not in before and in_scope
    info = {"generation_stop_ids": before, "end_of_turn_id": eot, "end_of_turn_token": token,
            "tokenizer_eos_id": text_tokenizer(tokenizer).eos_token_id,
            "rule": "stop also on the chat template end-of-turn id when generation does not already stop on it",
            "scope": "all cells" if SCOPE is None else sorted(f"{a}/{b}" for a, b in SCOPE),
            "in_scope": in_scope, "added": add, "stop_ids_used": before + [eot] if add else before}
    return (before + [eot] if add else None), info
