import argparse
from dataclasses import dataclass
from types import SimpleNamespace


def safe_getattr(obj, attr_name, default=None):
    try:
        return getattr(obj, attr_name)
    except Exception:
        return default


def safe_setattr(obj, attr_name, value):
    if value is None:
        return
    try:
        setattr(obj, attr_name, value)
    except Exception:
        pass


class BrokenHuggingFaceTokenizer:
    """
    Simulate the tokenizer wrapper we are seeing in runtime:
    - has inner tokenizer at `.tokenizer`
    - `eos` / `eod` property access raises
    - outer object itself has no encode/batch_decode
    """

    def __init__(self, inner_tokenizer):
        self.tokenizer = inner_tokenizer

    @property
    def eos(self):
        raise NotImplementedError("_HuggingFaceTokenizer has no attribute 'eos'")

    @property
    def eod(self):
        raise NotImplementedError("_HuggingFaceTokenizer has no attribute 'eod'")


@dataclass
class NormalizedTokenizerReport:
    eos_token_id: int | None
    pad_token_id: int | None
    bos_token_id: int | None
    has_encode: bool
    has_batch_decode: bool
    has_apply_chat_template: bool
    eod: int | None


def normalize_tokenizer_for_generate(tokenizer_obj, args, eos_token_id=None, pad_token_id=None, bos_token_id=None):
    """
    This function models the logic module.py should implement before calling
    tokenize_prompts() and later decode/truncate helpers.
    """
    from transformers import AutoTokenizer

    outer = tokenizer_obj
    inner = safe_getattr(outer, "tokenizer", outer)

    safe_setattr(outer, "pad_token_id", pad_token_id)
    safe_setattr(inner, "pad_token_id", pad_token_id)
    safe_setattr(outer, "eos_token_id", eos_token_id)
    safe_setattr(inner, "eos_token_id", eos_token_id)
    safe_setattr(outer, "bos_token_id", bos_token_id)
    safe_setattr(inner, "bos_token_id", bos_token_id)

    resolved_eos = safe_getattr(outer, "eos_token_id", None)
    if resolved_eos is None:
        resolved_eos = safe_getattr(inner, "eos_token_id", None)
    if resolved_eos is None:
        resolved_eos = safe_getattr(inner, "_eos_id", None)
    if resolved_eos is None:
        resolved_eos = safe_getattr(args, "eos_id", None)
    if resolved_eos is None:
        resolved_eos = safe_getattr(args, "eod_id", None)
    if resolved_eos is None and getattr(args, "tokenizer_name_or_path", None):
        hf_tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name_or_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        resolved_eos = hf_tokenizer.eos_token_id

    if resolved_eos is None:
        raise ValueError("Your tokenizer doesn't include eos_token.")

    safe_setattr(outer, "eos_token_id", resolved_eos)
    safe_setattr(inner, "eos_token_id", resolved_eos)
    safe_setattr(outer, "eod", resolved_eos)
    safe_setattr(inner, "eod", resolved_eos)
    args.eos_id = resolved_eos
    args.eod_id = resolved_eos

    resolved_pad = safe_getattr(outer, "pad_token_id", None)
    if resolved_pad is None:
        resolved_pad = safe_getattr(inner, "pad_token_id", None)
    if resolved_pad is None and getattr(args, "tokenizer_name_or_path", None):
        hf_tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name_or_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        resolved_pad = hf_tokenizer.pad_token_id
    if resolved_pad is None:
        resolved_pad = resolved_eos
    safe_setattr(outer, "pad_token_id", resolved_pad)
    safe_setattr(inner, "pad_token_id", resolved_pad)

    resolved_bos = safe_getattr(outer, "bos_token_id", None)
    if resolved_bos is None:
        resolved_bos = safe_getattr(inner, "bos_token_id", None)
    if resolved_bos is None and getattr(args, "tokenizer_name_or_path", None):
        hf_tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name_or_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        resolved_bos = hf_tokenizer.bos_token_id
    safe_setattr(outer, "bos_token_id", resolved_bos)
    safe_setattr(inner, "bos_token_id", resolved_bos)

    if not callable(safe_getattr(outer, "encode", None)):
        safe_setattr(outer, "encode", inner.encode)
    if not callable(safe_getattr(outer, "batch_decode", None)):
        safe_setattr(outer, "batch_decode", inner.batch_decode)
    if not callable(safe_getattr(outer, "decode", None)):
        safe_setattr(outer, "decode", inner.decode)
    if not callable(safe_getattr(outer, "apply_chat_template", None)) and callable(
        safe_getattr(inner, "apply_chat_template", None)
    ):
        safe_setattr(outer, "apply_chat_template", inner.apply_chat_template)

    return NormalizedTokenizerReport(
        eos_token_id=resolved_eos,
        pad_token_id=resolved_pad,
        bos_token_id=resolved_bos,
        has_encode=callable(safe_getattr(outer, "encode", None)),
        has_batch_decode=callable(safe_getattr(outer, "batch_decode", None)),
        has_apply_chat_template=callable(safe_getattr(outer, "apply_chat_template", None)),
        eod=safe_getattr(outer, "eod", None),
    )


def simulate_generate_front_half(tokenizer_obj, prompt, max_new_tokens, max_length, hf_chat_template=False):
    """
    CPU-only simulation of the front half of model.generate():
    - _init_tokenizer output contract
    - _encode_no_template / hf_chat_template path requirements
    - pad token usage
    - batch decode availability
    """
    tokenizer = tokenizer_obj

    if hf_chat_template:
        tokens = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )
    else:
        tokens = tokenizer.encode(prompt)

    padded = list(tokens)
    target_len = max(max_length, len(tokens) + max_new_tokens)
    padded.extend([tokenizer.pad_token_id] * (target_len - len(tokens)))

    decoded = tokenizer.batch_decode([tokens], skip_special_tokens=True)

    return {
        "input_token_count": len(tokens),
        "padded_token_count": len(padded),
        "first_tokens": tokens[:16],
        "decoded": decoded,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--prompt", default="nihao")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--hf-chat-template", action="store_true")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    real_hf = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    runtime_args = SimpleNamespace(
        tokenizer_name_or_path=args.tokenizer_path,
        eos_id=None,
        eod_id=None,
    )

    broken_outer = BrokenHuggingFaceTokenizer(real_hf)

    report = normalize_tokenizer_for_generate(
        broken_outer,
        runtime_args,
    )

    result = simulate_generate_front_half(
        broken_outer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        max_length=args.max_length,
        hf_chat_template=args.hf_chat_template,
    )

    print("normalized_report =", report)
    print("runtime_args.eos_id =", runtime_args.eos_id)
    print("runtime_args.eod_id =", runtime_args.eod_id)
    print("simulate_result =", result)
    print("PASS")


if __name__ == "__main__":
    main()
