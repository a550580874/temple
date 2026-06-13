import argparse
from types import SimpleNamespace


def safe_getattr(obj, attr_name, default=None):
    try:
        return getattr(obj, attr_name)
    except Exception:
        return default


def resolve_tokenizer_attr(tokenizer, inner_tokenizer, args, attr_name):
    candidates = [
        safe_getattr(tokenizer, attr_name, None),
        safe_getattr(inner_tokenizer, attr_name, None),
    ]

    if attr_name == "eos_token_id":
        candidates.extend(
            [
                safe_getattr(tokenizer, "_eos_id", None),
                safe_getattr(inner_tokenizer, "_eos_id", None),
                safe_getattr(args, "eos_id", None),
                safe_getattr(args, "eod_id", None),
            ]
        )
    elif attr_name == "pad_token_id":
        candidates.extend(
            [
                safe_getattr(tokenizer, "_pad_id", None),
                safe_getattr(inner_tokenizer, "_pad_id", None),
                safe_getattr(args, "pad_id", None),
            ]
        )
    elif attr_name == "bos_token_id":
        candidates.extend(
            [
                safe_getattr(tokenizer, "_bos_id", None),
                safe_getattr(inner_tokenizer, "_bos_id", None),
                safe_getattr(args, "bos_id", None),
            ]
        )

    for candidate in candidates:
        if candidate is not None:
            return candidate

    return None


def safe_setattr(obj, attr_name, value):
    if value is None:
        return
    try:
        setattr(obj, attr_name, value)
    except Exception:
        pass


def init_tokenizer_like_module(tokenizer_obj, args, eos_token_id=None, pad_token_id=None, bos_token_id=None):
    tokenizer = tokenizer_obj
    inner_tokenizer = safe_getattr(tokenizer, "tokenizer", tokenizer)

    safe_setattr(tokenizer, "pad_token_id", pad_token_id)
    safe_setattr(inner_tokenizer, "pad_token_id", pad_token_id)
    safe_setattr(tokenizer, "eos_token_id", eos_token_id)
    safe_setattr(inner_tokenizer, "eos_token_id", eos_token_id)
    safe_setattr(tokenizer, "bos_token_id", bos_token_id)
    safe_setattr(inner_tokenizer, "bos_token_id", bos_token_id)

    resolved_eos = resolve_tokenizer_attr(tokenizer, inner_tokenizer, args, "eos_token_id")
    if resolved_eos is None:
        raise ValueError("Your tokenizer doesn't include eos_token.")

    safe_setattr(tokenizer, "eos_token_id", resolved_eos)
    safe_setattr(inner_tokenizer, "eos_token_id", resolved_eos)
    args.eos_id = resolved_eos
    args.eod_id = resolved_eos

    resolved_pad = resolve_tokenizer_attr(tokenizer, inner_tokenizer, args, "pad_token_id")
    if resolved_pad is None:
        resolved_pad = resolved_eos

    safe_setattr(tokenizer, "pad_token_id", resolved_pad)
    safe_setattr(inner_tokenizer, "pad_token_id", resolved_pad)

    resolved_bos = resolve_tokenizer_attr(tokenizer, inner_tokenizer, args, "bos_token_id")
    if resolved_bos is not None:
        safe_setattr(tokenizer, "bos_token_id", resolved_bos)
        safe_setattr(inner_tokenizer, "bos_token_id", resolved_bos)

    return {
        "eos_token_id": resolved_eos,
        "pad_token_id": resolved_pad,
        "bos_token_id": resolved_bos,
    }


class RaisingOuterTokenizer:
    def __init__(self, inner_tokenizer):
        self.tokenizer = inner_tokenizer

    @property
    def eos(self):
        raise NotImplementedError("_HuggingFaceTokenizer has no attribute 'eos'")

    @property
    def eod(self):
        raise NotImplementedError("_HuggingFaceTokenizer has no attribute 'eod'")


def run_with_real_hf_tokenizer(tokenizer_path):
    from transformers import AutoTokenizer

    args = SimpleNamespace(
        tokenizer_name_or_path=tokenizer_path,
        eos_id=None,
        eod_id=None,
        bos_id=None,
        pad_id=None,
    )

    hf_tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    outer = RaisingOuterTokenizer(hf_tokenizer)
    result = init_tokenizer_like_module(outer, args)

    print("resolved =", result)
    print("outer.eos_token_id =", safe_getattr(outer, "eos_token_id", None))
    print("inner.eos_token_id =", safe_getattr(hf_tokenizer, "eos_token_id", None))
    print("args.eos_id =", args.eos_id)
    print("args.eod_id =", args.eod_id)
    print("PASS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-path", required=True)
    args = parser.parse_args()
    run_with_real_hf_tokenizer(args.tokenizer_path)


if __name__ == "__main__":
    main()
