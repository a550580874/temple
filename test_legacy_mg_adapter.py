from types import SimpleNamespace

import torch

from mindspeed_llm.tasks.inference.legacy_mg_adapter import (
    adapt_legacy_deepseek2_lite_checkpoint_if_needed,
)


def build_fake_legacy_state_dict():
    model = {}

    # layer 0 legacy MLA
    model["decoder.layers.0.self_attention.linear_q_proj.weight"] = torch.randn(6, 4)
    model["decoder.layers.0.self_attention.linear_kv_down_proj.weight"] = torch.randn(2, 4)
    model["decoder.layers.0.self_attention.linear_kv_up_proj.weight"] = torch.randn(4, 2)
    model["decoder.layers.0.self_attention.linear_kv_up_proj.layer_norm_weight"] = torch.randn(2)
    model["decoder.layers.0.mlp.linear_fc1.layer_norm_weight"] = torch.randn(4)

    # one MoE layer is enough to validate packing logic
    model["decoder.layers.1.self_attention.linear_q_proj.weight"] = torch.randn(6, 4)
    model["decoder.layers.1.self_attention.linear_kv_down_proj.weight"] = torch.randn(2, 4)
    model["decoder.layers.1.self_attention.linear_kv_up_proj.weight"] = torch.randn(4, 2)
    model["decoder.layers.1.self_attention.linear_kv_up_proj.layer_norm_weight"] = torch.randn(2)
    model["decoder.layers.1.mlp.router.weight"] = torch.randn(64, 4)

    for expert_idx in range(8):
        model[f"decoder.layers.1.mlp.experts.linear_fc1.weight{expert_idx}"] = torch.randn(8, 4)
        model[f"decoder.layers.1.mlp.experts.linear_fc2.weight{expert_idx}"] = torch.randn(4, 8)

    return {"model": model}


def main():
    args = SimpleNamespace(model_type_hf="deepseek2-lite")
    state_dict = build_fake_legacy_state_dict()
    adapted = adapt_legacy_deepseek2_lite_checkpoint_if_needed(state_dict, args)
    model = adapted["model"]

    must_have = [
        "decoder.layers.0.self_attention.linear_qkv.weight",
        "decoder.layers.0.self_attention.kv_layernorm.weight",
        "decoder.layers.0.pre_mlp_layernorm.weight",
        "decoder.layers.1.self_attention.linear_qkv.weight",
        "decoder.layers.1.self_attention.kv_layernorm.weight",
        "decoder.layers.1.mlp.experts.weight1",
        "decoder.layers.1.mlp.experts.weight2",
    ]
    must_not_have = [
        "decoder.layers.0.self_attention.linear_q_proj.weight",
        "decoder.layers.0.self_attention.linear_kv_down_proj.weight",
        "decoder.layers.0.self_attention.linear_kv_up_proj.layer_norm_weight",
        "decoder.layers.0.mlp.linear_fc1.layer_norm_weight",
        "decoder.layers.1.mlp.experts.linear_fc1.weight0",
        "decoder.layers.1.mlp.experts.linear_fc2.weight0",
    ]

    for key in must_have:
        assert key in model, f"missing expected key: {key}"
    for key in must_not_have:
        assert key not in model, f"unexpected leftover key: {key}"

    print("weight1 shape =", tuple(model["decoder.layers.1.mlp.experts.weight1"].shape))
    print("weight2 shape =", tuple(model["decoder.layers.1.mlp.experts.weight2"].shape))
    print("linear_qkv shape =", tuple(model["decoder.layers.0.self_attention.linear_qkv.weight"].shape))
    print("adapter test passed")


if __name__ == "__main__":
    main()
