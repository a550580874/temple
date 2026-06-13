import copy
import re

import torch


def _get_model_state_dict(state_dict):
    if not isinstance(state_dict, dict):
        return None
    model_state = state_dict.get("model")
    if isinstance(model_state, dict):
        return model_state
    return None


def _is_legacy_deepseek2_lite_schema(model_state):
    keys = set(model_state.keys())

    has_legacy_mla = (
        "decoder.layers.0.self_attention.linear_q_proj.weight" in keys
        and "decoder.layers.0.self_attention.linear_kv_down_proj.weight" in keys
        and "decoder.layers.0.self_attention.linear_kv_up_proj.weight" in keys
    )

    has_legacy_moe = any(
        re.match(r"decoder\.layers\.\d+\.mlp\.experts\.linear_fc1\.weight0$", k)
        for k in keys
    ) and any(
        re.match(r"decoder\.layers\.\d+\.mlp\.experts\.linear_fc2\.weight0$", k)
        for k in keys
    )

    return has_legacy_mla and has_legacy_moe


def _has_grouped_expert_schema(model_state):
    keys = set(model_state.keys())
    return any(k.endswith(".mlp.experts.weight1") for k in keys) and any(
        k.endswith(".mlp.experts.weight2") for k in keys
    )


def _ensure_router_expert_bias(model_state):
    router_weight_pat = re.compile(r"decoder\.layers\.(\d+)\.mlp\.router\.weight$")
    bias_tpl = "decoder.layers.{layer_idx}.mlp.router.expert_bias"

    for key, value in list(model_state.items()):
        matched = router_weight_pat.match(key)
        if matched is None:
            continue
        layer_idx = matched.group(1)
        bias_key = bias_tpl.format(layer_idx=layer_idx)
        if bias_key not in model_state:
            num_experts = value.shape[0]
            model_state[bias_key] = torch.zeros(
                num_experts, dtype=torch.float32, device=value.device
            )


def _alias_layer0_dense_norm_if_needed(model_state):
    src_key = "decoder.layers.0.mlp.linear_fc1.layer_norm_weight"
    dst_key = "decoder.layers.0.pre_mlp_layernorm.weight"
    if src_key in model_state and dst_key not in model_state:
        model_state[dst_key] = model_state[src_key].clone()


def _convert_grouped_experts_to_legacy_flat(model_state):
    weight1_pat = re.compile(r"decoder\.layers\.(\d+)\.mlp\.experts\.weight1$")
    weight2_pat = re.compile(r"decoder\.layers\.(\d+)\.mlp\.experts\.weight2$")

    layer_ids = set()
    for key in model_state.keys():
        matched = weight1_pat.match(key)
        if matched:
            layer_ids.add(int(matched.group(1)))
        matched = weight2_pat.match(key)
        if matched:
            layer_ids.add(int(matched.group(1)))

    for layer_idx in sorted(layer_ids):
        w1_key = f"decoder.layers.{layer_idx}.mlp.experts.weight1"
        w2_key = f"decoder.layers.{layer_idx}.mlp.experts.weight2"
        router_key = f"decoder.layers.{layer_idx}.mlp.router.weight"

        if w1_key not in model_state or w2_key not in model_state or router_key not in model_state:
            continue

        weight1 = model_state.pop(w1_key)
        weight2 = model_state.pop(w2_key)
        router_weight = model_state[router_key]

        num_experts = router_weight.shape[0]
        hidden_size = weight1.shape[0]

        flat_fc1_list = torch.chunk(weight1.reshape(-1), num_experts, dim=0)
        flat_fc2_list = torch.chunk(weight2.reshape(-1), num_experts, dim=0)

        for expert_idx in range(num_experts):
            fc1 = flat_fc1_list[expert_idx].view(hidden_size, -1).t().contiguous()
            fc2 = flat_fc2_list[expert_idx].view(-1, hidden_size).t().contiguous()

            model_state[
                f"decoder.layers.{layer_idx}.mlp.experts.linear_fc1.weight{expert_idx}"
            ] = fc1
            model_state[
                f"decoder.layers.{layer_idx}.mlp.experts.linear_fc2.weight{expert_idx}"
            ] = fc2


def adapt_legacy_deepseek2_lite_checkpoint_if_needed(state_dict, args):
    model_state = _get_model_state_dict(state_dict)
    if model_state is None:
        return state_dict

    model_type_hf = getattr(args, "model_type_hf", None)
    if model_type_hf not in (None, "deepseek2-lite"):
        return state_dict

    is_legacy = _is_legacy_deepseek2_lite_schema(model_state)
    is_grouped = _has_grouped_expert_schema(model_state)
    if not is_legacy and not is_grouped:
        return state_dict

    state_dict = copy.deepcopy(state_dict)
    model_state = state_dict["model"]

    _alias_layer0_dense_norm_if_needed(model_state)
    _ensure_router_expert_bias(model_state)

    if _has_grouped_expert_schema(model_state):
        _convert_grouped_experts_to_legacy_flat(model_state)

    return state_dict
