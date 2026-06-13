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


def _alias_layer0_norm(model_state):
    src_key = "decoder.layers.0.mlp.linear_fc1.layer_norm_weight"
    dst_key = "decoder.layers.0.pre_mlp_layernorm.weight"
    if src_key in model_state and dst_key not in model_state:
        model_state[dst_key] = model_state[src_key].clone()
    model_state.pop(src_key, None)


def _pack_legacy_attention_to_runtime(model_state):
    q_proj_pat = re.compile(r"decoder\.layers\.(\d+)\.self_attention\.linear_q_proj\.weight$")

    for key in list(model_state.keys()):
        matched = q_proj_pat.match(key)
        if matched is None:
            continue

        layer_idx = matched.group(1)
        q_key = f"decoder.layers.{layer_idx}.self_attention.linear_q_proj.weight"
        kv_down_key = f"decoder.layers.{layer_idx}.self_attention.linear_kv_down_proj.weight"
        kv_ln_src_key = f"decoder.layers.{layer_idx}.self_attention.linear_kv_up_proj.layer_norm_weight"

        qkv_dst_key = f"decoder.layers.{layer_idx}.self_attention.linear_qkv.weight"
        kv_ln_dst_key = f"decoder.layers.{layer_idx}.self_attention.kv_layernorm.weight"

        if q_key in model_state and kv_down_key in model_state and qkv_dst_key not in model_state:
            q_proj = model_state[q_key]
            kv_down_proj = model_state[kv_down_key]
            model_state[qkv_dst_key] = torch.cat([q_proj, kv_down_proj], dim=0).contiguous()

        if kv_ln_src_key in model_state and kv_ln_dst_key not in model_state:
            model_state[kv_ln_dst_key] = model_state[kv_ln_src_key].clone()

        model_state.pop(q_key, None)
        model_state.pop(kv_down_key, None)
        model_state.pop(kv_ln_src_key, None)


def _pack_legacy_experts_to_grouped(model_state):
    layer_pat = re.compile(r"decoder\.layers\.(\d+)\.mlp\.experts\.linear_fc1\.weight0$")

    layer_ids = []
    for key in model_state.keys():
        matched = layer_pat.match(key)
        if matched is not None:
            layer_ids.append(int(matched.group(1)))

    for layer_idx in sorted(set(layer_ids)):
        fc1_pat = re.compile(
            rf"decoder\.layers\.{layer_idx}\.mlp\.experts\.linear_fc1\.weight(\d+)$"
        )
        fc2_pat = re.compile(
            rf"decoder\.layers\.{layer_idx}\.mlp\.experts\.linear_fc2\.weight(\d+)$"
        )

        fc1_items = []
        fc2_items = []

        for key, value in model_state.items():
            matched = fc1_pat.match(key)
            if matched is not None:
                fc1_items.append((int(matched.group(1)), value))
            matched = fc2_pat.match(key)
            if matched is not None:
                fc2_items.append((int(matched.group(1)), value))

        fc1_items.sort(key=lambda x: x[0])
        fc2_items.sort(key=lambda x: x[0])

        if not fc1_items or not fc2_items:
            continue
        if len(fc1_items) != len(fc2_items):
            continue

        fc1_indices = [idx for idx, _ in fc1_items]
        fc2_indices = [idx for idx, _ in fc2_items]
        if fc1_indices != fc2_indices:
            continue

        fc1_list = [value for _, value in fc1_items]
        fc2_list = [value for _, value in fc2_items]

        weight1 = torch.cat([fc1.t().reshape(-1) for fc1 in fc1_list], dim=0).view(
            fc1_list[0].shape[1], -1
        ).contiguous()
        weight2 = torch.cat([fc2.t().reshape(-1) for fc2 in fc2_list], dim=0).view(
            -1, fc2_list[0].shape[0]
        ).contiguous()

        model_state[f"decoder.layers.{layer_idx}.mlp.experts.weight1"] = weight1
        model_state[f"decoder.layers.{layer_idx}.mlp.experts.weight2"] = weight2

        for expert_idx, _ in fc1_items:
            model_state.pop(
                f"decoder.layers.{layer_idx}.mlp.experts.linear_fc1.weight{expert_idx}",
                None,
            )
        for expert_idx, _ in fc2_items:
            model_state.pop(
                f"decoder.layers.{layer_idx}.mlp.experts.linear_fc2.weight{expert_idx}",
                None,
            )


def adapt_legacy_deepseek2_lite_checkpoint_if_needed(state_dict, args):
    model_state = _get_model_state_dict(state_dict)
    if model_state is None:
        return state_dict

    model_type_hf = getattr(args, "model_type_hf", None)
    if model_type_hf not in (None, "deepseek2-lite"):
        return state_dict

    if _has_grouped_expert_schema(model_state):
        return state_dict

    if not _is_legacy_deepseek2_lite_schema(model_state):
        return state_dict

    state_dict = copy.deepcopy(state_dict)
    model_state = state_dict["model"]

    _alias_layer0_norm(model_state)
    _pack_legacy_attention_to_runtime(model_state)
    _pack_legacy_experts_to_grouped(model_state)

    return state_dict
