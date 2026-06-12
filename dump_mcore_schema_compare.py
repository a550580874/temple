#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
from collections import Counter, defaultdict

import torch

from mindspeed_llm.tasks.checkpoint.model_builder import MegatronModel, HuggingFaceModel


GLOBAL_HINTS = [
    "embedding",
    "word_embeddings",
    "embed_tokens",
    "output_layer",
    "lm_head",
    "final_layernorm",
    "final_norm",
    "decoder.final",
    "model.norm",
    "shared_experts",
    "router",
    "mtp",
]

GLOBAL_ALIAS_RULES = [
    (["decoder.final_layernorm.weight", "final_layernorm.weight"], "model.norm.weight"),
    (["output_layer.weight"], "lm_head.weight"),
    (["word_embeddings.weight", "embedding.word_embeddings.weight"], "model.embed_tokens.weight"),
]


def load_data(file_path):
    return torch.load(file_path, map_location="cpu", weights_only=False)


def get_iter_path(ckpt_path, iteration=None):
    if iteration is None:
        latest_iter_file = os.path.join(ckpt_path, "latest_checkpointed_iteration.txt")
        if os.path.exists(latest_iter_file):
            with open(latest_iter_file, "r") as f:
                iteration = int(f.read().strip())
        else:
            raise FileNotFoundError(f"can not find {latest_iter_file}")
    return os.path.join(ckpt_path, f"iter_{iteration:07d}")


def get_pt_path(iter_path, tp_rank=0, pp_rank=0, ep_rank=0, pp_size=1, ep_size=1):
    mp_rank_path = os.path.join(iter_path, f"mp_rank_{tp_rank:02d}")
    if pp_size > 1:
        mp_rank_path = mp_rank_path + f"_{pp_rank:03d}"
    if ep_size > 1:
        mp_rank_path = mp_rank_path + f"_{ep_rank:03d}"
    return os.path.join(mp_rank_path, "model_optim_rng.pt")


def tensor_shape_str(t):
    if hasattr(t, "shape"):
        return list(t.shape)
    return None


def collect_actual_layer_keys(state_dict, local_idx):
    prefix = f"decoder.layers.{local_idx}."
    result = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            result[k] = {
                "shape": tensor_shape_str(v),
                "dtype": str(v.dtype) if hasattr(v, "dtype") else None,
            }
    return dict(sorted(result.items(), key=lambda x: x[0]))


def maybe_add(expected, group, name, key):
    if key:
        expected[group][name] = key


def maybe_add_hf(expected, group, name, key):
    if key:
        expected[group][name] = key


def expected_keys_for_layer(load_model, local_idx, hf_layer_idx):
    mg_weight_key = load_model.get_weight(local_idx)
    expected = defaultdict(dict)

    maybe_add(expected, "norm", "input_layernorm", mg_weight_key.get("layers_input_layernorm"))
    maybe_add(expected, "norm", "pre_mlp_layernorm", mg_weight_key.get("layers_self_attention_pre_mlp_layernorm"))
    maybe_add(expected, "norm", "post_attention_layernorm", mg_weight_key.get("layers_self_attention_post_attention_layernorm"))

    qkv_type = getattr(load_model, "qkv_type", None)
    expected["meta"]["qkv_type"] = qkv_type

    if qkv_type == "pack_mla":
        maybe_add(expected, "attn", "linear_qkv", mg_weight_key.get("layers_self_attention_linear_qkv"))
        maybe_add(expected, "attn", "linear_proj", mg_weight_key.get("layers_self_attention_linear_proj"))
        maybe_add(expected, "attn", "q_layernorm", mg_weight_key.get("layers_self_attention_q_layernorm"))
        maybe_add(expected, "attn", "kv_layernorm", mg_weight_key.get("layers_self_attention_kv_layernorm"))
        maybe_add(expected, "attn", "linear_q_up_proj", mg_weight_key.get("layers_self_attention_linear_q_up_proj"))
        maybe_add(expected, "attn", "linear_kv_up_proj", mg_weight_key.get("layers_self_attention_linear_kv_up_proj"))
        maybe_add(expected, "attn", "linear_qk_nope", mg_weight_key.get("layers_self_attention_linear_qk_nope"))
        maybe_add(expected, "attn", "linear_qk_rope", mg_weight_key.get("layers_self_attention_linear_qk_rope"))
        maybe_add(expected, "attn", "linear_kv_nope", mg_weight_key.get("layers_self_attention_linear_kv_nope"))
        maybe_add(expected, "attn", "linear_v", mg_weight_key.get("layers_self_attention_linear_v"))

    elif qkv_type in ("unpack", "pack_gqa"):
        maybe_add(expected, "attn", "linear_qkv", mg_weight_key.get("layers_self_attention_linear_qkv"))
        maybe_add(expected, "attn", "linear_proj", mg_weight_key.get("layers_self_attention_linear_proj"))
        maybe_add(expected, "attn", "q_layernorm", mg_weight_key.get("layers_self_attention_q_layernorm"))
        maybe_add(expected, "attn", "k_layernorm", mg_weight_key.get("layers_self_attention_k_layernorm"))

    elif qkv_type == "mix":
        maybe_add(expected, "attn", "linear_q_proj", mg_weight_key.get("layers_self_attention_linear_q_proj"))
        maybe_add(expected, "attn", "linear_k_proj", mg_weight_key.get("layers_self_attention_linear_k_proj"))
        maybe_add(expected, "attn", "linear_v_proj", mg_weight_key.get("layers_self_attention_linear_v_proj"))
        maybe_add(expected, "attn", "linear_proj", mg_weight_key.get("layers_self_attention_linear_proj"))

    maybe_add(expected, "mlp_dense", "linear_fc1", mg_weight_key.get("layers_mlp_linear_fc1"))
    maybe_add(expected, "mlp_dense", "linear_fc2", mg_weight_key.get("layers_mlp_linear_fc2"))

    maybe_add(expected, "mlp_moe", "router", mg_weight_key.get("layers_mlp_router"))
    maybe_add(expected, "mlp_moe", "router_bias", mg_weight_key.get("layers_mlp_router_bias"))
    maybe_add(expected, "mlp_moe", "shared_expert_gate", mg_weight_key.get("layers_mlp_shared_expert_gate"))
    maybe_add(expected, "mlp_moe", "shared_fc1", mg_weight_key.get("layers_mlp_shared_experts_linear_fc1"))
    maybe_add(expected, "mlp_moe", "shared_fc2", mg_weight_key.get("layers_mlp_shared_experts_linear_fc2"))
    maybe_add(expected, "mlp_moe", "experts_weight1", mg_weight_key.get("layers_mlp_experts_weight1"))
    maybe_add(expected, "mlp_moe", "experts_weight2", mg_weight_key.get("layers_mlp_experts_weight2"))

    return expected


def expected_hf_keys_for_layer(load_model, save_model, hf_layer_idx):
    hf_weight_key = save_model.get_weight(layer_idx=hf_layer_idx)
    expected = defaultdict(dict)

    maybe_add_hf(expected, "norm", "input_layernorm", hf_weight_key.get("layers_input_layernorm"))
    maybe_add_hf(
        expected,
        "norm",
        "pre_mlp_layernorm",
        hf_weight_key.get("layers_self_attention_pre_mlp_layernorm"),
    )

    qkv_type = getattr(load_model, "qkv_type", None)
    expected["meta"]["qkv_type"] = qkv_type

    if qkv_type == "pack_mla":
        maybe_add_hf(expected, "attn", "linear_q_proj", hf_weight_key.get("layers_self_attention_linear_q_proj"))
        maybe_add_hf(expected, "attn", "linear_kv_proj", hf_weight_key.get("layers_self_attention_linear_kv_proj"))
        maybe_add_hf(expected, "attn", "linear_proj", hf_weight_key.get("layers_self_attention_linear_proj"))
        maybe_add_hf(expected, "attn", "linear_kv_up_proj", hf_weight_key.get("layers_self_attention_linear_kv_up_proj"))
        maybe_add_hf(expected, "attn", "kv_layernorm", hf_weight_key.get("layers_self_attention_kv_layernorm"))
        maybe_add_hf(expected, "attn", "q_layernorm", hf_weight_key.get("layers_self_attention_q_layernorm"))
        maybe_add_hf(expected, "attn", "linear_q_up_proj", hf_weight_key.get("layers_self_attention_linear_q_up_proj"))
    elif qkv_type in ("unpack", "pack_gqa", "mix"):
        maybe_add_hf(expected, "attn", "linear_q_proj", hf_weight_key.get("layers_self_attention_linear_q_proj"))
        maybe_add_hf(expected, "attn", "linear_k_proj", hf_weight_key.get("layers_self_attention_linear_k_proj"))
        maybe_add_hf(expected, "attn", "linear_v_proj", hf_weight_key.get("layers_self_attention_linear_v_proj"))
        maybe_add_hf(expected, "attn", "linear_proj", hf_weight_key.get("layers_self_attention_linear_proj"))

    if hf_layer_idx < (getattr(load_model, "first_k_dense_replace", 0) or 0):
        maybe_add_hf(expected, "mlp_dense", "linear_fc1", hf_weight_key.get("layers_mlp_linear_fc1"))
        maybe_add_hf(expected, "mlp_dense", "gate_proj", hf_weight_key.get("layers_mlp_gate_proj"))
        maybe_add_hf(expected, "mlp_dense", "up_proj", hf_weight_key.get("layers_mlp_up_proj"))
        maybe_add_hf(expected, "mlp_dense", "linear_fc2", hf_weight_key.get("layers_mlp_linear_fc2"))
    elif getattr(load_model, "num_experts", None):
        maybe_add_hf(expected, "mlp_moe", "router", hf_weight_key.get("layers_mlp_router"))
        maybe_add_hf(expected, "mlp_moe", "router_bias", hf_weight_key.get("layers_mlp_router_bias"))
        maybe_add_hf(expected, "mlp_moe", "shared_expert_gate", hf_weight_key.get("layers_mlp_shared_expert_gate"))
        maybe_add_hf(
            expected,
            "mlp_moe",
            "shared_gate_proj",
            hf_weight_key.get("layers_mlp_shared_experts_gate_proj"),
        )
        maybe_add_hf(
            expected,
            "mlp_moe",
            "shared_up_proj",
            hf_weight_key.get("layers_mlp_shared_experts_up_proj"),
        )
        maybe_add_hf(
            expected,
            "mlp_moe",
            "shared_fc2",
            hf_weight_key.get("layers_mlp_shared_experts_linear_fc2"),
        )
        for expert_idx in range(getattr(load_model, "num_experts", 0) or 0):
            expert_weight_key = save_model.get_weight(layer_idx=hf_layer_idx, expert_idx=expert_idx)
            maybe_add_hf(
                expected,
                "mlp_moe_experts",
                f"expert_{expert_idx}_gate_proj",
                expert_weight_key.get("layers_mlp_experts_gate_proj"),
            )
            maybe_add_hf(
                expected,
                "mlp_moe_experts",
                f"expert_{expert_idx}_up_proj",
                expert_weight_key.get("layers_mlp_experts_up_proj"),
            )
            maybe_add_hf(
                expected,
                "mlp_moe_experts",
                f"expert_{expert_idx}_fc2",
                expert_weight_key.get("layers_mlp_experts_linear_fc2"),
            )

    return expected


def expected_hf_global_keys(load_model, save_model):
    hf_weight_key = save_model.get_weight()
    expected = defaultdict(dict)
    maybe_add_hf(expected, "global", "embedding", hf_weight_key.get("embedding_word_embeddings"))
    maybe_add_hf(expected, "global", "final_layernorm", hf_weight_key.get("final_layernorm"))
    if getattr(load_model, "untie_embeddings_and_output_weights", False):
        maybe_add_hf(expected, "global", "output_layer", hf_weight_key.get("output_layer"))
    return expected


def emittable_hf_keys_for_layer(load_model, save_model, actual_key_names, local_idx, hf_layer_idx):
    actual_set = set(actual_key_names)
    layer_prefix = f"decoder.layers.{local_idx}."
    hf_weight_key = save_model.get_weight(layer_idx=hf_layer_idx)
    emitted = defaultdict(dict)

    def has(name):
        return f"{layer_prefix}{name}" in actual_set

    if has("input_layernorm.weight"):
        maybe_add_hf(emitted, "norm", "input_layernorm", hf_weight_key.get("layers_input_layernorm"))

    if (
        has("pre_mlp_layernorm.weight")
        or has("mlp.linear_fc1.layer_norm_weight")
        or has("post_attn_norm.weight")
    ):
        maybe_add_hf(
            emitted,
            "norm",
            "pre_mlp_layernorm",
            hf_weight_key.get("layers_self_attention_pre_mlp_layernorm"),
        )

    if (
        has("self_attention.linear_q_proj.weight")
        and has("self_attention.linear_kv_down_proj.weight")
        and has("self_attention.linear_kv_up_proj.weight")
    ):
        maybe_add_hf(emitted, "attn", "linear_q_proj", hf_weight_key.get("layers_self_attention_linear_q_proj"))
        maybe_add_hf(emitted, "attn", "linear_kv_proj", hf_weight_key.get("layers_self_attention_linear_kv_proj"))
        maybe_add_hf(emitted, "attn", "linear_proj", hf_weight_key.get("layers_self_attention_linear_proj"))
        maybe_add_hf(emitted, "attn", "linear_kv_up_proj", hf_weight_key.get("layers_self_attention_linear_kv_up_proj"))
        if has("self_attention.linear_kv_up_proj.layer_norm_weight"):
            maybe_add_hf(emitted, "attn", "kv_layernorm", hf_weight_key.get("layers_self_attention_kv_layernorm"))
    elif has("self_attention.linear_qkv.weight"):
        maybe_add_hf(emitted, "attn", "linear_q_proj", hf_weight_key.get("layers_self_attention_linear_q_proj"))
        maybe_add_hf(emitted, "attn", "linear_kv_proj", hf_weight_key.get("layers_self_attention_linear_kv_proj"))
        maybe_add_hf(emitted, "attn", "linear_proj", hf_weight_key.get("layers_self_attention_linear_proj"))
        maybe_add_hf(emitted, "attn", "linear_kv_up_proj", hf_weight_key.get("layers_self_attention_linear_kv_up_proj"))
        maybe_add_hf(emitted, "attn", "kv_layernorm", hf_weight_key.get("layers_self_attention_kv_layernorm"))
        maybe_add_hf(emitted, "attn", "q_layernorm", hf_weight_key.get("layers_self_attention_q_layernorm"))
        maybe_add_hf(emitted, "attn", "linear_q_up_proj", hf_weight_key.get("layers_self_attention_linear_q_up_proj"))

    force_dense_layer = has("mlp.linear_fc1.weight") and has("mlp.linear_fc2.weight")

    if force_dense_layer and hf_layer_idx < (getattr(load_model, "first_k_dense_replace", 0) or 0):
        maybe_add_hf(emitted, "mlp_dense", "linear_fc1", hf_weight_key.get("layers_mlp_linear_fc1"))
        maybe_add_hf(emitted, "mlp_dense", "gate_proj", hf_weight_key.get("layers_mlp_gate_proj"))
        maybe_add_hf(emitted, "mlp_dense", "up_proj", hf_weight_key.get("layers_mlp_up_proj"))
        maybe_add_hf(emitted, "mlp_dense", "linear_fc2", hf_weight_key.get("layers_mlp_linear_fc2"))
    elif hf_layer_idx >= (getattr(load_model, "first_k_dense_replace", 0) or 0):
        if has("mlp.router.weight"):
            maybe_add_hf(emitted, "mlp_moe", "router", hf_weight_key.get("layers_mlp_router"))
        if has("mlp.router.expert_bias"):
            maybe_add_hf(emitted, "mlp_moe", "router_bias", hf_weight_key.get("layers_mlp_router_bias"))
        if has("mlp.shared_experts.gate_weight") and getattr(load_model, "shared_expert_gate", None):
            maybe_add_hf(emitted, "mlp_moe", "shared_expert_gate", hf_weight_key.get("layers_mlp_shared_expert_gate"))
        if (
            has("mlp.shared_experts.linear_fc1.weight")
            and has("mlp.shared_experts.linear_fc2.weight")
            and getattr(load_model, "n_shared_experts", None)
        ):
            maybe_add_hf(
                emitted,
                "mlp_moe",
                "shared_gate_proj",
                hf_weight_key.get("layers_mlp_shared_experts_gate_proj"),
            )
            maybe_add_hf(
                emitted,
                "mlp_moe",
                "shared_up_proj",
                hf_weight_key.get("layers_mlp_shared_experts_up_proj"),
            )
            maybe_add_hf(
                emitted,
                "mlp_moe",
                "shared_fc2",
                hf_weight_key.get("layers_mlp_shared_experts_linear_fc2"),
            )

        has_grouped = has("mlp.experts.weight1") and has("mlp.experts.weight2")
        has_legacy = any(has(f"mlp.experts.linear_fc1.weight{i}") for i in range(128))
        if has_grouped or has_legacy:
            for expert_idx in range(getattr(load_model, "num_experts", 0) or 0):
                expert_weight_key = save_model.get_weight(layer_idx=hf_layer_idx, expert_idx=expert_idx)
                maybe_add_hf(
                    emitted,
                    "mlp_moe_experts",
                    f"expert_{expert_idx}_gate_proj",
                    expert_weight_key.get("layers_mlp_experts_gate_proj"),
                )
                maybe_add_hf(
                    emitted,
                    "mlp_moe_experts",
                    f"expert_{expert_idx}_up_proj",
                    expert_weight_key.get("layers_mlp_experts_up_proj"),
                )
                maybe_add_hf(
                    emitted,
                    "mlp_moe_experts",
                    f"expert_{expert_idx}_fc2",
                    expert_weight_key.get("layers_mlp_experts_linear_fc2"),
                )

    return emitted


def emittable_hf_global_keys(load_model, save_model, all_keys):
    actual_set = set(all_keys)
    hf_weight_key = save_model.get_weight()
    emitted = defaultdict(dict)
    if (
        "embedding.word_embeddings.weight" in actual_set
        or "word_embeddings.weight" in actual_set
        or "embedding_word_embeddings" in actual_set
    ):
        maybe_add_hf(emitted, "global", "embedding", hf_weight_key.get("embedding_word_embeddings"))
    if (
        "decoder.final_layernorm.weight" in actual_set
        or "final_layernorm.weight" in actual_set
    ):
        maybe_add_hf(emitted, "global", "final_layernorm", hf_weight_key.get("final_layernorm"))
    if getattr(load_model, "untie_embeddings_and_output_weights", False) and (
        "output_layer.weight" in actual_set or "lm_head.weight" in actual_set
    ):
        maybe_add_hf(emitted, "global", "output_layer", hf_weight_key.get("output_layer"))
    return emitted


def build_hf_bridge_blockers(load_model, actual_key_names, local_idx, hf_layer_idx):
    actual_set = set(actual_key_names)
    layer_prefix = f"decoder.layers.{local_idx}."
    blockers = []

    shared_fc1 = f"{layer_prefix}mlp.shared_experts.linear_fc1.weight"
    shared_fc2 = f"{layer_prefix}mlp.shared_experts.linear_fc2.weight"
    if shared_fc1 in actual_set and shared_fc2 in actual_set and not getattr(load_model, "n_shared_experts", None):
        blockers.append({
            "type": "shared_experts_skipped_by_condition",
            "layer": hf_layer_idx,
            "reason": "actual shared_experts weights exist, but load_model.n_shared_experts is falsy so mg2hf will skip shared_experts HF outputs",
            "actual_keys": [shared_fc1, shared_fc2],
        })

    shared_gate = f"{layer_prefix}mlp.shared_experts.gate_weight"
    if shared_gate not in actual_set and getattr(load_model, "shared_expert_gate", None):
        blockers.append({
            "type": "optional_shared_gate_missing",
            "layer": hf_layer_idx,
            "reason": "converter config enables shared_expert_gate but actual ckpt has no shared_experts.gate_weight",
            "actual_keys": [],
        })

    return blockers


def detect_runtime_moe_schema(all_keys, load_model):
    first_k_dense_replace = getattr(load_model, "first_k_dense_replace", 0) or 0
    grouped_layers = []
    legacy_layers = []
    mixed_layers = []

    for local_idx in range(first_k_dense_replace, getattr(load_model, "num_layers", 0) or 0):
        prefix = f"decoder.layers.{local_idx}."
        layer_keys = [k for k in all_keys if k.startswith(prefix)]
        has_grouped = any(
            k == f"{prefix}mlp.experts.weight1" or k == f"{prefix}mlp.experts.weight2"
            for k in layer_keys
        )
        has_legacy = any(
            k.startswith(f"{prefix}mlp.experts.linear_fc1.weight") or k.startswith(f"{prefix}mlp.experts.linear_fc2.weight")
            for k in layer_keys
        )
        if has_grouped and has_legacy:
            mixed_layers.append(local_idx)
        elif has_grouped:
            grouped_layers.append(local_idx)
        elif has_legacy:
            legacy_layers.append(local_idx)

    recommendation = "unknown"
    if grouped_layers and not legacy_layers and not mixed_layers:
        recommendation = "generate_should_enable_moe_grouped_gemm"
    elif legacy_layers and not grouped_layers and not mixed_layers:
        recommendation = "generate_should_disable_moe_grouped_gemm"
    elif mixed_layers:
        recommendation = "checkpoint_has_mixed_moe_schema"

    return {
        "first_k_dense_replace": first_k_dense_replace,
        "grouped_layers": grouped_layers,
        "legacy_layers": legacy_layers,
        "mixed_layers": mixed_layers,
        "recommendation": recommendation,
    }


def get_local_expert_count(load_model):
    num_experts = getattr(load_model, "num_experts", 0) or 0
    ep_size = getattr(load_model, "expert_model_parallel_size", 1) or 1
    if num_experts and ep_size and num_experts % ep_size == 0:
        return num_experts // ep_size
    return num_experts


def expected_generate_moe_keys_for_layer(load_model, local_idx, runtime_grouped_gemm):
    prefix = f"decoder.layers.{local_idx}."
    keys = []

    first_k_dense_replace = getattr(load_model, "first_k_dense_replace", 0) or 0
    if local_idx < first_k_dense_replace:
        return keys

    num_experts = getattr(load_model, "num_experts", 0) or 0
    if not num_experts:
        return keys

    keys.append(f"{prefix}mlp.router.weight")
    keys.append(f"{prefix}mlp.router.expert_bias")

    n_shared_experts = getattr(load_model, "n_shared_experts", 0) or 0
    if n_shared_experts:
        keys.append(f"{prefix}mlp.shared_experts.linear_fc1.weight")
        keys.append(f"{prefix}mlp.shared_experts.linear_fc2.weight")
        if getattr(load_model, "shared_expert_gate", None):
            keys.append(f"{prefix}mlp.shared_experts.gate_weight")

    if runtime_grouped_gemm:
        keys.append(f"{prefix}mlp.experts.weight1")
        keys.append(f"{prefix}mlp.experts.weight2")
    else:
        for expert_idx in range(get_local_expert_count(load_model)):
            keys.append(f"{prefix}mlp.experts.linear_fc1.weight{expert_idx}")
            keys.append(f"{prefix}mlp.experts.linear_fc2.weight{expert_idx}")

    return keys


def build_generate_moe_compare(actual_keys, expected_keys):
    actual_set = set(actual_keys)
    expected_set = set(expected_keys)
    schema_actual = sorted(k for k in actual_set if ".mlp.experts." in k or ".mlp.router." in k or ".mlp.shared_experts." in k)
    unexpected_schema_actual = sorted(k for k in schema_actual if k not in expected_set)
    return {
        "required_keys": expected_keys,
        "required_key_count": len(expected_keys),
        "missing_required_keys": sorted(expected_set - actual_set),
        "unexpected_schema_keys": unexpected_schema_actual,
        "matches": not (expected_set - actual_set),
    }


def flatten_expected(expected):
    rows = []
    for group, items in expected.items():
        if group == "meta":
            continue
        for name, key in items.items():
            rows.append({
                "group": group,
                "name": name,
                "key": key,
            })
    return rows


def detect_schema(actual_keys, local_idx):
    prefix = f"decoder.layers.{local_idx}."
    s = set(actual_keys)

    def has(name):
        return f"{prefix}{name}" in s

    return {
        "legacy_mla": (
            has("self_attention.linear_q_proj.weight")
            and has("self_attention.linear_kv_down_proj.weight")
            and has("self_attention.linear_kv_up_proj.weight")
            and not has("self_attention.linear_qkv.weight")
        ),
        "standard_pack_mla": (
            has("self_attention.linear_qkv.weight")
            and has("self_attention.linear_kv_up_proj.weight")
        ),
        "dense_mlp": (
            has("mlp.linear_fc1.weight")
            and has("mlp.linear_fc2.weight")
        ),
        "legacy_moe_experts": any(
            f"{prefix}mlp.experts.linear_fc1.weight{i}" in s for i in range(128)
        ),
        "grouped_moe_experts": (
            has("mlp.experts.weight1")
            or has("mlp.experts.weight2")
        ),
        "fc1_layer_norm_fused": has("mlp.linear_fc1.layer_norm_weight"),
        "pre_mlp_layernorm": has("pre_mlp_layernorm.weight"),
        "kv_ln_fused": has("self_attention.linear_kv_up_proj.layer_norm_weight"),
        "kv_layernorm": has("self_attention.kv_layernorm.weight"),
        "q_layernorm": has("self_attention.q_layernorm.weight"),
        "q_up_proj": has("self_attention.linear_q_up_proj.weight"),
    }


def build_diff(expected, actual_keys, local_idx):
    expected_rows = flatten_expected(expected)
    actual_set = set(actual_keys)

    missing_expected = []
    for row in expected_rows:
        if row["key"] not in actual_set:
            missing_expected.append(row)

    expected_set = set(r["key"] for r in expected_rows)
    unexpected_actual = []
    for key in actual_keys:
        if key not in expected_set:
            unexpected_actual.append(key)

    suggestions = []
    prefix = f"decoder.layers.{local_idx}."

    if f"{prefix}mlp.linear_fc1.layer_norm_weight" in actual_set:
        suggestions.append({
            "type": "alias",
            "from": f"{prefix}mlp.linear_fc1.layer_norm_weight",
            "to": "pre_mlp_layernorm",
        })
    if f"{prefix}self_attention.linear_kv_up_proj.layer_norm_weight" in actual_set:
        suggestions.append({
            "type": "alias",
            "from": f"{prefix}self_attention.linear_kv_up_proj.layer_norm_weight",
            "to": "kv_layernorm",
        })
    if (
        f"{prefix}self_attention.linear_q_proj.weight" in actual_set
        and f"{prefix}self_attention.linear_kv_down_proj.weight" in actual_set
    ):
        suggestions.append({
            "type": "legacy_mla",
            "from": [
                f"{prefix}self_attention.linear_q_proj.weight",
                f"{prefix}self_attention.linear_kv_down_proj.weight",
                f"{prefix}self_attention.linear_kv_up_proj.weight",
            ],
            "to": "DeepSeek2-Lite HF MLA path",
        })
    if any(f"{prefix}mlp.experts.linear_fc1.weight{i}" in actual_set for i in range(128)):
        suggestions.append({
            "type": "legacy_moe_expert_naming",
            "from": f"{prefix}mlp.experts.linear_fc1.weight<i> / linear_fc2.weight<i>",
            "to": "legacy expert fallback",
        })
    if (
        f"{prefix}mlp.linear_fc1.weight" in actual_set
        and f"{prefix}mlp.linear_fc2.weight" in actual_set
    ):
        suggestions.append({
            "type": "dense_layer",
            "from": "dense fc1/fc2 present",
            "to": "force dense mlp path",
        })

    return {
        "missing_expected": missing_expected,
        "unexpected_actual": unexpected_actual,
        "suggestions": suggestions,
    }


def collect_global_actual_keys(all_keys):
    out = []
    for k in sorted(all_keys):
        if ".layers." in k:
            continue
        if any(h in k for h in GLOBAL_HINTS):
            out.append(k)
    return out


def is_probable_state_dict_key(key):
    if "." not in key:
        return False
    if key.startswith("layers_") or key.startswith("mtp_layers_"):
        return False
    if key.startswith("model.") or key.startswith("lm_head."):
        return True
    return any(h in key for h in GLOBAL_HINTS)


def grep_key_literals(code_text):
    patterns = [
        r'pop\("([^"]+)"\)',
        r"pop\('([^']+)'\)",
        r'get\("([^"]+)"\)',
        r"get\('([^']+)'\)",
        r'\["([^"]+)"\]',
        r"\['([^']+)'\]",
    ]
    found = set()
    for pattern in patterns:
        found.update(re.findall(pattern, code_text))
    return sorted(found)


def collect_code_expected_second_phase(code_text):
    keys = grep_key_literals(code_text)
    out = []
    for k in keys:
        if ".layers." in k:
            continue
        if is_probable_state_dict_key(k):
            out.append(k)
    return sorted(set(out))


def build_global_alias_suggestions(actual_keys, expected_keys):
    actual_set = set(actual_keys)
    expected_set = set(expected_keys)
    suggestions = []

    for actual_candidates, expected_name in GLOBAL_ALIAS_RULES:
        if expected_name not in expected_set:
            continue
        for candidate in actual_candidates:
            if candidate in actual_set:
                suggestions.append({
                    "type": "alias",
                    "from": candidate,
                    "to": expected_name,
                })
                break
    return suggestions


def diff_global(actual_keys, expected_keys):
    actual_set = set(actual_keys)
    expected_set = set(expected_keys)
    return {
        "actual": actual_keys,
        "expected": expected_keys,
        "missing_expected": sorted(expected_set - actual_set),
        "unexpected_actual": sorted(actual_set - expected_set),
        "suggestions": build_global_alias_suggestions(actual_keys, expected_keys),
    }


def summarize_keys(all_keys):
    counter = Counter()
    examples = {}
    for k in sorted(all_keys):
        if ".layers." in k:
            prefix = k.split(".layers.")[0] + ".layers.*"
        else:
            parts = k.split(".")
            prefix = ".".join(parts[:3]) if len(parts) >= 3 else k
        counter[prefix] += 1
        examples.setdefault(prefix, k)
    return {
        "counts": dict(sorted(counter.items())),
        "examples": examples,
    }


def build_full_expected_keys(layer_expected_map, global_expected_keys):
    expected_keys = set(global_expected_keys)
    for expected in layer_expected_map.values():
        for row in flatten_expected(expected):
            expected_keys.add(row["key"])
    return sorted(expected_keys)


def build_full_diff(actual_keys, expected_keys):
    actual_set = set(actual_keys)
    expected_set = set(expected_keys)
    return {
        "actual_key_count": len(actual_keys),
        "expected_key_count": len(expected_keys),
        "missing_expected": sorted(expected_set - actual_set),
        "unexpected_actual": sorted(actual_set - expected_set),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-dir", required=True)
    parser.add_argument("--model-type-hf", required=True)
    parser.add_argument("--hf-cfg-dir", required=True)
    parser.add_argument("--target-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--target-pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--target-expert-parallel-size", type=int, default=1)
    parser.add_argument("--expert-tensor-parallel-size", type=int, default=None)
    parser.add_argument("--moe-grouped-gemm", action="store_true")
    parser.add_argument("--transformer-impl", default="local")
    parser.add_argument("--iteration", type=int, default=None)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--pp-rank", type=int, default=0)
    parser.add_argument("--ep-rank", type=int, default=0)
    parser.add_argument("--num-layer-list", default=None)
    parser.add_argument("--noop-layers", default=None)
    parser.add_argument("--num-layers-per-virtual-pipeline-stage", type=int, default=None)
    parser.add_argument("--mtp-num-layers", type=int, default=0)
    parser.add_argument("--output-prefix", default="schema_compare")
    parser.add_argument("--mla-mm-split", action="store_true")
    parser.add_argument("--schedules-method", default=None)
    parser.add_argument("--first-k-dense-replace", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--convert-script", type=str, required=True, help="Path to convert_mg2hf.py")
    args = parser.parse_args()

    args.load_model_type = "mg"
    args.save_model_type = "hf"
    args.save_dir = "/tmp/unused_hf_out"

    load_model = MegatronModel(args)
    save_model = HuggingFaceModel(args)

    iter_path = get_iter_path(args.load_dir, args.iteration)
    ckpt_path = get_pt_path(
        iter_path,
        tp_rank=args.tp_rank,
        pp_rank=args.pp_rank,
        ep_rank=args.ep_rank,
        pp_size=load_model.pipeline_model_parallel_size,
        ep_size=load_model.expert_model_parallel_size,
    )

    ckpt = load_data(ckpt_path)
    state_dict = ckpt["model"]
    all_keys = set(state_dict.keys())

    sorted_all_actual_keys = sorted(all_keys)
    all_actual_summary = summarize_keys(sorted_all_actual_keys)
    global_actual = collect_global_actual_keys(all_keys)

    with open(args.convert_script, "r", encoding="utf-8") as f:
        convert_code = f.read()

    code_expected_second_phase = collect_code_expected_second_phase(convert_code)
    global_expected = code_expected_second_phase
    global_diff = diff_global(global_actual, global_expected)
    hf_global_expected = expected_hf_global_keys(load_model, save_model)
    hf_global_emittable = emittable_hf_global_keys(load_model, save_model, all_keys)
    runtime_moe_schema = detect_runtime_moe_schema(sorted_all_actual_keys, load_model)

    meta = {
        "ckpt_path": ckpt_path,
        "qkv_type": getattr(load_model, "qkv_type", None),
        "num_layers": getattr(load_model, "num_layers", None),
        "first_k_dense_replace": getattr(load_model, "first_k_dense_replace", None),
        "num_experts": getattr(load_model, "num_experts", None),
        "n_shared_experts": getattr(load_model, "n_shared_experts", None),
        "moe_grouped_gemm_flag": args.moe_grouped_gemm,
        "tp_rank": args.tp_rank,
        "pp_rank": args.pp_rank,
        "ep_rank": args.ep_rank,
        "runtime_moe_schema": runtime_moe_schema,
    }

    layer_expected_map = {}
    hf_layer_expected_map = {}
    hf_layer_emittable_map = {}
    hf_bridge_blockers = []

    expected_full = {
        "meta": meta,
        "all_expected_keys": [],
        "all_expected_key_summary": {},
        "hf2mg_required_hf_keys": [],
        "hf2mg_required_hf_key_summary": {},
        "generate_runtime_contract": runtime_moe_schema,
        "global": {
            "expected": global_expected,
        },
        "hf_global": {
            "expected": hf_global_expected,
        },
        "layers": {},
        "hf_layers": {},
        "generate_layers": {},
        "code_expected_second_phase": code_expected_second_phase,
    }

    actual_full = {
        "meta": {
            "ckpt_path": ckpt_path,
        },
        "all_actual_keys": sorted_all_actual_keys,
        "all_actual_key_summary": all_actual_summary,
        "mg2hf_emittable_hf_keys": [],
        "mg2hf_emittable_hf_key_summary": {},
        "generate_runtime_contract": runtime_moe_schema,
        "global": {
            "actual": global_actual,
        },
        "hf_global": {
            "emittable": hf_global_emittable,
        },
        "layers": {},
        "hf_layers": {},
        "generate_layers": {},
    }

    diff_full = {
        "meta": {
            "ckpt_path": ckpt_path,
        },
        "full_compare": {},
        "hf_bridge_compare": {},
        "generate_runtime_contract": runtime_moe_schema,
        "global": global_diff,
        "layers": {},
        "hf_layers": {},
        "generate_layers": {},
        "code_expected_second_phase": code_expected_second_phase,
    }

    grouped_generate_missing = 0
    legacy_generate_missing = 0

    for local_idx in range(load_model.num_layers):
        actual_keys = collect_actual_layer_keys(state_dict, local_idx)
        expected = expected_keys_for_layer(load_model, local_idx, local_idx)
        layer_expected_map[str(local_idx)] = expected
        hf_expected = expected_hf_keys_for_layer(load_model, save_model, local_idx)
        hf_emittable = emittable_hf_keys_for_layer(load_model, save_model, actual_keys.keys(), local_idx, local_idx)
        hf_layer_expected_map[str(local_idx)] = hf_expected
        hf_layer_emittable_map[str(local_idx)] = hf_emittable
        detected = detect_schema(actual_keys.keys(), local_idx)
        diff = build_diff(expected, actual_keys.keys(), local_idx)
        hf_layer_diff = build_full_diff(
            sorted(row["key"] for row in flatten_expected(hf_emittable)),
            sorted(row["key"] for row in flatten_expected(hf_expected)),
        )
        hf_blockers = build_hf_bridge_blockers(load_model, actual_keys.keys(), local_idx, local_idx)
        hf_bridge_blockers.extend(hf_blockers)
        grouped_generate_expected = expected_generate_moe_keys_for_layer(load_model, local_idx, runtime_grouped_gemm=True)
        legacy_generate_expected = expected_generate_moe_keys_for_layer(load_model, local_idx, runtime_grouped_gemm=False)
        grouped_generate_compare = build_generate_moe_compare(actual_keys.keys(), grouped_generate_expected)
        legacy_generate_compare = build_generate_moe_compare(actual_keys.keys(), legacy_generate_expected)
        grouped_generate_missing += len(grouped_generate_compare["missing_required_keys"])
        legacy_generate_missing += len(legacy_generate_compare["missing_required_keys"])

        expected_full["layers"][str(local_idx)] = {
            "expected": expected,
        }
        expected_full["hf_layers"][str(local_idx)] = {
            "expected": hf_expected,
        }
        expected_full["generate_layers"][str(local_idx)] = {
            "grouped_runtime_required": grouped_generate_expected,
            "legacy_runtime_required": legacy_generate_expected,
        }
        actual_full["layers"][str(local_idx)] = {
            "detected_schema": detected,
            "actual": actual_keys,
        }
        actual_full["hf_layers"][str(local_idx)] = {
            "detected_schema": detected,
            "emittable": hf_emittable,
        }
        actual_full["generate_layers"][str(local_idx)] = {
            "detected_schema": detected,
            "grouped_runtime_compare": grouped_generate_compare,
            "legacy_runtime_compare": legacy_generate_compare,
        }
        diff_full["layers"][str(local_idx)] = {
            "detected_schema": detected,
            "missing_expected": diff["missing_expected"],
            "unexpected_actual": diff["unexpected_actual"],
            "suggestions": diff["suggestions"],
        }
        diff_full["hf_layers"][str(local_idx)] = {
            "detected_schema": detected,
            "missing_required_hf": hf_layer_diff["missing_expected"],
            "emittable_but_not_required_hf": hf_layer_diff["unexpected_actual"],
            "blockers": hf_blockers,
        }
        diff_full["generate_layers"][str(local_idx)] = {
            "detected_schema": detected,
            "grouped_runtime_compare": grouped_generate_compare,
            "legacy_runtime_compare": legacy_generate_compare,
        }

    all_expected_keys = build_full_expected_keys(layer_expected_map, global_expected)
    expected_full["all_expected_keys"] = all_expected_keys
    expected_full["all_expected_key_summary"] = summarize_keys(all_expected_keys)
    diff_full["full_compare"] = build_full_diff(sorted_all_actual_keys, all_expected_keys)

    hf_required_hf_keys = build_full_expected_keys(hf_layer_expected_map, [row["key"] for row in flatten_expected(hf_global_expected)])
    hf_emittable_hf_keys = build_full_expected_keys(hf_layer_emittable_map, [row["key"] for row in flatten_expected(hf_global_emittable)])
    expected_full["hf2mg_required_hf_keys"] = hf_required_hf_keys
    expected_full["hf2mg_required_hf_key_summary"] = summarize_keys(hf_required_hf_keys)
    actual_full["mg2hf_emittable_hf_keys"] = hf_emittable_hf_keys
    actual_full["mg2hf_emittable_hf_key_summary"] = summarize_keys(hf_emittable_hf_keys)
    diff_full["hf_bridge_compare"] = {
        "required_hf_key_count": len(hf_required_hf_keys),
        "emittable_hf_key_count": len(hf_emittable_hf_keys),
        "missing_required_hf": sorted(set(hf_required_hf_keys) - set(hf_emittable_hf_keys)),
        "emittable_but_not_required_hf": sorted(set(hf_emittable_hf_keys) - set(hf_required_hf_keys)),
        "blockers": hf_bridge_blockers,
    }
    diff_full["generate_load_compare"] = {
        "grouped_runtime_missing_total": grouped_generate_missing,
        "legacy_runtime_missing_total": legacy_generate_missing,
        "recommended_runtime": runtime_moe_schema["recommendation"],
    }

    expected_path = f"{args.output_prefix}_expected.json"
    actual_path = f"{args.output_prefix}_actual.json"
    diff_path = f"{args.output_prefix}_diff.json"

    with open(expected_path, "w", encoding="utf-8") as f:
        json.dump(expected_full, f, ensure_ascii=False, indent=2)

    with open(actual_path, "w", encoding="utf-8") as f:
        json.dump(actual_full, f, ensure_ascii=False, indent=2)

    with open(diff_path, "w", encoding="utf-8") as f:
        json.dump(diff_full, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(f"Expected schema: {expected_path}")
    print(f"Actual ckpt schema: {actual_path}")
    print(f"Diff summary: {diff_path}")


if __name__ == "__main__":
    main()
