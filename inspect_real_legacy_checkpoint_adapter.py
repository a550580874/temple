import argparse
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from mindspeed_llm.tasks.inference.legacy_mg_adapter import (
    adapt_legacy_deepseek2_lite_checkpoint_if_needed,
)


HIDDEN_SIZE = 4
Q_PROJ_ROWS = 6
KV_DOWN_ROWS = 2
KV_LORA_RANK = 2
FFN_HIDDEN_SIZE = 8
VOCAB_SIZE = 16
NUM_LAYERS = 27
NUM_LOCAL_EXPERTS = 8


class WeightHolder(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(*shape))


class NormHolder(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(size))


class SelfAttentionRuntime(nn.Module):
    def __init__(self, qkv_rows, hidden_size, kv_lora_rank):
        super().__init__()
        self.linear_qkv = WeightHolder(qkv_rows, hidden_size)
        self.linear_kv_up_proj = WeightHolder(hidden_size, kv_lora_rank)
        self.kv_layernorm = NormHolder(kv_lora_rank)
        self.linear_proj = WeightHolder(hidden_size, hidden_size)


class DenseMLPRuntime(nn.Module):
    def __init__(self, ffn_hidden_size, hidden_size):
        super().__init__()
        self.linear_fc1 = WeightHolder(ffn_hidden_size, hidden_size)
        self.linear_fc2 = WeightHolder(hidden_size, ffn_hidden_size)


class RouterRuntime(nn.Module):
    def __init__(self, num_local_experts, hidden_size):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(num_local_experts, hidden_size))
        self.expert_bias = nn.Parameter(torch.zeros(num_local_experts))


class ExpertsRuntime(nn.Module):
    def __init__(self, hidden_size, ffn_hidden_size, num_local_experts):
        super().__init__()
        self.weight1 = nn.Parameter(torch.zeros(hidden_size, num_local_experts * ffn_hidden_size))
        self.weight2 = nn.Parameter(torch.zeros(num_local_experts * ffn_hidden_size, hidden_size))


class SharedExpertsRuntime(nn.Module):
    def __init__(self, ffn_hidden_size, hidden_size):
        super().__init__()
        self.linear_fc1 = WeightHolder(ffn_hidden_size, hidden_size)
        self.linear_fc2 = WeightHolder(hidden_size, ffn_hidden_size)


class MoeMLPRuntime(nn.Module):
    def __init__(self, hidden_size, ffn_hidden_size, num_local_experts):
        super().__init__()
        self.router = RouterRuntime(num_local_experts, hidden_size)
        self.experts = ExpertsRuntime(hidden_size, ffn_hidden_size, num_local_experts)
        self.shared_experts = SharedExpertsRuntime(ffn_hidden_size, hidden_size)


class DecoderLayerRuntime(nn.Module):
    def __init__(self, layer_idx, qkv_rows, hidden_size, kv_lora_rank, ffn_hidden_size, num_local_experts):
        super().__init__()
        self.input_layernorm = NormHolder(hidden_size)
        self.self_attention = SelfAttentionRuntime(qkv_rows, hidden_size, kv_lora_rank)
        self.pre_mlp_layernorm = NormHolder(hidden_size)
        if layer_idx == 0:
            self.mlp = DenseMLPRuntime(ffn_hidden_size, hidden_size)
        else:
            self.mlp = MoeMLPRuntime(hidden_size, ffn_hidden_size, num_local_experts)


class EmbeddingRuntime(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.word_embeddings = WeightHolder(vocab_size, hidden_size)


class DecoderRuntime(nn.Module):
    def __init__(self, num_layers, qkv_rows, hidden_size, kv_lora_rank, ffn_hidden_size, num_local_experts):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DecoderLayerRuntime(
                    i, qkv_rows, hidden_size, kv_lora_rank, ffn_hidden_size, num_local_experts
                )
                for i in range(num_layers)
            ]
        )
        self.final_layernorm = NormHolder(hidden_size)


class OutputLayerRuntime(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(vocab_size, hidden_size))


class DummyGPTModelInferRuntime(nn.Module):
    def __init__(self, num_layers, qkv_rows, hidden_size, kv_lora_rank, ffn_hidden_size, num_local_experts, vocab_size):
        super().__init__()
        self.embedding = EmbeddingRuntime(vocab_size, hidden_size)
        self.decoder = DecoderRuntime(
            num_layers, qkv_rows, hidden_size, kv_lora_rank, ffn_hidden_size, num_local_experts
        )
        self.output_layer = OutputLayerRuntime(vocab_size, hidden_size)


def infer_shapes_from_legacy_model(model_state):
    q_proj = model_state["decoder.layers.0.self_attention.linear_q_proj.weight"]
    kv_down = model_state["decoder.layers.0.self_attention.linear_kv_down_proj.weight"]
    kv_up = model_state["decoder.layers.0.self_attention.linear_kv_up_proj.weight"]
    embed = model_state["embedding.word_embeddings.weight"]
    layer0_fc1 = model_state["decoder.layers.0.mlp.linear_fc1.weight"]
    layer1_fc1 = model_state["decoder.layers.1.mlp.experts.linear_fc1.weight0"]

    hidden_size = q_proj.shape[1]
    qkv_rows = q_proj.shape[0] + kv_down.shape[0]
    kv_lora_rank = kv_up.shape[1]
    vocab_size = embed.shape[0]
    ffn_hidden_size = layer0_fc1.shape[0]
    num_local_experts = 0
    while f"decoder.layers.1.mlp.experts.linear_fc1.weight{num_local_experts}" in model_state:
        num_local_experts += 1

    if num_local_experts == 0:
        num_local_experts = 8

    assert layer1_fc1.shape[0] == ffn_hidden_size, "layer0/layer1 ffn hidden size mismatch"

    return {
        "hidden_size": hidden_size,
        "qkv_rows": qkv_rows,
        "kv_lora_rank": kv_lora_rank,
        "vocab_size": vocab_size,
        "ffn_hidden_size": ffn_hidden_size,
        "num_local_experts": num_local_experts,
    }


def summarize_keys(model_state):
    grouped_layers = []
    legacy_layers = []
    attn_runtime_layers = []
    attn_legacy_layers = []

    for layer_idx in range(NUM_LAYERS):
        if f"decoder.layers.{layer_idx}.mlp.experts.weight1" in model_state:
            grouped_layers.append(layer_idx)
        if f"decoder.layers.{layer_idx}.mlp.experts.linear_fc1.weight0" in model_state:
            legacy_layers.append(layer_idx)
        if f"decoder.layers.{layer_idx}.self_attention.linear_qkv.weight" in model_state:
            attn_runtime_layers.append(layer_idx)
        if f"decoder.layers.{layer_idx}.self_attention.linear_q_proj.weight" in model_state:
            attn_legacy_layers.append(layer_idx)

    return {
        "grouped_layers": grouped_layers,
        "legacy_layers": legacy_layers,
        "attn_runtime_layers": attn_runtime_layers,
        "attn_legacy_layers": attn_legacy_layers,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to model_optim_rng.pt")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model" not in raw or not isinstance(raw["model"], dict):
        raise ValueError("checkpoint does not contain a dict model state")

    raw_model = raw["model"]
    print("raw_num_keys =", len(raw_model))
    print("raw_summary =", summarize_keys(raw_model))

    runtime_args = SimpleNamespace(model_type_hf="deepseek2-lite")
    adapted = adapt_legacy_deepseek2_lite_checkpoint_if_needed(raw, runtime_args)
    adapted_model = adapted["model"]

    print("adapted_num_keys =", len(adapted_model))
    print("adapted_summary =", summarize_keys(adapted_model))

    shapes = infer_shapes_from_legacy_model(raw_model)
    print("inferred_shapes =", shapes)

    runtime_model = DummyGPTModelInferRuntime(
        num_layers=NUM_LAYERS,
        qkv_rows=shapes["qkv_rows"],
        hidden_size=shapes["hidden_size"],
        kv_lora_rank=shapes["kv_lora_rank"],
        ffn_hidden_size=shapes["ffn_hidden_size"],
        num_local_experts=shapes["num_local_experts"],
        vocab_size=shapes["vocab_size"],
    )

    missing, unexpected = runtime_model.load_state_dict(adapted_model, strict=False)
    print("missing_count =", len(missing))
    print("unexpected_count =", len(unexpected))

    if missing:
        print("missing_keys =")
        for key in missing:
            print(key)

    if unexpected:
        print("unexpected_keys =")
        for key in unexpected:
            print(key)

    if not missing and not unexpected:
        print("adapter_runtime_contract_check = PASS")
    else:
        print("adapter_runtime_contract_check = FAIL")


if __name__ == "__main__":
    main()
