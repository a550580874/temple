from pathlib import Path
from tempfile import TemporaryDirectory
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
    def __init__(self):
        super().__init__()
        self.linear_qkv = WeightHolder(Q_PROJ_ROWS + KV_DOWN_ROWS, HIDDEN_SIZE)
        self.linear_kv_up_proj = WeightHolder(HIDDEN_SIZE, KV_LORA_RANK)
        self.kv_layernorm = NormHolder(KV_LORA_RANK)
        self.linear_proj = WeightHolder(HIDDEN_SIZE, HIDDEN_SIZE)


class DenseMLPRuntime(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_fc1 = WeightHolder(FFN_HIDDEN_SIZE, HIDDEN_SIZE)
        self.linear_fc2 = WeightHolder(HIDDEN_SIZE, FFN_HIDDEN_SIZE)


class RouterRuntime(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(NUM_LOCAL_EXPERTS, HIDDEN_SIZE))
        self.expert_bias = nn.Parameter(torch.zeros(NUM_LOCAL_EXPERTS))


class ExpertsRuntime(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight1 = nn.Parameter(torch.zeros(HIDDEN_SIZE, NUM_LOCAL_EXPERTS * FFN_HIDDEN_SIZE))
        self.weight2 = nn.Parameter(torch.zeros(NUM_LOCAL_EXPERTS * FFN_HIDDEN_SIZE, HIDDEN_SIZE))


class SharedExpertsRuntime(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_fc1 = WeightHolder(FFN_HIDDEN_SIZE, HIDDEN_SIZE)
        self.linear_fc2 = WeightHolder(HIDDEN_SIZE, FFN_HIDDEN_SIZE)


class MoeMLPRuntime(nn.Module):
    def __init__(self):
        super().__init__()
        self.router = RouterRuntime()
        self.experts = ExpertsRuntime()
        self.shared_experts = SharedExpertsRuntime()


class DecoderLayerRuntime(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        self.input_layernorm = NormHolder(HIDDEN_SIZE)
        self.self_attention = SelfAttentionRuntime()
        self.pre_mlp_layernorm = NormHolder(HIDDEN_SIZE)
        if layer_idx == 0:
            self.mlp = DenseMLPRuntime()
        else:
            self.mlp = MoeMLPRuntime()


class EmbeddingRuntime(nn.Module):
    def __init__(self):
        super().__init__()
        self.word_embeddings = WeightHolder(VOCAB_SIZE, HIDDEN_SIZE)


class DecoderRuntime(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayerRuntime(i) for i in range(NUM_LAYERS)])
        self.final_layernorm = NormHolder(HIDDEN_SIZE)


class OutputLayerRuntime(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(VOCAB_SIZE, HIDDEN_SIZE))


class DummyGPTModelInferRuntime(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = EmbeddingRuntime()
        self.decoder = DecoderRuntime()
        self.output_layer = OutputLayerRuntime()


def _randn(*shape):
    return torch.randn(*shape, dtype=torch.float32)


def build_full_fake_legacy_model_state():
    model = {}
    model["embedding.word_embeddings.weight"] = _randn(VOCAB_SIZE, HIDDEN_SIZE)
    model["decoder.final_layernorm.weight"] = _randn(HIDDEN_SIZE)
    model["output_layer.weight"] = _randn(VOCAB_SIZE, HIDDEN_SIZE)

    for layer_idx in range(NUM_LAYERS):
        prefix = f"decoder.layers.{layer_idx}"
        model[f"{prefix}.input_layernorm.weight"] = _randn(HIDDEN_SIZE)
        model[f"{prefix}.self_attention.linear_q_proj.weight"] = _randn(Q_PROJ_ROWS, HIDDEN_SIZE)
        model[f"{prefix}.self_attention.linear_kv_down_proj.weight"] = _randn(KV_DOWN_ROWS, HIDDEN_SIZE)
        model[f"{prefix}.self_attention.linear_kv_up_proj.weight"] = _randn(HIDDEN_SIZE, KV_LORA_RANK)
        model[f"{prefix}.self_attention.linear_kv_up_proj.layer_norm_weight"] = _randn(KV_LORA_RANK)
        model[f"{prefix}.self_attention.linear_proj.weight"] = _randn(HIDDEN_SIZE, HIDDEN_SIZE)

        if layer_idx == 0:
            model[f"{prefix}.mlp.linear_fc1.layer_norm_weight"] = _randn(HIDDEN_SIZE)
            model[f"{prefix}.mlp.linear_fc1.weight"] = _randn(FFN_HIDDEN_SIZE, HIDDEN_SIZE)
            model[f"{prefix}.mlp.linear_fc2.weight"] = _randn(HIDDEN_SIZE, FFN_HIDDEN_SIZE)
            continue

        model[f"{prefix}.pre_mlp_layernorm.weight"] = _randn(HIDDEN_SIZE)
        model[f"{prefix}.mlp.router.weight"] = _randn(NUM_LOCAL_EXPERTS, HIDDEN_SIZE)
        model[f"{prefix}.mlp.router.expert_bias"] = _randn(NUM_LOCAL_EXPERTS)
        model[f"{prefix}.mlp.shared_experts.linear_fc1.weight"] = _randn(FFN_HIDDEN_SIZE, HIDDEN_SIZE)
        model[f"{prefix}.mlp.shared_experts.linear_fc2.weight"] = _randn(HIDDEN_SIZE, FFN_HIDDEN_SIZE)
        for expert_idx in range(NUM_LOCAL_EXPERTS):
            model[f"{prefix}.mlp.experts.linear_fc1.weight{expert_idx}"] = _randn(
                FFN_HIDDEN_SIZE, HIDDEN_SIZE
            )
            model[f"{prefix}.mlp.experts.linear_fc2.weight{expert_idx}"] = _randn(
                HIDDEN_SIZE, FFN_HIDDEN_SIZE
            )

    return model


def main():
    with TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "model_optim_rng.pt"
        torch.save({"model": build_full_fake_legacy_model_state()}, ckpt_path)

        args = SimpleNamespace(model_type_hf="deepseek2-lite")
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        adapted = adapt_legacy_deepseek2_lite_checkpoint_if_needed(state_dict, args)

        model = DummyGPTModelInferRuntime()
        missing, unexpected = model.load_state_dict(adapted["model"], strict=False)

        assert not missing, f"missing keys: {missing}"
        assert not unexpected, f"unexpected keys: {unexpected}"

        print("full synthetic checkpoint load passed")
        print("num_state_keys =", len(adapted["model"]))
        print(
            "layer1 weight1 shape =",
            tuple(adapted["model"]["decoder.layers.1.mlp.experts.weight1"].shape),
        )
        print(
            "layer1 linear_qkv shape =",
            tuple(adapted["model"]["decoder.layers.1.self_attention.linear_qkv.weight"].shape),
        )


if __name__ == "__main__":
    main()
