#!/bin/bash

# DeepSeek2-Lite HF -> MindSpeed-LLM mcore checkpoint convert
# Filled from the successful mg2hf command:
#   HF weights dir: /apdcephfs_jn2/share_304376610/lsy/ckpt_mcore2hf/
#   Source mcore dir: /apdcephfs_jn2/share_304376610/lsy/tmp
#   Tokenizer cfg dir: /apdcephfs_nj11/share_304376610/lqc/ds2_lite_config/ds2_tokenizer

source /usr/local/Ascend/ascend-toolkit/set_env.sh

python3 convert_ckpt_v2.py \
    --moe-grouped-gemm \
    --model-type-hf deepseek2-lite \
    --load-model-type hf \
    --save-model-type mg \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --target-expert-parallel-size 8 \
    --load-dir /apdcephfs_jn2/share_304376610/lsy/ckpt_mcore2hf/ \
    --save-dir /apdcephfs_jn2/share_304376610/lsy/ckpt_hf2mcore_ep8/ \
    | tee ./convert_hf2mcore.log
