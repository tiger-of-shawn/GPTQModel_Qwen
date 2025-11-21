#!/bin/bash

# qwen2.5-omni thinker-only
# python3 '/nas/yuehu/NEW/GPTQModel_Qwen/ali/quant_qwen_omni.py' --model_path /nas/yuehu/models/omni/Qwen2.5-Omni-3B --cuda_device 0 --quant_type thinker-only --dataset_count 8


# qwen2.5-omni thinker-talker
python3 '/nas/yuehu/NEW/GPTQModel_Qwen/ali/quant_qwen_omni.py' \
        --model_path /nas/yuehu/models/omni/Qwen2.5-Omni-3B \
        --cuda_device 0 \
        --quant_type thinker-talker \
        --dataset_count 8
