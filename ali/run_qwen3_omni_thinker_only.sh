#!/bin/bash

# qwen2.5-omni
# python3 '/nas/yuehu/NEW/GPTQModel_Qwen/ali/quant_qwen_omni.py' --model_path /nas/yuehu/models/omni/Qwen2.5-Omni-3B --cuda_device 0 --quant_type thinker-only --dataset_count 8

# qwen3-omni thinker-only
export PYTHONPATH="/nas/yuehu/NEW_H20/transformers-internal-q3o-dense-3d1a4f5e34753e51cb85052539c6ef10cab9a5c1/src:$PYTHONPATH"
python3 '/nas/yuehu/NEW/GPTQModel_Qwen/ali/quant_qwen_omni3.py' \
        --model_path /nas/yuehu/models/omni/Qwen3-Omni-4B-Instruct-multilingual \
        --cuda_device 2 --quant_type thinker-only --dataset_count 8


