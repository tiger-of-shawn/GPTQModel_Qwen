#!/bin/sh
#****************************************************************#
# ScriptName: run.sh
# Author: @alibaba-inc.com
# Create Date: 2025-06-10 23:01
# Modify Author: @alibaba-inc.com
# Modify Date: 2025-06-10 23:01
# Function: 
#***************************************************************#

python3 ./ali/quant_qwen_omni.py --model_path /nas/yuehu/models/omni/Qwen2.5-Omni-3B --cuda_device 2 --quant_type thinker-only
