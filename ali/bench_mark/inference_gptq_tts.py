import torch
from typing import Any, Dict
import sys
# 将本地路径插入到 sys.path 的最前面
sys.path.insert(0, '/nas/yuehu/NEW/GPTQModel_Qwen')
import os
import argparse
from functools import partial
from tts import tts as _tts
import soundfile as sf

from transformers import (
    Qwen2_5OmniForConditionalGeneration, 
    Qwen2_5OmniProcessor, 
)
from transformers.utils.hub import cached_file
from transformers.generation.utils import GenerateOutput

from gptqmodel import GPTQModel, QuantizeConfig, BACKEND
from gptqmodel.models.base import BaseGPTQModel
from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models._const import CPU

from qwen_omni_utils import process_mm_info


import argparse
import json
from more_itertools import batched

USE_AUDIO_IN_VIDEO = False

parser = argparse.ArgumentParser()
parser.add_argument("--input-file")
parser.add_argument("--output_path")
parser.add_argument("--model-path")
parser.add_argument("--start-index", default=-1, type=int)
parser.add_argument("--end-index", default=99999999, type=int)
parser.add_argument("--batch-size", default=4, type=int)
args = parser.parse_args()

model_path = args.model_path

layers_node = []
if 'thinker-only' in model_path:
    layers_node = ["thinker.model.layers"]
elif 'thinker-talker' in model_path:
    layers_node = ["thinker.model.layers", "talker.model.layers"]
else:
    print(f'model not support yet.')
    exit(0)   
    
# FP Model
if 'GPTQ' in model_path:
    model = GPTQModel.load(
        model_path, 
        device_map="auto", 
        torch_dtype="auto",   
        attn_implementation="flash_attention_2", layers_node_user=layers_node
    )
    spk_path = model_path + '/spk_dict.pt'
    model.model.load_speakers(spk_path)
else:
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )

from qwen_omni_utils import process_mm_info
processor = Qwen2_5OmniProcessor.from_pretrained(model_path)

tts = partial(_tts, model=model, processor=processor, speaker="Chelsie")

os.makedirs(args.output_path, exist_ok=True)
index = 0
with open(args.input_file) as f:
    for line in f:
        if index in range(args.start_index, args.end_index):
            data = json.loads(line)
            text = data["gt"]
            
            audio = tts(text=text)
            sf.write(
                os.path.join(args.output_path, f"{data['id']}.wav"),
                audio.reshape(-1).detach().cpu().numpy(),
                samplerate=24000,
            )
            print(f'inference index: {index}')
        index += 1