import torch
from typing import Any, Dict
import sys
# 将本地路径插入到 sys.path 的最前面
sys.path.insert(0, '/nas/yuehu/NEW/GPTQModel_Qwen')

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
parser.add_argument("--output-file")
parser.add_argument("--model_path")
parser.add_argument("--batch-size", default=4, type=int)
args = parser.parse_args()

model_path = args.model_path

# FP Model
if 'GPTQ' in model_path:
    model = GPTQModel.load(
        model_path, 
        device_map="auto", 
        torch_dtype="auto",   
        attn_implementation="flash_attention_2"
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

with open(args.input_file) as f, open(args.output_file, "w") as fw:
    for lines in batched(f, args.batch_size):
        datas = [json.loads(line) for line in lines]

        conversations = [data["prompt"] for data in datas]

        text = processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=False,
        )
        audios, images, videos = process_mm_info(
            conversations, use_audio_in_video=USE_AUDIO_IN_VIDEO
        )

        inputs = processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=USE_AUDIO_IN_VIDEO,
        )
        inputs = inputs.to(model.device).to(model.dtype)
        text_ids = model.generate(
            **inputs,
            use_audio_in_video=USE_AUDIO_IN_VIDEO,
            thinker_do_sample=False,
            return_audio=False,
            repetition_penalty=1.0,
        )
        generated_ids_list = [
            text_ids[i][len(inputs["input_ids"][i]) :] for i in range(len(datas))
        ]
        response_text = processor.batch_decode(
            generated_ids_list,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        for data, response in zip(datas, response_text):
            fw.write(json.dumps(data | {"response": response}) + "\n")
            fw.flush()
