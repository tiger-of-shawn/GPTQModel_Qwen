
import sys
# 将本地路径插入到 sys.path 的最前面
sys.path.insert(0, '/nas/yuehu/NEW/GPTQModel_Qwen')
import argparse

from gptqmodel.models.definitions.base_qwen2_5_omni import BaseQwen2_5_OmniGPTQ

import gptqmodel
import shutil

from gptqmodel import GPTQModel
from gptqmodel.utils.eval import EVAL
from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig
from transformers import AutoProcessor
from qwen_omni_utils import process_mm_info
import soundfile as sf
import torch
import os 
import json
def process_json_file(json_path, audio_data_path, n_sample):
    result = []
    # 打开并逐行读取文件
    with open(json_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 去除每行末尾的换行符
            line = line.strip()
            if not line:
                continue
            
            # 解析 JSON 数据
            try:
                json_obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                continue
            
            # 遍历 messages 寻找并更新 audio 字段
            if 'messages' in json_obj:
                for message in json_obj['messages']:
                    if 'content' in message:
                        for content_item in message['content']:
                            if 'audio' in content_item:
                                # 修改 audio 路径
                                original_audio_path = content_item['audio']
                                content_item['audio'] = "file://" + os.path.join(audio_data_path, original_audio_path)
                                content_item['type'] = 'audio'
                            if 'text' in content_item:
                                content_item['type'] = 'text'
                            if 'raw_text' in content_item:
                                content_item['text'] = content_item['raw_text']
            # 将处理过的 JSON 对象添加到结果列表
            result.append(json_obj['messages'])
            if len(result) >= n_sample:
                break
    print(f'result:  {result}')
    return result

# 示例调用
# json_file_path = "path/to/your/json_file.json"
# audio_path = "path/to/your/audio_data"
# processed_data = process_json_file(json_file_path, audio_path)
# print(processed_data)

# data_type: 可选为
# text-image: 图文对
# text: 仅文字
# text-video: 视频文本音频对
def prepare_dataset(n_sample: int = 8, data_type: str = 'text-image') -> list[list[dict]]:
    from datasets import load_dataset

    if data_type == 'text':
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=f"train[:{n_sample}]")
        return [
            [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                    ],
                },            
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": sample['text']},
                    ],
                }
            ]
            for sample in dataset
        ]
    elif data_type == 'text-video':
        json_file = '/nas/yuehu/assets/dataset/omni/calibration_data/filtered_output_100.json'
        pass
    elif data_type == 'audio':
        audio_path = '/nas/yuehu/assets/dataset/omni/calibration_data/data'
        json_file = '/nas/yuehu/assets/dataset/omni/calibration_data/audio.json.out.fixed_10.json'
        
        return process_json_file(json_file, audio_path, n_sample)
        
    elif data_type == 'text-image':
        dataset = load_dataset("laion/220k-GPT4Vision-captions-from-LIVIS", split=f"train[:{n_sample}]")
        return [
            [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                    ],
                },            
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": sample["url"]},
                        {"type": "text", "text": "generate a caption for this image"},
                    ],
                },
                {"role": "assistant", "content": sample["caption"]},
            ]
            for sample in dataset
        ]        
    else:
        print(f'data_type: {data_type} is not supported yet.')
        return []



def quantize(model_path, quant_path, layers_to_convert, n_samples=8):

    # calibration_dataset = load_dataset(
    #     "wikitext",
    #     "wikitext-2-raw-v1",
    #     split="train"
    # ).select(range(1024))["text"]
    calibration_dataset = prepare_dataset(n_sample=n_samples)
    
    quant_config = QuantizeConfig(bits=4, group_size=128)

    model = GPTQModel.load(model_path, quant_config)

    layers_node = []
    layers_modules_tree = []
    if 'thinker' in layers_to_convert:
        layers_node.append("thinker.model.layers")
        layers_modules_tree.append([
        "thinker",
        "model",
        "layers",
        "#",
        {
            "self_attn": ("k_proj", "v_proj", "q_proj", "o_proj"),
            "mlp": ("up_proj", "gate_proj", "down_proj"),
        }
        ])
    if 'talker' in layers_to_convert:
        layers_node.append("talker.model.layers")
        layers_modules_tree.append([
        "talker",
        "model",
        "layers",
        "#",
        {
            "self_attn": ("k_proj", "v_proj", "q_proj", "o_proj"),
            "mlp": ("up_proj", "gate_proj", "down_proj"),
        }
        ])
    model.layers_node = layers_node
    model.layers_modules_tree = layers_modules_tree
    # increase `batch_size` to match gpu/vram specs to speed up quantization
    model.quantize(calibration_dataset, batch_size=4)

    model.save(quant_path)
    
    spk_dict_path = model_path + '/spk_dict.pt'
    shutil.copy(spk_dict_path, quant_path)     

def inference(quant_path, post_fix, layers_to_convert):
    device = 'cuda:0'

    layers_node = []
    if 'thinker' in layers_to_convert:
        layers_node.append("thinker.model.layers")
    if 'talker' in layers_to_convert:
        layers_node.append("talker.model.layers")
            
    model = GPTQModel.load(quant_path,   
        attn_implementation="flash_attention_2", layers_node_user=layers_node)
    processor = AutoProcessor.from_pretrained(quant_path)
    
    spk_path = quant_path + '/spk_dict.pt'
    model.model.load_speakers(spk_path)
        
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"},
                # {"type": "text", "text": "generate a caption for this image"},
                {"type": "text", "text": "描述一下图片内容"},
            ],
        },
    ]
    USE_AUDIO_IN_VIDEO = False

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    audios, images, videos = process_mm_info(messages, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)

    model.model.thinker.model.embed_tokens = model.model.thinker.model.embed_tokens.to(device)
    model.model.thinker.visual = model.model.thinker.visual.to(device)
    model.model.thinker.audio_tower = model.model.thinker.audio_tower.to(device)
    model.model.thinker.visual.rotary_pos_emb = model.model.thinker.visual.rotary_pos_emb.to(device)
    model.model.thinker.model.rotary_emb = model.model.thinker.model.rotary_emb.to(device)

    for layer in model.model.thinker.model.layers:
        layer.self_attn.rotary_emb = layer.self_attn.rotary_emb.to(device)

    inputs = inputs.to(device)

    return_audio = True
    if return_audio:
        with torch.no_grad():
            text_ids, audio = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO, max_new_tokens=512, return_audio = True)
        sf.write(
            f"output_{post_fix}.wav",
            audio.reshape(-1).detach().cpu().numpy(),
            samplerate=24000,
        )
    else:
        text_ids = model.generate(**inputs, max_new_tokens=512, return_audio = False)

    text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    print(f'generation_output:  {text}')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize a model with specified parameters.")

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the original model."
    )
    parser.add_argument(
        "--cuda_device",
        type=str,
        default='0', # Default to '0' if not specified
        help="CUDA device to use (e.g., '0', '1')."
    )
    parser.add_argument(
        "--quant_type",
        type=str,
        choices=['thinker-talker', 'thinker-only', 'talker-only'],
        required=True,
        help="Type of quantization: 'thinker-talker', 'thinker-only', or 'talker-only'."
    )
    parser.add_argument(
        "--dataset_count",
        type=int,
        default=8,
        help="采用几条数据集进行量化"
    )
    args = parser.parse_args()

    # Automatically construct quant_path based on model_path and quant_type
    model_base_name = os.path.basename(args.model_path)
    model_dir = os.path.dirname(args.model_path)
    
    if args.quant_type == 'thinker-talker':
        quant_suffix = 'GPTQ-thinker-talker'
        layers_to_convert = ['thinker', 'talker']
    elif args.quant_type == 'thinker-only':
        quant_suffix = 'GPTQ-thinker-only'
        layers_to_convert = ['thinker']
    elif args.quant_type == 'talker-only':
        quant_suffix = 'GPTQ-talker-only'
        layers_to_convert = ['talker']
    else:
        # This case should ideally not be reached due to 'choices' in argparse
        raise ValueError("Invalid quant_type specified.")

    quant_path = os.path.join(model_dir, f"{model_base_name}-{quant_suffix}-text-image-pair-{args.dataset_count}")
    print(f'quant_path:  {quant_path}')

    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
    
    quantize(model_path=args.model_path, quant_path=quant_path, layers_to_convert=layers_to_convert, n_samples=args.dataset_count)

    inference(quant_path, quant_suffix, layers_to_convert=layers_to_convert)
