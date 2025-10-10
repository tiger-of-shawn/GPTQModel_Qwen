
import torch
import time
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info # Assuming this is available

model_path = "/nas/yuehu/models/vl/Qwen2.5-VL-3B-Instruct"
# model_path = "/mnt/yuehu/models/vl/qwen2_5_vl_3b_abacus_clock_count_solve_eval_sample_grpo"
device = "cuda:1" # 明确指定设备，简化代码

# ----------------- 1. 模型和处理器加载 -----------------
print("Loading model and processor...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto", # 使用 device_map="auto" 时，模型已经在GPU上
)
processor = AutoProcessor.from_pretrained(model_path)

# 确保模型和输入在同一个设备上，如果 model_map="auto" 已经处理，这一行可能可选，但为保险起见保留
model = model.to(device) 
print("Model loaded.")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "描述一下图片内容"},
        ],
    }
]

# ----------------- 2. 输入准备 (Prefill Token Count) -----------------
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)

inputs = inputs.to(device)
prefill_token_count = inputs.input_ids.shape[1] # Prefill token 数量
print(f"Prefill token count: {prefill_token_count}")

# ----------------- 3. Prefill 速度统计 -----------------
# 确保 GPU 操作完成，以便准确计时
if torch.cuda.is_available():
    torch.cuda.synchronize()
prefill_start_time = time.time()

# 使用 model.forward() 进行 Prefill (首次前向传播)
with torch.no_grad():
    # 只需要计算首次前向传播的结果，不需要 generate 的全部逻辑
    # forward 已经包含了所有 vision tower 和 text model 的 prefill 计算
    outputs = model(**inputs, return_dict=True)

if torch.cuda.is_available():
    torch.cuda.synchronize()
prefill_end_time = time.time()
prefill_time = prefill_end_time - prefill_start_time
prefill_speed = prefill_token_count / prefill_time if prefill_time > 0 else float('inf')


# ----------------- 4. Decode 速度统计 -----------------
max_new_tokens = 128
# 确保 GPU 操作完成，以便准确计时
if torch.cuda.is_available():
    torch.cuda.synchronize()
decode_start_time = time.time()

# Inference: Generation of the output
# 注意：这里调用 generate 会包含 prefill + decode 过程
# 但由于我们前面已经单独计时了 prefill，这里主要关注总时间减去 prefill 时间（近似）
# 或者更准确的做法是：只用 generate 计时，然后减去 prefill 耗时
generated_ids = model.generate(
    **inputs,
    max_new_tokens=max_new_tokens,
    return_dict_in_generate=True, # 返回更详细的信息
    output_scores=True, # 确保返回生成的信息
)

if torch.cuda.is_available():
    torch.cuda.synchronize()
decode_end_time = time.time()
total_generation_time = decode_end_time - decode_start_time

# 计算实际生成的 token 数量 (Decode Token Count)
# total_output_ids = generated_ids.sequences
total_output_ids = generated_ids.sequences
# 实际生成的 token 数量 = 总输出 token - 输入 token (prefill token)
actual_decoded_token_count = total_output_ids.shape[1] - prefill_token_count
print(f"Actual decoded token count: {actual_decoded_token_count}")

# 近似 Decode Time: 总生成时间 - Prefill 时间
decode_time_approx = total_generation_time - prefill_time
decode_speed = actual_decoded_token_count / decode_time_approx if decode_time_approx > 0 else float('inf')

# ----------------- 5. 输出结果 -----------------
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, total_output_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

print("\n" + "="*50)
print("Output Text:")
print(output_text)
print("="*50)

print("\n" + "⭐️ **Performance Metrics** ⭐️")
print(f"| Metric | Value | Unit |")
print(f"|---|---|---|")
print(f"| **Prefill Tokens** | {prefill_token_count} | tokens |")
print(f"| **Prefill Time** | {prefill_time:.4f} | seconds |")
print(f"| **Prefill Speed** | **{prefill_speed:.2f}** | tokens/s |")
print(f"| **---** | **---** | **---** |")
print(f"| **Decode Tokens** | {actual_decoded_token_count} | tokens |")
print(f"| **Total Generation Time** | {total_generation_time:.4f} | seconds |")
print(f"| **Approx. Decode Time** | {decode_time_approx:.4f} | seconds |")
print(f"| **Decode Speed** | **{decode_speed:.2f}** | tokens/s |")
print("="*50)