import soundfile as sf
import os 
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor, AutoProcessor, AutoModelForTextToWaveform
from qwen_omni_utils import process_mm_info

model_path = '/nas/yuehu/models/omni/Qwen2.5-Omni-3B-AWQ-thinker'
# model_path = '/nas/yuehu/models/omni/Qwen2.5-Omni-3B'

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# default: Load the model on the available device(s)
# model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto", device_map="auto")

# We recommend enabling flash_attention_2 for better acceleration and memory saving.
model = AutoModelForTextToWaveform.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_path)

conversation = [
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
            {"type": "text", "text": "描述一下图片内容"},
        ],
    },
]

# set use audio in video
USE_AUDIO_IN_VIDEO = False

# Preparation for inference
text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)

inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)

# from transformers import AutoConfig, AutoTokenizer, PretrainedConfig
# tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# tokenized = tokenizer(text, return_tensors="pt")

inputs = inputs.to(model.device).to(model.dtype)

# Inference: Generation of the output text and audio
return_audio = True
if return_audio:
    text_ids, audio = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO, return_audio = True)
    sf.write(
        "output.wav",
        audio.reshape(-1).detach().cpu().numpy(),
        samplerate=24000,
    )
else:
    text_ids = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO, return_audio = False)
    
text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(text)
