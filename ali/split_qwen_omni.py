from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor, AutoProcessor, AutoModelForTextToWaveform

model_path = '/nas/yuehu/models/omni/Qwen2.5-Omni-3B'

thinker_model_saved_path = '/nas/yuehu/models/omni/Qwen2.5-Omni-3B-thinker'
talker_model_saved_path = '/nas/yuehu/models/omni/Qwen2.5-Omni-3B-talker'
token2wav_model_saved_path = '/nas/yuehu/models/omni/Qwen2.5-Omni-3B-token2wav'

# default: Load the model on the available device(s)
# model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto", device_map="auto")

# We recommend enabling flash_attention_2 for better acceleration and memory saving.
model = AutoModelForTextToWaveform.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2",
)
processor = AutoProcessor.from_pretrained(model_path)

model.thinker.save_pretrained(thinker_model_saved_path)
model.talker.save_pretrained(talker_model_saved_path)
model.token2wav.save_pretrained(token2wav_model_saved_path)




