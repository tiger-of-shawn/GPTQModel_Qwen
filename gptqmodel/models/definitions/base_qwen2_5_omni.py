# Copyright 2024-2025 ModelCloud.ai
# Copyright 2024-2025 qubitium@modelcloud.ai
# Contact: qubitium@modelcloud.ai, x.com/qubitium
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, Optional

from PIL import Image
from transformers import AutoModelForTextToWaveform, AutoProcessor, ProcessorMixin

from ...utils.calibration import batched
from ...utils.image import extract_vision_info, fetch_image
from ...utils.model import MODALITY, move_to
from .._const import CPU
from ..base import BaseGPTQModel
from qwen_omni_utils import process_mm_info

class BaseQwen2_5_OmniGPTQ(BaseGPTQModel):
    loader = AutoModelForTextToWaveform

    base_modules = ["talker.model.embed_tokens", "talker.model.norm"]
    pre_lm_head_norm_module = "talker.model.norm"

    layers_node = ["thinker.model.layers", "talker.model.layers"]

    layer_modules = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        ["mlp.up_proj", "mlp.gate_proj"],
        ["mlp.down_proj"],
    ]

    layers_modules_tree = [[
        "thinker",
        "model",
        "layers",
        "#",
        {
            "self_attn": ("k_proj", "v_proj", "q_proj", "o_proj"),
            "mlp": ("up_proj", "gate_proj", "down_proj"),
        }
    ],[
        "talker",
        "model",
        "layers",
        "#",
        {
            "self_attn": ("k_proj", "v_proj", "q_proj", "o_proj"),
            "mlp": ("up_proj", "gate_proj", "down_proj"),
        }
    ]]

    modality = [MODALITY.TEXT, MODALITY.IMAGE_TO_TEXT]

    require_load_processor = True

    support_batch_quantize = False
    
    def move_model_to_device(self, model, device):
        model.to(device)
        # thinker
        thinker = model.thinker
        thinker.audio_tower.conv1.to(device)
        thinker.audio_tower.conv2.to(device)
        thinker.audio_tower.positional_embedding.to(device)
        thinker.audio_tower.audio_bos_eos_token.to(device)
        for layer in thinker.audio_tower.layers:
            layer.to(device)
        thinker.audio_tower.ln_post.to(device)
        thinker.audio_tower.avg_pooler.to(device)
        thinker.audio_tower.proj.to(device)

        thinker.visual.patch_embed.to(device)
        thinker.visual.rotary_pos_emb.to(device)
        for block in thinker.visual.blocks:
            block.to(device)
        thinker.visual.merger.to(device)

        thinker.model.embed_tokens.to(device)
        for layer in thinker.model.layers:
            layer.to(device)
        thinker.model.norm.to(device)
        thinker.model.rotary_emb.to(device)
        thinker.lm_head.to(device)

        # talker
        talker = model.talker
        talker.thinker_to_talker_proj.to(device)
        talker.model.embed_tokens.to(device)
        for layer in talker.model.layers:
            layer.to(device)
        talker.model.norm.to(device)
        talker.model.rotary_emb.to(device)
        talker.codec_head.to(device)

        # token2wav
        token2wav = model.token2wav
        token2wav.code2wav_dit_model.to(device)
        token2wav.code2wav_bigvgan_model.to(device)
    def pre_quantize_generate_hook_start(self):
        self.move_model_to_device(self.model, self.quantize_config.device)
    def pre_quantize_generate_hook_end(self):
        self.move_model_to_device(self.model, CPU)
       
    @staticmethod
    def process_vision_info(
            conversations: list[dict] | list[list[dict]],
    ) -> Optional[list[Image.Image]]:
        vision_infos = extract_vision_info(conversations)
        # Read images
        image_inputs = []
        for vision_info in vision_infos:
            if "image" in vision_info or "image_url" in vision_info:
                image_inputs.append(fetch_image(vision_info))
            else:
                raise ValueError("image, image_url should in content.")
        if len(image_inputs) == 0:
            image_inputs = None
        return image_inputs

    def preprocess_dataset(self, sample: Dict) -> Dict:
        return sample

    def load_processor(self) -> ProcessorMixin:
        return AutoProcessor.from_pretrained(self.model_local_path)

    def prepare_dataset(self, calibration_dataset, calibration_dataset_concat_size=None, batch_size: int = 1):
        processor = self.load_processor()
        
        calib_data = []
        for batch in batched(calibration_dataset, batch_size, process_func=self.preprocess_dataset):
            text = processor.apply_chat_template(batch, tokenize=False, add_generation_prompt=True)
            audios, images, videos = process_mm_info(batch, use_audio_in_video=False)
            inputs = processor(text=text, images=images, videos=videos, audio=audios, padding=True, return_tensors="pt")
            
            calib_data.append(inputs)
        del processor
        return calib_data
        