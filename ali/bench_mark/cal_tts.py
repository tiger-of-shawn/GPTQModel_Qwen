import sys
import os
import json
from tqdm import tqdm
from jiwer import compute_measures
import numpy as np
from zhon.hanzi import punctuation
import string
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import soundfile as sf
import scipy
import zhconv
from funasr import AutoModel

punctuation_all = punctuation + string.punctuation

wav_res_text_path = sys.argv[1]
# wav_path = f'output/seed_tts_eval_hard'
wav_path = sys.argv[2]
lang = sys.argv[3]  # zh or en
lang = 'zh'
device = "cuda:0"


def load_en_model():
    model_id = "openai/whisper-large-v3"
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)
    return processor, model


def load_zh_model():
    model = AutoModel(model="paraformer-zh", disable_update=True)
    return model


def process_one(hypo, truth):
    raw_truth = truth
    raw_hypo = hypo

    for x in punctuation_all:
        if x == "'":
            continue
        truth = truth.replace(x, "")
        hypo = hypo.replace(x, "")

    truth = truth.replace("  ", " ")
    hypo = hypo.replace("  ", " ")

    if lang == "zh":
        truth = " ".join([x for x in truth])
        hypo = " ".join([x for x in hypo])
    elif lang == "en":
        truth = truth.lower()
        hypo = hypo.lower()
    else:
        raise NotImplementedError

    measures = compute_measures(truth, hypo)
    ref_list = truth.split(" ")
    wer = measures["wer"]
    subs = measures["substitutions"] / len(ref_list)
    dele = measures["deletions"] / len(ref_list)
    inse = measures["insertions"] / len(ref_list)
    return (raw_truth, raw_hypo, wer, subs, dele, inse)


def run_asr(wav_res_text_path, wav_path):
    if lang == "en":
        processor, model = load_en_model()
    elif lang == "zh":
        model = load_zh_model()

    params = []
    for line in open(wav_res_text_path).readlines():
        data = json.loads(line.strip())

        wav_res_path = os.path.join(wav_path, f"{data['id']}.wav")
        if not os.path.exists(wav_res_path):
            continue
        params.append((wav_res_path, data["gt"]))

    wers = []
    for wav_res_path, text_ref in tqdm(params):
        if lang == "en":
            wav, sr = sf.read(wav_res_path)
            if sr != 16000:
                wav = scipy.signal.resample(wav, int(len(wav) * 16000 / sr))
            input_features = processor(
                wav, sampling_rate=16000, return_tensors="pt"
            ).input_features
            input_features = input_features.to(device)
            forced_decoder_ids = processor.get_decoder_prompt_ids(
                language="english", task="transcribe"
            )
            predicted_ids = model.generate(
                input_features, forced_decoder_ids=forced_decoder_ids
            )
            transcription = processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0]
        elif lang == "zh":
            res = model.generate(input=wav_res_path, batch_size_s=300)
            transcription = res[0]["text"]
            transcription = zhconv.convert(transcription, "zh-cn")

        raw_truth, raw_hypo, wer, subs, dele, inse = process_one(
            transcription, text_ref
        )
        wers.append(wer)
    print(np.mean(wers))


run_asr(wav_res_text_path, wav_path)
