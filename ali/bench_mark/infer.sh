# Omnibench
python inference.py --input-file data/jsonls/omnibench.jsonl --output-file output/omnibench.jsonl --batch-size 4
python cal_omni.py output/omnibench.jsonl

# LibriSpeech test-other
python inference.py --input-file data/jsonls/libri_test_other.jsonl --output-file output/libri_test_other.jsonl --batch-size 4
python cal_wer.py output/libri_test_other.jsonl

# WenetSpeech test-net
python inference.py --input-file data/jsonls/wenetspeech_test_net.jsonl --output-file output/wenetspeech_test_net.jsonl --batch-size 4
python cal_wer.py output/wenetspeech_test_net.jsonl

# MMLU-Pro
python inference.py --input-file data/jsonls/mmlu_pro.jsonl --output-file output/mmlu_pro.jsonl --batch-size 4
python cal_mmlu.py output/mmlu_pro.jsonl

# VideoMME
python inference.py --input-file data/jsonls/videomme_50.jsonl --output-file output/videomme_50.jsonl --batch-size 1
python cal_vidmme.py output/videomme_50.jsonl

# SeedTTS-Eval-Hard
python inference_tts.py --input-file data/jsonls/seedtts_eval_hard.jsonl --output-path output/seed_tts_eval_hard
python cal_tts.py data/jsonls/seedtts_eval_hard.jsonl output/seed_tts_eval_hard zh


