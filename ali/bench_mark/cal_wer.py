import json
import sys
from metrics import calculate_wer

preds, gts = [], []
lang = None
with open(sys.argv[1]) as f:
    for line in f:
        data = json.loads(line)
        preds.append(data["response"])
        gts.append(data["gt"])
        lang = data["lang"]
        task = data["task"]

wer = calculate_wer(preds, gts, lang)

print(f"{task} WER is {wer:.4f}")
