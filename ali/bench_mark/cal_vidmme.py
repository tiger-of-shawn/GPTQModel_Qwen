import json
import sys
from metrics import videomme_extract


correct, total, invalid = 0, 0, 0

with open(sys.argv[1], "r") as f:
    for line in f:
        data = json.loads(line)
        pred = data["response"]
        gt = data["gt"]
        task = data["task"]

        pred_ans = videomme_extract(pred)

        if pred_ans is None:
            invalid += 1
            continue

        total += 1
        if pred_ans == gt:
            correct += 1

print(f"VideoMME Acc: {correct / total}, invalid: {invalid}")
