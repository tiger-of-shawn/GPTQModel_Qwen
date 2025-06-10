import json
import sys
from collections import defaultdict

from metrics import omnibench_parse


types = defaultdict(lambda: defaultdict(lambda: 0))

with open(sys.argv[1], "r") as f:
    for line in f:
        data = json.loads(line)
        pred = data["response"]
        gt = data["gt"]
        task = data["task"]
        audio_type = data["audio_type"]
        task_type = data["task_type"]

        pred_ans = omnibench_parse(pred)
        gt_ans = omnibench_parse(gt)

        if pred_ans == gt_ans:
            types[audio_type]["correct"] += 1
        types[audio_type]["total"] += 1

correct, total = 0, 0
for t, d in types.items():
    print(f"{t}: {d['correct']} / {d['total']}, Acc: {d['correct'] / d['total']}")
    correct += d["correct"]
    total += d["total"]
print(f"OmniBench: {correct} / {total}, Acc: {correct / total}")
