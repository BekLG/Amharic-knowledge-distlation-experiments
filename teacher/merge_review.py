#!/usr/bin/env python3
"""
Merge the human-reviewed rows back into the training/eval files.

In review_queue.csv, set decision_keep_yes_no to "yes" for rows to keep. If you corrected
the answer, put the corrected text in fixed_response (it replaces the teacher's answer).
Rows marked anything other than "yes" are left out.

  python merge_review.py --review filtered/review_queue.csv --teacher teacher_outputs.jsonl --outdir filtered
"""
import argparse, csv, json, os

ap = argparse.ArgumentParser()
ap.add_argument("--review", required=True)
ap.add_argument("--teacher", required=True)
ap.add_argument("--outdir", default="data/sft")
args = ap.parse_args()

teacher = {}
with open(args.teacher, encoding="utf-8") as f:
    for l in f:
        if l.strip():
            r = json.loads(l)
            teacher[r["id"]] = r

existing = set()
for name in ("sft_train.jsonl", "eval_teacher.jsonl"):
    p = os.path.join(args.outdir, name)
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            existing |= {json.loads(l)["id"] for l in f if l.strip()}

added = {"train": 0, "eval": 0}
skipped = 0
with open(args.review, encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        if row.get("decision_keep_yes_no", "").strip().lower() not in ("yes", "y", "አዎ"):
            skipped += 1
            continue
        r = teacher.get(row["id"])
        if not r or row["id"] in existing:
            continue
        resp = row.get("fixed_response", "").strip() or r["response"]
        rec = {"id": r["id"], "category": r.get("category"), "language_mode": r.get("language_mode"),
               "messages": r["messages"] + [{"role": "assistant", "content": resp}]}
        name = "sft_train.jsonl" if r.get("split") == "train" else "eval_teacher.jsonl"
        with open(os.path.join(args.outdir, name), "a", encoding="utf-8") as out:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        added["train" if r.get("split") == "train" else "eval"] += 1

print(f"added to train: {added['train']}, added to eval: {added['eval']}, not kept: {skipped}")
