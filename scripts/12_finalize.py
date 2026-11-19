"""Step 13 + outputs: write the JSONL files, removed.csv, changed.csv, review_sample.csv and
teacher_answer_review_ids.txt to data/out/."""
import csv
import json
import random
from collections import Counter, defaultdict

import pandas as pd

from common import WORK, OUT, RAW, read_jsonl, write_jsonl, read_csv, norm_text

REVIEW_SAMPLE_N = 150


def needs_review(r):
    if r["split"] == "eval" or r["status"] == "new":
        return True
    if r["category"] == "safety" or r.get("sensitive"):
        return True
    ct = r.get("change_type") or ""
    if r["source"] == "v2_rewritten" or any(x in ct for x in (
            "translated", "rewritten", "fixed", "instruction_naturalized", "instruction_rephrased",
            "merged", "form_label_removed", "varied_data")):
        return True
    return bool(r.get("needs_human_review"))


def to_record(r):
    return {
        "id": r["id"], "category": r["category"], "subcategory": r["subcategory"] or r["category"],
        "language_mode": r["language_mode"], "messages": [{"role": "user", "content": r["prompt"]}],
        "source": r["source"], "origin_id": r["origin_id"], "needs_human_review": needs_review(r),
        "split": r["split"],
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train = read_jsonl(WORK / "train_dedup.jsonl")
    evalr = read_jsonl(WORK / "eval_final.jsonl")
    for r in train:
        r["split"] = "train"
    write_jsonl(OUT / "pilot_train.jsonl", [to_record(r) for r in train])
    write_jsonl(OUT / "pilot_eval.jsonl", [to_record(r) for r in evalr])

    # removed.csv: every dropped input row with its reason
    removed = read_jsonl(WORK / "removed_all.jsonl") + read_jsonl(WORK / "eval_removed.jsonl") + \
        read_jsonl(WORK / "step12_removed.jsonl")
    kept_origin = {r["origin_id"] for r in train + evalr if r["origin_id"]} | {r["id"] for r in train + evalr}
    rem, seen = [], set()
    for r in removed:
        if r["id"] in seen or r["id"] in kept_origin:
            continue
        seen.add(r["id"])
        rem.append({"id": r["id"], "original_prompt": r["original_prompt"], "reason": r["reason"]})
    pd.DataFrame(rem).to_csv(OUT / "removed.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)
    # rows dropped after a step had already changed them (e.g. dedup) are logged in removed only
    # changed.csv: every kept row whose text or label changed
    ch = []
    for r in train + evalr:
        if r["status"] == "changed" or (r.get("change_type") and r["status"] != "new"):
            ch.append({"id": r["id"], "original_prompt": r["original_prompt"], "new_prompt": r["prompt"],
                       "change_type": r.get("change_type") or "", "reason": r.get("reason") or ""})
    pd.DataFrame(ch).to_csv(OUT / "changed.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)

    # teacher answers to read before training: safety + sensitive prompts (both splits)
    ids = [r["id"] for r in train + evalr if r["category"] == "safety" or r.get("sensitive")
           or r.get("loaded_terms")]
    (OUT / "teacher_answer_review_ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")

    # review_sample.csv: 150 rows needing review, stratified by category, every eval category present
    rnd = random.Random(13)
    pool = [r for r in train + evalr if needs_review(r)]
    by_cat = defaultdict(list)
    for r in pool:
        by_cat[r["category"]].append(r)
    cats = sorted(by_cat)
    alloc = {c: 3 for c in cats}
    rest = REVIEW_SAMPLE_N - sum(alloc.values())
    total = sum(len(v) for v in by_cat.values())
    for c in cats:
        alloc[c] += int(rest * len(by_cat[c]) / total)
    i = 0
    while sum(alloc.values()) < REVIEW_SAMPLE_N:
        alloc[cats[i % len(cats)]] += 1
        i += 1
    sample = []
    for c in cats:
        rows = by_cat[c][:]
        rnd.shuffle(rows)
        ev = [r for r in rows if r["split"] == "eval"]
        tr = [r for r in rows if r["split"] == "train"]
        take = ev[:1] + tr[: alloc[c] - min(1, len(ev))]
        if len(take) < alloc[c]:
            take += ev[1: 1 + alloc[c] - len(take)]
        sample += take[: alloc[c]]
    pd.DataFrame([{"id": r["id"], "split": r["split"], "category": r["category"],
                   "subcategory": r["subcategory"], "language_mode": r["language_mode"], "prompt": r["prompt"],
                   "natural_yes_no": "", "correct_amharic_yes_no": "", "notes": ""} for r in sample]) \
        .to_csv(OUT / "review_sample.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)
    print({"train": len(train), "eval": len(evalr), "removed": len(rem), "changed": len(ch),
           "teacher_review": len(ids), "review_sample": len(sample)})


if __name__ == "__main__":
    main()
