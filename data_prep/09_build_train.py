"""Step 10: assemble the train set from every step, pass through untouched rows,
add the hand-written gap-filling prompts (data_prep/manual/new_prompts/*.txt) and trim
categories that exceed their range.

Writes data/work/train_assembled.jsonl, data/work/removed_all.jsonl, step10_counts.json
"""
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from common import WORK, MANUAL, TRAIN_TARGETS, row, dropped, write_jsonl, read_jsonl, new_id

STEP_FILES = ["step2_rows.jsonl", "step3_rows.jsonl", "step4_rows.jsonl", "step5_format_rows.jsonl",
              "step5b_latin_rows.jsonl", "step6_knowledge_rows.jsonl", "step7_news_rows.jsonl"]
REMOVED_FILES = ["step2_removed.jsonl", "step3_removed.jsonl", "step4_removed.jsonl",
                 "step5_format_removed.jsonl", "step5b_latin_removed.jsonl", "step6_knowledge_removed.jsonl"]
HANDLED_FAMILIES = {"en", "dolly_ctx", "pair", "gen1", "format", "format_text", "latin", "passage_qa",
                    "news_sum", "news_cls", "tr_am_en"} | {f"math_{x}" for x in
                   "speed percentage profit savings division perimeter average box_packing fraction multiply".split()}


def load_new_prompts():
    """data_prep/manual/new_prompts/*.txt: '## category/subcategory' headers, one prompt per line."""
    out = []
    for f in sorted((MANUAL / "new_prompts").glob("*.txt")):
        cat = sub = None
        for line in f.read_text(encoding="utf-8").split("\n"):
            if line.startswith("## "):
                cat, sub = line[3:].strip().split("/")
                continue
            if not line.strip() or line.startswith("#"):
                continue
            p = line.replace("\\n", "\n").strip()
            split = "eval" if f.name.startswith("eval_") else "train"
            out.append(row(new_id(f"{f.name}|{p}"), cat, sub, p, "v2_generated", None, True, split=split,
                           status="new", file=f.name))
    return out


def main():
    df = pd.read_parquet(WORK / "labelled.parquet")
    tr = df[df.split_in == "train"]
    rows, removed = [], []
    for f in STEP_FILES + ["step6_knowledge_eval_pool.jsonl"]:
        rows += read_jsonl(WORK / f)
    for f in REMOVED_FILES:
        removed += read_jsonl(WORK / f)
    done = {r["id"] for r in rows} | {r["id"] for r in removed}

    # pass-through rows not owned by a step script
    for r in tr.itertuples():
        if r.id in done:
            continue
        if r.family in ("knowledge_tail", "residual") and r.note.startswith("D"):
            removed.append(dropped(r.id, r.prompt, r.note[2:], "step1_labelling"))
            continue
        if r.family == "residual" and r.category == "knowledge_qa":
            continue  # handled by step 6 (either kept or logged there)
        if r.category == "unassigned":
            raise ValueError(r.id)
        needs = bool(r.sensitive)
        rows.append(row(r.id, r.category, r.subcategory or r.family, r.prompt, r.source, r.id, needs,
                        status="kept", original_prompt=r.prompt, sensitive=bool(r.sensitive)))
    rows = [r for r in rows if r["split"] == "train"] + [r for r in rows if r["split"] == "eval"]
    new = load_new_prompts()
    rows += new
    ids = Counter(r["id"] for r in rows)
    dup_ids = [k for k, v in ids.items() if v > 1]
    assert not dup_ids, dup_ids[:5]

    train = [r for r in rows if r["split"] == "train"]
    evalr = [r for r in rows if r["split"] == "eval"]
    counts = Counter(r["category"] for r in train)
    write_jsonl(WORK / "train_assembled.jsonl", train)
    write_jsonl(WORK / "eval_from_steps.jsonl", evalr)
    write_jsonl(WORK / "removed_all.jsonl", removed)
    gaps = {c: (counts.get(c, 0), lo, hi, (lo - counts.get(c, 0)) if counts.get(c, 0) < lo else
                (hi - counts.get(c, 0) if counts.get(c, 0) > hi else 0)) for c, (lo, hi) in TRAIN_TARGETS.items()}
    out = {"train_total": len(train), "eval_from_steps": len(evalr), "removed": len(removed),
           "new_prompts": len(new), "by_category": {c: v[0] for c, v in gaps.items()},
           "out_of_range": {c: v for c, v in gaps.items() if v[3]}}
    (WORK / "step10_counts.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
