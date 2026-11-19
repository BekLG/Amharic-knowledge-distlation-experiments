"""Step 5: Latin-script Amharic. The v1 rows reused 51 situations x 35 instructions;
they are replaced by hand-written distinct prompts with varied romanisation."""
import json

import pandas as pd

from common import WORK, MANUAL, row, dropped, write_jsonl, new_id

TRAIN_N = 150


def main():
    df = pd.read_parquet(WORK / "labelled.parquet")
    old = df[(df.split_in == "train") & (df.category == "latin_amharic")].sort_values("id")
    new = [l.replace("\\n", "\n") for l in (MANUAL / "latin_prompts.txt").read_text(encoding="utf-8").split("\n")
           if l.strip() and not l.startswith("#")]
    assert len(set(new)) == len(new)
    rows, drops = [], []
    olds = list(old.itertuples())
    for i, p in enumerate(new):
        split = "train" if i < TRAIN_N else "eval"
        if split == "train" and i < len(olds):
            o = olds[i]
            rows.append(row(o.id, "latin_amharic", "romanized", p, "v2_rewritten", o.id, True, status="changed",
                            original_prompt=o.prompt, change_type="rewritten_latin",
                            reason="v1 latin rows reused 51 situations x 35 instructions; rewritten as a distinct prompt",
                            language_mode_="latin_am"))
        else:
            rows.append(row(new_id("latin|" + p), "latin_amharic", "romanized", p, "v2_generated", None, True,
                            split=split, status="new", language_mode_="latin_am"))
    for o in olds[TRAIN_N:]:
        drops.append(dropped(o.id, o.prompt, "latin template repetition", "step5_latin"))
    write_jsonl(WORK / "step5b_latin_rows.jsonl", rows)
    write_jsonl(WORK / "step5b_latin_removed.jsonl", drops)
    print({"old": len(olds), "train": sum(r["split"] == "train" for r in rows),
           "eval": sum(r["split"] == "eval" for r in rows), "distinct": len(new)})


if __name__ == "__main__":
    main()
