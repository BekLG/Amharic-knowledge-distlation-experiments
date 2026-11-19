"""Write data/out/REPORT.md from the work files and validation results."""
import json
from collections import Counter, defaultdict

import pandas as pd

from common import WORK, OUT, TRAIN_TARGETS, EVAL_TARGETS, read_jsonl


def j(name):
    return json.loads((WORK / name).read_text())


def table(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    lab = pd.read_parquet(WORK / "labelled.parquet")
    s1 = j("step1_counts.json")
    train = read_jsonl(OUT / "pilot_train.jsonl")
    evalr = read_jsonl(OUT / "pilot_eval.jsonl")
    removed = pd.read_csv(OUT / "removed.csv")
    changed = pd.read_csv(OUT / "changed.csv")
    val = j("validation.json")
    work_train = {r["id"]: r for r in read_jsonl(WORK / "train_dedup.jsonl")}
    work_eval = {r["id"]: r for r in read_jsonl(WORK / "eval_final.jsonl")}

    tc = Counter(r["category"] for r in train)
    ec = Counter(r["category"] for r in evalr)
    orig_cat = dict(zip(lab.id, lab.category))
    orig_split = dict(zip(lab.id, lab.split_in))

    # per-category kept / changed / new / dropped (train)
    per = defaultdict(Counter)
    for r in train:
        w = work_train[r["id"]]
        per[r["category"]]["new" if w["status"] == "new" else "changed" if w["status"] == "changed" else "kept"] += 1
    for rid in removed.id:
        if orig_split.get(rid) == "train":
            c = orig_cat.get(rid, "?")
            per[c if c in TRAIN_TARGETS else "(labelled as dropped / other)"]["dropped"] += 1
    cat_rows = []
    for c, (lo, hi) in TRAIN_TARGETS.items():
        cat_rows.append([c, tc[c], f"{lo}–{hi}", per[c]["kept"], per[c]["changed"], per[c]["new"], per[c]["dropped"],
                         "yes" if lo <= tc[c] <= hi else "NO"])
    eval_rows = []
    for k, v in EVAL_TARGETS.items():
        cats = k if isinstance(k, tuple) else (k,)
        eval_rows.append([" + ".join(cats), sum(ec[c] for c in cats), v])

    def dist(rows, key):
        c = Counter(r[key] for r in rows)
        n = len(rows)
        return [[k, v, f"{v / n:.1%}"] for k, v in c.most_common()]

    s2, s3, s4 = j("step2_stats.json"), j("step3_stats.json"), j("step4_math_stats.json")
    s6, s7, s12 = j("step6_knowledge_stats.json"), j("step7_news_stats.json"), j("step12_stats.json")
    s10 = j("step10_counts.json")
    rv = pd.read_csv(WORK / "pair_review.csv")
    verdicts = rv.verdict.value_counts().to_dict()
    nh = sum(1 for r in train + evalr if r["needs_human_review"])
    teacher = len((OUT / "teacher_answer_review_ids.txt").read_text().split())
    rem_by_step = Counter(r["reason"].split(":")[0][:70] for r in read_jsonl(WORK / "removed_all.jsonl"))

    steps = [
        ["1 Load & label", "5,000 train + 300 eval rows", f"{s1['flagged_for_drop_in_labelling']} rows flagged unusable while labelling (logged in removed.csv)"],
        ["2 English prompts", f"{s2['input']} English rows", f"{s2['translated']} translated, {s2['kept_with_request']} kept + Amharic request "
         f"({s2['request_phrasings_used']} phrasings, max {s2['max_uses']} uses each), {s2['dropped']} dropped, {s2['new_english']} new English prompts"],
        ["3 Situation + instruction pairs", f"{s3['input_rows']} rows over {s3['situations']} situations",
         f"verdicts {verdicts}; {s3['kept_rows']} kept; {s3['merged_share']:.0%} of kept 2-line pairs merged; "
         f"form labels removed in {s3['label_removed_share']:.0%}; 'don't' constraints left: {s3['dont_share']:.0%}; "
         f"max instruction-line uses {s3['max_instruction_line_uses']}"],
        ["4 Math", f"{s4['dropped_templated']} templated rows (10 families)",
         f"{s4['written']} hand-written problems over {s4['types']} types → {s4['train']} train / {s4['eval']} eval; "
         f"train styles {s4['style_train']}; {s4['status_train']}; {s4['non_integer_answers']} non-integer answers"],
        ["4b Format", "202 list/JSON rows", f"{tc['format']} kept with new data and wording; 15 hand-written eval rows"],
        ["5 Latin-script Amharic", "100 rows (51 situations × 35 instructions)", f"{tc['latin_amharic']} distinct rewritten train prompts + 13 new eval"],
        ["6 Code-switched", "61 natural rows after the step-3 caps", f"{tc['code_switch']} after adding new code-switched prompts"],
        ["7 Knowledge", "751 knowledge-labelled rows", f"{s6['knowledge_rows']} kept ({s6['by_sub']}); {s6['promoted_to_translated']} Dolly instructions → translated_instruction; "
         f"trivia rows with a generic tail {s6['trivia_rows_with_tail']}/{s6['trivia_rows_originally_with_tail']} originally tailed"],
        ["8 News", f"{s7['rows']} news / translation / wiki rows", f"source credits removed {s7.get('removed_source_credit', 0)}, URL sentences {s7.get('removed_url_sentence', 0)}, "
         f"datelines {s7.get('removed_trailing_dateline', 0)}, merged sentences split {s7.get('split_merged_sentences', 0)}, "
         f"loaded-term flags {s7.get('loaded_flag', 0)}, political/war share of news {s7['political_share_of_news']:.0%}, "
         f"instruction lines varied {s7.get('instruction_varied', 0)} (max {s7['max_instruction_uses']} uses)"],
        ["9 Safety", "150 generated safety rows", f"{tc['safety']} train / {ec['safety']} eval after caps and new prompts; all listed for teacher-answer review"],
        ["10 Fill & rebalance", f"{s10['train_total'] - s10['new_prompts'] + sum(1 for r in read_jsonl(WORK / 'eval_from_steps.jsonl'))} pre-fill rows",
         f"{s10['new_prompts']} new hand-written prompts (train + eval)"],
        ["11 Eval", "300 old eval rows", f"rebuilt: 50 native, knowledge 20 (Gurnah premise fixed), 20 summarisation, "
         f"29 rewritten situations, fresh prompts; 10 surplus natives moved to train"],
        ["12 Dedup / leakage", f"{s10['train_total'] + 10} train rows", f"{s12['train']} after removing near-duplicates ({s12.get('near_dup_train', 0)}) "
         f"and rows sharing a line with eval ({s12.get('shared_line_eval', 0)})"],
    ]

    unsure = [
        "**No `prompts.csv` was available**, so `source` was recovered by reading the rows: Aya/AddisGPT vs. NQ-Open vs. Dolly vs. "
        "v2-generated is my judgement (see `data/manual/residual_labels.txt`). Short native-style trivia without an NQ answer tail was "
        "labelled `aya_dataset`/`native_chat`; native_chat ends at 500 vs. the ~472 exact matches the spec mentions.",
        "**English prompts:** only ~300 English rows existed (the other ~100 of the \"about 400\" were romanised Amharic), so I "
        "translated 78 and filled `translated_instruction` with the 82 Amharic Dolly translations already present plus 58 more "
        "translated Dolly rows; 47 new English user-text prompts were written so `english_instruction` reaches 200.",
        "**Totals are tight:** with native_chat at ~500, nearly every other category had to sit at the top of its range to reach "
        "5,000 ± 50; train is 4,961. Any further drops (e.g. after native review) need replacements in categories below their maximum.",
        "**Language mode rule:** an English prompt with one of the Amharic answer-request sentences is `en`, and ALL-CAPS acronyms, "
        "snake_case keys and programming-language names (JSON, CSV, Python, Excel…) are not counted as English words. Without this "
        "rule mixed + latin would exceed 9%.",
        "**One-liner definition:** \"about 30% one-liners\" is read as single-line prompts of at most 9 words (33.1%); 63% of "
        "prompts are single lines of any length because many paired prompts were merged into one paragraph.",
        "**Instruction variants and merges** in the generated rows were written by me; all changed generated rows are flagged "
        "`needs_human_review`. Native speakers should especially check the casual register and the Latin-script spellings.",
        "**News:** merged words were only split after a closed list of sentence-final verbs; other run-together words remain. "
        f"{s7.get('loaded_flag', 0)} news rows with loaded political terms are flagged and listed for teacher-answer review.",
        "**Kept sensitive native prompts** (political tweets, ethnic/political questions, betting, religion) are flagged and listed "
        "in `teacher_answer_review_ids.txt` rather than dropped.",
        "**Amharic reading-comprehension rows** whose question shares under 20% of its words with the passage were dropped as "
        "probable mismatches; some may be fine (spelling variants).",
        "**Math:** answers in `data/work/math_answers.csv` were computed in Python, including Ethiopian↔Gregorian date and "
        "Ethiopian-clock conversions; missing/contradictory problems store the check value or an empty answer and a note.",
    ]

    md = [
        "# Pilot prompt-set fix — report",
        "",
        f"Train: **{len(train)}** rows · Eval: **{len(evalr)}** rows · Removed: **{len(removed)}** input rows · "
        f"Changed: **{len(changed)}** rows · needs_human_review: **{nh}** · teacher-answer review list: **{teacher}** ids.",
        "",
        "Everything is reproducible from `data/raw/` with `scripts/run_all.sh`. Hand-made judgements and prompts live in "
        "`data/manual/` (labels, pair verdict overrides, instruction variants, translations, math, Latin-script and new prompts).",
        "",
        "## Validation (step 14)",
        "",
        table([[k, "PASS" if v["pass"] else "FAIL",
                ", ".join(f"{a}={b}" for a, b in v.items() if a != "pass" and not isinstance(b, (list, dict)))[:160]]
               for k, v in val.items() if k != "all_pass"], ["check", "result", "details"]),
        "",
        f"**All checks pass: {val['all_pass']}**",
        "",
        "## Final train composition",
        "",
        table(cat_rows, ["category", "train", "target", "kept as-is", "changed", "new", "dropped (input rows)", "in range"]),
        "",
        "## Final eval composition",
        "",
        table(eval_rows, ["category", "eval", "target"]),
        "",
        "## Language mode",
        "",
        "Train:",
        "",
        table(dist(train, "language_mode"), ["language_mode", "rows", "share"]),
        "",
        "Eval:",
        "",
        table(dist(evalr, "language_mode"), ["language_mode", "rows", "share"]),
        "",
        "## Source",
        "",
        table(dist(train, "source"), ["source (train)", "rows", "share"]),
        "",
        table(dist(evalr, "source"), ["source (eval)", "rows", "share"]),
        "",
        "## Step 1 labels before any change (train input)",
        "",
        table([[k, v] for k, v in s1["category"].items()], ["category", "rows"]),
        "",
        table([[k, v] for k, v in s1["language_mode"].items()], ["language_mode", "rows"]),
        "",
        table([[k, v] for k, v in s1["source"].items()], ["source", "rows"]),
        "",
        table([[k, v] for k, v in s1["family"].items()], ["detected generator family", "rows"]),
        "",
        "## Before / after by step",
        "",
        table(steps, ["step", "before", "after / what changed"]),
        "",
        "## Things a person should look at",
        "",
        "\n".join(f"- {u}" for u in unsure),
        "",
        "## Files",
        "",
        "- `data/out/pilot_train.jsonl`, `data/out/pilot_eval.jsonl` — final sets (schema in the spec).",
        "- `data/out/removed.csv` — every dropped input row with its reason; `data/out/changed.csv` — every modified row.",
        "- `data/out/review_sample.csv` — 150 rows for native-speaker review (stratified, all eval categories).",
        "- `data/out/teacher_answer_review_ids.txt` — safety and sensitive prompts whose teacher answers must be read.",
        "- `data/work/labelled.parquet`, `data/work/pair_review.csv`, `data/work/math_answers.csv`, `data/work/validation.json`.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("REPORT.md written")


if __name__ == "__main__":
    main()
