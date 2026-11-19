"""Step 3 (+ step 6 code-switch and step 9 safety for generated rows):
review the mechanically combined situation + instruction prompts.

* verdicts: data/manual/pair_overrides.tsv (drop / rewrite); unlisted pairs fit -> keep
* each situation is kept at most 3 times (extras dropped)
* AI-test guardrails removed by swapping in natural variants (data/manual/instruction_variants.tsv)
* no instruction line > 8 uses; >= 40 % of kept 2-line pairs merged into one paragraph
* form labels ዝግጅታችን፦ / ርዕሰ ጉዳዬ፦ / የጻፍኩት፦ rephrased in >= 70 % of rows
Writes data/work/pair_review.csv and step3_rows / step3_removed.
"""
import json
import random
import re
from collections import Counter, defaultdict

import pandas as pd

from common import WORK, MANUAL, row, dropped, parse_tsv, write_jsonl, lines_of

KS_TAIL = "ምንም ማብራሪያ ሳትጨምር መልሱን ብቻ ስጠኝ።"
MAX_PER_SITUATION = 3
MAX_PER_LINE = 8
MERGE_SHARE = 0.47
OPENERS = ["ሰላም፣ ", "ጤና ይስጥልኝ፣ ", "እባክህ ", "አንድ ጥያቄ ነበረኝ። ", "ሰላም ነው? ", "እስኪ እርዳኝ፣ "]


def load_variants():
    v = {}
    for key, txt in parse_tsv(MANUAL / "instruction_variants.tsv", 2):
        v[key] = [x.strip() for x in txt.split("||")]
    v["KS"] = [KS_TAIL, "ዝም ብለህ መልሱን ብቻ ንገረኝ።", "ማብራሪያ አያስፈልገኝም፣ መልሱን ብቻ።"]
    return v


def strip_ye(x):
    return x[1:] if x.startswith("የ") else x


def rephrase_label(sit, k, rnd):
    """Return (new_situation, label_removed: bool)."""
    m = re.match(r"^ርዕሰ ጉዳዬ፦ (.+?)።?$", sit)
    if m:
        t = m.group(1)
        opts = [f"ስለ {strip_ye(t)} ድርሰት መጻፍ አለብኝ።", f"የክፍል ስራዬ ርዕስ «{t}» ነው።",
                f"ለትምህርት ቤት «{t}» በሚል ርዕስ ልጽፍ ነው።", f"መምህራችን ስለ {strip_ye(t)} እንድንጽፍ ነግሮናል።",
                f"«{t}» በሚል ርዕስ ጽሑፍ እያዘጋጀሁ ነው።"]
        return (sit, False) if k % 4 == 3 else (opts[k % len(opts)], True)
    m = re.match(r"^ዝግጅታችን፦ (.+?)።?$", sit)
    if m:
        t = m.group(1)
        opts = [f"{t} እያዘጋጀን ነው።", f"በሚቀጥለው ሳምንት {t} አለን።", f"በሰፈራችን {t} ልናደርግ አስበናል።",
                f"ለ{strip_ye(t)} አስተባባሪ ሆኛለሁ።", f"ቅዳሜ {t} ይኖረናል።"]
        return (sit, False) if k % 4 == 3 else (opts[k % len(opts)], True)
    m = re.match(r"^የጻፍኩት፦ (.+)$", sit)
    if m:
        t = m.group(1)
        opts = [f"ይህን መልእክት ልልክ ነው፦ {t}", f"{t} ብዬ ጽፌያለሁ።", f"ይሄን ጽሑፍ ተመልከትልኝ፦ {t}",
                f"ለማስታወቂያ ሰሌዳ {t} የሚል ጽፌያለሁ።", f"{t} — ይሄን ነው የጻፍኩት።"]
        return (sit, False) if k % 4 == 3 else (opts[k % len(opts)], True)
    return sit, False


AI_TEST_RE = re.compile("አታባብስ|ምርመራ አታድርግ|ዋጋ አትገምት|ሳትፈጥር|አትፍጠር|አትገምት|የማታውቀውን ጠይቀኝ|"
                        "እንደማታውቅ አስታውስ|አትጨምር|አታቅርብ|እንደሆነ አታስብ|አታወሳስብ")


def main():
    rnd = random.Random(3)
    df = pd.read_parquet(WORK / "labelled.parquet")
    instr_ids = json.loads((WORK / "instruction_ids.json").read_text())
    id_instr = {v: k for k, v in instr_ids.items()}
    sit_ids = {s: t for s, t in parse_tsv(MANUAL / "situation_ids.tsv", 2)}
    sit_ids = {s: (t[: -len(" Any tips?")] if t.endswith(" Any tips?") else t) for s, t in sit_ids.items()}
    over = {}
    for s, i, v, txt in parse_tsv(MANUAL / "pair_overrides.tsv", 4):
        over[(sit_ids[s], i)] = (v, txt)
    variants = load_variants()

    g = df[(df.split_in == "train") & (df.instr_key != "") & (df.family != "latin")].copy()
    g = g[~g.instr_group.isin(["latin", "format_csv"])]  # CSV rows are format tasks (step 4b)
    g = g.sort_values(["situation", "instr_key", "id"]).reset_index(drop=True)

    review, rows, drops = [], [], []
    instr_use = Counter()
    by_sit = defaultdict(list)
    for r in g.itertuples():
        by_sit[r.situation].append(r)

    # 1. verdicts + situation cap
    chosen = []
    for sit, rs in by_sit.items():
        cands = []
        for r in rs:
            v, txt = over.get((sit, r.instr_key), ("k", ""))
            if v == "d":
                review.append((r, "drop", txt, ""))
                drops.append(dropped(r.id, r.prompt, f"pair does not fit: {txt}", "step3_pairs"))
            else:
                cands.append((r, v, txt))
        # rewrites first, then least-used instructions (balances instruction usage)
        cands.sort(key=lambda c: (c[1] != "r", instr_use[c[0].instr_key], rnd.random()))
        for j, (r, v, txt) in enumerate(cands):
            if j < MAX_PER_SITUATION:
                instr_use[r.instr_key] += 1
                chosen.append((r, v, txt))
            else:
                review.append((r, "drop", "situation already used 3 times", ""))
                drops.append(dropped(r.id, r.prompt, "situation line repeated more than 3 times "
                                     "(kept the 3 best-fitting instructions)", "step3_pairs"))

    # 2. build prompts
    line_use = Counter()
    variant_use = Counter()
    label_k = Counter()
    pairs_2line = [c for c in chosen if c[0].family == "pair"]
    merge_ids = set(r.id for r, _, _ in rnd.sample(pairs_2line, round(MERGE_SHARE * len(pairs_2line))))
    stats = Counter()
    for r, v, txt in chosen:
        sit = r.situation
        lab = re.match(r"^(ዝግጅታችን|ርዕሰ ጉዳዬ|የጻፍኩት)፦", sit)
        if lab:
            label_k[lab.group(1)] += 1
            sit, removed = rephrase_label(sit, label_k[lab.group(1)], rnd)
            stats["label_rows"] += 1
            stats["label_removed"] += removed
        if v == "r":
            ins = txt
        else:
            pool = variants[r.instr_key]
            ins = min(pool, key=lambda x: (variant_use[(r.instr_key, x)], line_use[x], rnd.random()))
            variant_use[(r.instr_key, ins)] += 1
        orig_ins = KS_TAIL if r.instr_key == "KS" else id_instr.get(r.instr_key, "")
        merged = r.family != "pair" or r.id in merge_ids or line_use[ins] >= MAX_PER_LINE
        if merged:
            opener = OPENERS[stats["opener"] % len(OPENERS)] if rnd.random() < 0.12 else ""
            stats["opener"] += bool(opener)
            new = f"{opener}{sit} {ins}"
            if rnd.random() < 0.08 and new[-1] in "።":
                new = new[:-1]  # users often skip the final full stop
        else:
            new = f"{sit}\n{ins}"
            line_use[ins] += 1
        stats["merged_pairs"] += (r.family == "pair" and merged)
        stats["pairs"] += r.family == "pair"
        changes = []
        if v == "r":
            changes.append("instruction_rewritten_to_fit")
        elif ins != orig_ins:
            changes.append("instruction_naturalized" if AI_TEST_RE.search(orig_ins) else "instruction_rephrased")
        if r.family == "pair" and merged:
            changes.append("merged_into_one_paragraph")
        if lab and sit != r.situation:
            changes.append("form_label_removed")
        reason = {"instruction_rewritten_to_fit": "instruction did not fit the situation",
                  "instruction_naturalized": "removed AI-test guardrail phrasing",
                  "instruction_rephrased": "broke up template repetition",
                  "merged_into_one_paragraph": "varied prompt shape",
                  "form_label_removed": "replaced form-style label with natural phrasing"}
        sub = r.subcategory
        needs = v == "r" or r.category == "safety"
        rows.append(row(r.id, r.category, sub, new, "v2_rewritten" if v == "r" else "v2_generated", r.id,
                        needs, status="changed" if new != r.prompt else "kept",
                        original_prompt=r.prompt, change_type="+".join(changes),
                        reason="; ".join(reason[c] for c in changes), situation=r.situation,
                        instr_key=r.instr_key))
        review.append((r, "rewrite" if v == "r" else "keep",
                       "instruction rewritten to fit the situation" if v == "r" else "fits the situation", new))

    rv = pd.DataFrame([{"id": r.id, "situation": r.situation, "instruction_key": r.instr_key,
                        "original_prompt": r.prompt, "verdict": verdict, "reason": reason, "new_prompt": new}
                       for r, verdict, reason, new in review])
    rv.to_csv(WORK / "pair_review.csv", index=False)
    write_jsonl(WORK / "step3_rows.jsonl", rows)
    write_jsonl(WORK / "step3_removed.jsonl", drops)
    gen_rows = [x for x in rows]
    dont = sum(1 for x in gen_rows if AI_TEST_RE.search(x["prompt"]))
    out = {"input_rows": len(g), "situations": len(by_sit), "verdicts": rv.verdict.value_counts().to_dict(),
           "kept_rows": len(rows), "two_line_pairs_kept": stats["pairs"], "merged_pairs": stats["merged_pairs"],
           "merged_share": round(stats["merged_pairs"] / max(1, stats["pairs"]), 3),
           "label_rows": stats["label_rows"], "label_removed": stats["label_removed"],
           "label_removed_share": round(stats["label_removed"] / max(1, stats["label_rows"]), 3),
           "rows_with_dont_constraint": dont, "dont_share": round(dont / len(rows), 3),
           "max_instruction_line_uses": max(line_use.values()),
           "category": Counter(x["category"] for x in rows)}
    (WORK / "step3_stats.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
