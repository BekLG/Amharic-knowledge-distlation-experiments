"""Step 7: knowledge and trivia.

* amharic_qa reading-comprehension rows: keep those whose question matches the passage
* NQ-Open / Dolly machine-translated questions: hand labels from data_prep/manual (fix or drop)
* at most 150 non-Ethiopian trivia questions (science / world geography preferred)
* generic answer-style tails kept on at most 30 % of the tail-carrying prompts
* some Dolly open-ended instructions move to translated_instruction (they are translated Dolly)
"""
import json
import random
import re
from collections import Counter

import pandas as pd

from common import WORK, MANUAL, row, dropped, write_jsonl, lines_of, parse_tsv, strip_credits

TRIVIA_CAP = 120          # non-Ethiopian trivia kept in train (spec max 150)
PASSAGE_CAP = 175         # reading comprehension kept in train
CONCEPT_CAP = 74          # Dolly concept questions kept as knowledge_qa
PROMOTE_TO_TRANSLATED = 58
TAIL_SHARE = 0.30
ETH = re.compile("ኢትዮጵ|አዲስ አበባ|አክሱም|ላሊበላ|ጎንደር|አባይ|ጣና|ምኒልክ|ሚኒሊክ|ጣይቱ|ቴዎድሮስ|ኃይለ ሥላሴ|አማርኛ|ግዕዝ|ግእዝ")
SCIENCE = re.compile("ፕላኔት|ፀሐይ|ፀሀይ|ውሃ|አጥንት|ሕዋስ|ህዋስ|ኬሚካል|ንጥረ|ብርሃን|ምድር|ደም|ልብ|አካል|ሙቀት|ኤሌክትሪክ|ጨረቃ|"
                     "ማዕድን|ወፍ|እንስሳ|ተክል|ቫይረስ|በሽታ|ብረት|አየር|ዲግሪ|ፍጥነት|ማግኔ")
GEO = re.compile("ወንዝ|ተራራ|ባሕር|ባህር|አህጉር|ዋና ከተማ|ደሴት|በረሃ|ሀገር|አገር|ውቅያኖስ|ሐይቅ|ሀይቅ")
TAILS = ["በአጭሩ አስረዳ፤ እርግጠኛ ካልሆንክ ግለጽ።", "መልሱንና አንድ ተዛማጅ ማብራሪያ ስጠኝ።", "ለጀማሪ የሚገባ ማብራሪያ ስጥ።",
         "መልሱን በሁለት ዓረፍተ ነገር አስረዳ።", "አጭር ማብራሪያም ጨምር።", "ስም ወይም ቁጥር ብቻ ሳይሆን ትንሽ አብራራ።"]
NEW_TAILS = ["በአጭሩ አስረዳ።", "አጭር ማብራሪያም ጨምር።", "መልሱን በሁለት ዓረፍተ ነገር ንገረኝ።", "ትንሽ አብራራልኝ።",
             "ለምን እንደሆነም ንገረኝ።", "እርግጠኛ ካልሆንክ ንገረኝ።"]
HOWTO = re.compile("እንዴት|ለምን|ምን ጥቅም|ምን ማድረግ|መንገዶች|ዘዴዎች|ይሻላል|ልማር|ላድርግ|ልዩነት|ጥቅሞች|ጉዳቶች|ምን ምን")


def overlap(p):
    q, ps = p.split("\nጽሑፍ፦\n", 1)
    qw = set(re.findall(r"[ሀ-፿]{3,}", q))
    pw = set(re.findall(r"[ሀ-፿]{3,}", ps))
    return len(qw & pw) / max(1, len(qw))


def trivia_score(q):
    return 3 if SCIENCE.search(q) else 2 if GEO.search(q) else 1


def main():
    rnd = random.Random(7)
    df = pd.read_parquet(WORK / "labelled.parquet")
    tr = df[df.split_in == "train"]
    rows, drops, eval_pool = [], [], []

    # --- reading comprehension (amharic_qa) ---
    pq = tr[tr.family == "passage_qa"].copy()
    pq["ov"] = pq.prompt.map(overlap)
    pq = pq.sort_values("ov", ascending=False)
    good = pq[pq.ov >= 0.2]
    for r in pq[pq.ov < 0.2].itertuples():
        drops.append(dropped(r.id, r.prompt, "reading-comprehension question does not match its passage", "step7"))
    # reserve some well-matched rows for eval, keep PASSAGE_CAP in train
    good = good.sample(frac=1, random_state=7)
    reserve = good.iloc[:20]
    keep = good.iloc[20:20 + PASSAGE_CAP]
    rest = good.iloc[20 + PASSAGE_CAP:]
    def clean(p):
        q, ps = p.split("\nጽሑፍ፦\n", 1)
        return f"{q}\nጽሑፍ፦\n{strip_credits(ps.strip())}"
    for r in reserve.itertuples():
        c = clean(r.prompt)
        eval_pool.append(row(r.id, "knowledge_qa", "reading_comprehension", c, "amharic_qa", r.id, True,
                             split="eval", status="changed" if c != r.prompt else "kept", original_prompt=r.prompt,
                             change_type="removed_source_credit" if c != r.prompt else "",
                             reason="removed source credit / promo sentence" if c != r.prompt else ""))
    for r in keep.itertuples():
        c = clean(r.prompt)
        rows.append(row(r.id, "knowledge_qa", "reading_comprehension", c, "amharic_qa", r.id, False,
                        status="changed" if c != r.prompt else "kept", original_prompt=r.prompt,
                        change_type="removed_source_credit" if c != r.prompt else "",
                        reason="removed source credit / promo sentence" if c != r.prompt else ""))
    for r in rest.itertuples():
        drops.append(dropped(r.id, r.prompt, "knowledge_qa cap: enough reading-comprehension rows", "step7"))

    # --- trivia: NQ residual + knowledge-tail rows ---
    triv = []
    for r in tr[(tr.category == "knowledge_qa") & (tr.family.isin(["residual", "knowledge_tail"]))].itertuples():
        if r.note.startswith("D"):
            drops.append(dropped(r.id, r.prompt, r.note[2:], "step7"))
            continue
        q = lines_of(r.prompt)[0]
        fixed = r.note[4:] if r.note.startswith("FIX|") else None
        if fixed:
            q = fixed
        triv.append((r, q, fixed))
    nq = [t for t in triv if t[0].source == "nq_open"]
    dolly = [t for t in triv if t[0].source == "dolly"]

    nq.sort(key=lambda t: (-trivia_score(t[1]), rnd.random()))
    tail_rows = [t for t in nq[:TRIVIA_CAP] if t[0].family == "knowledge_tail"]
    keep_tail = set(t[0].id for t in rnd.sample(tail_rows, int(TAIL_SHARE * len(tail_rows))))
    tail_use = Counter()
    for i, (r, q, fixed) in enumerate(nq):
        if i >= TRIVIA_CAP:
            drops.append(dropped(r.id, r.prompt, "non-Ethiopian trivia cap (kept science/geography first)", "step7"))
            continue
        changes, new = [], q
        if r.family == "knowledge_tail":
            if r.id in keep_tail:
                tail = min(NEW_TAILS, key=lambda x: (tail_use[x], rnd.random()))
                tail_use[tail] += 1
                new = f"{q}\n{tail}" if i % 2 else f"{q} {tail}"
                changes.append("generic_tail_varied")
            else:
                changes.append("generic_tail_removed")
        if fixed:
            changes.append("fixed_machine_translation")
        rows.append(row(r.id, "knowledge_qa", "trivia", new, "nq_open", r.id, bool(fixed),
                        status="changed" if new != r.prompt else "kept", original_prompt=r.prompt,
                        change_type="+".join(changes),
                        reason="; ".join({"generic_tail_removed": "repeated generic tail removed (<=30% keep a tail)",
                                          "generic_tail_varied": "generic tail rephrased",
                                          "fixed_machine_translation": "garbled machine translation fixed"}[c]
                                         for c in changes)))

    # --- Dolly concept questions: some become translated_instruction, cap the rest ---
    howto = [t for t in dolly if HOWTO.search(t[1])]
    rnd.shuffle(howto)
    others = [t for t in dolly if not HOWTO.search(t[1]) and not re.search(r"[A-Za-z]", t[1])]
    rnd.shuffle(others)
    promote = set(t[0].id for t in (howto + others)[:PROMOTE_TO_TRANSLATED])
    rest = [t for t in dolly if t[0].id not in promote]
    # prefer questions without Latin-script jargon
    rest.sort(key=lambda t: (bool(re.search(r"[A-Za-z]", t[1])), rnd.random()))
    for i, (r, q, fixed) in enumerate(dolly):
        if r.id in promote:
            rows.append(row(r.id, "translated_instruction", "dolly_mt", q, "dolly", r.id, bool(fixed),
                            status="changed" if q != r.prompt else "kept", original_prompt=r.prompt,
                            change_type="recategorized", reason="open-ended translated Dolly instruction"))
    for i, (r, q, fixed) in enumerate(rest):
        if i < CONCEPT_CAP:
            rows.append(row(r.id, "knowledge_qa", "concept", q, "dolly", r.id, bool(fixed),
                            status="changed" if q != r.prompt else "kept", original_prompt=r.prompt,
                            change_type="fixed_machine_translation" if fixed else "",
                            reason="garbled machine translation fixed" if fixed else ""))
        else:
            drops.append(dropped(r.id, r.prompt, "knowledge_qa cap: lower-priority translated concept question",
                                 "step7"))

    write_jsonl(WORK / "step6_knowledge_rows.jsonl", rows)
    write_jsonl(WORK / "step6_knowledge_eval_pool.jsonl", eval_pool)
    write_jsonl(WORK / "step6_knowledge_removed.jsonl", drops)
    kq = [r for r in rows if r["category"] == "knowledge_qa"]
    tails_now = sum(1 for r in kq if r["subcategory"] == "trivia" and any(t in r["prompt"] for t in NEW_TAILS + TAILS))
    tail_base = sum(1 for r in kq if r["subcategory"] == "trivia" and r["original_prompt"] and
                    any(r["original_prompt"].endswith(t) for t in TAILS))
    out = {"knowledge_rows": len(kq), "by_sub": Counter(r["subcategory"] for r in kq),
           "promoted_to_translated": len(promote), "dropped": len(drops),
           "trivia_rows_with_tail": tails_now, "trivia_rows_originally_with_tail": tail_base,
           "eval_reserve": len(eval_pool)}
    (WORK / "step6_knowledge_stats.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
