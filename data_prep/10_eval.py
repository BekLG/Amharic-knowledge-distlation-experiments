"""Step 11: rebuild the 300-row eval set so it covers what train teaches.

Sources: native rows of the old eval file (50; surplus natives go to train), held-out
reading-comprehension rows + the Gurnah row with its premise fixed (knowledge 20),
old eval passages turned into summarisation tasks (20), one natural rewrite per old
templated eval situation, and fresh eval prompts (data_prep/manual/new_prompts/eval_*.txt),
plus the eval rows produced by the math / format / latin steps.
"""
import json
import random
from collections import Counter

import pandas as pd

from common import WORK, MANUAL, EVAL_TARGETS, row, dropped, write_jsonl, read_jsonl, parse_tsv, lines_of, new_id, strip_credits

SUMM_INSTR = [
    "ይህን ጽሑፍ በሦስት ዓረፍተ ነገር አጠቃልልልኝ።", "ዋና ዋና ሀሳቦቹን በነጥብ ዘርዝርልኝ።",
    "ይሄን ለ12 ዓመት ልጅ በሚገባ ቀላል ቋንቋ በአጭሩ ንገረኝ።", "ጊዜ የለኝም፣ ዋናውን ሀሳብ በአንድ አንቀጽ ብቻ ንገረኝ።",
    "ከዚህ ጽሑፍ የማይረሱ ሦስት ቁም ነገሮችን አውጣልኝ።", "ይህን ጽሑፍ ለክፍል ጓደኞቼ በምናገረው አጭር ማጠቃለያ ቀይረው።",
    "ጽሑፉ ስለ ምን እንደሆነ በሁለት ዓረፍተ ነገር ንገረኝ።", "ለማስታወሻዬ የሚሆን አጭር ማጠቃለያ ጻፍልኝ።",
    "ይሄን ረጅም ጽሑፍ አሳጥረህ ዋናውን ብቻ አስቀርልኝ።", "የዚህን ጽሑፍ ቁልፍ መረጃዎች በአጭሩ አቅርብልኝ።",
]
SENSITIVE_NATIVE = {"53150f55", "125320e9", "dc86727c", "617025c4", "3330e375", "a3675030"}
GURNAH_OLD = "የዚምባቡዌና የእንግሊዝ ዜግነት ያለው ማነው?"
GURNAH_NEW = "የታንዛኒያና የእንግሊዝ ዜግነት ያለውና በዛንዚባር የተወለደው ማነው?"
MISMATCHED = ["ውሃን ከተማ ከተማ የቻይና", "በሕንድ ግዛት ውስጥ ተከስቶ"]


def main():
    rnd = random.Random(11)
    df = pd.read_parquet(WORK / "labelled.parquet")
    ev = df[df.split_in == "eval"].copy()
    c = Counter(lines_of(p)[-1] for p in ev.prompt if len(lines_of(p)) == 2)

    def kind(r):
        p, ls = r.prompt, lines_of(r.prompt)
        if r.family == "passage_qa":
            return "P"
        if r.family == "latin" or r.language_mode == "latin_am":
            return "L"
        if p.startswith("ዝርዝር፦"):
            return "F"
        if p.startswith("በክፍል ") and "ተማሪዎች አሉ" in p:
            return "M"
        if p.startswith("ጽሑፌ፦") or (len(ls) == 2 and c[ls[1]] >= 2):
            return "G"
        return "N"
    ev["k"] = ev.apply(kind, axis=1)
    dec = {k: (d, reason) for k, d, reason in parse_tsv(MANUAL / "eval_decisions.tsv", 3)}

    out, to_train, removed = [], [], []
    # natives
    nat = []
    for r in ev[ev.k == "N"].itertuples():
        d = dec.get(r.id[6:14])
        if d:
            removed.append(dropped(r.id, r.prompt, d[1], "step11_eval"))
        else:
            nat.append(r)
    rnd.shuffle(nat)
    # natives sharing a line with a train prompt cannot be eval rows: they go to train instead
    from dedup import shareable_lines
    train_lines = {l for r in read_jsonl(WORK / "train_assembled.jsonl") for l in shareable_lines(r["prompt"])}
    nat.sort(key=lambda r: any(l in train_lines for l in shareable_lines(r.prompt)))
    for i, r in enumerate(nat):
        sens = r.id[6:14] in SENSITIVE_NATIVE
        if i < 50:
            out.append(row(r.id, "native_chat", "native", r.prompt, "aya_dataset", r.id, True, split="eval",
                           status="kept", original_prompt=r.prompt, sensitive=sens))
        else:
            to_train.append(row(r.id, "native_chat", "native", r.prompt, "aya_dataset", r.id, sens,
                                status="kept", original_prompt=r.prompt, note="moved from old eval file (surplus native)",
                                sensitive=sens))
    # latin rows of the old eval file
    for r in ev[ev.k == "L"].itertuples():
        out.append(row(r.id, "latin_amharic", "romanized", r.prompt, "v2_generated", r.id, True, split="eval",
                       status="kept", original_prompt=r.prompt, language_mode_="latin_am"))
    # templated math / format rows of the old eval file are replaced
    for r in ev[ev.k.isin(["M", "F"])].itertuples():
        removed.append(dropped(r.id, r.prompt, "templated eval math/format item; replaced by hand-written ones",
                               "step11_eval"))
    # templated situation pairs: one natural rewrite per situation
    rew = {s: (cs, p) for s, cs, p in parse_tsv(MANUAL / "eval_rewrites.tsv", 3)}
    used = set()
    for r in ev[ev.k == "G"].sort_values("id").itertuples():
        sit = lines_of(r.prompt)[0]
        if sit in rew and sit not in used:
            used.add(sit)
            cat, sub = rew[sit][0].split("/")
            out.append(row(r.id, cat, sub, rew[sit][1], "v2_rewritten", r.id, True, split="eval", status="changed",
                           original_prompt=r.prompt, change_type="rewritten_natural",
                           reason="templated eval pair rewritten as one natural prompt (one per situation)"))
        else:
            removed.append(dropped(r.id, r.prompt, "templated eval pair: situation already covered by one "
                                                   "rewritten prompt", "step11_eval"))
    assert used == set(rew), set(rew) - used
    # passages: Gurnah premise fix -> knowledge; others -> summarisation
    pas = ev[ev.k == "P"].sort_values("id")
    summ_i = 0
    for r in pas.itertuples():
        q, passage = r.prompt.split("\nጽሑፍ፦\n", 1)
        passage = strip_credits(passage.strip())
        if GURNAH_OLD in q:
            newp = f"{q.replace(GURNAH_OLD, GURNAH_NEW)}\nጽሑፍ፦\n{passage}"
            out.append(row(r.id, "knowledge_qa", "reading_comprehension", newp, "amharic_qa", r.id, True,
                           split="eval", status="changed", original_prompt=r.prompt, change_type="fixed_premise",
                           reason="Gurnah was born in Zanzibar, Tanzania (not Zimbabwe); premise fixed"))
            continue
        if any(m in q for m in MISMATCHED):
            removed.append(dropped(r.id, r.prompt, "question does not match its passage", "step11_eval"))
            continue
        if summ_i >= 20:
            removed.append(dropped(r.id, r.prompt, "eval summarisation quota (20) already filled", "step11_eval"))
            continue
        instr = SUMM_INSTR[summ_i % len(SUMM_INSTR)]
        newp = f"{instr}\n\n{passage.strip()}"
        summ_i += 1
        out.append(row(r.id, "summarization", "passage", newp, "amharic_qa", r.id, True, split="eval",
                       status="changed", original_prompt=r.prompt, change_type="converted_to_summarization",
                       reason="old eval reading-comprehension passage reused as a summarisation task"))
    # rows produced by other steps for eval (math, format, latin, knowledge reserve, fresh eval prompts)
    step_eval = read_jsonl(WORK / "eval_from_steps.jsonl")
    kn = [r for r in step_eval if r["category"] == "knowledge_qa"]
    for r in kn[19:]:  # the Gurnah row is the 20th knowledge item
        removed.append(dropped(r["id"], r["original_prompt"], "held-out passage not needed: eval knowledge quota "
                                                            "filled", "step11_eval"))
    step_eval = [r for r in step_eval if r["category"] != "knowledge_qa"] + kn[:19]
    out += step_eval

    # enforce exact eval targets (fresh prompts are trimmed when a category has spares)
    target = {}
    for k, v in EVAL_TARGETS.items():
        for cat in (k if isinstance(k, tuple) else (k,)):
            target[cat] = (k, v)
    grp = Counter()
    final = []
    for r in out:
        k, v = target[r["category"]]
        if grp[k] < v:
            grp[k] += 1
            final.append(r)
    short = {str(k): (grp[k], v) for k, v in EVAL_TARGETS.items() if grp[k] != v}
    assert not short, short
    assert len(final) == 300, len(final)
    write_jsonl(WORK / "eval_built.jsonl", final)
    write_jsonl(WORK / "eval_to_train.jsonl", to_train)
    write_jsonl(WORK / "eval_removed.jsonl", removed)
    print({"eval": len(final), "natives_to_train": len(to_train), "removed_old_eval": len(removed),
           "by_cat": Counter(r["category"] for r in final)})


if __name__ == "__main__":
    main()
