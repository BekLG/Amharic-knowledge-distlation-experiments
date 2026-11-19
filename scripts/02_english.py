"""Step 2: English prompts. Translate a useful subset into Amharic, drop US-only /
low-value ones, and add a varied Amharic-answer request to the rest."""
import json
import random
import re
from collections import Counter

import pandas as pd

from common import WORK, MANUAL, row, dropped, parse_tsv, write_jsonl, new_id

REQUESTS = [  # (text, kind) ; >=12 phrasings, mixed English / Amharic
    "Please answer in Amharic.", "Reply in Amharic, keep it short.", "Explain this to me in Amharic.",
    "Can you answer in Amharic? Simple language please.", "Answer in Amharic please.",
    "I'd like the answer in Amharic.", "Respond in Amharic, I understand it better.", "In Amharic please 🙏",
    "Write your answer in Amharic.", "Amharic answer please, not English.",
    "መልሱን በአማርኛ ስጠኝ።", "በአማርኛ አስረዳኝ እባክህ።", "በአማርኛ መልስልኝ።", "እባክህ መልሱን በአማርኛ ጻፍልኝ።",
    "አማርኛ ላይ ብታስረዳኝ ይሻለኛል።", "በአማርኛ ቢሆን ደስ ይለኛል።",
]
MAX_PER_PHRASE = 25


def add_request(prompt, phrase, position):
    if "\nContext:" in prompt:
        instr, ctx = prompt.split("\nContext:", 1)
        instr = instr.strip()
        if position == "start":
            instr = f"{phrase} {instr}"
        elif position == "end":
            instr = f"{instr} {phrase}"
        else:
            instr = f"{instr}\n{phrase}"
        return f"{instr}\n\nContext:{ctx}"
    if position == "start":
        return f"{phrase} {prompt}"
    if position == "end":
        return f"{prompt} {phrase}"
    return f"{prompt}\n\n{phrase}"


def main():
    rnd = random.Random(2)
    df = pd.read_parquet(WORK / "labelled.parquet")
    e = df[(df.split_in == "train") & (df.family.isin(["en", "dolly_ctx"]))].reset_index(drop=True)
    dec = {int(i): (d, t) for i, d, t in parse_tsv(MANUAL / "english_decisions.tsv", 3)}
    new_en = parse_tsv(MANUAL / "new_english.tsv", 2)

    rows, drops = [], []
    use = Counter()
    keep_idx = [i for i in range(len(e)) if i not in dec]
    items = [(e.id[i], e.prompt[i], "dolly", e.id[i], False) for i in keep_idx]
    items += [(new_id("en|" + p), p.replace("\\n", "\n"), "v2_generated", None, True) for sub, p in new_en]
    for k, (rid, p, src, origin, new) in enumerate(items):
        phrase = min(REQUESTS, key=lambda x: (use[x], rnd.random()))
        use[phrase] += 1
        assert use[phrase] <= MAX_PER_PHRASE
        pos = ["start", "end", "own_line"][k % 3]
        newp = add_request(p, phrase, pos)
        if new:
            rows.append(row(rid, "english_instruction", "user_text", newp, src, None, True, status="new"))
        else:
            sub = "context_qa" if "\nContext:" in p else "general"
            rows.append(row(rid, "english_instruction", sub, newp, src, origin, False, status="changed",
                            original_prompt=p, change_type="added_amharic_request",
                            reason=f"English prompt had no Amharic request; added one ({pos})"))
    for i, (d, t) in dec.items():
        rid, p = e.id[i], e.prompt[i]
        if d == "D":
            drops.append(dropped(rid, p, t, "step2_english"))
        else:
            rows.append(row(rid, "translated_instruction", "dolly_translated", t, "dolly", rid, True,
                            status="changed", original_prompt=p, change_type="translated_to_amharic",
                            reason="English instruction translated into natural Amharic"))
    write_jsonl(WORK / "step2_rows.jsonl", rows)
    write_jsonl(WORK / "step2_removed.jsonl", drops)
    stats = {"input": len(e), "kept_with_request": len(keep_idx), "new_english": len(new_en),
             "translated": sum(1 for d, _ in dec.values() if d == "T"),
             "dropped": len(drops), "request_phrasings_used": len(use), "max_uses": max(use.values())}
    (WORK / "step2_stats.json").write_text(json.dumps(stats, indent=1))
    print(stats)


if __name__ == "__main__":
    main()
