"""Step 4: drop the templated math families, load the 275 hand-written problems,
compute every answer in Python and split 250 train / 25 eval.

Writes data/work/math_answers.csv, step4_rows.jsonl, step4_removed.jsonl.
"""
import datetime as dt
import json
import math
import random
from collections import Counter

import pandas as pd

from common import WORK, MANUAL, row, dropped, write_jsonl, new_id

ETH_EPOCH = 1724221  # JDN of Meskerem 1, year 1 (Amete Mihret)
DAYS = ["ሰኞ", "ማክሰኞ", "ረቡዕ", "ሐሙስ", "ዓርብ", "ቅዳሜ", "እሑድ"]


def eth_to_jdn(y, m, d):
    return ETH_EPOCH - 1 + 365 * (y - 1) + y // 4 + 30 * (m - 1) + d


def jdn_to_greg(j):
    return dt.date.fromordinal(j - 1721425)


def greg_to_jdn(y, m, d):
    return dt.date(y, m, d).toordinal() + 1721425


def eth_to_greg(y, m, d):
    return jdn_to_greg(eth_to_jdn(y, m, d)).isoformat()


def greg_to_eth(y, m, d):
    j = greg_to_jdn(y, m, d)
    r = (j - ETH_EPOCH) % 1461
    n = r % 365 + 365 * (r // 1460)
    year = 4 * ((j - ETH_EPOCH) // 1461) + r // 365 - r // 1460 + 1
    month = n // 30 + 1
    day = n % 30 + 1
    return (year, month, day)


def eth_hour(h24):
    """International 24h clock -> Ethiopian 12h clock (hours since 6 o'clock)."""
    e = (h24 - 6) % 12
    return e if e >= 1 else e + 12


def intl_hour(eth, part):
    base = {"morning": 6, "day": 6, "evening": 18, "night": 18}[part]
    return (base + eth) % 24


def eth_add(eth_start_morning, hours):
    """Ethiopian daytime clock start (counted from 6 AM) plus a duration -> Ethiopian clock."""
    h24 = (6 + eth_start_morning) % 24 + hours
    return round(eth_hour(h24 % 24), 4)


def weekday_after(day, n):
    return DAYS[(DAYS.index(day) + n) % 7]


ENV = {"math": math, "eth_to_greg": eth_to_greg, "greg_to_eth": greg_to_eth, "eth_hour": eth_hour,
       "intl_hour": intl_hour, "eth_add": eth_add, "weekday_after": weekday_after, "min": min}


def self_test():
    assert eth_to_greg(2018, 1, 1) == "2025-09-11"
    assert eth_to_greg(2016, 1, 1) == "2023-09-12"      # after an Ethiopian leap year
    assert greg_to_eth(2026, 1, 7) == (2018, 4, 29)     # Genna
    assert eth_hour(7) == 1 and eth_hour(18) == 12 and eth_hour(12) == 6


def fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}".rstrip("0").rstrip(".")
    if isinstance(v, tuple):
        return "(" + ", ".join(fmt(x) for x in v) + ")"
    return str(v)


def main():
    self_test()
    rnd = random.Random(4)
    df = pd.read_parquet(WORK / "labelled.parquet")
    old = df[(df.split_in == "train") & (df.family.str.startswith("math_"))]
    drops = [dropped(r.id, r.prompt, f"templated math family '{r.family[5:]}' (fixed numbers / "
                                     f"unrealistic prices); replaced by hand-written problems", "step4_math")
             for r in old.itertuples()]

    probs = []
    for part in ["math_part1.py", "math_part2.py", "math_part3.py", "math_part4.py", "math_part5.py"]:
        ns = {}
        exec((MANUAL / part).read_text(encoding="utf-8"), ns)
        probs += ns["P"]
    recs = []
    for typ, style, q, expr, status in probs:
        ans = None
        if expr is not None:
            ans = eval(expr, ENV)  # every numeric problem must evaluate
        if status == "ok" and ans is None:
            raise ValueError(f"no answer for solvable problem: {q}")
        recs.append({"id": new_id("math|" + q), "type": typ, "style": style, "status": status, "prompt": q,
                     "computed_answer": fmt(ans) if ans is not None else "",
                     "note": {"ok": "", "missing": "information missing: the model should ask / point it out",
                              "contradiction": "data is inconsistent: the model should point it out"}[status]})
    assert len({r["prompt"] for r in recs}) == len(recs), "duplicate math prompt"
    # stratified eval pick: 25 problems over different types/styles
    by_type = {}
    for r in recs:
        by_type.setdefault(r["type"], []).append(r)
    eval_ids = set()
    types = sorted(by_type)
    rnd.shuffle(types)
    while len(eval_ids) < 25:
        for t in types:
            pool = [r for r in by_type[t] if r["id"] not in eval_ids]
            if pool and len(eval_ids) < 25:
                eval_ids.add(rnd.choice(pool)["id"])
    rows = []
    for r in recs:
        split = "eval" if r["id"] in eval_ids else "train"
        r["split"] = split
        rows.append(row(r["id"], "math", r["type"], r["prompt"], "v2_generated", None, True, split=split,
                        status="new"))
    pd.DataFrame(recs).to_csv(WORK / "math_answers.csv", index=False)
    write_jsonl(WORK / "step4_rows.jsonl", rows)
    write_jsonl(WORK / "step4_removed.jsonl", drops)
    tr = [r for r in recs if r["split"] == "train"]
    out = {"dropped_templated": len(drops), "written": len(recs), "train": len(tr), "eval": len(recs) - len(tr),
           "types": len(by_type), "style_train": Counter(r["style"] for r in tr),
           "status_train": Counter(r["status"] for r in tr),
           "non_integer_answers": sum(1 for r in recs if r["computed_answer"] and "." in r["computed_answer"])}
    (WORK / "step4_math_stats.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
