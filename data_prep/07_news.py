"""Step 8: news-based prompts (summarisation, Amharic->English translation,
classification / entity lists). Strip trailing source junk and URLs, split merged
words at confident sentence boundaries, flag loaded political terms and check the
political/war share."""
import json
import re
from collections import Counter

import pandas as pd

from common import WORK, MANUAL, row, write_jsonl, parse_tsv

NEWS_FAMILIES = ["news_sum", "news_cls", "tr_am_en", "tr_en_am", "wiki_split"]
# sentence-final verb forms that, when glued to a following word, mark a lost "።"
LONG_FINALS = ["ተገልጿል", "አስታውቋል", "ገልጸዋል", "ገልፀዋል", "ተናግረዋል", "ገልጿል", "ገልጿል", "ተናግሯል", "ይታወሳል",
               "ታውቋል", "አስታውቀዋል", "ተብሏል", "አመልክቷል", "ጠቁመዋል", "አስታወቀ", "ተገልጧል", "ተገልጻል",
               "አረጋግጠዋል", "አሳስበዋል", "ያመለክታሉ", "ያመላክታሉ", "ተዘግቧል", "አስረድተዋል"]
SHORT_FINALS = ["ቀረበ", "ተለየ", "ገለፀ", "ገለጸ", "ተናገረ", "ተካሄደ", "ተጠናቀቀ", "ተመረቀ", "ተፈረመ", "ተጀመረ"]
LOADED = ["ጁንታ", "ወያኔ", "ሸኔ", "አሸባሪው", "ጸረ ሰላም", "ፀረ ሰላም", "ተስፋፊ", "ባንዳ", "ጽንፈኛ", "ፅንፈኛ"]
POLITICAL = ["ፖለቲካ", "ምርጫ", "ጦርነት", "ፓርቲ", "ታጣቂ", "ግጭት", "ጦር ሰራዊት", "ወታደር", "ጠቅላይ ሚኒስትር",
             "ፕሬዚዳንት", "ፕሬዝዳንት", "ፕሬዚደንት", "ተቃዋሚ", "አማፂ", "ጥቃት", "ሽብር", "መፈንቅለ", "ፓርላማ",
             "ምክር ቤት", "ሚኒስትር", "ወረራ", "ጦር"]


def clean_body(t):
    notes = []
    sents = re.split(r"(?<=።)\s*", t)
    keep = [s for s in sents if not re.search(r"https?|www", s)]
    if len(keep) != len(sents):
        notes.append("removed_url_sentence")
    t = " ".join(s for s in keep if s).strip()
    # trailing source credit: ...ምንጭ ቢቢሲ። / ...ምንጭአልጀዚራ። (not the city name አርባምንጭ)
    new = re.sub(r"(?<!አርባ)\s*ምንጭ\s*[:፦]?[^።]{0,60}።?\s*$", "", t)
    if new != t:
        notes.append("removed_source_credit")
        t = new.rstrip()
        if t and t[-1] not in "።?!":
            t += "።"
    new = re.sub(r"(\s*(ኤፍቢሲ|ኢዜአ|FBC|ቢቢሲ\s*FBC)\s*።?\s*)+$", "", t)
    if new != t:
        notes.append("removed_trailing_dateline")
        t = new.rstrip()
        if t and t[-1] not in "።?!":
            t += "።"
    # split glued sentence boundaries
    def split_long(m):
        return f"{m.group(1)}። {m.group(2)}"
    pat_long = re.compile(r"(%s)([ሀ-፿]{2,})" % "|".join(sorted(set(LONG_FINALS), key=len, reverse=True)))
    pat_short = re.compile(r"(?<=\s)(%s)([የበለከሲ][ሀ-፿]{2,})" % "|".join(SHORT_FINALS))
    n0 = t
    t = pat_long.sub(split_long, t)
    t = pat_short.sub(split_long, t)
    if t != n0:
        notes.append("split_merged_sentences")
    return t, notes


def load_pools():
    return {k: [x.strip() for x in v.split("||")] for k, v in parse_tsv(MANUAL / "passage_instruction_pools.tsv", 2)}


def main():
    pools = load_pools()
    use = Counter()
    df = pd.read_parquet(WORK / "labelled.parquet")
    n = df[(df.split_in == "train") & (df.family.isin(NEWS_FAMILIES + ["residual"]))]
    n = n[(n.family != "residual") | (n.category == "summarization")]
    rows, stats = [], Counter()
    for r in n.itertuples():
        if "\n\n" in r.prompt:
            instr, body = r.prompt.split("\n\n", 1)
        else:  # aya "ጽሑፉን ጠቅለል አድርገህ ተናገር፦ <body>"
            instr, body = r.prompt.split("፦", 1) if "፦" in r.prompt else ("", r.prompt)
            instr = instr + "፦"
        if r.family in ("tr_en_am", "wiki_split"):
            body2, notes = body.strip(), []
            if r.family == "tr_en_am":
                b = re.sub(r"\s+'\s+s\b", "'s", body2)
                b = re.sub(r"(\w)\s+'\s+", r"\1' ", b)
                b = re.sub(r"\s+--\s+", " – ", b)
                b = re.sub(r"\s+([,.;:!?])", r"\1", b)
                if b != body2:
                    notes.append("detokenized")
                    body2 = b
        else:
            body2, notes = clean_body(body.strip())
        pool_key = r.family
        if r.family == "news_cls":
            pool_key = "news_cls_entities" if "ስሞች" in instr else "news_cls_topic"
        if r.family == "wiki_split" and "\n\n" not in r.prompt:
            instr, body2 = r.prompt.split("\n", 1)
        if pool_key in pools:
            v = min(pools[pool_key], key=lambda x: (use[x], x != instr.strip()))
            use[v] += 1
            if v != instr.strip():
                notes.append("instruction_varied")
            instr = v
        sep = "\n" if r.family == "wiki_split" and "\n\n" not in r.prompt else "\n\n"
        new = f"{instr}{sep}{body2}" if "\n" in r.prompt else f"{instr} {body2}"
        loaded = [w for w in LOADED if w in body2]
        political = r.source == "news" and sum(body2.count(w) for w in POLITICAL) >= 2
        for x in notes:
            stats[x] += 1
        stats["loaded_flag"] += bool(loaded)
        stats["political"] += political
        changed = new != r.prompt
        rows.append(row(r.id, r.category, r.subcategory, new, r.source, r.id, bool(loaded),
                        status="changed" if changed else "kept", original_prompt=r.prompt,
                        change_type="+".join(notes), reason=("varied repeated instruction line; " if "instruction_varied" in notes else "") +
                        ("cleaned news text: " + ", ".join(n for n in notes if n != "instruction_varied") if [n for n in notes if n != "instruction_varied"] else ""),
                        political=political, loaded_terms=",".join(loaded), family=r.family))
    write_jsonl(WORK / "step7_news_rows.jsonl", rows)
    news_rows = [x for x in rows if x["source"] == "news"]
    out = {"rows": len(rows), **stats, "political_share_of_news": round(stats["political"] / len(news_rows), 3),
           "max_instruction_uses": max(use.values()),
           "by_category": Counter(x["category"] for x in rows)}
    (WORK / "step7_news_stats.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
