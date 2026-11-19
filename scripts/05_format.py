"""Step 4b: list / JSON / table format tasks. Cap at 120 in train, vary the numbers,
item lists and notices (the v1 set reused the same 10 number lists), and vary the
instruction wording. 15 extra rows are generated for eval."""
import json
import random
from collections import Counter

import pandas as pd

from common import WORK, row, dropped, write_jsonl, new_id, lines_of

TRAIN_N, EVAL_N = 118, 0
ITEMS = [  # name, typical unit price range in birr
    ("ዳቦ", 10, 15), ("እንቁላል", 14, 18), ("ወተት (ሊትር)", 70, 95), ("ስኳር (ኪሎ)", 120, 150), ("ጤፍ (ኪሎ)", 130, 170),
    ("ዘይት (ሊትር)", 280, 340), ("ሽንኩርት (ኪሎ)", 60, 120), ("ቲማቲም (ኪሎ)", 40, 90), ("ድንች (ኪሎ)", 35, 60),
    ("ሩዝ (ኪሎ)", 130, 160), ("ፓስታ (ፓኬት)", 55, 75), ("ሳሙና", 35, 55), ("ደብተር", 35, 60), ("እስክሪብቶ", 12, 25),
    ("ቡና (ኪሎ)", 750, 1000), ("ሻይ ቅጠል (ፓኬት)", 30, 45), ("ምስር (ኪሎ)", 140, 190), ("በርበሬ (ኪሎ)", 450, 700),
    ("ሙዝ (ኪሎ)", 45, 70), ("ብርቱካን (ኪሎ)", 60, 110), ("አቮካዶ", 15, 30), ("ውሃ (ጠርሙስ)", 20, 30),
    ("ለስላሳ", 30, 45), ("ሶፍት (ጥቅል)", 25, 40), ("ቅቤ (ኪሎ)", 900, 1300), ("ጨው (ኪሎ)", 25, 40),
    ("ማር (ኪሎ)", 550, 800), ("ነጭ ሽንኩርት (ኪሎ)", 180, 300), ("ሎሚ (ኪሎ)", 70, 120), ("ከሰል (ኩንታል)", 900, 1400),
]
NOTICES = [
    "ኮርሱ ነፃ ነው። ቅድመ ምዝገባ ግን ያስፈልጋል።", "የውሃ መስመሩ ተጠግኗል። አገልግሎቱ ከነገ ጀምሮ ይመለሳል።",
    "ቤተ መጻሕፍቱ ቅዳሜ ግማሽ ቀን ይሰራል። እሁድ ዝግ ነው።", "የወላጆች ስብሰባ ሐሙስ ከሰዓት ነው። ሁሉም ወላጅ እንዲገኝ ይጠየቃል።",
    "የክትባት ዘመቻ በጤና ጣቢያው ይካሄዳል። ከአምስት ዓመት በታች ያሉ ህፃናት ይከተባሉ።", "መብራት ነገ ከጠዋት እስከ ቀትር ይቋረጣል። ለጥገና ስራ ነው።",
    "የእድር መዋጮ እስከ ወሩ መጨረሻ መከፈል አለበት። ያልከፈሉ አባላት ቅጣት አለባቸው።", "አዲሱ የታክሲ ተራ በመገናኛ ተከፍቷል። የቀድሞው ተራ ተዘግቷል።",
    "የእግር ኳስ ውድድሩ ለሳምንት ተራዝሟል። ሜዳው በዝናብ ተበላሽቷል።", "ሱቃችን አዲስ እቃዎች አስገብቷል። እስከ አርብ 10% ቅናሽ አለ።",
    "ፈተናው በ8 ሰዓት ይጀምራል። ተማሪዎች መታወቂያ ይዘው ይምጡ።", "የሰፈሩ የጽዳት ዘመቻ እሁድ ጠዋት ነው። መጥረጊያ ይዘው ይምጡ።",
    "ክሊኒኩ ቅዳሜ አገልግሎት አይሰጥም። ለድንገተኛ ጉዳይ ሆስፒታሉን ይጠቀሙ።", "የቤት ኪራይ ከሚቀጥለው ወር ይጨምራል። አዲሱ ዋጋ በደብዳቤ ይገለጻል።",
    "የኮምፒውተር ስልጠና ለሴቶች ቅድሚያ ይሰጣል። ቦታው ለ20 ሰው ብቻ ነው።", "ባንኩ በበዓል ቀን ዝግ ነው። የኤቲኤም አገልግሎት ግን ይሰራል።",
    "የገበያው ቀን ከረቡዕ ወደ ሐሙስ ተቀይሯል። ነጋዴዎች ቦታቸውን ቀድመው ይያዙ።", "የዛፍ ችግኞች በነፃ ይታደላሉ። እያንዳንዱ ቤት ሁለት ይወስዳል።",
    "የመንገድ ጥገና ለሁለት ሳምንት ይቆያል። አሽከርካሪዎች ሌላ መንገድ ይጠቀሙ።", "የቡና ማፍላት ስልጠና ቅዳሜ ይሰጣል። ተሳታፊዎች 100 ብር ይከፍላሉ።",
    "ስብሰባው በኦንላይን ይካሄዳል። ሊንኩ ጠዋት በቴሌግራም ይላካል።", "የትምህርት ቤቱ ቤተ ሙከራ ታድሷል። ተማሪዎች ከሰኞ ጀምሮ መጠቀም ይችላሉ።",
    "የውሃ ክፍያ በሞባይል ባንኪንግ መክፈል ተችሏል። ደረሰኙን ማስቀመጥ አይርሱ።", "አውቶቡሱ ዘግይቷል። መንገዱ በጥገና ላይ ነው።",
]
NUM_INSTR = {
    "sort": ["{nums} ቁጥሮቹን ከትንሽ ወደ ትልቅ ደርድር። ቁጥሮቹን ብቻ በኮማ ለይተህ መልስ።",
             "እነዚህን ቁጥሮች ከትልቅ ወደ ትንሽ ደርድራቸው፦ {nums}",
             "{nums}\nበቅደም ተከተል (ከትንሽ ወደ ትልቅ) አስቀምጣቸው፤ ሌላ ነገር አትጻፍ።"],
    "odd": ["{nums}\nጎዶሎ ቁጥሮችን ብቻ በተሰጡበት ቅደም ተከተል በኮማ ለይተህ ጻፍ።",
            "ከእነዚህ ውስጥ ጥንድ ቁጥሮቹን ብቻ አውጣ፦ {nums}"],
    "maxmin": ["{nums}\nትልቁንና ትንሹን ቁጥር የያዘ JSON ብቻ ስጠኝ።",
               "ከ {nums} ውስጥ ትልቁ፣ ትንሹ እና ድምራቸው ስንት ነው? በJSON መልስ።"],
    "dedupe_json": ["{nums}\nየተደገሙትን አስወግድና የመጀመሪያውን ቅደም ተከተል ጠብቀህ በJSON ዝርዝር ብቻ መልስ።",
                    "ከዚህ ዝርዝር ድግግሞሹን አጥፋልኝ፤ ውጤቱ በJSON ይሁን፦ {nums}"],
    "unique_count": ["[{nums}]\nበዚህ ዝርዝር ውስጥ ስንት የተለያዩ እሴቶች አሉ? ቁጥሩን ብቻ መልስ።",
                     "{nums} — እዚህ ውስጥ ስንት የማይደገሙ ቁጥሮች አሉ?"],
    "freq_table": ["{nums}\nየእያንዳንዱን ቁጥር ድግግሞሽ በሁለት አምድ ሰንጠረዥ አሳይ (ቁጥር፣ ብዛት)።",
                   "እያንዳንዱ ቁጥር ስንት ጊዜ እንደተደገመ በMarkdown ሰንጠረዥ አሳየኝ፦ {nums}"],
    "py_sum": ["[{nums}]\nየዚህን ዝርዝር ድምር የሚመልስ ተግባር በPython ጻፍ፤ ውጤቱን አታትም፣ ይመልስ ብቻ።",
               "Python ውስጥ የ[{nums}]ን አማካይ የሚመልስ አጭር ተግባር ጻፍልኝ።"],
    "js_array": ["[{nums}]\nይህን ዝርዝር ወደ JavaScript ቀይረው፤ አንድ መስመር ኮድ ብቻ ስጥ።",
                 "{nums} የሚሉትን ቁጥሮች JavaScript ላይ ከትንሽ ወደ ትልቅ የሚደረድር አጭር ኮድ ጻፍ።"],
    "py_dedupe": ["[{nums}]\nበPython ድግግሞሹን የሚያስወግድና ቅደም ተከተሉን የሚጠብቅ አጭር ኮድ ጻፍ (ከአራት መስመር አይብለጥ)።",
                  "በPython [{nums}] ውስጥ ያሉትን ድግግሞሾች ቅደም ተከተሉን ጠብቄ እንዴት አስወግዳለሁ? አጭር ኮድ ብቻ።"],
    "py_sort": ["[{nums}] ይህን የPython ዝርዝር ከትንሽ ወደ ትልቅ የሚደረድር አንድ መስመር ኮድ ብቻ ስጥ።",
                "[{nums}] የሚለውን ዝርዝር ከትልቅ ወደ ትንሽ በPython እንዴት ልደርድረው? አንድ መስመር ብቻ።"],
}
ITEM_INSTR = {
    "item_json": ["ስም፣ ብዛት፣ ነጠላ ዋጋ በብር፦ {item}፣ {q}፣ {p}\nየእቃውን ስምና ጠቅላላ ዋጋ የያዘ JSON ስጥ፤ ጠቅላላው ቁጥር ይሁን።",
                  "{q} {item} በ{p} ብር ገዛሁ። ስም፣ ብዛት፣ ነጠላ ዋጋና ጠቅላላ ዋጋ ያሉት JSON ስጠኝ።"],
    "md_table": ["{rows}\nይህን መረጃ በሰንጠረዥ አሳየኝ፤ አምዶቹ ስም፣ ብዛት፣ ነጠላ ዋጋ እና ጠቅላላ ዋጋ ይሁኑ።",
                 "የዛሬ ግዢዬ፦\n{rows}\nበሰንጠረዥ አድርግልኝና መጨረሻ ላይ አጠቃላይ ድምሩን ጨምር።"],
    "one_line": ["ስም፣ ብዛት፣ ነጠላ ዋጋ በብር፦ {item}፣ {q}፣ {p}\nየእቃውን ስምና ጠቅላላ ዋጋ ብቻ በአንድ መስመር ጻፍ፤ መካከላቸው ኮሎን ይኑር።"],
    "csv": ["{rows}\nይህን በCSV አቅርብ፤ የአምዶቹ ስም እቃ፣ ብዛት፣ ነጠላ ዋጋ ይሁን።",
            "የሱቄን እቃዎች ወደ CSV ቀይርልኝ፤ ጠቅላላ ዋጋም ይጨመር፦\n{rows}"],
    "item_json_list": ["{rows}\nእያንዳንዱን እቃ ስም፣ ብዛትና ነጠላ ዋጋ የያዘ JSON ዝርዝር አድርገህ ስጠኝ፤ ማብራሪያ አያስፈልግም።"],
}
NOTICE_INSTR = {
    "title": ["«{n}»\nለዚህ ማስታወቂያ ከአምስት ቃላት የማይበልጥ ርዕስ ብቻ ጻፍ።", "ለዚህ ጽሑፍ አጭር ርዕስ ስጠው፦ {n}"],
    "bullets": ["«{n}»\nከጽሑፉ ሁለት ዋና ነጥቦችን አውጣ፤ እያንዳንዱን በ- ጀምር።", "{n}\nዋና ዋናዎቹን በነጥብ ዘርዝራቸው።"],
    "one_sentence": ["{n} ዋናውን ሀሳብ በአንድ ዓረፍተ ነገር ብቻ ጻፍ።", "ይህን በአንድ አጭር ዓረፍተ ነገር ጠቅልለው፦ «{n}»"],
    "missing_q": ["«{n}»\nበዚህ ማስታወቂያ ውስጥ ያልተጠቀሰ አንድ ጠቃሚ መረጃ ለማወቅ የምጠይቀውን ጥያቄ ብቻ ጻፍ።"],
    "title_facts": ["«{n}»\nርዕስና ሁለት ዋና መረጃዎችን የያዘ JSON ስጥ።",
                    "ይህን ማስታወቂያ ወደ JSON ቀይረው፤ ርዕስና መረጃዎች ይኑሩት፦ {n}"],
}


def nums(rnd, dup=False):
    k = rnd.randint(4, 8)
    scale = rnd.choice([(1, 20), (10, 99), (100, 999), (1, 50)])
    xs = [rnd.randint(*scale) for _ in range(k)]
    if dup or rnd.random() < 0.5:
        xs[rnd.randrange(k)] = xs[rnd.randrange(k)]
    return ", ".join(map(str, xs))


def item_rows(rnd, k):
    out = []
    for name, lo, hi in rnd.sample(ITEMS, k):
        out.append(f"{name}፣ {rnd.randint(1, 12)}፣ {rnd.randint(lo, hi)}")
    return out


def make(task, rnd):
    if task in NUM_INSTR:
        return rnd.choice(NUM_INSTR[task]).format(nums=nums(rnd, dup=task in ("dedupe_json", "py_dedupe",
                                                                               "unique_count", "freq_table")))
    if task in ITEM_INSTR:
        name, lo, hi = rnd.choice(ITEMS)
        return rnd.choice(ITEM_INSTR[task]).format(item=name, q=rnd.randint(1, 15), p=rnd.randint(lo, hi),
                                                   rows="\n".join(item_rows(rnd, rnd.randint(2, 4))))
    return rnd.choice(NOTICE_INSTR[task]).format(n=rnd.choice(NOTICES))


def main():
    rnd = random.Random(5)
    df = pd.read_parquet(WORK / "labelled.parquet")
    f = df[(df.split_in == "train") & (df.category == "format")]
    defin = f[f.prompt.map(lambda p: len(lines_of(p)) == 1 and p.endswith("ምንድን ነው?"))]
    f = f[~f.id.isin(defin.id)].sample(frac=1, random_state=5)
    tasks = list(NUM_INSTR) + list(ITEM_INSTR) + list(NOTICE_INSTR)
    rows, drops, seen = [], [], set()
    for i, r in enumerate(f.itertuples()):
        if i >= TRAIN_N:
            drops.append(dropped(r.id, r.prompt, "format tasks capped at 120; repeated number lists", "step4_format"))
            continue
        while True:
            p = make(tasks[i % len(tasks)], rnd)
            if p not in seen:
                seen.add(p)
                break
        rows.append(row(r.id, "format", tasks[i % len(tasks)], p, "v2_rewritten", r.id, False, status="changed",
                        original_prompt=r.prompt, change_type="varied_data_and_wording",
                        reason="v1 reused the same 10 number lists / item lines; regenerated data and wording"))
    for j in range(EVAL_N):
        task = tasks[(j * 3 + 1) % len(tasks)]
        while True:
            p = make(task, rnd)
            if p not in seen:
                seen.add(p)
                break
        rows.append(row(new_id("fmt-eval|" + p), "format", task, p, "v2_generated", None, True, split="eval",
                        status="new"))
    for r in defin.itertuples():  # "JSON ምንድን ነው?" style rows are translated Dolly questions, not format tasks
        rows.append(row(r.id, "translated_instruction", "dolly_mt", r.prompt, "dolly", r.id, False, status="kept",
                        original_prompt=r.prompt, change_type="recategorized", reason="definition question, not a format task"))
    write_jsonl(WORK / "step5_format_rows.jsonl", rows)
    write_jsonl(WORK / "step5_format_removed.jsonl", drops)
    print({"train": sum(1 for r in rows if r["split"] == "train" and r["category"] == "format"),
           "eval": EVAL_N, "dropped": len(drops), "tasks": Counter(r["subcategory"] for r in rows).most_common(5)})


if __name__ == "__main__":
    main()
