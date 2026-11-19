"""Rule-based detection of the generator families in the v1 pilot files.

Each input row gets a `family` describing how it was produced. Families drive the
default category/source; rows in the `residual` family are labelled by hand
(data_prep/manual/residual_labels.csv).
"""
import re
from collections import Counter

from common import lines_of, language_mode

NEWS_SUM_INSTR = {
    "በሁለት ወይም በሦስት ዓረፍተ ነገሮች የጽሑፉን ሀሳብ ግለጽ።", "ከዚህ ዜና ውስጥ ዋናውን ክስተትና አስፈላጊ ዝርዝሩን አውጣ።",
    "ቀጣዩን ጽሑፍ በአንድ አጭር አንቀጽ አሳጥረው።", "የጽሑፉን ቁልፍ መረጃዎች በሦስት ነጥቦች አቅርብ።",
    "ለዜና ክትትል ማስታወሻ የሚሆን ማጠቃለያ አዘጋጅ።", "የዘገባውን አስፈላጊ ነጥቦች ብቻ አስቀር። ከጽሑፉ ውጪ አትጨምር።",
    "በቀላል አነጋገር የዚህን ዘገባ ዋና መልእክት አስተላልፍ።", "ስለዚህ ዜና ለጓደኛ ለመንገር የሚሆን ማጠቃለያ ጻፍ።",
    "ጊዜ ለሌለው አንባቢ የሚሆን አጭር ማጠቃለያ ጻፍ።", "ይህ ዜና በአጭሩ ምን ይላል? ዋናውን ሀሳብ አጠቃልል።",
}
NEWS_CLS_INSTR = {
    "በጽሑፉ የተጠቀሱ የቦታና የድርጅት ስሞችን ለይተህ ዘርዝር። የሌለ ስም አትጨምር።",
    "ከዘገባው ውስጥ የቦታ እና የተቋም ስሞችን በሁለት ዝርዝሮች አውጣ። ያልተጠቀሰን አትገምት።",
    "ይህ ዘገባ ስለ ምን ነው? ከኢኮኖሚ፣ ከስፖርት፣ ከባህልና መዝናኛ፣ ከፖለቲካ ወይም ከአጠቃላይ ዜና አንዱን ለይ።",
    "የዜናውን ምድብ ምረጥ፦ ስፖርት፣ ቢዝነስ፣ መዝናኛ፣ ፖለቲካ፣ ሀገር አቀፍ ዜና፣ ዓለም አቀፍ ዜና።",
}
WIKI_SPLIT_INSTR = {
    "አዲስ እውነታ ሳትጨምር ይህን ዓረፍተ ነገር በበለጠ ውስብስብ አወቃቀር ጻፈው።",
    "ሀሳቡን ጠብቀህ የዓረፍተ ነገሩን አወቃቀር አስፋፋ።",
    "ቀጣዩን ዓረፍተ ነገር በተሟላ አገላለጽ ደግመህ ጻፍ፤ ከተሰጠው መረጃ አትውጣ።",
    "ትርጉሙን ሳትቀይር ዓረፍተ ነገሩን የበለጠ የተሟላ አድርገህ ጻፍ።",
}
# templated math families (tail sentence or distinctive pattern)
MATH_PATTERNS = [
    ("speed", r"ኪሎሜትር በ.*ሰዓት ተጓዝን"),
    ("percentage", r"ዋጋው በ20% ቀንሶ"),
    ("profit", r"የተጣራ ትርፉ ስንት ነው"),
    ("savings", r"በሳምንት .* ብር እቆጥባለሁ"),
    ("division", r"በማባዛት መልሱን አረጋግጥ"),
    ("perimeter", r"ዙሪያውን ለመሸፈን ስንት ሜትር"),
    ("average", r"የሦስት ቀን ሽያጭ"),
    ("box_packing", r"ስንት ሳጥን ያስፈልጋል"),
    ("fraction", r"በቀላል ክፍልፋይና በመቶኛ"),
    ("multiply", r"እያንዳንዱ .* ብር ከሆነ ጠቅላላው ስንት"),
]
FORMAT_HINTS = [
    r"JSON", r"\bCSV\b", r"Markdown", r"Python", r"JavaScript", r"ኮማ የለያቸው ቁጥሮች",
    r"ጎዶሎ ቁጥሮች", r"ከትንሽ ወደ ትልቅ ደርድር", r"ልዩ እሴቶች", r"ድግግሞሽ", r"^ስም፣ ብዛት፣ ነጠላ ዋጋ",
    r"ከአምስት ቃላት የማይበልጥ ርዕስ", r"ሁለት ነጥቦችን አውጣ", r"በአንድ መስመር ጻፍ",
    r"ዋናውን ሀሳብ በአንድ ዓረፍተ ነገር ብቻ መልስ", r"ማብራሪያ አትጨምር።$", r"የሌለ አንድ ጠቃሚ መረጃ",
]
# one-line generated prompts: situation sentence + a fixed tail
GEN1_TAILS = {
    "ዋናው ልዩነቱ ምንድን ነው?", "አጭር መግቢያ ስጠኝ።", "ለጀማሪ ምን ትነግረዋለህ?", "አንድ የሚስብ መግቢያ አንቀጽ ጻፍ።",
    "ለጀማሪ በሚገባ አስረዳኝ።", "ከየት ልጀምር?", "በቀላሉ እንዴት ልጀምር?", "መሠረታዊ ሀሳቡን አስረዳኝ።",
    "ተግባራዊ መፍትሔ አለህ?", "ለዚህ አሳታፊ ርዕስ ጻፍ።", "ለዚህ አጭር መልእክት ጻፍልኝ።", "ምን ብዬ ልናገር?",
    "ከቀላሉ እንጀምር።", "ያለ ብዙ ወጪ ምን ላድርግ?", "ውሳኔዬን በምን ላስመስርት?", "በዚህ ሀሳብ አጭር ታሪክ ጻፍ።",
    "እንዴት በጥንቃቄ ልረዳ?", "እርዳኝ።", "ትርጉሙን ጠብቀህ አሳጥረው።", "Min timikir alegn?",
}
KNOWLEDGE_TAILS = [
    "በአጭሩ አስረዳ፤ እርግጠኛ ካልሆንክ ግለጽ።", "መልሱንና አንድ ተዛማጅ ማብራሪያ ስጠኝ።", "ለጀማሪ የሚገባ ማብራሪያ ስጥ።",
    "መልሱን በሁለት ዓረፍተ ነገር አስረዳ።", "አጭር ማብራሪያም ጨምር።", "ስም ወይም ቁጥር ብቻ ሳይሆን ትንሽ አብራራ።",
    "ምንም ማብራሪያ ሳትጨምር መልሱን ብቻ ስጠኝ።",
]


FORMAT_INSTR_HINTS = [
    r"JSON", r"\bCSV\b", r"Markdown", r"JavaScript", r"ኮማ የለያቸው ቁጥሮች", r"ጎዶሎ ቁጥሮችን ብቻ",
    r"ከትንሽ ወደ ትልቅ", r"ልዩ እሴቶች", r"ድግግሞሽ", r"ከአምስት ቃላት የማይበልጥ ርዕስ", r"ሁለት ነጥቦችን አውጣ",
    r"በአንድ መስመር ጻፍ", r"ዋናውን ሀሳብ በአንድ ዓረፍተ ነገር ብቻ መልስ", r"የሌለ አንድ ጠቃሚ መረጃ",
    r"Python function", r"በPython ድግግሞሽን",
]
DATA_LINE = re.compile(r"^(\[?[\d, ]+\]?|ስም፣ ብዛት፣ ነጠላ ዋጋ በብር፦.*)")


def is_format(ls):
    text = " ".join(ls)
    if not any(re.search(h, text) for h in FORMAT_INSTR_HINTS):
        return False
    if DATA_LINE.match(ls[0]):
        return True
    # short two-sentence fact line + format instruction
    return len(ls[0]) < 80 and any(re.search(h, ls[-1] if len(ls) > 1 else ls[0]) for h in FORMAT_INSTR_HINTS) \
        and not re.search(r"error|Excel|app|phone", text)


def build_instruction_counter(prompts):
    c = Counter()
    for p in prompts:
        ls = lines_of(p)
        if len(ls) >= 2:
            c[ls[-1]] += 1
    return c


def detect(row, instr_counter):
    p, direction, task, domain = row["prompt"], row["direction"], row["task"], row["domain"]
    ls = lines_of(p)
    first = ls[0] if ls else ""
    last = ls[-1] if ls else ""
    mode = language_mode(p)
    if direction == "EN2AM":
        return "tr_en_am"
    if direction == "AM2EN":
        return "tr_am_en"
    if first in NEWS_SUM_INSTR:
        return "news_sum"
    if first in NEWS_CLS_INSTR:
        return "news_cls"
    if first in WIKI_SPLIT_INSTR:
        return "wiki_split"
    if first == "ጽሑፍ፦":
        return "format_text"  # passage + format instruction
    for name, pat in MATH_PATTERNS:
        if re.search(pat, p):
            return "math_" + name
    if "\nContext:" in p and mode == "en":
        return "dolly_ctx"
    if mode == "latin_am":
        return "latin"
    if mode == "en":
        return "en"
    if "\nጽሑፍ፦\n" in p:
        return "passage_qa"
    if "ከታች ያለውን ክፍል ተመልከቱ" in p:
        return "aya_ctx"
    if is_format(ls):
        return "format"
    if any(p.endswith(t) for t in KNOWLEDGE_TAILS) and len(ls) >= 1:
        return "knowledge_tail"
    if len(ls) == 2 and instr_counter[last] >= 10:
        return "pair"
    if len(ls) == 1:
        for t in GEN1_TAILS:
            if p.endswith(t) and p != t:
                return "gen1"
    return "residual"
