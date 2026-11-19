# Pilot prompt-set fix — report

Train: **4961** rows · Eval: **300** rows · Removed: **1592** input rows · Changed: **2717** rows · needs_human_review: **3264** · teacher-answer review list: **224** ids.

Everything is reproducible from `data/raw/` with `scripts/run_all.sh`. Hand-made judgements and prompts live in `data/manual/` (labels, pair verdict overrides, instruction variants, translations, math, Latin-script and new prompts).

## Validation (step 14)

| check | result | details |
|---|---|---|
| 1_schema | PASS |  |
| 2_unique_ids | PASS |  |
| 3_counts | PASS | train=4961, eval=300, eval_latin_am=18 |
| 4_duplicates | PASS | exact_duplicates=0, near_dup_train_pairs=0, near_dup_train_eval_pairs=0, near_dup_within_eval=0 |
| 5_repetition | PASS | max_line_uses=8, max_line=የጽሑፉን ቁልፍ መረጃዎች በሦስት ነጥቦች አቅርብ።, max_five_word_opening=11, opening=ስም፣ ብዛት፣ ነጠላ ዋጋ በብር፦, limit_opening=49, generated_with_dont_constra |
| 6_english_requests | PASS |  |
| 7_length | PASS | max_tokens=950.3, share_over_600=0.005, share_one_liners=0.331, one_liner_definition=single line of at most 9 words (about one phone-screen line in Amharic), sh |
| 8_no_pii_urls | PASS |  |
| 9_language_mix | PASS | am=0.8256, en=0.0907, mixed_plus_latin=0.0837 |
| 10_math_answers | PASS | math_rows=285 |
| 11_safety_listed | PASS | listed_total=224 |
| x_all_inputs_accounted | PASS | inputs=5300 |

**All checks pass: True**

## Final train composition

| category | train | target | kept as-is | changed | new | dropped (input rows) | in range |
|---|---|---|---|---|---|---|---|
| native_chat | 500 | 400–520 | 500 | 0 | 0 | 3 | yes |
| translated_instruction | 220 | 180–220 | 142 | 78 | 0 | 0 | yes |
| english_instruction | 200 | 150–200 | 0 | 153 | 47 | 69 | yes |
| summarization | 305 | 280–320 | 27 | 278 | 0 | 0 | yes |
| translation_en_am | 250 | 230–250 | 29 | 221 | 0 | 0 | yes |
| translation_am_en | 200 | 190–200 | 36 | 164 | 0 | 0 | yes |
| news_classification | 100 | 90–110 | 9 | 91 | 0 | 0 | yes |
| knowledge_qa | 350 | 250–350 | 198 | 152 | 0 | 240 | yes |
| everyday | 850 | 700–850 | 44 | 466 | 340 | 310 | yes |
| ethiopia_culture | 479 | 400–480 | 8 | 220 | 251 | 152 | yes |
| writing | 479 | 400–480 | 13 | 326 | 140 | 161 | yes |
| math | 260 | 240–260 | 0 | 0 | 260 | 250 | yes |
| format | 118 | 100–120 | 0 | 118 | 0 | 82 | yes |
| code_switch | 220 | 180–220 | 6 | 55 | 159 | 40 | yes |
| latin_amharic | 150 | 120–150 | 0 | 100 | 50 | 0 | yes |
| creative | 130 | 100–130 | 2 | 58 | 70 | 40 | yes |
| safety | 150 | 120–150 | 5 | 85 | 60 | 60 | yes |

## Final eval composition

| category | eval | target |
|---|---|---|
| native_chat | 50 | 50 |
| everyday | 30 | 30 |
| ethiopia_culture | 25 | 25 |
| writing | 25 | 25 |
| math | 25 | 25 |
| code_switch + latin_amharic | 25 | 25 |
| safety | 20 | 20 |
| creative | 15 | 15 |
| format | 15 | 15 |
| knowledge_qa | 20 | 20 |
| summarization | 20 | 20 |
| translation_en_am + translation_am_en | 20 | 20 |
| english_instruction + translated_instruction | 10 | 10 |

## Language mode

Train:

| language_mode | rows | share |
|---|---|---|
| am | 4096 | 82.6% |
| en | 450 | 9.1% |
| mixed | 265 | 5.3% |
| latin_am | 150 | 3.0% |

Eval:

| language_mode | rows | share |
|---|---|---|
| am | 250 | 83.3% |
| latin_am | 18 | 6.0% |
| mixed | 17 | 5.7% |
| en | 15 | 5.0% |

## Source

| source (train) | rows | share |
|---|---|---|
| v2_generated | 2550 | 51.4% |
| news | 600 | 12.1% |
| aya_dataset | 489 | 9.9% |
| dolly | 447 | 9.0% |
| wiki_split | 350 | 7.1% |
| v2_rewritten | 233 | 4.7% |
| amharic_qa | 172 | 3.5% |
| nq_open | 104 | 2.1% |
| addisgpt | 16 | 0.3% |

| source (eval) | rows | share |
|---|---|---|
| v2_generated | 181 | 60.3% |
| aya_dataset | 50 | 16.7% |
| amharic_qa | 40 | 13.3% |
| v2_rewritten | 29 | 9.7% |

## Step 1 labels before any change (train input)

| category | rows |
|---|---|
| everyday | 820 |
| knowledge_qa | 667 |
| writing | 500 |
| native_chat | 493 |
| ethiopia_culture | 380 |
| summarization | 305 |
| english_instruction | 300 |
| translation_en_am | 250 |
| math | 250 |
| format | 202 |
| translation_am_en | 200 |
| safety | 150 |
| code_switch | 101 |
| creative | 100 |
| news_classification | 100 |
| latin_amharic | 100 |
| translated_instruction | 82 |

| language_mode | rows |
|---|---|
| am | 4075 |
| en | 550 |
| mixed | 275 |
| latin_am | 100 |

| source | rows |
|---|---|
| v2_generated | 2503 |
| dolly | 690 |
| news | 600 |
| aya_dataset | 482 |
| wiki_split | 350 |
| amharic_qa | 206 |
| nq_open | 153 |
| addisgpt | 16 |

| detected generator family | rows |
|---|---|
| pair | 1555 |
| residual | 907 |
| gen1 | 370 |
| news_sum | 300 |
| tr_en_am | 250 |
| en | 243 |
| passage_qa | 206 |
| tr_am_en | 200 |
| format | 192 |
| knowledge_tail | 165 |
| news_cls | 100 |
| latin | 100 |
| wiki_split | 100 |
| dolly_ctx | 57 |
| math_profit | 25 |
| math_percentage | 25 |
| math_speed | 25 |
| math_fraction | 25 |
| math_division | 25 |
| math_perimeter | 25 |
| math_average | 25 |
| math_box_packing | 25 |
| math_savings | 25 |
| math_multiply | 25 |
| aya_ctx | 5 |

## Before / after by step

| step | before | after / what changed |
|---|---|---|
| 1 Load & label | 5,000 train + 300 eval rows | 180 rows flagged unusable while labelling (logged in removed.csv) |
| 2 English prompts | 300 English rows | 78 translated, 153 kept + Amharic request (16 phrasings, max 13 uses each), 69 dropped, 47 new English prompts |
| 3 Situation + instruction pairs | 1930 rows over 390 situations | verdicts {'keep': 1153, 'drop': 762, 'rewrite': 15}; 1168 kept; 47% of kept 2-line pairs merged; form labels removed in 75%; 'don't' constraints left: 0%; max instruction-line uses 4 |
| 4 Math | 250 templated rows (10 families) | 285 hand-written problems over 28 types → 260 train / 25 eval; train styles {'casual': 126, 'steps': 82, 'answer': 52}; {'ok': 234, 'missing': 16, 'contradiction': 10}; 82 non-integer answers |
| 4b Format | 202 list/JSON rows | 118 kept with new data and wording; 15 hand-written eval rows |
| 5 Latin-script Amharic | 100 rows (51 situations × 35 instructions) | 150 distinct rewritten train prompts + 13 new eval |
| 6 Code-switched | 61 natural rows after the step-3 caps | 220 after adding new code-switched prompts |
| 7 Knowledge | 751 knowledge-labelled rows | 350 kept ({'reading_comprehension': 172, 'trivia': 104, 'concept': 74}); 58 Dolly instructions → translated_instruction; trivia rows with a generic tail 30/101 originally tailed |
| 8 News | 955 news / translation / wiki rows | source credits removed 47, URL sentences 5, datelines 7, merged sentences split 81, loaded-term flags 11, political/war share of news 20%, instruction lines varied 823 (max 8 uses) |
| 9 Safety | 150 generated safety rows | 150 train / 20 eval after caps and new prompts; all listed for teacher-answer review |
| 10 Fill & rebalance | 3993 pre-fill rows | 1160 new hand-written prompts (train + eval) |
| 11 Eval | 300 old eval rows | rebuilt: 50 native, knowledge 20 (Gurnah premise fixed), 20 summarisation, 29 rewritten situations, fresh prompts; 10 surplus natives moved to train |
| 12 Dedup / leakage | 4965 train rows | 4961 after removing near-duplicates (1) and rows sharing a line with eval (3) |

## Things a person should look at

- **No `prompts.csv` was available**, so `source` was recovered by reading the rows: Aya/AddisGPT vs. NQ-Open vs. Dolly vs. v2-generated is my judgement (see `data/manual/residual_labels.txt`). Short native-style trivia without an NQ answer tail was labelled `aya_dataset`/`native_chat`; native_chat ends at 500 vs. the ~472 exact matches the spec mentions.
- **English prompts:** only ~300 English rows existed (the other ~100 of the "about 400" were romanised Amharic), so I translated 78 and filled `translated_instruction` with the 82 Amharic Dolly translations already present plus 58 more translated Dolly rows; 47 new English user-text prompts were written so `english_instruction` reaches 200.
- **Totals are tight:** with native_chat at ~500, nearly every other category had to sit at the top of its range to reach 5,000 ± 50; train is 4,961. Any further drops (e.g. after native review) need replacements in categories below their maximum.
- **Language mode rule:** an English prompt with one of the Amharic answer-request sentences is `en`, and ALL-CAPS acronyms, snake_case keys and programming-language names (JSON, CSV, Python, Excel…) are not counted as English words. Without this rule mixed + latin would exceed 9%.
- **One-liner definition:** "about 30% one-liners" is read as single-line prompts of at most 9 words (33.1%); 63% of prompts are single lines of any length because many paired prompts were merged into one paragraph.
- **Instruction variants and merges** in the generated rows were written by me; all changed generated rows are flagged `needs_human_review`. Native speakers should especially check the casual register and the Latin-script spellings.
- **News:** merged words were only split after a closed list of sentence-final verbs; other run-together words remain. 11 news rows with loaded political terms are flagged and listed for teacher-answer review.
- **Kept sensitive native prompts** (political tweets, ethnic/political questions, betting, religion) are flagged and listed in `teacher_answer_review_ids.txt` rather than dropped.
- **Amharic reading-comprehension rows** whose question shares under 20% of its words with the passage were dropped as probable mismatches; some may be fine (spelling variants).
- **Math:** answers in `data/work/math_answers.csv` were computed in Python, including Ethiopian↔Gregorian date and Ethiopian-clock conversions; missing/contradictory problems store the check value or an empty answer and a note.

## Files

- `data/out/pilot_train.jsonl`, `data/out/pilot_eval.jsonl` — final sets (schema in the spec).
- `data/out/removed.csv` — every dropped input row with its reason; `data/out/changed.csv` — every modified row.
- `data/out/review_sample.csv` — 150 rows for native-speaker review (stratified, all eval categories).
- `data/out/teacher_answer_review_ids.txt` — safety and sensitive prompts whose teacher answers must be read.
- `data/work/labelled.parquet`, `data/work/pair_review.csv`, `data/work/math_answers.csv`, `data/work/validation.json`.
