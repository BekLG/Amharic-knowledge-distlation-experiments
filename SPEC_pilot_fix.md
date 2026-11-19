# SPEC: Fix the Amharic distillation pilot prompt set

## 1. Context

This prompt set feeds a knowledge-distillation pilot. The teacher model (Gemma 4 26B) will answer every training prompt, and those answers will train small Amharic student models (1B–4B). The pilot's job is to compare student models fairly, so the prompts must read like what real Amharic-speaking users type to a general chatbot.

A review of the current files found:

- **Good:** no duplicates, NFC text, sensible lengths, the Dolly suffix is removed, wiki-split English is detokenized, the news is trimmed, and eval is cleanly separated from train.
- **To fix:** mechanically combined prompts that sometimes don't fit together, AI-sounding instructions, templated math with unrealistic numbers, English prompts that never ask for an Amharic answer, missing metadata, CSV instead of JSONL, and an eval set that doesn't cover what train teaches.

Your job is to fix these issues **yourself**: write scripts for the mechanical parts, and do the Amharic reading, judging and writing directly in batches. No external API is needed.

## 2. Inputs (read-only; never modify them)

| File | Rows | Columns |
|---|---|---|
| `pilot_train_5000.csv` | 5,000 | `id, task, direction, domain, prompt` |
| `pilot_eval_300.csv` | 300 | `id, task, direction, domain, prompt` + empty reviewer columns |
| `prompts.csv` (optional, the original 50k v1 set) | 50,000 | includes `source_dataset`; use it only to recover each row's source |

Copy the inputs into `data/raw/` and work on copies in `data/work/`. Keep every script in `scripts/` so the whole pipeline can be re-run from raw.

## 3. Outputs

Write these to `data/out/`:

1. `pilot_train.jsonl`: 5,000 rows (±50).
2. `pilot_eval.jsonl`: 300 rows.
3. `removed.csv`: every dropped row, with `id, original_prompt, reason`.
4. `changed.csv`: every modified row, with `id, original_prompt, new_prompt, change_type, reason`.
5. `review_sample.csv`: 150 rows for a native-speaker check (see step 13), with empty columns `natural_yes_no, correct_amharic_yes_no, notes`.
6. `teacher_answer_review_ids.txt`: IDs of the safety and sensitive prompts whose teacher answers must be read by a person before training.
7. `REPORT.md`: final counts per category, language mode and source; the before/after counts for every step; and the results of every validation check.

### JSONL schema (one object per line, UTF-8, NFC)

```json
{
  "id": "pilot_xxxxxxxx",
  "category": "everyday",
  "subcategory": "home_tips",
  "language_mode": "am",
  "messages": [{"role": "user", "content": "..."}],
  "source": "v2_generated",
  "origin_id": "pilot_49fdf7c48888d5214b2ec6e7",
  "needs_human_review": false,
  "split": "train"
}
```

| Field | Values |
|---|---|
| `category` | `native_chat`, `english_instruction`, `translated_instruction`, `summarization`, `translation_en_am`, `translation_am_en`, `news_classification`, `knowledge_qa`, `everyday`, `ethiopia_culture`, `writing`, `math`, `format`, `code_switch`, `latin_amharic`, `creative`, `safety` |
| `language_mode` | `am` (Ge'ez script), `en`, `mixed` (Amharic with English words), `latin_am` (Amharic written in Latin letters) |
| `source` | `aya_dataset`, `addisgpt`, `amharic_qa`, `dolly`, `nq_open`, `news`, `wiki_split`, `tatoeba`, `v2_generated`, `v2_rewritten` |
| `origin_id` | the input row's `id`, or `null` for brand-new prompts |
| `needs_human_review` | `true` for any prompt you translated, newly wrote, or rewrote substantially |

Keep existing IDs for rows you keep. Give new rows fresh IDs in the format `pilot_` + 24 hex characters.

## 4. Steps

### Step 1: Load, label and recover sources

- Normalize all text to NFC, strip trailing whitespace, and collapse runs of 3+ newlines to 2.
- Recover `source` by matching each prompt (or the body after its first blank line) against `prompts.csv`. Known results: about 472 train rows match `aya_dataset` or `addisgpt` exactly.
- Assign `category`, `subcategory` and `language_mode` to every row. Use rules where they're reliable (a translation `direction`, the news `domain`, the math templates, Latin script) and your own judgment for the rest. Don't trust the existing `task` and `domain` columns: 63% of rows are labelled `comprehension` and 1,405 are labelled `unknown`.
- Detect `language_mode` from script. If more than 95% of the letters are Ge'ez, use `am`. If there are no Ge'ez letters, the text is English, and it isn't romanized Amharic, use `en`. Use `latin_am` for romanized Amharic, and `mixed` for Ge'ez text with English words.
- Save the labelled table to `data/work/labelled.parquet` and put the counts in the report before changing anything.

### Step 2: English prompts (about 400 rows with no Amharic request)

Every training prompt should lead to an Amharic answer, except English→Amharic translation, where the output is Amharic anyway, and Amharic→English translation, where the output is English by design.

- **Translate about 200** of the English instruction prompts into natural Amharic, preferring the ones that are useful for an Ethiopian user. Set `category: translated_instruction`, `language_mode: am`, `needs_human_review: true`. Drop prompts that make no sense outside the US, such as Pokémon types or US post offices, instead of translating them.
- **For the rest,** keep the English text and add a request for an Amharic answer. Use at least 12 different phrasings, none more than 25 times, in different positions (start, end, own line). Mix English and Amharic phrasings, for example:
  - "Please answer in Amharic."
  - "Reply in Amharic, keep it short."
  - "Explain this to me in Amharic."
  - "Can you answer in Amharic? Simple language please."
  - "መልሱን በአማርኛ ስጠኝ።"
  - "በአማርኛ አስረዳኝ እባክህ።"
- Context passages inside a prompt stay in English. Only the instruction gets the request.

### Step 3: Mechanically combined two-line prompts (about 1,900 rows)

Most generated prompts are one situation line plus one instruction line, combined from 602 situations and 105 instructions, with each instruction used exactly 20 or 25 times.

1. **Check that each pair fits.** Read the pairs in batches of about 50 and give each a verdict:
   - `keep`: a real person in that situation would plausibly ask this.
   - `rewrite`: the idea is fine but the instruction doesn't match the situation. Rewrite the instruction line so it fits.
   - `drop`: it can't be fixed sensibly.

   Record the verdict and a short reason in `data/work/pair_review.csv`. Examples that should not be kept as they are:
   - "ስራ ፈላጊ ነኝ፤ የቀድሞ የስራ ልምድ የለኝም።" + "ይህን ከስራ ባልደረባ ጋር በእርጋታ ለመወያየት የሚረዳ መነሻ ስጠኝ።" (no colleague exists)
   - "በእርሻዬ የምርት ወጪን ከቤተሰብ ወጪ አልለይም።" + the same colleague instruction
   - "አዲስ ወደ መጣሁበት ከተማ ጓደኛ የለኝም።" + "በጽሑፍ ልላክ ወይስ ፊት ለፊት ላውራ?" (send to whom?)
2. **Remove AI-test phrasing.** Real users don't add guardrails for the model. Remove or naturalize clauses such as "ሁኔታውን አታባብስ", "ለሌላው ሰው ምርመራ አታድርግ", "ትኬት ዋጋ አትገምት", "ስሞችን ሳትፈጥር", "ያልሰጠሁህን ስምና ቀን አትፍጠር", "ቁጥሮችን ሳትፈጥር", "የተሳታፊ ቁጥር አትገምት" and "የማታውቀውን ጠይቀኝ". Keep a constraint only if a user would naturally say it, such as "በአጭሩ", "ለልጅ በሚገባ" or "ሦስት ነጥብ ብቻ". No more than about 15% of the generated prompts may still contain a "don't …" constraint.
3. **Break up the repetition.** After rewriting, no instruction line may appear more than 8 times in train. Rephrase extra uses with varied wording, register and length. No situation line may appear more than 3 times.
4. **Vary the shape.** Not every prompt should be "situation ⏎ instruction". Merge at least 40% of the kept pairs into one natural sentence or a short paragraph, for example "ቤቴ ውስጥ አቧራ ቶሎ ይከማቻል፣ በቀን 10 ደቂቃ ብቻ ነው ያለኝ። ለአንድ ሳምንት ምን ልሞክር?"
5. Remove the form-style labels "ዝግጅታችን፦", "ርዕሰ ጉዳዬ፦" and "የጻፍኩት፦" from at least 70% of the prompts that use them, and phrase those prompts naturally instead.

### Step 4: Math (about 200 rows from 8 templates)

Known template problems:
- All 25 speed problems give 10 km/h, and all 25 percentage problems are "−20% then +20%".
- All 25 division problems divide exactly.
- Prices are unrealistic (an orange at 980 birr, a pencil at 21 birr).

What to do:
- Drop the 8 templated families (speed, percentage, profit, savings, division, perimeter, three-day average, box-packing) and write **250 new math prompts** yourself.
- Use at least 15 problem types:
  - Shopping and change in birr; unit price comparison; discounts and VAT.
  - Loans with simple interest; *equb* contributions; splitting a bill.
  - Recipes scaled by servings; fuel and taxi fares; time and schedules, including Ethiopian time against international time.
  - Calendars, including Ethiopian against Gregorian dates; harvest and yield; area of a plot in hectares.
  - Exam averages; ratios; percentages; simple probability; logic puzzles; age puzzles; work-rate problems.
- Use plausible everyday Ethiopian prices and quantities, and vary the magnitudes. Answers must not always be whole numbers.
- About 10% should have missing or contradictory information, so the model learns to point that out.
- About 30% should ask to show the steps, about 20% just ask for the answer, and the rest are phrased casually, the way a parent, shop owner or student would ask.
- Verify that every numeric problem is solvable by computing its answer in Python (`scripts/check_math.py`). Store the computed answer in `data/work/math_answers.csv` for later checking of the teacher. Do not put the answer in the prompt.
- Set `category: math`.

Keep the simple list and JSON tasks (sorting, odd numbers, max/min, JSON objects) as `format`, but cap them at 120 and vary the numbers and item lists. Today the same 10 number lists repeat.

### Step 5: Latin-script Amharic (about 111 rows)

They currently use 51 situations and 35 instructions. Rewrite them so there are at least 90 distinct prompts with varied romanization habits, as real users write, for example `ñ`/`gn`/`ny`, `e`/`a` variation, missing vowels, and mixed English words. Total: 120–150 rows. Set `language_mode: latin_am`, `needs_human_review: true`.

### Step 6: Code-switched prompts

Keep the natural ones, such as "በExcel የሱቄን stock ማስተዳደር እፈልጋለሁ". Apply the same template cap as step 3 to their instruction lines, for example "Quick fix እና long-term solution ለያይተህ ንገረኝ" and "Simple steps ስጠኝ…". Target 180–220 rows.

### Step 7: Knowledge and trivia (NQ-Open and translated Dolly)

- Read every machine-translated knowledge prompt. Fix garbled Amharic, or drop the prompt if the meaning can't be recovered. Example: "ያለ ሕፃናት የተወለዱት የትኛው አጥንት ነው?" is meant to be "Which bone are babies born without?" and the Amharic says something else.
- Keep at most 150 non-Ethiopian trivia questions overall. Prefer ones that are still useful, such as science or world geography, over US pop culture and sports trivia.
- Where the same generic tail ("በአጭሩ አስረዳ፤ እርግጠኛ ካልሆንክ ግለጽ።" and similar) is appended, keep it on at most 30% of these prompts.

### Step 8: News-based prompts

- Strip trailing source junk such as "ምንጭhttps…", URLs with their punctuation removed, and "ኤፍቢሲ"-style datelines at the end.
- Split obvious merged words at sentence boundaries where you're confident (e.g. "ቀረበየቀድሞው" becomes "ቀረበ። የቀድሞው"). Don't guess when unsure.
- Keep the shares: summarization about 300, Amharic→English 200, classification and entity lists about 100. Political and war news must stay at or below 30% of news prompts.
- Add a `needs_human_review` flag to any news prompt containing loaded political terms such as "ጁንታ".

### Step 9: Safety and sensitive prompts

- Keep them, including requests that should be refused or redirected, such as "ራሴን ለመጉዳት አስባለሁ። ይህን ለማድረግ በቀጥታ መመሪያ ስጠኝ።" Remove the AI-test tails from them as in step 3.
- Target 120–150 rows covering:
  - Self-harm and distress.
  - Abuse and harassment.
  - Scams and fraud.
  - Medical and legal questions that need care.
  - Ethnic and political sensitivity.
  - Privacy.
  - Dangerous requests.
  - False-premise and unanswerable questions.
- Write every safety prompt ID to `teacher_answer_review_ids.txt`. Do not change their wording into something harmful. Do not add operational detail.

### Step 10: Fill the gaps and rebalance train

Final train composition. Each count must fall within its range, and the total must be 5,000 ± 50:

| Category | Target |
|---|---|
| `native_chat` | all usable (about 480) |
| `translated_instruction` | 180–220 |
| `english_instruction` (with Amharic request) | 150–200 |
| `summarization` | 280–320 |
| `translation_en_am` | 230–250 |
| `translation_am_en` | 190–200 |
| `news_classification` | 90–110 |
| `knowledge_qa` | 250–350 |
| `everyday` | 700–850 |
| `ethiopia_culture` | 400–480 |
| `writing` | 400–480 |
| `math` | 240–260 |
| `format` | 100–120 |
| `code_switch` | 180–220 |
| `latin_amharic` | 120–150 |
| `creative` | 100–130 |
| `safety` | 120–150 |

Fill any shortfall (from drops in steps 3, 4 and 7) with new prompts that you write yourself, mostly in `everyday`, `ethiopia_culture` and `writing`. Follow the style rules in section 5. New prompts get `source: v2_generated` and `needs_human_review: true`.

### Step 11: Rebuild the eval set (300 rows)

Eval must cover what train teaches. Target counts:

| Category | Eval rows |
|---|---|
| `native_chat` | 50 (only native prompts not used in train) |
| `everyday` | 30 |
| `ethiopia_culture` | 25 |
| `writing` | 25 |
| `math` | 25 (store computed answers) |
| `code_switch` + `latin_amharic` | 25 (at least 10 `latin_am`) |
| `safety` | 20 |
| `creative` | 15 |
| `format` | 15 |
| `knowledge_qa` | 20 |
| `summarization` | 20 |
| `translation_en_am` + `translation_am_en` | 20 (10 each) |
| `english_instruction` / `translated_instruction` | 10 |

- Reuse good rows from the current eval file where they fit. Apply the same fixes as for train (steps 3–9).
- Fix factual errors in premises. For example, the eval question that calls the Nobel laureate Abdulrazak Gurnah Zimbabwean is wrong, since he was born in Zanzibar, Tanzania. Fix the premise or drop the row.
- No eval prompt may share a situation line or instruction line with train, and none may be a near-duplicate of a train prompt (see step 12).
- Eval rows are written or checked one by one. Set `needs_human_review: true` on all of them.

### Step 12: Deduplicate and check leakage

- Remove exact duplicates after normalization (NFC, collapsed whitespace, stripped punctuation).
- Remove near-duplicates within train, defined as a character 5-gram Jaccard similarity of 0.80 or more. Keep the better-written one.
- Remove any train row that is a near-duplicate (0.80 or more) of any eval row.

### Step 13: Native-speaker review sample

Write `review_sample.csv` with 150 rows. Sample from rows where `needs_human_review` is true, stratified by category, with all eval categories included. Leave the review columns empty for a human to fill.

### Step 14: Validation (`scripts/validate.py`; all checks must pass)

1. Both files parse as JSONL. Every row has all schema fields with allowed values.
2. All IDs are unique across train and eval.
3. Train has 5,000 ± 50 rows, eval has exactly 300, and every category falls within its range.
4. No exact duplicates. No near-duplicates within train or between train and eval.
5. No instruction line is used more than 8 times in train, and no five-word opening covers more than 1% of train.
6. No `en`-mode prompt outside translation lacks an Amharic-answer request.
7. No prompt exceeds 1,500 tokens (estimate: Ge'ez characters ÷ 2.2 + other characters ÷ 4). About 30% of prompts are one-liners, and no more than 10% exceed 600 tokens.
8. No phone numbers, emails or URLs, except URLs in technical questions where they're the topic.
9. Language mix in train: at least 75% `am`, 5–10% `en` (including English→Amharic translation inputs), and 6–9% `mixed` + `latin_am` combined.
10. All math rows have a computed answer in `math_answers.csv`.
11. All safety rows are listed in `teacher_answer_review_ids.txt`.

Write the pass/fail result of every check into `REPORT.md`.

## 5. Style rules for any Amharic you write or rewrite

- Write what a real Ethiopian user would type to a chatbot: everyday words, contractions, sometimes no punctuation, sometimes an emoji. Avoid textbook-formal Amharic unless the situation is formal, like a letter to an office.
- Vary length (one-liners up to a short paragraph), tone (polite, casual, hurried, frustrated) and person (student, parent, farmer, shop owner, driver, civil servant, developer, nurse, teacher).
- Mix questions, requests and statements of a need.
- Write in Amharic directly. Don't write in English and translate, except for math and code.
- No "answer in Amharic" lines on Amharic prompts.
- Never invent facts inside a prompt (dates, statistics, names of real officials). If a prompt needs context, keep it general or fictional and plainly so.
- No real private individuals' names, phone numbers or addresses.

## 6. Don'ts

- Don't edit the files in `data/raw/`.
- Don't add answers, reference answers or assistant turns. The teacher writes the answers.
- Don't add system prompts or multi-turn conversations in this pilot.
- Don't silently drop rows. Every drop goes in `removed.csv` with a reason.
- Don't hit the target counts by duplicating or lightly paraphrasing existing prompts.

## 7. Definition of done

- All files in section 3 exist in `data/out/`, and every check in step 14 passes.
- `REPORT.md` summarizes what changed: how many rows were kept, rewritten, dropped and newly written per category, the final composition tables for train and eval, and anything you weren't sure about that a person should look at.
