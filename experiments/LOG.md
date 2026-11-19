# Experiment log

One entry per run, newest first. Each entry should let someone rerun the experiment exactly.

## Template

### YYYY-MM-DD — <short name>
- **Goal:**
- **Command(s):**
- **Inputs:** (dataset file and version or tag, model and version, provider)
- **Settings:** (seed, temperature, caps, LoRA rank, learning rate, epochs, …)
- **Cost and time:**
- **Results:** (numbers, plus a link to the output files)
- **Decision and next step:**
- **Open questions:**

---

### 2026-11-19 — System-prompt fix + train smoke v2 (2 × 40 prompts, DeepInfra)
- **Goal:** check the tightened system prompt (no bracketed English glosses; ዓ.ም never on foreign-event years; translations output only the translation).
- **Command(s):** same as the train smoke with `--out data/teacher/smoke_train_v2_same.jsonl` (the same 40 prompts), then again with `--out data/teacher/smoke_train_v2_next.jsonl` pre-filled with the first 40 rows (so the next 40 prompts ran).
- **Inputs / settings:** as before; system prompt changed in `teacher/gen_teacher.py`.
- **Cost and time:** $0.009, about 1.5 min.
- **Results:**
  - Glosses on the same 40 prompts: 46 in 18 answers before, 12 in 8 answers after. Fresh 40 prompts: 2 in 2 answers. Most of what remains is technical terms.
  - Translation am→en: fixed. All 5 answers are English only, with no Amharic preface and no restated source. Average answer length dropped from 250 to about 200–230 tokens.
  - Calendar: still "ከ1841 እስከ 1846 ዓ.ም" for the British prime-minister question (the filter flags it for review).
  - New: 1 of 80 answers looped ("passed passed …") until the 900 cap (ethiopia_culture, Irreecha). The filter rejects it as looping/truncated.
  - Teacher factual errors the filter can't catch: names Magellan instead of Balboa for the 1513 Pacific sighting; says ልዑል አለማየሁ was taken "በ1932 ዓ.ም" (it was 1860 ዓ.ም / እ.ኤ.አ. 1868); dates a news story to 2024 instead of 2020.
- **Decision and next step:** keep the new system prompt. The full run needs the project lead's go-ahead. Factual accuracy will need the native-speaker spot check; regenerate filter-rejected rows once after the run.
- **Open questions:** a few train prompts write Gregorian years with ዓ.ም themselves (e.g. pilot_8c3a08f3701998f66b12f112, a passage about ልዑል አለማየሁ), which may push the teacher toward the same mistake.

### 2026-11-19 — OpenRouter train smoke test (40 prompts, DeepInfra pinned)
- **Goal:** cover every category on train prompts with one pinned provider.
- **Command(s):** `python teacher/gen_teacher.py --train data/prompts/pilot_train.jsonl --eval data/prompts/pilot_eval.jsonl --out data/teacher/smoke_train.jsonl --openrouter --only train --limit 40 --workers 8 --provider DeepInfra --quantizations bf16,fp8 --log_every 15`, then `filter_teacher.py` on it (output in a scratch dir).
- **Inputs:** first 40 train rows in run order (all 17 categories); `google/gemma-4-26b-a4b-it` on DeepInfra.
- **Settings:** seed 42, temperature 0.6, top_p 0.95, default system prompt, thinking off.
- **Cost and time:** $0.005, about 45 s. That puts the full 5,261-prompt run at roughly $0.65 and 1–1.5 h.
- **Results:** 40/40 `stop`, all on DeepInfra, no thinking, no HTML. Filter: 36 accepted, 3 sent to review (2 safety, 1 calendar), 1 rejected.
  - English glosses in brackets in 18/40 answers, including basic words (e.g. መደመር (Addition), ማባዛት (Multiplication)).
  - Gregorian years labelled ዓ.ም: "ከ1841 እስከ 1846 ዓ.ም" for a British prime-minister question. The calendar flag caught it.
  - Amharic→English translation: in 1 of 2 answers the teacher first rewrote the passage in Amharic, then translated it (428/450 tokens). The filter rejected it as wrong language. The other answer opened with an Amharic preface before the English.
- **Decision and next step:** proposed a system-prompt fix for glosses, Gregorian ዓ.ም and English-only translation output, then another train smoke on different prompts before the full run. Waiting on the project lead.

### 2026-11-19 — OpenRouter smoke test (20 prompts)
- **Goal:** check the OpenRouter path end to end before the full pilot run.
- **Command(s):** `python teacher/gen_teacher.py --train data/prompts/pilot_train.jsonl --eval data/prompts/pilot_eval.jsonl --out data/teacher/smoke.jsonl --openrouter --workers 8 --limit 20 --log_every 10`
- **Inputs:** first 20 rows in run order (all eval `native_chat`); `google/gemma-4-26b-a4b-it`; no provider pinned.
- **Settings:** seed 42, temperature 0.6, top_p 0.95, default system prompt, thinking off.
- **Cost and time:** $0.002, about 40 s.
- **Results:** 20/20 `stop`, none cut off (longest 673/900 tokens), no thinking, no HTML, no ዓ.ም on Gregorian years. `cost_usd` and `provider` recorded. Requests went to 6 different providers. English glosses in brackets in 13/20 answers (26 in total). Some are reasonable (names, exercise names, IAEA), but many follow ordinary Amharic words, against the system prompt. One answer had Ge'ez share 0.67, below the 0.70 filter threshold, because it named English websites.
- **Decision and next step:** pin one provider before the full run. Do a train-only smoke across categories, because this one covered only eval `native_chat`.
- **Open questions:** should the system prompt be tightened further on glosses, or should the filter flag them? Should the Ge'ez threshold allow answers that name English sites or products?

### 2026-11-19 — Repo restructure; pilot prompt set rebuilt
- **Goal:** move the pilot-fix pipeline into `data_prep/` and its outputs into `data/prompts/`, and check the set still reproduces.
- **Command(s):** `bash data_prep/run_all.sh` (Python from `.venv`)
- **Inputs:** `data/raw/pilot_train_5000.csv`, `data/raw/pilot_eval_300.csv`, `data_prep/manual/`
- **Results:** all validation checks pass. `pilot_train.jsonl` (4,961), `pilot_eval.jsonl` (300), `removed.csv`, `changed.csv`, `review_sample.csv` and `teacher_answer_review_ids.txt` are byte-identical to the earlier build; only path strings in `REPORT.md` changed. The OpenRouter teacher scripts ran end to end (generate → filter) against a local mock server.
- **Decision and next step:** this build of `data/prompts/` is the pilot v1 prompt set. Next: OpenRouter smoke test.

### 2026-11-11 — Local teacher smoke test (llama.cpp Vulkan, laptop)
- **Goal:** check the teacher's speed and answer quality on local hardware.
- **Inputs:** 20 eval prompts (all `native_chat`); `unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_M`; 2 parallel slots.
- **Results:** 15.8 tok/s total. 8 of 20 answers were cut off at the 600-token cap. Gregorian years were labelled ዓ.ም. One wrong multiple-choice answer, one made-up answer to a question whose passage was missing, and one stray HTML tag.
- **Decision:** tightened the system prompt, raised the caps, and added HTML and calendar checks to the filter. The full run moves to OpenRouter.
