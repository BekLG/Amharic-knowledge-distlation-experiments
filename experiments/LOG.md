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
