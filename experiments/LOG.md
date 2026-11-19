# Experiment log

One entry per run: what was run, the exact command, config (seed, model versions, provider), result, and the decision taken. Newest first. Open questions go here too.

## 2026-11-19 — Repo restructure; pilot prompt set rebuilt

- **What:** moved the pilot-fix pipeline to `data_prep/` and its outputs to `data/prompts/`, then rebuilt from raw to check nothing changed.
- **Command:** `bash data_prep/run_all.sh` (Python from `.venv`)
- **Result:** all validation checks pass; `pilot_train.jsonl` (4,961), `pilot_eval.jsonl` (300), `removed.csv`, `changed.csv`, `review_sample.csv` and `teacher_answer_review_ids.txt` are byte-identical to the previous `data/out/`. Only path strings in `REPORT.md` changed.
- **Decision:** `data/prompts/` from this build is the pilot v1 prompt set.
