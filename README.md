# Amharic knowledge distillation experiments

Distilling Gemma 4 26B A4B-it into a small (1B–4B) Amharic chat model via response-level distillation: build Amharic prompts, have the teacher answer them, filter and review the answers, then fine-tune and compare student models.

- `data_prep/` builds and validates the prompt sets (`bash data_prep/run_all.sh` rebuilds `data/prompts/` from `data/raw/`).
- `teacher/` generates, filters and merges teacher answers (see `docs/RUN_teacher_local.md`).
- `data/prompts/` holds the pilot set: 4,961 train and 300 eval prompts. See `data/prompts/REPORT.md`.
- `experiments/LOG.md` records every run.

Project context, decisions and working rules are in [CLAUDE.md](CLAUDE.md).
