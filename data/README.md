# data/

| Folder | Contents | Committed? |
|---|---|---|
| `raw/` | Pilot inputs from the team member (`pilot_train_5000.csv`, `pilot_eval_300.csv`). Read-only. | Yes |
| `work/` | Intermediate files written by `data_prep/` (rebuilt by `run_all.sh`). | Yes |
| `prompts/` | Final prompt sets and their logs: `pilot_train.jsonl`, `pilot_eval.jsonl`, `math_answers.csv`, `teacher_answer_review_ids.txt`, `removed.csv`, `changed.csv`, `review_sample.csv`, `REPORT.md`. Never edit by hand. | Yes |
| `teacher/` | Teacher outputs (`teacher_outputs.jsonl`), smoke runs, error logs. | Yes, while under ~50 MB |
| `sft/` | Filter output: `sft_train.jsonl`, `eval_teacher.jsonl`, `review_queue.csv`, `rejected.jsonl`, `filter_report.md`. | Yes |
| `review/` | Filled native-speaker review sheets. | Yes |

Files over about 50 MB go to a private Hugging Face repo or Git LFS instead; record the link in `experiments/LOG.md`.

The sources in the pilot have not all had their licences checked, so keep this repo private.
