#!/usr/bin/env bash
# Re-run the whole pipeline from data/raw (inputs are never modified).
set -euo pipefail
cd "$(dirname "$0")"
PY=../.venv/bin/python
for step in 01_load_label 02_english 03_pairs check_math 05_format 05b_latin 06_knowledge 07_news \
            09_build_train 10_eval 11_dedup 12_finalize validate 13_report; do
  echo "== $step"
  $PY -W ignore $step.py
done
