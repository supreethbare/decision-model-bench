#!/bin/zsh
# Download the upstream public datasets into bench/raw/ (not redistributed in this repo).
# Then run: python build_datasets.py
set -e
cd "${0:A:h}"
mkdir -p raw && cd raw

HF=https://huggingface.co/datasets
curl -sL -o bitext.parquet \
  "$HF/bitext/Bitext-customer-support-llm-chatbot-training-dataset/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
curl -sL -o clinc_plus_test.parquet \
  "$HF/clinc/clinc_oos/resolve/refs%2Fconvert%2Fparquet/plus/test/0000.parquet"
curl -sL -o civil_test.parquet \
  "$HF/google/civil_comments/resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet"

BFCL=$HF/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/main
curl -sL -o bfcl_multiple.json "$BFCL/BFCL_v3_multiple.json"
curl -sL -o bfcl_multiple_answers.json "$BFCL/possible_answer/BFCL_v3_multiple.json"

ls -la
