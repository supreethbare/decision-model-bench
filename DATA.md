# Data sources, licences and attribution

The four test sets in `bench/data/` are **samples drawn from public datasets**, built by
`bench/build_datasets.py` with a fixed seed (42). Each source keeps its own licence, listed
below. The upstream files themselves are not redistributed here - `bench/fetch_raw.sh`
downloads them.

| Test set | Source dataset | Licence | Sample |
|---|---|---|---|
| `triage.jsonl` | [bitext/Bitext-customer-support-llm-chatbot-training-dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) | CDLA-Sharing-1.0 | 18 per category × 11 categories = 198 |
| `intent_oos.jsonl` | [clinc/clinc_oos](https://huggingface.co/datasets/clinc/clinc_oos) (config `plus`, test split) | CC BY 3.0 | 150 in-scope + 25 out-of-scope + 25 near-domain = 200 |
| `toxicity.jsonl` | [google/civil_comments](https://huggingface.co/datasets/google/civil_comments) (test split) | CC0-1.0 | 100 toxic + 100 non-toxic = 200 |
| `tool_select.jsonl` | [gorilla-llm/Berkeley-Function-Calling-Leaderboard](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard) (`BFCL_v3_multiple`) | Apache-2.0 | all 200 |

## Terms that travel with the data

- **Bitext (CDLA-Sharing-1.0).** The rows sampled into `bench/data/triage.jsonl`, and any
  further derivative you publish of them, stay under CDLA-Sharing-1.0. The repo's MIT
  licence covers the code, not this data.
- **CLINC150 (CC BY 3.0).** Attribution required: Larson et al., *An Evaluation Dataset for
  Intent Classification and Out-of-Scope Prediction*, EMNLP 2019.
- **Civil Comments (CC0-1.0).** Public domain dedication. Note the content: this is a
  toxicity dataset, so `bench/data/toxicity.jsonl` contains real comments that are
  offensive by design. The `annotator_toxicity` field carries the original crowd score.
- **BFCL (Apache-2.0).** Attribution: Patil et al., Gorilla / Berkeley Function Calling
  Leaderboard.

## Labels

Labels are the datasets' own, not ours, with two framing decisions made by
`build_datasets.py` and documented here:

1. **Out-of-scope.** We defined "in scope" as 15 CLINC banking intents. 25 cases drawn from
   CLINC's *credit card* intents (`report_lost_card`, `credit_limit`, `card_declined`,
   `apr`, `new_card`) are labelled `out_of_scope` because a banking-only assistant should
   decline them. They are genuinely near-misses, and reasonable people could scope them
   differently.
2. **Toxicity threshold.** Binarised at the standard `toxicity >= 0.5` crowd score.

## Models evaluated

| Model | Source | Licence |
|---|---|---|
| Jev | [TypeSafe](https://typesafe.ai) - proprietary, cloud API | vendor terms |
| Open-Jev-2B | [Zefan-Cai/Open-Jev](https://github.com/Zefan-Cai/Open-Jev) - code MIT, adapters Apache-2.0 over pinned Qwen weights | MIT / Apache-2.0 |
| Needle 3 | [cactus-compute/needle](https://github.com/cactus-compute/needle) | Apache-2.0 |

No model weights are redistributed in this repo.
