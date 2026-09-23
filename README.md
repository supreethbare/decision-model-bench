# decision-model-bench

Benchmark harness comparing **typed decision models** on four enterprise tasks, with
per-case predictions committed as evidence.

A typed decision model doesn't generate text: you give it state and a typed question
(pick one of N, a yes/no probability, a rubric score) and it returns that type with a
calibrated confidence. This repo asks a practical question about them - *how much traffic
can you route automatically, and can you trust the confidence score you'd threshold on?* -
and answers it with the same ~200 cases per task sent to every model.

The models are compared as a **decision layer in front of an LLM**, not as a replacement
for one: the confidence score decides what is handled automatically, what needs an LLM, and
what needs a human.

| Model | What it is | Ran on |
|---|---|---|
| **Jev** 1.13.0 | TypeSafe's proprietary "System One" decision model | cloud API |
| **Open-Jev-2B** | open reproduction of the idea (Qwen3.5-2B + LoRA + decision head) | MacBook Air M4 24 GB, Apple GPU |
| **Needle 3** 3.0.4 | 121M-param on-device tool-calling / extraction model | same MacBook, CPU |

## Results

798 cases per model, one run, September 2026. Zero request errors.

**Accuracy** (declined answers count as wrong)

| Task | Jev | Open-Jev-2B | Needle 3 |
|---|---|---|---|
| Ticket triage (11 categories) | **93.4%** | 88.4% | 28.3% |
| Intent + out-of-scope (15 + 1) | **91.0%** | 74.5% | 20.0% |
| Toxicity guardrail (yes/no) | **78.5%** | 58.5% | 32.0% |
| Tool selection (2-4 functions) | **99.5%** | **99.5%** | 95.0% |

**Share handled with no human and no LLM**, i.e. answers above the confidence threshold
that keeps accuracy at 95%

| Task | Jev | Open-Jev-2B | Needle 3 |
|---|---|---|---|
| Ticket triage | **98.0%** | 88.4% | 1.0% |
| Intent + out-of-scope | **62.5%** | 43.5% | 0.0% |
| Toxicity guardrail | **20.5%** | 1.5% | 0.0% |
| Tool selection | **100%** | **100%** | **100%** |

**Calibration error** (ECE, 10 bins - how far stated confidence sits from actual accuracy;
lower is better): Jev 0.006-0.073 · Open-Jev 0.076-0.263 · Needle 0.078-0.543.

**Median latency**: Jev ~150 ms (includes the network round trip) · Needle 30-164 ms (CPU)
· Open-Jev 465-2642 ms (Apple GPU, without the CUDA kernels it uses on NVIDIA hardware;
its authors measure ~85 ms on an H100).

Full table with macro-F1, p95 latency and per-task notes: [`bench/results/summary.md`](bench/results/summary.md)
and [`summary.csv`](bench/results/summary.csv). Open
[`bench/results/comparison.html`](bench/results/comparison.html) locally for the
interactive chart.

### What we read from this

- **Jev leads on every task**, and by the widest margin where the decision needs judgement.
  It is also the only model whose confidence we would threshold on without per-task tuning.
- **Open-Jev-2B is a credible self-hosted option for routing-shaped work** (triage, tool
  selection) and much weaker on judgement tasks. Its confidence needs calibrating per task.
- **Needle 3 is not a classifier and we don't claim it should be.** On the task it is built
  for - picking a function - it scored 95%, and it filled the arguments correctly 61% of the
  time. On classification it declined 28-55% of cases while reporting 0.9-1.0 confidence.
- **Toxicity is hard for all three.** Even the best model automates only 20.5% of cases at
  95% precision. That task needs a second opinion, not a threshold.

### Caveats - read before quoting these numbers

- **One run, ~200 cases per task.** Differences of 2-3 points are noise. The large gaps
  aren't.
- **Public data, not production data.** Real enterprise tickets are messier and
  domain-specific. These numbers show relative strength, not what you'd get in production.
- **Latency is not like-for-like.** Cloud API vs laptop GPU vs laptop CPU, measured from one
  client on one machine.
- **No prompt tuning.** Every model got the same short instructions and category
  descriptions. Better descriptions would likely lift all three.
- **Needle was used outside its design.** Each classification was posed to it as a tool to
  call, the closest fit its interface allows.

## Reproducing

```sh
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./bench/fetch_raw.sh                 # download the public datasets into bench/raw/
python bench/build_datasets.py       # rebuild bench/data/ (seed 42; should match what's committed)
```

Then run the models. Each writes one JSON line per case to `bench/results/<model>/<task>.jsonl`;
re-running resumes and retries failed cases.

```sh
# Jev - needs a TypeSafe API key
export TYPESAFE_API_KEY=...          # prompts if unset
python bench/run.py --model jev

# Needle 3 - local, downloads ~35 MB of weights on first use
python bench/run.py --model needle

# Open-Jev-2B - clone it, then start its server (see below) and:
python bench/run.py --model openjev
```

Score and chart:

```sh
python bench/report.py               # -> results/summary.csv + summary.md
python bench/make_chart.py           # -> results/comparison.html
python bench/make_blog_charts.py     # -> results/results.png
```

`bench/run_local.sh` runs Needle and Open-Jev back to back (never in parallel, so latencies
aren't skewed) and starts and stops the Open-Jev server itself.

### Open-Jev setup

```sh
git clone https://github.com/Zefan-Cai/Open-Jev.git && cd Open-Jev
python -m venv .venv && source .venv/bin/activate && pip install -e '.[train]'
hf download ZefanCai/Open-Jev-2B \
  --revision 0c7aa498b1627be8da4acf34c863ff0ee0a92785 --local-dir models/Open-Jev-2B
python -m jev.server --checkpoint models/Open-Jev-2B/package/checkpoint \
  --device mps --max-length 4096 --batch-size 16 --no-prefix-cache   # --device cuda:0 on NVIDIA
```

It downloads the pinned Qwen3.5-2B base weights (~4.6 GB) on first run. Set `OPENJEV_DIR` if
your checkout isn't at `../Open-Jev`, and `OPENJEV_ENDPOINT` if the server isn't on
`127.0.0.1:8791`.

## Using your own data

The harness doesn't care where cases come from. One JSON object per line:

```json
{"id": "ticket_1",
 "task": "triage",
 "state": {"customer_message": "I was charged twice for September"},
 "questions": {"route": {"type": "choice",
                         "instructions": "Which team should own this ticket?",
                         "criteria": {"billing": "Charges, refunds, invoices.",
                                      "technical": "Bugs, outages, API problems."}}},
 "gold": {"route": "billing"}}
```

`type` is `choice`, `noul` (yes/no) or `score` (rubric). Drop the file in `bench/data/`, add
its name to `TASKS` in `run.py` and `report.py`, and every model adapter handles it
unchanged - each one translates the same typed question into its own interface.

## Layout

```
bench/build_datasets.py   samples the four test sets from the public data
bench/models.py           one adapter per model (Jev / Open-Jev / Needle)
bench/run.py              runs a model over the tasks, resumable
bench/report.py           scores everything -> summary.csv / summary.md
bench/make_chart.py       interactive HTML comparison
bench/data/               the sampled test sets (see DATA.md for licences)
bench/results/            per-case predictions + the reports
examples/                 three minimal single-ticket scripts, one per model
```

## Metric definitions

- **Accuracy** - exact match against the gold label. A declined answer counts as wrong.
- **Macro-F1** - F1 averaged over classes, so rare categories count as much as common ones.
  Not reported for tool selection, where each case has its own function set.
- **Auto @95%** - rank answers by confidence, take from the top, stop where accuracy on the
  taken set would fall below 95%; report that share of *all* cases.
- **ECE** - expected calibration error over 10 confidence bins, using the model's
  confidence in the answer it chose.
- **Brier** - for yes/no tasks, mean squared error of the probability.

## Licence and attribution

Code is MIT (see [LICENSE](LICENSE)). The sampled test sets keep their upstream licences -
CDLA-Sharing-1.0, CC BY 3.0, CC0-1.0 and Apache-2.0 - with terms and citations in
[DATA.md](DATA.md). No model weights are redistributed here.

This is an independent evaluation. It is not affiliated with or endorsed by TypeSafe,
Cactus Compute, or the Open-Jev authors.
