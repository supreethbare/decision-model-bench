| Task | Model | n | Accuracy | Macro-F1 | Auto @95% | ECE | Brier | Declined | p50 ms | p95 ms | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| triage | jev | 198 | 93.4% | 0.932 | 98.0% | 0.037 | - | 0 | 152 | 204 |  |
| triage | openjev | 198 | 88.4% | 0.888 | 88.4% | 0.263 | - | 0 | 1590 | 1852 |  |
| triage | needle | 198 | 28.3% | 0.375 | 1.0% | 0.180 | - | 108 | 30 | 50 |  |
| intent_oos | jev | 200 | 91.0% | 0.922 | 62.5% | 0.073 | - | 0 | 151 | 210 | out-of-scope recall 0.84, in-scope acc 0.93 |
| intent_oos | openjev | 200 | 74.5% | 0.739 | 43.5% | 0.250 | - | 0 | 2642 | 2848 | out-of-scope recall 0.82, in-scope acc 0.72 |
| intent_oos | needle | 200 | 20.0% | 0.266 | 0.0% | 0.543 | - | 56 | 37 | 91 | out-of-scope recall 0.00, in-scope acc 0.27; acc 0.38 if declines count as out_of_scope |
| toxicity | jev | 200 | 78.5% | 0.792 | 20.5% | 0.069 | 0.149 | 0 | 151 | 202 | toxic precision 0.77, recall 0.82 |
| toxicity | openjev | 200 | 58.5% | 0.325 | 1.5% | 0.208 | 0.273 | 0 | 465 | 1120 | toxic precision 0.87, recall 0.20 |
| toxicity | needle | 200 | 32.0% | 0.544 | 0.0% | 0.359 | 0.376 | 76 | 48 | 114 | toxic precision 0.56, recall 0.53 |
| tool_select | jev | 200 | 99.5% | - | 100.0% | 0.006 | - | 0 | 148 | 217 |  |
| tool_select | openjev | 200 | 99.5% | - | 100.0% | 0.076 | - | 0 | 1076 | 1768 |  |
| tool_select | needle | 200 | 95.0% | - | 100.0% | 0.078 | - | 0 | 164 | 277 | function + arguments correct 0.61 |
