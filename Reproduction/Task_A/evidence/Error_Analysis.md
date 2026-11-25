# Holdout error analysis

Examples below are selected from per-video predictions. Label metadata and scores support the observations; no causal attribution is made from visual content.

## Majority + median onset

Clip false positives **50**, clip false negatives **0**, and detected positives outside the timing window **12**.

| Failure | Video | Label subtype | Lighting | Reference onset (s) | Predicted onset (s) | Score |
|---|---|---|---|---:|---:|---:|
| False flag | vid0005 | normal | high | — | 7.0 | 1.000 |
| False flag | vid0018 | normal | high | — | 7.0 | 1.000 |
| False flag | vid0021 | normal | high | — | 7.0 | 1.000 |
| False flag | vid0038 | normal | high | — | 7.0 | 1.000 |
| False flag | vid0040 | normal | high | — | 7.0 | 1.000 |
| Out-of-window | vid0075 | static dumping | high | 16.0 | 7.0 | 1.000 |
| Out-of-window | vid0085 | dynamic dumping | high | 16.0 | 7.0 | 1.000 |
| Out-of-window | vid0101 | static dumping | high | 13.0 | 7.0 | 1.000 |
| Out-of-window | vid0115 | static dumping | high | 15.0 | 7.0 | 1.000 |
| Out-of-window | vid0166 | dynamic dumping | high | 15.0 | 7.0 | 1.000 |

## Motion SVM

Clip false positives **31**, clip false negatives **7**, and detected positives outside the timing window **16**.

| Failure | Video | Label subtype | Lighting | Reference onset (s) | Predicted onset (s) | Score |
|---|---|---|---|---:|---:|---:|
| False flag | vid0005 | normal | high | — | 14.75 | 0.698 |
| False flag | vid0021 | normal | high | — | 5.5 | 0.629 |
| False flag | vid0038 | normal | high | — | 21.25 | 0.904 |
| False flag | vid0040 | normal | high | — | 1.0 | 0.985 |
| False flag | vid0050 | normal | high | — | 5.0 | 0.963 |
| Missed event | vid0069 | dynamic dumping | high | 3.0 | — | -0.045 |
| Missed event | vid0088 | static dumping | high | 6.0 | — | -0.469 |
| Missed event | vid0199 | dynamic dumping | high | 4.0 | — | -0.907 |
| Missed event | vid0297 | dynamic dumping | high | 34.0 | — | -0.834 |
| Missed event | vid0328 | static dumping | high | 9.0 | — | -0.188 |
| Out-of-window | vid0023 | dynamic dumping | high | 2.0 | 12.75 | 0.834 |
| Out-of-window | vid0075 | static dumping | high | 16.0 | 1.25 | 0.012 |
| Out-of-window | vid0085 | dynamic dumping | high | 16.0 | 1.0 | 0.969 |
| Out-of-window | vid0086 | static dumping | low | 6.0 | 0.0 | 0.963 |
| Out-of-window | vid0101 | static dumping | high | 13.0 | 1.0 | 0.770 |

## ResNet-18 + GRU

Clip false positives **11**, clip false negatives **14**, and detected positives outside the timing window **7**.

| Failure | Video | Label subtype | Lighting | Reference onset (s) | Predicted onset (s) | Score |
|---|---|---|---|---:|---:|---:|
| False flag | vid0018 | normal | high | — | 26.0 | 0.364 |
| False flag | vid0021 | normal | high | — | 2.0 | 0.311 |
| False flag | vid0063 | normal | low | — | 11.75 | 0.007 |
| False flag | vid0094 | normal | high | — | 9.0 | 0.717 |
| False flag | vid0224 | normal | high | — | 6.75 | 1.808 |
| Missed event | vid0086 | static dumping | low | 6.0 | — | -0.042 |
| Missed event | vid0127 | static dumping | high | 9.0 | — | -0.139 |
| Missed event | vid0163 | dynamic dumping | high | 3.0 | — | -1.045 |
| Missed event | vid0192 | static dumping | high | 11.0 | — | -0.112 |
| Missed event | vid0215 | dynamic dumping | high | 25.0 | — | -0.007 |
| Out-of-window | vid0012 | dynamic dumping | high | 2.0 | 13.75 | 1.829 |
| Out-of-window | vid0023 | dynamic dumping | high | 2.0 | 13.0 | 2.991 |
| Out-of-window | vid0075 | static dumping | high | 16.0 | 9.5 | 1.904 |
| Out-of-window | vid0085 | dynamic dumping | high | 16.0 | 53.0 | 1.691 |
| Out-of-window | vid0180 | static dumping | high | 2.0 | 15.5 | 1.452 |

The complete error rows and the candidate scene-overlap checks are stored alongside the predictions.
