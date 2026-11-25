# Error analysis

The crops show the deposited view on the left and the inspection on the right. The analysis covers every unique forced-error pair from the four methods. Causes are plausible interpretations of the visible crops and recorded scores. No causal feature ablation was run.

## Retrospective comparison

| Pair | Errors | Visible evidence and interpretation |
|---|---|---|
| 04 medium partial | Image Difference falsely cleared | The composition changed around the pole, but a tray and packaging remain. The inspection is wider, changing the scale and position of paving and shadow. Global MAE 0.14782 crossed the 0.14199 threshold. |
| 10 small removed | All four falsely called waste present | Carton and the red container are gone. Wall, tile and a diagonal shadow dominate the inspection. MAE 0.04329 misses a localized material change. The SVM margin was negative and inside its review band. |
| 10 medium partial | Forced SVM falsely cleared | The large black bag is gone, but a bottle and packaging remain visible. The inspection crop shows more of the wall and gate. Small remnants can contribute weakly to an aggregated descriptor. The fixed SVM deferred this pair. |
| 10 medium removed | All four falsely called waste present | The footprint is clear. The wall–floor balance and shadow boundary shift. MAE 0.05548 stayed below the reference threshold. The SVM's negative score was inside the review band. |
| 10 large removed | Image Difference, MLP and majority falsely called waste present | The mixed pile disappears. MAE 0.13593 was close to and below the fixed cutoff. The compact MLP produced a negative logit, as it did on every other pair. The frozen SVM accepted clearance here. |

The majority method makes no image measurement. Matching its decisions does not show that the MLP's weights are constant or identify why it missed clearances. Its globally pooled statistics, position-sensitive differences, limited data and the class mix are plausible constraints. There is no evidence that further tuning on the same pairs would generalize.

## Development comparison

| Pair | Errors | Visible evidence and interpretation |
|---|---|---|
| 02 small removed | Image Difference, MLP and majority falsely called waste present | The staged pile is gone. Sand texture and small background marks remain throughout. The reference score stayed below the cutoff. Background detail remains valid even when deposited waste is removed. |
| 08 small partial | Forced SVM falsely cleared | The small white sack remains on textured ground. The existing error analysis noted that fixed-grid pooling can dilute a small retained object. The fixed SVM deferred. |
| 08 small removed | All four falsely called waste present | The sack is gone. Texture, framing and illumination differ across views. The SVM's negative score was inside the review band. |

## Applicability

The six retrospective waste-present pairs are genuine partial cleanups. Existing rubbish and incidental background marks are separate from deposited objects, and these experiments lack independent estimates for the effects of pre-existing rubbish, moved waste and replacement. A shifted camera or changing shadow can raise or lower a difference score without changing cleanup status. The fixed abstention policy offers a human-review action and does not guarantee safety. With all three clearances from Location 10 and dependent views within sites, the results are descriptive.

[Development error crops](recorded_run/development_error_crops.jpg) · [Retrospective error crops](recorded_run/retrospective_error_crops.jpg) · [Retrospective predictions](recorded_run/retrospective_predictions.csv)
