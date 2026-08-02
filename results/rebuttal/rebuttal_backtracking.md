## Backtracking & RLHF: controlled comparison within the OLMo 32B family (Spatial-Gym)

Δ Acc. = acc(backtracking) − acc(no backtracking) ("Correctly Solved" %). Backtracking ratio = median(steps_taken / path_edges) (Figure 5 metric; steps < 100, path_edges > 0).

| Model | Tuning | No-BT Acc. (%) | BT Acc. (%) | Δ Acc. (pp) | Median ratio | Mean ratio | N (BT valid)$^a$ |
|---|---|---:|---:|---:|---:|---:|---:|
| OLMo 3 32B Think-SFT | SFT only | 9.4 | 8.6 | -0.8 | 1.29 | 1.91 | 413 |
| OLMo 3.1 32B Think | RLVF | 11.4 | 9.6 | -1.8 | 1.25 | 1.87 | 458 |
| OLMo 3.1 32B Instruct-SFT | SFT only | 2.6 | 4.8 | +2.2 | 2.94 | 3.64 | 311 |
| OLMo 3.1 32B Instruct | RLVF | 8.4 | 5.8 | -2.6 | 1.43 | 1.71 | 466 |

**Notes.**
- $^a$ Backtracking runs cover fewer than 500 puzzles for some variants (Think-SFT 360, Instruct-SFT 436, Instruct 320); accuracies are run-level percentages and the median ratio is over puzzles with a valid path.
- The 7B Instruct models are excluded per the 32B-only scope.

**Observation.** In both pairs the SFT-only (non-RLHF) variant backtracks at least as much as, and is helped more by backtracking than, its RLVF sibling. The effect is clearest for the Instruct pair: the SFT model *gains* from backtracking (+2.2 pp) and backtracks far more (median ratio 2.94), while the RLVF model *loses* accuracy (−1.5 pp) and backtracks much less (1.39). For the Think pair both variants decline, but the RLVF one declines more (−1.8 vs −0.8 pp). This is consistent with the RLHF hypothesis, but with only two pairs that differ in other ways it remains supporting context for a hypothesis rather than a tested causal claim.
