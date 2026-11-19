# Evaluation standard for same-room decisions (all pairs)

Purpose: evaluate a model's ability to "find the panoramas of the same room in a whole batch and split them into rooms". It can be applied to new datasets as is. Chinese version: [EVALUATION_STANDARD_ZH.md](EVALUATION_STANDARD_ZH.md).
Last updated: 2026-10-23 (replaces the earlier "group-member" scheme that counted only grouped panoramas; see the appendix).

## Ground truth

- All indoor panoramas of a house (task) form a "room partition".
- A person puts the panoramas of each room into a room group (at least 2 panoramas); a panorama not in any group is a room of its own.
- Recommended procedure: compute once with the manual groups as GT; then blind-review the pairs that any model judged same room but GT says different room (model source hidden, control pairs with known answers mixed in), correct the GT with the review and recompute. Report the numbers before and after the review, and the agreement on the control pairs.

## Level 1: same-room decision (evaluates the decision itself)

- Scope: **all** unordered panorama pairs within the same house and the same floor (including unknown floor), with no pre-filtering.
- Each pair: the model says "same room" or "different room", compared with GT.
  - Precision = judged same and GT same / judged same
  - Recall = judged same and GT same / GT same
  - F1 = 2PR / (P + R)
- Also report the "hard-negative false-positive rate": the share of pairs directly connected by a tour hotspot but in different GT rooms (mostly adjacent rooms visible through a door) that are judged same room.

## Level 2: clustering (evaluates the split into rooms)

- Take the connected components of the pairs judged same room; each component is a predicted room.
- Clustering (pairs): Precision = pairs in the same predicted room that are GT same room / pairs in the same predicted room; Recall = GT same-room pairs that end up in the same predicted room / GT same-room pairs.
- Rooms: Precision = predicted rooms (≥ 2 panoramas) identical to a GT room / predicted rooms; Recall = GT rooms (≥ 2 panoramas) recovered exactly / GT rooms.

## Reporting rules

1. One row of P / R / F1 and TP, FP, FN per model and operating point (rule or threshold).
2. A model run only on sampled pairs is weighted by 1 / sampling probability and marked as an estimate.
3. When a threshold is chosen on the same data, also report leave-one-house-out cross-validation.
4. List the number of GT groups, panoramas and pairs, the annotation file version, the size of the blind review and the agreement on the control pairs.

## Implementation (this dataset; scripts of the same_room_benchmark project, not in this package)

- Level 1: `scripts/pair_decision.py` → `work/pair_decision_adj.json`
- Level 2: `scripts/cluster_metrics.py` → `work/cluster_metrics_adj.json`
- Blind review: `scripts/build_adjudication.py`, `scripts/adjudicate.py`
- Calibration table of the adopted classifier: `scripts/wedge_calibration.py` → `work/wedge_calibration_adj.json`

## Appendix: group-member scheme (retired)

Counted only pairs where both panoramas belong to a manual group. It drops most negatives (only 667 pairs left in this dataset, versus about 7,500 in the whole batch), especially adjacent rooms seen through a door, and badly overestimates precision (e.g. floor ≥ 750 goes from 0.27 to 0.80). Code and results are kept in `scripts/group_standard.py` and `work/group_standard_*.json` for comparison only.
