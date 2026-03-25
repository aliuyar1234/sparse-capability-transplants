# CLAIMS_MATRIX.md

This file is the central anti-overclaiming mechanism for the repo.

Status values:
- **intended** — claim is part of the target contribution but has no supporting evidence yet
- **partially supported** — some evidence exists but required support is incomplete
- **supported** — required evidence exists and has been logged
- **unsupported** — evidence does not support the claim
- **weakened** — only a narrower version should be stated

Evidence must be logged through milestone evidence notes and reflected here.

---

## Claim summary table

| ID | Short name | Type | Initial status |
|---|---|---|---|
| C1 | Same-size existence proof | empirical / method | intended |
| C2 | Cross-scale transplant works | empirical / method | intended |
| C3 | Gains are semantic, not format-only | empirical / robustness | intended |
| C4 | Better data-efficiency than small-data recipient tuning | empirical / efficiency | intended |
| C5 | Narrow transfer with low collateral damage | empirical / robustness | intended |
| C6 | Sparse causal subset is sufficient | empirical / ablation / interpretability | intended |
| C7 | Locked mechanism matters; shortcuts fail | empirical / ablation | intended |
| C8 | Scope-limited contribution only | limitation / guardrail | supported by design |

## Variant activation

- **V24 active claims:** `C1, C3, C5, C6, C7, C8`. In `V24`, `C2` and `C4` are inactive and must not be claimed.
- **V48 active claims:** `C1-C8`, but `C2` and `C4` are only eligible after the same-size gate passes.
- Changing variants requires a `docs/STATUS.md` update plus an evidence note explaining the budget change.

## Current evidence checkpoint

As of `2026-03-25`, the repo has completed the full `V24` same-size experiment path through `M8`: donor-gap establishment, locked rough same-size candidate selection, selected-checkpoint frozen eval, pruning/random-subset controls, shortcut controls, sparse multiseed confirmation, matched dense multiseed comparison, M7 analysis artifacts, and the final M8 claim audit. Base and donor eval artifacts remain complete, and the locked `R20` gate still stands with base `0.0375`, donor `0.178125`, delta `0.140625`, and `95% CI [0.120833, 0.160430]`.

The strongest single-seed sparse same-size checkpoint (`layer 12 / gain 1.25`) reached frozen primary strict `0.2067708333`, donor-gap recovery `1.2037`, and control drop `-0.0140625`. The completed pruning run selected feature `104` as a `1`-feature subset that retained `47.1%` of the full sparse frozen-manifest gain and beat four random `1`-feature controls. The completed dense shortcut control reached frozen primary strict `0.1854166667`, donor-gap recovery `1.0519`, and control drop `-0.0015625`, while the steering shortcut control reached `0.1078125`, donor-gap recovery `0.5`, and control drop `-0.00625`. The sparse multiseed aggregate remained positive across seeds with primary strict mean `0.1657986111`, donor-gap recovery mean `0.9123456790`, and control drop mean `-0.0057291667`. However, the matched dense multiseed aggregate exceeded sparse with primary strict mean `0.2177083333`, donor-gap recovery mean `1.2814814815`, and control drop mean `-0.0109375`.

That means:
- `C1` is now `weakened`
- `C3` is now `weakened`
- `C5` is now `weakened`
- `C6` is now `weakened`
- `C7` is now `weakened`
- `C2` and `C4` remain inactive under `V24`
- `C8` remains `supported`

---

## C1 — Same-size existence proof

- **Exact claim text**  
  A sparse fixed transplant inserted into the same-size untuned base model can recover a nontrivial fraction of the donor model's function-calling advantage on the primary metric.

- **Claim type**  
  Empirical / method

- **Status**  
  weakened

- **Variant scope**  
  V24 and V48

- **Required evidence**  
  1. Donor-base gap established on the primary metric.  
  2. Same-size transplant mean donor-gap recovery reported across seeds with confidence intervals.  
  3. Added-parameter budget reported.  
  4. Same-size transplant beats random-layer, random-feature, and dense parameter-matched controls.

- **Experiments / analyses needed**  
  E1 donor-gap establishment, E4 same-size transplant, E8 control ablations.

- **Implementation dependencies**  
  R1–R10, R13–R24.

- **Risk of overclaiming**  
  High if donor gap is weak, if improvement is only IID, or if random controls are not included.

- **How to weaken if evidence is insufficient**  
  If recovery is positive but small, weaken to:  
  “A same-size sparse transplant can recover a limited but nonzero portion of donor gain on this task.”  
  Drop efficiency/narrowness language unless separately supported.

---

## C2 — Cross-scale transplant works

- **Exact claim text**  
  A stitched sparse transplant can improve the smaller recipient model on the primary metric using gain calibration only, without full recipient retraining.

- **Claim type**  
  Empirical / method

- **Status**  
  intended

- **Variant scope**  
  V48 only

- **Required evidence**  
  1. C1 at least partially supported first.  
  2. Cross-scale transplant improves recipient base on primary metric across seeds.  
  3. Calibration is limited to scalar gains or equally constrained parameters defined in the method spec.  
  4. Comparison against small-data LoRA and random/shortcut controls.

- **Experiments / analyses needed**  
  E6 stitch-map fitting, E7 cross-scale transplant, E8 ablations.

- **Implementation dependencies**  
  R1–R12, R13–R24.

- **Risk of overclaiming**  
  Very high if hidden tuning sneaks in, if only IID improves, or if the same-size proof is weak.

- **How to weaken if evidence is insufficient**  
  If only same-size works, weaken to a same-size-only paper.  
  If cross-scale gain is weak but above controls, weaken to:  
  “Preliminary evidence suggests limited within-family cross-scale transfer.”

---

## C3 — Gains are semantic, not format-only

- **Exact claim text**  
  The transplant improves semantic function calling rather than merely JSON formatting, as shown by gains on SchemaShift, Distractor, and NoCall slices plus strict-versus-semantic metric analysis.

- **Claim type**  
  Empirical / robustness

- **Status**  
  weakened

- **Required evidence**  
  1. Improvement on the primary metric built from SchemaShift + NoCall.  
  2. Gains remain when function/argument names are renamed and distractors are added.  
  3. Strict and semantic metrics are both reported.  
  4. Error analysis shows more than format-only improvements.

- **Experiments / analyses needed**  
  E0 data harness validation, E4/E7 main evaluations, E9 error analysis.

- **Implementation dependencies**  
  R1–R4, R13–R24.

- **Risk of overclaiming**  
  High if evaluation leaks alias names, if negatives are weak, or if only JSON validity rises.

- **How to weaken if evidence is insufficient**  
  If semantic gains are weak, say only:  
  “The method improves structured output behavior on this task setup.”  
  Do not use “semantic transfer”.

---

## C4 — Better data-efficiency than small-data recipient tuning

- **Exact claim text**  
  Under a matched or stricter parameter budget, the transplant is more data-efficient than recipient calibration-only fine-tuning on the primary OOD metric.

- **Claim type**  
  Empirical / efficiency

- **Status**  
  intended

- **Variant scope**  
  V48 only

- **Required evidence**  
  1. Parameter accounting for transplant and baseline LoRA.  
  2. Small-data LoRA baseline trained only on the calibration set.  
  3. Cross-scale transplant beats or materially matches that baseline on the primary metric.  
  4. Sensitivity analysis if parameter budgets are not exactly identical.

- **Experiments / analyses needed**  
  E2 recipient baselines, E7 cross-scale transplant, E8 parameter-budget sensitivity.

- **Implementation dependencies**  
  R5–R7, R11–R24.

- **Risk of overclaiming**  
  High if parameter budgets differ substantially, if the budget audit is missing, or if only IID is used.

- **How to weaken if evidence is insufficient**  
  If transplant does not beat the baseline, weaken to:  
  “The transplant is competitive with a small-data recipient baseline while providing stronger mechanistic localization.”  
  If not competitive, drop efficiency claim entirely.

---

## C5 — Narrow transfer with low collateral damage

- **Exact claim text**  
  The transplant is relatively narrow: it improves the target capability while causing less damage on non-tool control tasks than dense or full-tuning alternatives.

- **Claim type**  
  Empirical / robustness

- **Status**  
  weakened

- **Required evidence**  
  1. Frozen non-tool control suite with exact-match metrics.  
  2. Control-task performance for the active variant's systems: always source base / donor / same-size transplant / dense and steering controls; add recipient and tuning baselines only in `V48`.  
  3. Reported control drop and a predeclared acceptability threshold.  
  4. Example-based failure analysis if damage is nontrivial.

- **Experiments / analyses needed**  
  E0 control suite validation, E4 or E7 main eval depending on variant, E8 dense/shortcut comparisons, E9 error analysis.

- **Implementation dependencies**  
  R4, R6, R9–R24.

- **Risk of overclaiming**  
  Medium-high because “narrow” is easy to say and hard to operationalize.

- **How to weaken if evidence is insufficient**  
  Replace “narrow” with a numeric statement only, e.g.  
  “Under this control suite, the transplant caused X pp average drop.”  
  Avoid broader narrowness language.

---

## C6 — Sparse causal subset is sufficient

- **Exact claim text**  
  A relatively small selected subset of transplant features retains most of the useful effect, while random subsets and single-feature heuristics do not.

- **Claim type**  
  Empirical / ablation / interpretability

- **Status**  
  weakened

- **Required evidence**  
  1. Feature ranking and pruning protocol locked before final test reporting.  
  2. Retained-gain curve vs number of kept features.  
  3. Random-subset controls.  
  4. Single-feature or leave-one-out ablation evidence on validation/test slices.  
  5. Explicit note that sufficiency does not imply uniqueness or full identifiability.

- **Experiments / analyses needed**  
  E5 feature pruning, E8 random-control ablations.

- **Implementation dependencies**  
  R9–R11, R13–R24.

- **Risk of overclaiming**  
  Very high if the paper slips into “the true circuit is these K features”.

- **How to weaken if evidence is insufficient**  
  Use only:  
  “A small selected subset was sufficient for much of the observed effect on this task.”  
  Do not claim uniqueness, completeness, or ground-truth localization.

---

## C7 — Locked mechanism matters; shortcuts fail

- **Exact claim text**  
  Progressive fitting, stitch maps, and the sparse structural module each contribute meaningfully; shortcut alternatives such as random placement, dense parameter-matched adapters, or one-vector steering do not explain the results.

- **Claim type**  
  Empirical / ablation

- **Status**  
  weakened

- **Required evidence**  
  1. No-progressive ablation.  
  2. Dense parameter-matched adapter baseline.  
  3. One-vector steering baseline.  
  4. Random-layer/random-feature controls.  
  5. Add no-stitch / bad-stitch or naive layer-match ablations only when `V48` is active.

- **Experiments / analyses needed**  
  E8 ablation suite, plus E6/E7 for stitch-specific ablations only in `V48`.

- **Implementation dependencies**  
  R8–R12, R13–R24.

- **Risk of overclaiming**  
  Medium-high if ablations are missing or weak.

- **How to weaken if evidence is insufficient**  
  Limit to the specific shortcut ablations that were actually run.  
  Do not say “the mechanism matters” in general.

---

## C8 — Scope-limited contribution only

- **Exact claim text**  
  The project demonstrates, at most, a narrow within-family capability-transfer result for single-turn function calling. It does not claim universal cross-model transfer, general model editing, or safety guarantees.

- **Claim type**  
  Limitation / guardrail

- **Status**  
  supported

- **Required evidence**  
  This is supported by the locked scope and must be preserved in writing.

- **Experiments / analyses needed**  
  None; this is a boundary condition.

- **Implementation dependencies**  
  None beyond scope discipline.

- **Risk of overclaiming**  
  High if the paper language drifts.

- **How to weaken if evidence is insufficient**  
  N/A. This claim should not be strengthened; it is a cap on ambition.

---

## Claims that are intentionally NOT made

The paper must not implicitly make these claims unless later added here with evidence requirements:

- “The method works across model families.”
- “The selected features are the unique ground-truth circuit.”
- “The method is better than full-data fine-tuning overall.”
- “The method improves general reasoning.”
- “The method is a safety technique.”
- “The method works for multi-turn agents.”
- “The method proves identifiability of internal features.”
