# EVALUATION_PLAN.md

This file defines how claims will be tested. It is the main guard against “implementation progress = research validation”.

---

## 1) Evaluation goals

The evaluation must answer five questions:
1. Does the donor actually learn the target capability?
2. Can a same-size sparse transplant recover nontrivial donor gain?
3. Can a stitched cross-scale transplant improve the smaller recipient?
4. Are gains semantic rather than format-only?
5. Do the sparse mechanism and documented method choices matter relative to shortcuts?


## 1.1) Execution variants and minimum evidence

- **V24 (default)** requires `E0, E1, E3, E4, E5, E8, E9`. In this variant, `C2` and `C4` are inactive and cross-scale experiments are not required for the main paper.
- **V48** adds `E2, E6, E7` and activates `C2` and `C4`.
- Both variants must keep deterministic scoring, SchemaShift, NoCall, and shortcut controls.

---

## 2) Dataset and benchmark plan

### 2.1 Main task data
Base task data is derived from Mobile Actions-style single-turn function-calling traces, then canonicalized and augmented.

### 2.2 Frozen split policy
- build stable `example_id` per canonical row
- raw `eval` rows become the seed IID evaluation pool
- raw `train` rows are sorted by `sha1(example_id)` and split into:
  - first `512` rows -> donor validation
  - next `1024` rows -> recipient calibration
  - remainder -> train
- if canonical train pool `< 4096`, fallback to `10%` val, `15%` calib, remainder train

### 2.3 Frozen evaluation slices
- **IID** — canonical eval examples
- **SchemaShift** — held-out function/argument aliases and description templates
- **Distractor** — SchemaShift plus extra distractor tools and shuffled order
- **NoCall** — missing-tool negatives + unsupported-intent negatives
- **Control** — non-tool exact-match tasks

### 2.4 Leakage risks to check explicitly
- alias surface forms reused across train and test
- paraphrase templates reused across train and test
- unsupported-intent negatives accidentally mapped to existing tools
- SchemaShift examples accidentally retaining canonical names in descriptions
- calibration rows leaking into eval slices

---

## 3) Evaluation objects

### Systems to evaluate
- `B1` source base
- `D1` donor
- `R0` recipient base (V48 only on the main path)
- recipient baselines: small-data LoRA, full-data LoRA, full-data full fine-tune (V48 main path; V24 optional appendix only)
- same-size transplant
- cross-scale transplant (V48 only)
- random-feature transplant
- random-layer transplant
- dense parameter-matched transplant
- steering baseline
- bad-stitch / no-progressive ablations

### Notes
- same-size transplant is a main result in its own right
- cross-scale transplant should not be reported without same-size context
- appendix external BFCL evaluation is optional and separate

---

## 4) Metrics

### 4.1 Primary metric
**Strict Full-Call Success on `SchemaShift union NoCall`**

This is the main metric because:
- SchemaShift resists schema memorization,
- NoCall tests abstention,
- strict scoring discourages partial-credit inflation.

### 4.2 Secondary metrics
- strict full-call success on IID
- strict full-call success on Distractor
- semantic full-call success on all slices
- JSON validity
- call/no-call F1
- tool-selection accuracy conditional on call
- argument exact match
- control-suite exact-match average
- added parameter count
- inference latency overhead

### 4.3 Supporting derived metrics
- donor-gap recovery
- retained-gain percentage after feature pruning
- calibration sensitivity curves over layer gains
- subset-size vs retained-gain tradeoff

Operational rule:
- if donor-gap denominator is `< 0.05`, do not headline recovery; report raw deltas instead

---

## 5) Claims-to-experiments map

| Experiment ID | Purpose | Claims affected |
|---|---|---|
| E0 | data/scorer integrity checks | C3, C5, C7 |
| E1 | donor gap establishment | C1, C3 |
| E2 | recipient baseline training | C2, C4, C5 |
| E3 | rough layer scan | C1, C7 |
| E4 | same-size transplant end-to-end | C1, C3, C5 |
| E5 | feature pruning + subset controls | C6, C7 |
| E6 | stitch map fitting | C2, C7 |
| E7 | cross-scale transplant end-to-end | C2, C3, C4, C5 |
| E8 | ablation suite / shortcut controls | C1, C2, C6, C7 |
| E9 | robustness + error analysis | C3, C5 |
| E10 | optional external appendix check | none unless explicitly promoted |

---

## 6) Required baselines

These are mandatory for main-paper claims.

### Always required (V24 and V48)
- source base `B1`
- donor `D1`
- same-size transplant
- random-feature transplant
- random-layer transplant
- dense parameter-matched same-size transplant
- one-vector steering baseline
- no-progressive same-size ablation

### Required only when `V48` is active
- recipient base `R0`
- small-data LoRA on calibration split only
- full-data LoRA
- full-data full fine-tune (or documented infeasibility)
- cross-scale transplant
- bad-stitch or naive-stitch baseline
- dense parameter-matched cross-scale transplant

### Parameter-matching rule
A baseline counts as parameter-matched only if its trainable parameter count is within `±10%` of the transplant budget. If exact matching is impossible, run one lower and one upper budget baseline and report both.

---

## 7) Main ablation plan

### A1 — Layer count / placement
- 1 selected layer
- 2 selected layers
- 3 selected layers
- rough candidate scan across 4 fractional-depth sites `{0.25, 0.50, 0.65, 0.85}`

### A2 — Sparsity budget
- vary latent width / TopK / retained feature count
- plot retained gain vs added parameters

### A3 — Progressive vs independent
- independent per-layer fitting only
- progressive fitting with refreshed caches

### A4 — Stitch design
- fractional-depth pairing only
- local searched pairing
- bad-stitch/random stitch control
- reduced rank vs nominal rank

### A5 — Calibration budget
- no calibration
- gain-only calibration
- prohibited richer calibration (diagnostic only; not main method)

### A6 — Dense vs sparse
- sparse transplant
- dense parameter-matched adapter transplant

### A7 — Feature subset controls
- selected subset
- random subset same size
- top-activation-only subset
- leave-one-out feature ablations

---

## 8) Error analysis plan

For the main same-size and cross-scale runs, collect example-level error labels:
- parsing failure
- wrong call vs `NO_TOOL`
- wrong tool under call
- correct tool, wrong argument key
- correct tool, wrong argument value
- distractor confusion
- alias confusion
- control-task damage example

Each error example should store:
- example ID
- prompt contract version
- input prompt content
- available tools
- gold output
- raw model output
- parsed output
- error category
- active selected layers / gains / feature subset version

Qualitative review should inspect at least:
- 20 same-size success cases
- if `V48` is active: 20 cross-scale success cases
- 20 failures per major slice
- if `V48` is active: cases where transplant helps or hurts relative to recipient base
- always: cases where the transplant helps or hurts relative to its direct controls

---

## 9) Support, weak support, and failure bars

### C1 Same-size existence proof
- **Support:** recovery >= `0.35` on the primary metric, 95% CI lower bound above `0`, and clear wins over random/dense controls
- **Weak support:** recovery in `[0.10, 0.35)` or positive but unstable across seeds
- **Failure:** recovery < `0.10`, or random/dense controls match it

### C2 Cross-scale transplant
- **Variant scope:** V48 only
- **Support:** recovery >= `0.15` on the primary metric with gain-only calibration and wins over shortcut controls
- **Weak support:** recovery in `[0.05, 0.15)` or positive only on some slices
- **Failure:** recovery < `0.05`, or richer calibration is required

### C3 Semantic transfer
- **Support:** primary metric improves and error analysis shows real tool/argument improvements on SchemaShift + NoCall
- **Weak support:** strict gains mostly from formatting with only modest semantic lift
- **Failure:** gains vanish under SchemaShift/NoCall or only JSON validity rises

### C4 Data efficiency
- **Variant scope:** V48 only
- **Support:** transplant beats or materially matches calibration-only LoRA on the primary OOD metric under matched or stricter budget
- **Weak support:** competitive only on some slices or only under bracketed budget comparisons
- **Failure:** calibration-only LoRA clearly dominates

### C5 Narrowness / low collateral damage
- **Support:** control drop <= `2.0` percentage points and lower than dense/full-tuning alternatives
- **Weak support:** control drop in `(2.0, 5.0]` pp and well characterized
- **Failure:** control drop > `5.0` pp or worse than dense/full-tuning

### C6 Sparse causal subset
- **Support:** a small subset retains >= `80%` of same-size gain and random subsets fail clearly
- **Weak support:** subset helps, but retained-gain or sparsity advantage is unstable
- **Failure:** random subsets perform similarly or many features are required

### C7 Mechanism matters
- **Support:** no-progressive, bad-stitch, dense, or steering shortcuts lose materially relative to the locked method
- **Weak support:** some ablations matter, some do not
- **Failure:** shortcut variants explain the same result

These thresholds are operational bars for claim status. They do not guarantee acceptance, but they prevent vague “it kind of worked” reporting.

---

## 10) Statistical treatment

### Seed policy by variant
- **V24**
  - discovery/search runs may use `1` seed
  - the final same-size setting and donor-gap check must receive at least `1` confirmatory rerun if budget allows
  - if only one seed exists for the final result, no affected claim may rise above **partially supported**
- **V48**
  - discovery/search runs may use `1` seed
  - final same-size and final cross-scale settings should receive `2` confirmatory seeds minimum, `3` preferred
  - if only one seed exists for the final cross-scale result, `C2` and `C4` may be at most **partially supported**

Shared rules:
- report mean and 95% bootstrap confidence intervals over examples where multi-seed aggregation exists
- for pairwise main comparisons, use paired permutation tests where applicable
- use Holm correction across families of secondary metrics
- do not promote a claim based on a single lucky seed

When donor-gap recovery is reported:
- ensure denominator is sufficiently above zero
- if denominator is too small, report raw deltas instead and mark recovery as unstable

---

## 11) Resource / compute considerations

### Variant-specific priority order
- **V24**
  1. preserve donor/base metrics
  2. preserve same-size transplant
  3. preserve random/dense/steering controls
  4. reduce search breadth
  5. cut cross-scale entirely (already inactive)
- **V48**
  1. preserve donor/base/recipient baselines
  2. preserve same-size transplant
  3. preserve cross-scale transplant
  4. preserve random/shortcut controls
  5. reduce hyperparameter search breadth and appendix checks

### Do not cut first
- control suite
- NoCall negatives
- held-out alias banks
- random-feature/random-layer controls
- parameter-matching audit for any active efficiency claim
- same-size existence proof

---

## 12) Reproducibility notes specific to evaluation

- every evaluation must be tied to a frozen data manifest
- primary metric script must be deterministic and versioned
- raw outputs must be saved for error analysis
- every result table should be reproducible from saved prediction files + scorer version
- figures should be generated from saved metric artifacts, not manually edited
- every table row must record the prompt contract version

---

## 13) Expected evaluation artifacts

### Tables
- donor/base baseline table
- if `V48` is active: recipient baseline table
- same-size vs controls table
- if `V48` is active: cross-scale vs recipient baselines table
- strict vs semantic metrics table
- ablation table
- error-category table
- parameter-budget table (same-size always; recipient comparison only in `V48`)

### Plots
- donor-gap recovery vs parameters
- retained gain vs number of selected features
- control drop vs primary metric gain
- calibration sensitivity curves
- per-slice comparison plots

### Traces / examples
- success/failure example appendix
- parsed-output examples
- per-feature or per-layer qualitative traces if useful

---

## 14) Optional external evaluation

An external BFCL single-turn check may be added **only after** the main harness is stable.

Rules:
- do not let it redefine the paper story
- do not upgrade any claim based solely on appendix external data
- if included, report it as external stress validation, not the main evidence
