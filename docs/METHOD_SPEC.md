# METHOD_SPEC.md

This file is the exact method source of truth. If code or experiments disagree with this file, either:
1. update this file intentionally and record the reason in `docs/STATUS.md`, or
2. stop and record a blocker.

The main method is a **sparse structural transplant** for a narrow capability. It is not generic distillation, generic LoRA, or pure inference-time steering.

---

## 1) Problem setup

We work with three model roles in one family:

- **Source base** `B1`: untuned same-size base model (default `Gemma 3 1B-IT`)
- **Donor** `D1`: `B1` after task-specific function-calling training
- **Recipient** `R0`: smaller untuned model (default `Gemma 3 270M-IT`)

The task is **single-turn function calling**.

Each example consists of:
- `u`: user request text
- `T = {t_1, ..., t_n}`: available tool schemas shown to the model
- `y*`: gold target, either:
  - `{"name": "<TOOL_SURFACE_NAME>", "arguments": {...}}`, or
  - `{"name": "NO_TOOL", "arguments": {}}`

The project aims to learn a sparse structural transplant `T_sparse` such that:
1. `B1 + T_sparse` recovers part of the donor's function-calling advantage,
2. a stitched version of `T_sparse` improves `R0` without full recipient retraining,
3. the gain is semantic under schema shift and NoCall cases, not just formatting.

---

## 2) Locked inputs, outputs, and contracts

### 2.1 Inputs
- canonicalized function-calling train/val/calibration/eval data
- frozen alias banks and distractor-tool libraries
- frozen control suite
- source base / donor / recipient checkpoints
- activation caches from matched teacher-forced executions
- configs specifying layers, ranks, widths, seeds, and gain search

### 2.2 Outputs
- donor checkpoint(s)
- recipient baseline checkpoints
- same-size transplant module(s)
- pruned sparse feature subset(s)
- stitch-map checkpoint(s)
- cross-scale transplant module(s)
- deterministic evaluation artifacts
- claim-linked evidence notes

### 2.3 Locked prompt-content contract

The repo must separate **message content** from **model-specific chat serialization**.

Locked message-content contract:
- system message: instruct the model to return exactly one JSON object and no prose
- user message body contains:
  1. a tool inventory block,
  2. the user request block,
  3. explicit JSON contract,
  4. abstention rule using `NO_TOOL`
- assistant target during supervised training and teacher forcing is the canonical JSON object only

Locked user message body template:

```text
Available tools:
<JSON array of tool schemas exactly as shown to the model>

User request:
<user request string>

Return exactly one JSON object with keys "name" and "arguments".
If no tool applies, return {"name":"NO_TOOL","arguments":{}}.
Do not return prose, Markdown, or multiple JSON objects.
```

Operational rule:
- use each tokenizer/model's official chat template to serialize the same message content
- do not hand-insert model-specific control tokens directly in multiple places
- save a `prompt_contract_version` with every run manifest

### 2.4 Locked split-selection and calibration policy

Base source rows come from the raw dataset's own split metadata if available.

Default deterministic selection policy:
1. import raw rows and build a stable `example_id` for every canonical example
2. keep all raw `eval` rows as the seed IID evaluation pool
3. within raw `train` rows only, sort by `sha1(example_id)` ascending
4. reserve the first `512` rows for donor validation
5. reserve the next `1024` rows for recipient calibration
6. use the remainder as donor/recipient training data

Fallback if the raw `train` pool is unexpectedly small (`< 4096` rows after canonicalization):
- validation = first `10%` of hashed rows
- calibration = next `15%`
- train = remainder

Rules:
- calibration rows are never used for evaluation slices
- SchemaShift, Distractor, and NoCall are generated from the IID evaluation pool only
- alias banks are frozen **after** split selection, not before


### 2.5 Locked compute-budget execution variants

Planning GPU-hour budgets are **execution caps**, not empirical results. They exist to stop unbounded branching. Actual consumed GPUh must be logged in evidence notes.

- **V24 (default)**
  - planning cap: `<=24 GPUh`
  - active claims: `C1, C3, C5, C6, C7, C8`
  - paper scope: same-size transplant only
  - recipient-specific cross-scale claims are inactive
  - no donor escalation above `1B` unless the project is explicitly re-scoped

- **V48 (extension)**
  - planning cap: `<=48 GPUh`
  - active claims: `C1-C8`
  - paper scope: same-size + cross-scale transplant
  - may activate recipient baselines and cross-scale experiments only after the same-size gate passes
  - donor escalation to `4B` is allowed only if the budget still fits and the change is recorded

Operational rule:
- freeze `execution_variant` before donor training begins
- do not silently switch variants to rescue a weak result

---

## 3) Locked requirements (R*)

### Data and task requirements
- **R1** Canonicalize all targets into one strict JSON format with reserved `NO_TOOL`.
- **R2** Build a deterministic parser and scorer before any method work.
- **R3** Freeze disjoint alias banks for train/val/test on tool names, argument names, and descriptions.
- **R4** Build two negative types: missing-tool negatives and unsupported-intent negatives.
- **R5** Build a non-tool exact-match control suite.

### Model and training requirements
- **R6** Train a donor that establishes a clear donor-base gap on the primary metric before any transplant work.
- **R7** When `V48` is active, train recipient baselines: base, small-data LoRA, full-data LoRA, and full-data full fine-tune (or record infeasibility explicitly). In `V24`, recipient baselines are optional scaffolds only and not required for main claims.
- **R8** Capture matched activation caches for donor/base and recipient/base alignment corpora.

### Method requirements
- **R9** Same-size transplant must target selected MLP-site output deltas, not generic residual edits.
- **R10** Same-size fitting must be progressive: recollect downstream activations after earlier injections.
- **R11** Feature selection must keep causal sufficiency separate from descriptive salience.
- **R12** Cross-scale transfer must use low-rank stitch maps and gain-only calibration; no hidden dense retuning.

### Evaluation and discipline requirements
- **R13** Primary metric is deterministic and built from SchemaShift + NoCall.
- **R14** Main baselines and ablations are mandatory; no shortcut-free result is publishable.
- **R15** Report strict vs semantic metrics separately.
- **R16** Multi-seed evaluation, frozen configs, and evidence logging are required for claim updates.

### Additional locked requirements added by audit
- **R17** Freeze and version the prompt-content contract and the assistant target serialization.
- **R18** Freeze split-selection, validation, and calibration manifests before training begins.
- **R19** Parameter-matched baselines must match added trainable parameter count within `±10%`, or use a documented lower/upper budget bracket.
- **R20** Donor-gap gate is numeric: proceed only if donor beats source base on the primary metric by at least `15` percentage points, or by `>= 5` points with a 95% CI lower bound above `0`; otherwise stop, strengthen the task/training, or escalate donor size and record the decision.
- **R21** Freeze `execution_variant in {V24, V48}` before donor training; the default is `V24`.
- **R22** Under `V24`, cross-scale transfer and recipient data-efficiency claims are out of scope; do not run or claim them on the main path.
- **R23** Under `V48`, cross-scale work is allowed only after Gates `G1` and `G2` pass.
- **R24** Every experiment slot has a planning GPU-hour cap; exceeding it requires a `docs/STATUS.md` update and a new evidence note before more compute is spent.

---

## 4) Data flow and control flow overview

1. **Build task data**
   - canonicalize source data
   - freeze split manifests
   - generate SchemaShift, Distractor, NoCall, and control-suite slices
   - freeze scorer fixtures

2. **Train models**
   - train donor from `B1`
   - train recipient baselines from `R0`

3. **Establish donor gap**
   - if donor gap fails `R20`, stop and repair task/training before interpretability work

4. **Same-size transplant**
   - cache donor/base activations on matched data
   - scan candidate layers
   - fit sparse delta modules
   - progressively refit with recollected activations
   - prune features into a causal subset
   - verify same-size recovery
   - if `execution_variant == V24`, stop the main method here and move directly to ablations / paper artifacts

5. **Cross-scale transplant** (V48 only)
   - fit stitch maps between `R0` and `B1`
   - compose donor-space delta module with input/output stitch maps
   - calibrate per-layer gains on the calibration set only
   - evaluate against recipient baselines

6. **Ablate and analyze**
   - random/shortcut controls
   - dense parameter-matched control
   - no-progressive ablation
   - no-stitch / naive-stitch ablations
   - control damage and error analysis

---

## 5) Core mechanism

### 5.1 Same-size sparse delta transplant

#### Intuition
The donor and source base share architecture and parameter count. The donor differs mainly because training changed some internal computations. Rather than retrain the base again, we learn a sparse module that predicts the donor's extra useful MLP output at a few layers.

#### Precise mechanism
For selected donor/source-base layer `l`:
- `x_l^B`: input to the MLP sublayer in `B1`
- `u_l^B`: MLP output of `B1`
- `u_l^D`: MLP output of `D1`
- target delta `Delta_l = u_l^D - u_l^B`

Train a sparse module `T_l` such that:
- `T_l(x_l^B) ~= Delta_l`

At inference in `B1`, replace:
- `u_l^B` with `u_l^B + lambda_l * T_l(x_l^B)`

where `lambda_l` is a scalar gain.

#### Implementation hooks
- hook point is the **MLP output tensor before it is added back into the residual stream**
- cache `x_l^B`, `u_l^B`, `u_l^D`, token-position labels, prompt/example IDs, and cache version
- do not substitute some nearby residual tensor if a clean MLP hook exists

### 5.2 Sparse delta module

#### Intuition
The module should be small, sparse, and inspectable. It must not silently become a dense adapter.

#### Precise mechanism
For layer `l` with hidden width `d_l` and latent width `m_l`:
- `W_enc_l` has shape `[m_l, d_l]`
- `b_enc_l` has shape `[m_l]`
- `W_dec_l` has shape `[d_l, m_l]`
- `a_l = SiLU(W_enc_l x + b_enc_l)`
- `z_l = TopK(a_l, k_l)` where only the top `k_l` values per token are kept
- `delta_hat_l = W_dec_l z_l`

Training objective family is locked:
- weighted MSE to donor delta target
- optional `L1` penalty on `z_l`
- optional decoder weight penalty

#### Implementation hooks
- `TopK` is value-based **per token**, not across the batch
- store active feature indices and activations for analysis
- token weights emphasize decision/tool/argument tokens
- write shape assertions at every hook boundary

### 5.3 Progressive fitting

#### Intuition
If an earlier transplant changes later hidden states, later modules trained on the untouched base are targeting the wrong distribution.

#### Precise mechanism
1. Fit rough independent modules at candidate layers.
2. Rank layers by validation objective.
3. Insert the best layer into `B1`.
4. Recollect activations for the next chosen layer under the partially transplanted model.
5. Fit the next layer on refreshed caches.
6. Repeat until the selected layer budget is reached.

#### Implementation hooks
- keep cache versions explicit: untouched-base vs partially-transplanted
- never mix stale and refreshed caches silently
- record the selected layer order and the cache version used for each layer

### 5.4 Causal feature pruning

#### Intuition
Highly active features are not necessarily useful. Useful features may be redundant. We need a small subset that keeps gain while avoiding control damage.

#### Precise mechanism
After same-size modules are trained:
1. compute feature statistics on validation data
2. build a shortlist using causal utility and semantic specificity signals
3. cluster near-duplicates by activation correlation
4. greedily add features if they improve a validation objective that rewards primary-metric gain and penalizes control damage and feature count
5. freeze the final subset before any final test reporting

Suggested validation objective:
`J(S) = PrimaryVal(S) - alpha * max(0, ControlDrop(S) - tau_c) - beta * |S|`

#### Implementation hooks
- shortlist generation and greedy selection must be deterministic
- log every subset evaluation
- random-subset controls are mandatory
- subset freeze happens before final test metrics are opened

### 5.5 Cross-scale stitch maps

#### Intuition
The donor-space delta module expects donor-space inputs and emits donor-space deltas. The smaller recipient needs maps into and out of that space.

#### Precise mechanism
For selected donor layer `l_D` and aligned recipient layer `l_R`:
- learn input stitch `A_in: R^(d_R) -> R^(d_D)`
- learn output stitch `A_out: R^(d_D) -> R^(d_R)`

Default alignment:
- donor candidate layers are chosen by fractional depth positions `{0.25, 0.50, 0.65, 0.85}` of donor depth
- recipient layer `l_R` is initialized by fractional-depth alignment
- bounded local search radius defaults to `1` neighboring recipient layer on either side

Default alignment corpus composition:
- `10k` positive tool-use prompts
- `5k` NoCall negatives
- `5k` control prompts
If fewer examples are available, sample proportionally with a fixed seed and record the realized counts.

Use low-rank affine maps:
- `A_in(x) = U_in (V_in x) + b_in`
- `A_out(y) = U_out (V_out y) + b_out`

#### Implementation hooks
- fit input and output stitches separately unless the config explicitly ties them
- save nominal fractional-depth pairing and any searched alternatives
- prefer source-base `B1` activations for stitch fitting unless a justified variant is being studied

### 5.6 Cross-scale transplant composition

#### Intuition
Map recipient inputs into donor space, run the donor-space sparse delta module, then map the predicted donor delta back into recipient space and inject it at the recipient MLP site.

#### Precise mechanism
At recipient layer `l_R`:
1. read recipient MLP input `x_l^R`
2. donor-space input `x_tilde = A_in(x_l^R)`
3. donor-space delta `delta_tilde = T_l(x_tilde)`
4. recipient-space delta `delta_R = A_out(delta_tilde)`
5. modified recipient MLP output: `u_l^R + lambda_l * delta_R`

`lambda_l` is calibrated on the calibration set only.

#### Implementation hooks
- main-method calibration may tune only small explicit parameters defined in config (default: one scalar per selected layer, optional global scalar)
- no gradient updates to recipient base weights in the main method
- store calibrated gains with the run manifest

---

## 6) Baseline definitions and parameter accounting

### 6.1 Recipient baselines
Mandatory only when `V48` is active:
- recipient base `R0`
- small-data LoRA on calibration split only
- full-data LoRA on full train split
- full-data full fine-tune (or documented infeasibility)

In `V24`, recipient baselines may exist as optional scaffolds but are not required for the main paper path.

### 6.2 Dense parameter-matched transplant baseline
This control must use the **same hook locations** and the **same calibration rules** as the main method, but replace the sparse module with a dense trainable module of comparable parameter count.

Default dense baseline at one layer `l`:
- two-layer dense MLP: `Dense_l(x) = W2 * SiLU(W1 x + b1) + b2`
- choose hidden width so that the dense module's total trainable params are within `±10%` of the sparse module params for that layer
- for cross-scale dense control, keep the same stitch maps and replace only the sparse donor-space module

### 6.3 One-vector steering baseline
This control exists to show that the result is not explained by a crude direction injection.

Default steering vector at one layer `l`:
- `v_l = mean(Delta_l)` over target-token rows in the training cache
- inject `u_l <- u_l + lambda_l * v_l`
- use the same gain grid and the same calibration split as the main method

### 6.4 Parameter-count formulas
Sparse same-size module at layer `l`:
- encoder params: `m_l * d_l + m_l`
- decoder params: `d_l * m_l`
- gains: `1` scalar per transplanted layer
- total approx: `2 * d_l * m_l + m_l + 1`

Low-rank stitch maps for one donor/recipient pair:
- `A_in`: `r * d_R + d_D * r + d_D`
- `A_out`: `r * d_D + d_R * r + d_R`

Total cross-scale added parameters:
- sum of sparse donor-space module params over selected layers
- plus sum of stitch-map params over selected layer pairs
- plus calibrated scalars

Operational rule for C4:
- if exact budget matching is not possible, run one baseline just below and one just above the transplant budget and report bracketed sensitivity; do not quietly compare against a much larger or much smaller baseline

---

## 7) Training / inference / optimization flow

### 7.1 Donor training
- teacher-forced supervised training on the canonical training split
- greedy decoding for evaluation
- donor must satisfy `R20` before transplant work continues
- if `1B` donor fails the gate, strengthen the task or escalate donor to `4B` and record the decision

### 7.2 Recipient baselines
- small-data LoRA uses the calibration split only
- full-data LoRA uses full train split
- full-data full fine-tune is an upper bound, not the main comparison target

### 7.3 Delta-module optimization
- optimize on cached donor/base activations
- use weighted token loss emphasizing decision/tool/argument tokens
- validate on both held-out activation rows and end-to-end task metrics

### 7.4 Stitch optimization
- optimize on recipient/source-base activation pairs from the alignment corpus
- validate both reconstruction error and end-to-end cross-scale transfer

### 7.5 Inference and evaluation
- no stochastic decoding in main reported metrics
- always use deterministic prompt formatting and parsing rules
- save raw outputs, parsed outputs, and scorer version

---

## 8) Edge cases that must be handled explicitly

- model outputs prose before or after JSON
- model outputs invalid JSON
- model emits multiple JSON objects
- argument order differs but semantics are the same
- schema aliases collide or become ambiguous
- tool list omits the gold tool
- unsupported user requests should map to `NO_TOOL`
- recipient and donor layer counts differ
- some selected donor layers may map poorly to recipient layers
- stitched transplant may improve JSON validity but hurt NoCall behavior
- tokenizer offset mapping may split tool names or argument values across multiple pieces

---

## 9) Complexity targets and bounded search defaults

These are locked defaults for v1 unless intentionally changed.

### Candidate layers (all variants)
- donor candidate layers are the MLP blocks nearest fractional depths `{0.25, 0.50, 0.65, 0.85}`
- do not expand beyond these `4` candidates on the main path without revising the compute plan

### V24 defaults (same-size-only, `<=24 GPUh`)
- selected layer budget: at most `2`
- sparse latent width `m_l`: start at `256`
- TopK `k_l`: start at `8`, may expand to `16` only if within slot budget
- discovery cache subset: `2048-4096` examples, task-relevant token spans only
- no stitch maps, no recipient baselines on the main path
- seed policy: `1` discovery seed + `1` confirmatory rerun on the final same-size setting if budget permits

### V48 defaults (same-size + cross-scale, `<=48 GPUh`)
- selected layer budget: at most `3`
- sparse latent width `m_l`: start at `512`
- TopK `k_l`: start at `16`
- stitch rank `r`: start at `64`
- local recipient-layer search radius: `1`
- alignment corpus: up to `20k` examples with the locked positive/NoCall/control mix
- seed policy: `1` discovery seed + `2` confirmatory reruns on the final same-size and cross-scale settings if budget permits

### Shared calibration defaults
- calibration gain grid: `{0.0, 0.25, 0.5, 0.75, 1.0, 1.25}`
- optional global gain grid: same as layer gains

### Resource rules
- activation caching must be chunked and resumable
- if compute is tight, reduce search breadth before reducing baseline integrity
- do not exceed the active variant's planned GPU-hour cap without a status update and an evidence note
- target well under `1%` added parameters if possible, but do not overclaim if the realized budget is higher

---

## 10) Places where shortcutting will break the science

- treating function-calling success as JSON validity only
- skipping held-out alias banks
- training the recipient heavily and still calling it calibration
- fitting all layers independently and calling it progressive
- selecting features after seeing final test results
- using random or convenience layer matches without logging them
- omitting random controls or dense baselines
- quietly changing the primary metric to the best-looking one
- comparing against an unmatched baseline budget and still calling it parameter-matched
- changing prompt content between donor/base/recipient cache collection

---

## 11) Failure modes and what they mean

- **Weak donor gap**  
  The task/training is not strong enough yet; method claims cannot be evaluated.

- **Same-size transplant fails**  
  Cross-scale work is not justified as a main positive claim; either the capability is too diffuse or the method is wrong.

- **Cross-scale fails but same-size works**  
  The paper may still succeed as a same-size transplant paper; cross-scale becomes negative evidence.

- **SchemaShift gains disappear**  
  The method may only learn formatting or schema memorization.

- **Calibration-only LoRA dominates**  
  Data-efficiency claim fails; transplant may still matter mechanistically if other claims survive.

- **Control damage is high**  
  Narrowness claim fails; the module may be too entangled or the control suite too weak.

- **Random subsets match selected subsets**  
  Sparse causal localization claim fails.

---

## 12) Decision gates

- **Gate G1 — Donor gap gate**  
  Do not start transplant work until the donor satisfies `R20` on the primary metric.

- **Gate G2 — Same-size gate**  
  Do not treat cross-scale transfer as a main positive claim unless same-size transplant is meaningfully positive.

- **Gate G3 — Semantic gate**  
  Do not use “semantic transfer” language unless SchemaShift + NoCall gains exist.

- **Gate G4 — Efficiency gate**  
  Do not use “data-efficient” language unless the transplant is competitive with the calibration-only baseline under a matched or stricter budget.

- **Gate G5 — Narrowness gate**  
  Do not use “narrow” language unless control damage is low and lower than dense/full-tuning alternatives.

- **Gate G6 — Variant gate**  
  Under `V24`, stop after same-size evidence and do not reopen cross-scale claims unless the variant is explicitly changed to `V48`.

- **Gate G7 — Budget gate**  
  If a planned slot exhausts its GPU-hour cap without meeting its proceed condition, stop and record the failure instead of quietly broadening the search.
