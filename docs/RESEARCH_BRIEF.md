# RESEARCH_BRIEF.md

## 1) Core problem statement

Current post-training practice can produce a stronger model, but usually does **not** tell us which internal computation actually changed, how narrow that change is, or whether the useful part can be moved into a cheaper model without fully retraining it.

This project studies one narrow, operationally meaningful capability:

**single-turn function calling** within the Gemma family.

We ask whether a donor model's function-calling skill can be isolated as a **sparse internal delta** and transplanted into a smaller recipient model via a **tiny fixed structural module**, rather than via generic full fine-tuning, distillation, or test-time steering.

## 2) Why this matters

If the project works, it would matter for at least four reasons:

1. **Economic value**  
   It offers a route to transferring a narrow useful capability into a cheaper model.

2. **Mechanistic value**  
   It turns interpretability into a constructive intervention instead of a descriptive afterthought.

3. **Post-training science**  
   It forces separation between useful computation and collateral style/formatting changes.

4. **Trustworthiness**  
   The evaluation can deterministically distinguish semantic transfer from formatting-only gains.

## 3) Proposed contribution in plain language

Train a stronger donor model to do function calling well. Compare it to its untuned same-size base model. Learn a sparse module that predicts the donor's extra useful internal computation at a few layers. Show that this module can:
- recover some of the donor's gain when inserted back into the same-size base model,
- then be mapped into a smaller recipient model with only small stitch maps and gain calibration,
- while improving semantic function calling under schema shift and abstention cases.

## 4) Locked facts / locked scope

These are locked unless explicitly revised in this file and `docs/STATUS.md`.

### Locked facts (F*)
- **F1** Main task is single-turn function calling with deterministic grading.
- **F2** Main model family is Gemma.
- **F3** Main donor path is `Gemma 3 1B-IT` base -> donor fine-tune.
- **F4** Main recipient is `Gemma 3 270M-IT`.
- **F5** Same-size existence proof comes before cross-scale transplant.
- **F6** Main output format is strict JSON with a reserved `NO_TOOL` object for abstention.
- **F7** Main evaluation slices are IID, SchemaShift, Distractor, NoCall, and a non-tool control suite.
- **F8** Main transplant mechanism is a sparse fixed module inserted at selected MLP sites.
- **F9** Cross-scale transfer uses learned low-rank stitch maps plus gain calibration only.
- **F10** Main metrics are deterministic; no LLM judge is allowed for primary claims.
- **F11** The paper is not allowed to become “another benchmark paper”; evaluation exists to falsify fake wins.
- **F12** Optional external BFCL evaluation is appendix-only unless later promoted by explicit decision.
- **F13** Default execution variant is `V24`: same-size-only paper path under a planning cap of `<=24 GPUh`.
- **F14** `V48` is an explicitly gated extension path: same-size + cross-scale paper under a planning cap of `<=48 GPUh`.
- **F15** `V24` may not claim cross-scale transfer or recipient data-efficiency; those claims are inactive unless the project is promoted to `V48`.

### Locked non-goals
- Multi-turn planning agents
- Cross-family transfer as a main claim
- Universal model editing
- Full theory of feature identifiability
- Safety / policy alignment claims
- Replacing all recipient tuning with transplants in general
- Benchmark leadership as the main paper story

## 5) Target users / readers / evaluators

Primary readers:
- ML / NLP reviewers for empirical method papers
- interpretability researchers interested in constructive interventions
- post-training / efficiency researchers
- engineers who care about moving one narrow capability into a cheaper model

Secondary evaluators:
- reproducibility-minded reviewers who will check leakage, baselines, and overclaiming
- implementation agents resuming work in later sessions

## 6) Success conditions

### Strong compute-bounded success (`V24`)
The default `V24` path counts as a success if all of these are true:

- A same-size sparse transplant recovers a meaningful fraction of donor gain.
- Gains survive schema renaming, distractors, and NoCall cases.
- Random-feature / random-layer / dense / steering shortcuts fail clearly enough that the sparse mechanism matters.
- Control-task damage stays low.
- The paper can honestly stop without any cross-scale claim.

### Full success (`V48`)
The `V48` extension path counts as a full success if `V24` success already holds and, in addition:

- A cross-scale stitched transplant improves the smaller recipient on the primary semantic metric.
- The transplant beats or is at least competitive with a parameter-matched small-data recipient baseline on the primary OOD metric.
- Cross-scale gains remain compatible with the locked calibration-only rule.

### Partial success
The project still matters if:
- same-size transplant works but cross-scale transfer fails,
- cross-scale transfer works only weakly but clearly above random/shortcut controls,
- gains are narrow but honest and interpretable,
- the negative result itself identifies why structural transfer fails.

### Partial failure
This project is a partial failure if:
- gains come only from JSON formatting,
- same-size transplant does not work well enough to justify cross-scale work,
- a tiny recipient LoRA on the calibration set easily dominates the transplant when `V48` is active,
- control damage is large,
- the method only works with hidden full-data recipient tuning.

## 7) Hypotheses (H*) and assumptions (A*)

### Hypotheses
These are things we aim to test, not established results.

- **H1** Function-calling gain is sparse enough at a few MLP sites to admit a useful same-size transplant.
- **H2** Progressive fitting matters because earlier injections change later layer inputs.
- **H3** Low-rank within-family stitch maps preserve the capability-relevant subspace well enough for cross-scale transfer.
- **H4** SchemaShift + NoCall evaluation is strong enough to separate semantics from formatting.
- **H5** A small selected feature subset can retain most of the transplant gain.
- **H6** `Gemma 3 1B-IT` is strong enough as the donor. If not, a 4B donor is the fallback.

### Assumptions
These are operational assumptions that must be checked.

- **A1** Donor/base/recipient tokenization and prompt formatting can be aligned tightly enough for matched-position caching.
- **A2** Augmented aliases and paraphrases remain semantically faithful.
- **A3** Mobile Actions-derived supervision is rich enough to establish a meaningful donor gap.
- **A4** Gain calibration can stay “calibration only” and not silently become another tuning stage.
- **A5** The chosen control tasks are sufficient to detect major collateral damage.
- **A6** Available hardware permits at least the chosen execution variant (`V24` or `V48`) plus its required confirmatory reruns.

## 8) What must be true for claims to remain honest

This repo must always preserve the distinction below:

### “We aim to show”
- sparse capability transplants can work within a family,
- the moved capability is semantic and narrow,
- the mechanism matters more than simple formatting or trivial fine-tuning shortcuts.

### “We have shown”
Nothing yet. At project start, all positive contribution claims are **intended only**. Claim status changes happen only through `docs/CLAIMS_MATRIX.md` with linked evidence.

## 9) Open questions that are allowed to remain open temporarily

These are not locked; implementation can investigate them within the stated scope.

- Exact donor training recipe if the 1B donor gap is weak
- Exact selected layer count and positions
- Exact latent width / TopK / rank values
- Exact feature scoring weights for pruning
- Whether output-space and input-space stitch maps should be tied or separate
- Whether BFCL appendix adds value beyond the main evaluation harness

## 10) Success philosophy for this repo

The goal is not to force a positive result. The goal is to produce a paper-quality answer to:

> Is a sparse structural transplant a real way to move one narrow capability into a cheaper model, or not?

A clean negative answer is acceptable. Hidden overclaiming is not.
