# REPRODUCIBILITY_AND_ENV.md

This file defines the minimum reproducibility standard for the project.

---

## 1) Expected environment

Initial target environment:
- Python 3.11
- Linux workstation environment preferred
- CUDA-capable GPU environment
- PyTorch 2.4+ (exact pin to be frozen once code exists)
- Transformers / PEFT / TRL versions pinned in lockfiles once code exists
- `datasets`, `accelerate`, `safetensors`, `numpy`, `pandas`, `pyyaml`, `pytest`, and lint/format tooling

Important:
- exact package versions are not yet frozen in this build pack
- once implementation starts, create a lockfile and record it in `docs/STATUS.md`

---

## 2) Hardware assumptions

Expected primary hardware:
- single workstation GPU with enough memory for donor training, recipient baselines, and activation caching
- CPU RAM sufficient for dataset processing and chunked cache manifests
- disk space sufficient for cached activations, checkpoints, and raw predictions

Operational rule:
- if compute becomes constrained, reduce search breadth before reducing baseline integrity or evaluation integrity
- every run used in the main paper must record `execution_variant`, `slot_id`, planned GPUh, and actual GPUh

---

## 3) Seed policy

Use fixed named seeds for all main experiments:
- `17`
- `29`
- `43`

Rules:
- every run config must declare its seed
- data-manifest generation must also be seeded and saved
- if an experiment is single-seed due to cost, mark it explicitly as preliminary
- claim upgrades require the variant-specific multi-seed standard described in `docs/EVALUATION_PLAN.md`

---

## 4) Config policy

- all nontrivial runs must be config-driven
- configs must freeze:
  - model IDs / revisions
  - data manifest IDs
  - layer sets
  - latent width / TopK / rank
  - optimization hyperparameters
  - calibration budget
  - eval slices
  - seed
  - prompt contract version
- if a config changes a mathematical object (hook point, dense vs sparse, calibration degrees of freedom), treat it as a method variant, not a routine hyperparameter tweak
- save an immutable copy of the config with every run

---

## 5) Dataset location assumptions

Suggested layout:
- `data/raw/mobile_actions/...`
- `data/processed/canonical/<manifest_id>/...`
- `data/processed/control_suite/<manifest_id>/...`

Rules:
- raw data is read-only after initial import
- processed manifests are versioned and immutable
- all eval runs must record the manifest IDs used
- every manifest should also record a manifest hash over canonical rows + alias banks + prompt contract version
- alias-bank and schema map files are part of the manifest and must be saved

---

## 6) Checkpoint and artifact storage

Suggested layout:
- `artifacts/checkpoints/<run_id>/...`
- `artifacts/caches/<cache_id>/...`
- `artifacts/predictions/<run_id>/...`
- `artifacts/metrics/<run_id>.json`
- `artifacts/figures/<figure_id>.png|pdf`
- `artifacts/evidence/<date>_<milestone>.md`

Every saved run should include:
- prompt contract version
- config snapshot
- seed
- git commit
- hostname / device summary
- start/end timestamps
- model revision identifiers
- data manifest identifiers

---

## 7) Experiment tracking expectations

At minimum, track:
- run ID
- milestone
- command used
- config hash
- seed
- training/eval metrics
- checkpoint path
- prediction path
- notes / anomalies

A lightweight JSONL or SQLite registry is acceptable. External tracking services are optional, not required.

---

## 8) What must be saved for later inspection

For any run used in claim support:
- final config snapshot
- raw predictions on all evaluated slices
- parsed predictions if parsing is separate
- aggregate metrics
- selected feature subset manifest
- calibrated gains
- stitch map stats
- layer-order / cache-version information
- baseline comparison outputs
- error-analysis sample lists

For same-size and cross-scale transplant runs:
- exact selected layers
- exact feature IDs kept
- exact calibration parameters
- cache IDs used for fitting

---

## 9) Minimal rerun instructions

A minimally reproducible result for this project means another session can:
1. recreate the environment from the lockfile,
2. load the frozen manifest and config,
3. rerun the command,
4. obtain materially matching metrics and artifacts.

Planned command style:
- `make train-donor CONFIG=configs/donor_main.yaml`
- `make fit-same-size-transplant CONFIG=configs/same_size_main.yaml`
- `make fit-cross-scale-transplant CONFIG=configs/cross_scale_main.yaml`
- `make eval-main CONFIG=configs/eval_main.yaml`

---

## 10) Minimal reproducible result definitions

### For M1
- canonical manifest regeneration yields identical manifest hash
- scorer outputs identical strict/semantic labels on golden fixtures

### For M3
- donor/base/recipient baseline metrics reproduce within expected seed variation
- saved predictions re-score identically

### For M5
- same-size transplant metrics and selected subset reproduce from saved module + subset manifest
- random-subset controls can be regenerated from saved seed/config

### For M6
- cross-scale transplant metrics reproduce from saved stitch maps + donor sparse module + gain manifest

---

## 11) Artifact naming conventions

Use machine-readable names:
- run IDs: `<date>_<variant>_<slot>_<milestone>_<exp_name>_s<seed>`
- manifests: `manifest_<purpose>_<version>`
- caches: `cache_<modelpair>_<layer>_<manifest>_<version>`
- feature subsets: `subset_<same_size_run>_<version>.json`
- stitch maps: `stitch_<donorlayer>_<recipientlayer>_<run_id>.pt`

Avoid ad-hoc names like `final2_reallyfinal.pt`.

---

## 12) Reproducibility red flags

If any of these occur, results are not reproducible enough for claim support:
- no frozen manifest ID
- no saved config snapshot
- no seed recorded
- no raw predictions saved
- subset selected after looking at final test results
- code path used in a notebook but not captured in `src/`
- results only recoverable from console output or screenshots

---

## 13) Environment unknowns to resolve early

These are open and must be pinned once coding begins:
- exact package versions that support the chosen Gemma checkpoints and hooks cleanly
- preferred hook library or in-house hooks
- exact tokenizer/prompt handling for all selected models
- checkpoint storage budget and cache chunk sizes
