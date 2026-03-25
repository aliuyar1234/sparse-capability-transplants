# Sparse Capability Transplants

Code and core documentation for a research project on same-size capability transfer in function calling.

## Overview

This repository studies whether a narrow function-calling capability can be transferred from a stronger donor model into an untuned weaker model using a fixed internal module rather than full recipient retraining.

The public release is intentionally narrow:

- within-family Gemma transfer
- single-turn function calling
- deterministic JSON grading
- same-size transfer only

## Main result

The strongest final result is:

- same-size transfer is real on this task
- sparse interventions are not noise and show partial localization structure
- dense parameter-matched controls outperform sparse interventions on the matched multiseed comparison

So the public story is not sparse superiority. The interesting result is a performance-versus-localization tradeoff.

## Public contents

This public repo includes:

- core implementation under `src/`
- CLI/task wrappers under `scripts/`
- tests and fixtures under `tests/`
- a small curated doc surface under `docs/`

Public docs kept in this release:

- `docs/RESEARCH_BRIEF.md`
- `docs/METHOD_SPEC.md`
- `docs/EVALUATION_PLAN.md`
- `docs/REPRODUCIBILITY_AND_ENV.md`
- `docs/CLAIMS_MATRIX.md`

## Intentionally omitted

This GitHub-facing release is intentionally minimal.

It does not include:

- generated run outputs
- cached activations
- local checkpoints
- downloaded raw datasets
- internal milestone logs
- internal paper-packaging bundles
- machine-specific experiment configs

Those materials existed in the internal research workspace, but they are not part of this first public cut.

## Repo layout

- `src/` - model, eval, train, and analysis code
- `scripts/` - thin task wrappers
- `tests/` - unit and integration coverage plus small fixtures
- `docs/` - public research, method, and evaluation docs

## Local development

This repo targets Python `3.11`.

Typical setup:

```bash
python -m venv .venv
.venv\\Scripts\\activate
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check src tests scripts
```

## Reproducibility note

The public release keeps the method and evaluation logic, but not the full internal experiment workspace. Some internal experiments depended on local model checkpoints and local run artifacts that are intentionally not published here.

If a later reproducibility package is prepared, it should use sanitized example configs rather than the original machine-specific run configs.

## Public framing

If you are reading this repo as a paper companion, the intended takeaway is:

> same-size capability transfer is real in this setting, but the best-performing same-size mechanism is not maximally sparse.

That is the strongest honest claim supported by the finished project.
