# Sparse Capability Transplants

[![Preprint PDF](https://img.shields.io/badge/Preprint-PDF-B31B1B?style=flat-square&logo=adobeacrobatreader&logoColor=white)](https://github.com/aliuyar1234/sparse-capability-transplants/raw/main/same-size-capability-transfer-tradeoff-preprint.pdf)
[![Paper Source](https://img.shields.io/badge/Paper-LaTeX-008080?style=flat-square&logo=latex&logoColor=white)](https://github.com/aliuyar1234/sparse-capability-transplants/tree/main/paper)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://github.com/aliuyar1234/sparse-capability-transplants/blob/main/pyproject.toml)
[![Evaluation](https://img.shields.io/badge/Evaluation-Deterministic-1F6FEB?style=flat-square)](https://github.com/aliuyar1234/sparse-capability-transplants/blob/main/docs/EVALUATION_PLAN.md)
[![Scope](https://img.shields.io/badge/Scope-Same--Size%20Transfer-5B4B8A?style=flat-square)](https://github.com/aliuyar1234/sparse-capability-transplants/blob/main/docs/RESEARCH_BRIEF.md)

Research code and core documentation for studying same-size capability transfer in function calling.

## Overview

This project asks a focused question:

Can a narrow function-calling capability be transferred from a stronger donor model into an untuned weaker model using a fixed internal module, rather than full recipient retraining?

The public release is intentionally narrow:

- within-family Gemma transfer
- single-turn function calling
- deterministic JSON grading
- same-size transfer only

## Key finding

The final result is not a sparse-superiority paper.

The strongest honest takeaway is:

- same-size transfer is real on this task
- sparse interventions are not noise and show partial localization structure
- dense parameter-matched controls outperform sparse interventions on the matched multiseed comparison

The interesting result is therefore a performance-versus-localization tradeoff, not a clean sparse win.

## Main numbers

The most important headline numbers from the closed `V24` path are:

- donor-gap primary metric: base `0.0375`, donor `0.178125`
- best sparse single-seed frozen eval: `0.2068`
- sparse multiseed mean: `0.1658`
- dense multiseed mean: `0.2177`
- selected `1`-feature subset retained-gain fraction: `0.4708`

## What this repository contains

This repository is meant to be a clean paper-companion codebase. It includes:

- `paper/` - public LaTeX source for the manuscript
- `src/` - model, eval, train, and analysis code
- `scripts/` - thin task wrappers
- `tests/` - unit and integration coverage plus small fixtures
- `docs/` - curated public research, method, and evaluation docs

The public docs to read first are:

1. `docs/RESEARCH_BRIEF.md`
2. `docs/METHOD_SPEC.md`
3. `docs/EVALUATION_PLAN.md`
4. `docs/CLAIMS_MATRIX.md`
5. `docs/REPRODUCIBILITY_AND_ENV.md`

## How to read this repo

If you want the high-level research story:

- start with `docs/RESEARCH_BRIEF.md`
- then read `docs/CLAIMS_MATRIX.md`

If you want the exact method:

- read `docs/METHOD_SPEC.md`

If you want the evaluation contract:

- read `docs/EVALUATION_PLAN.md`

If you want to work with the code:

- start in `src/`
- use `scripts/task.py` as the CLI entrypoint
- use `tests/` for expected behavior and small fixtures

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

## Reproducibility

The repository contains the core method and evaluation code used in the project. Environment and reproduction details are summarized in `docs/REPRODUCIBILITY_AND_ENV.md`.

## Paper framing

If you are reading this repo as a paper companion, the intended interpretation is:

> same-size capability transfer is real in this setting, but the best-performing same-size mechanism is not maximally sparse.

That is the strongest honest claim supported by the finished project.
