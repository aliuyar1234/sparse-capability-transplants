# Same-Size Capability Transfer Reveals a Performance-Localization Tradeoff in Deterministic Function Calling

[![Paper PDF](https://img.shields.io/badge/Paper-PDF-B31B1B?style=flat-square&logo=adobeacrobatreader&logoColor=white)](https://github.com/aliuyar1234/sparse-capability-transplants/raw/main/ali-uyar-same-size-capability-transfer.pdf)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19226753.svg)](https://doi.org/10.5281/zenodo.19226753)
[![Manuscript Source](https://img.shields.io/badge/LaTeX-source-1D4ED8?style=flat-square&logo=latex&logoColor=white)](paper/main.tex)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white)](pyproject.toml)
[![Evaluation](https://img.shields.io/badge/Evaluation-Deterministic-1F6FEB?style=flat-square)](docs/EVALUATION_PLAN.md)
[![Scope](https://img.shields.io/badge/Scope-Same--Size%20Transfer-5B4B8A?style=flat-square)](#scope)

Ali Uyar
Independent Researcher

**Paper title:** *Same-Size Capability Transfer Reveals a Performance-Localization Tradeoff in Deterministic Function Calling*

This repository accompanies a methods paper on constructive capability transfer. It asks whether a narrow function-calling capability can be transplanted from a stronger donor into an untuned same-size base model using a fixed internal module instead of full recipient retraining, and how sparse, dense, and steering interventions compare under matched parameter budgets.

## Abstract

We study whether task-specific function-calling behavior can be transplanted from a donor back into an untuned same-size base model using fixed modules instead of full retraining. Our setting is intentionally narrow and fully deterministic: within-family Gemma models, single-turn function calling, frozen manifests, exact JSON grading, and a primary metric defined on SchemaShift and NoCall stress slices. The donor substantially outperforms the base on the primary strict metric, from 0.0375 to 0.1781 (delta 0.1406, 95% CI [0.1208, 0.1604]). A sparse same-size intervention selected by a locked discovery pipeline reaches 0.2068 strict success on the discovery seed and 0.1658 mean strict success across three seeds, while a matched dense shortcut reaches 0.2177 mean strict success across the same seeds. Sparse structure is nevertheless non-random: a locked one-feature subset retains 47.1% of the full sparse gain and beats random one-feature controls, and one-vector steering is clearly weaker at 0.1078 strict success. The resulting picture is therefore not sparse superiority. Instead, we find a real performance-localization tradeoff: same-size transfer is real in this setting, dense parameter-matched modules recover more raw task performance, and sparse modules provide partial localization signal. We present this deterministic function-calling setup as a concrete testbed for studying constructive capability transfer under tight claim discipline.

## Main Finding

The final result is not a sparse-superiority paper. Same-size transfer is real on this task, and the best-performing same-size mechanism is not maximally sparse.

The most important headline numbers from the closed `V24` path on the primary strict metric:

| Configuration | Strict success |
| ------------- | -------------- |
| base (untuned)                                    | 0.0375 |
| donor (task-trained same-size)                    | 0.1781 |
| sparse intervention, best single-seed frozen eval | 0.2068 |
| sparse intervention, multiseed mean               | 0.1658 |
| dense parameter-matched shortcut, multiseed mean  | **0.2177** |
| one-vector steering control                       | 0.1078 |
| locked 1-feature subset, retained-gain fraction   | 0.4708 |

- Same-size transfer is real on this task.
- Sparse interventions are not noise and show partial localization structure.
- Dense parameter-matched controls outperform sparse interventions on the matched multiseed comparison.

The interesting result is therefore a performance-versus-localization tradeoff, not a clean sparse win.

## Contributions

1. A locked same-size transfer testbed for deterministic single-turn function calling with frozen manifests, exact JSON grading, and explicit SchemaShift and NoCall stress slices.
2. Evidence that same-size transfer is real in this setting, with a donor gap of 0.1406 on the primary metric and positive sparse transfer across three seeds.
3. A matched comparison of sparse, dense, and steering interventions at the same hook site under effectively identical parameter budgets.
4. An honest negative result: dense parameter-matched modules win the final multiseed comparison, while sparse modules retain partial localization signal through pruning and random-subset control analyses.

## Scope

This public release is intentionally narrow and claim-safe.

- within-family Gemma transfer
- single-turn function calling
- deterministic JSON grading
- same-size transfer only

We do not claim cross-scale transfer, recipient data efficiency, universal model editing, uniquely identified sparse circuits, or clean schema-shift semantic transfer. The contribution is narrower and, we think, cleaner: a well-measured same-size transfer result that reveals a real tradeoff.

## Paper

- Compiled PDF: [`ali-uyar-same-size-capability-transfer.pdf`](ali-uyar-same-size-capability-transfer.pdf)
- LaTeX source: [`paper/main.tex`](paper/main.tex)
- Main results tables: [`paper/tables/main_results.tex`](paper/tables/main_results.tex), [`paper/tables/localization.tex`](paper/tables/localization.tex)

If you are reading this repo as a paper companion, the intended interpretation is:

> same-size capability transfer is real in this setting, but the best-performing same-size mechanism is not maximally sparse.

## Repository Layout

- [`paper/`](paper/) — public LaTeX source for the manuscript
- [`src/`](src/) — model, eval, train, and analysis code
- [`scripts/`](scripts/) — thin task wrappers
- [`tests/`](tests/) — unit and integration coverage plus small fixtures
- [`docs/`](docs/) — curated public research, method, and evaluation docs

Public docs to read first:

1. [`docs/RESEARCH_BRIEF.md`](docs/RESEARCH_BRIEF.md)
2. [`docs/METHOD_SPEC.md`](docs/METHOD_SPEC.md)
3. [`docs/EVALUATION_PLAN.md`](docs/EVALUATION_PLAN.md)
4. [`docs/CLAIMS_MATRIX.md`](docs/CLAIMS_MATRIX.md)
5. [`docs/REPRODUCIBILITY_AND_ENV.md`](docs/REPRODUCIBILITY_AND_ENV.md)

## Reproducibility

This repo targets Python `3.11`. The repository contains the core method and evaluation code used in the project. Environment and reproduction details are summarized in [`docs/REPRODUCIBILITY_AND_ENV.md`](docs/REPRODUCIBILITY_AND_ENV.md).

Typical setup:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check src tests scripts
```

## Citation

```bibtex
@misc{uyar2026samesizecapabilitytransfer,
  author = {Uyar, Ali},
  title  = {Same-Size Capability Transfer Reveals a Performance--Localization Tradeoff in Deterministic Function Calling},
  year   = {2026},
  doi    = {10.5281/zenodo.19226753},
  url    = {https://doi.org/10.5281/zenodo.19226753},
  note   = {Independent research}
}
```

Machine-readable citation metadata is also available in [`CITATION.cff`](CITATION.cff).
