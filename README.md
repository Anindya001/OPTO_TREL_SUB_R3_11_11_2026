# Optocoupler thermal-cycling reliability — R3

Computational archive for IEEE Transactions on Reliability manuscript TR-2026-412.R3.
It reproduces the statistical analysis, the twelve tables and the eight
code-generated figures from the 48 recorded lifetimes.

## Install

Python 3.12 or later (tested with 3.12.14 and 3.13.18). From the repository root:

```bash
python -m pip install -r requirements-figures.txt
```

## Run

Each command writes to a new, empty output folder and never changes the inputs.

| Command | What it does | Time* |
|---|---|---|
| `python code/reproduce_all.py --from-retained --output out_quick` | Tables and figures from the retained posterior draws | < 1 min |
| `python code/reproduce_all.py --output out_main` | Also regenerates the four posterior settings and compares them with the retained draws | about 3 min |
| `python code/reproduce_all.py --secondary --output out_full` | Also regenerates all secondary analyses (comparator models, group bootstrap, LOSO, Firth-type fit, Bartlett check) | about 4 min |

\*On a two-core machine.

The run stops with a non-zero exit status if any step or check fails. Each
output folder contains `execution_report.json` and a log for every step.

## Check

```bash
python code/verify_package.py      # files present, hashes valid
python code/verify_results.py      # 45 statistical checks
python code/test_revision.py       # Kaplan–Meier display checks
python code/independent_checks.py --nodes 160   # independent numerical integration
```

## Outputs

| Folder in the output | Contents |
|---|---|
| `tables/` | `Table_01.csv` … `Table_12.csv`, numbered as in the manuscript, and `table_manifest.json` |
| `figures/` | Figures 2, 3 and 6–11 as PDF and PNG |
| `posterior/` | Regenerated chains, comparison with the retained draws, diagnostics |
| `secondary/` | Secondary analyses and `secondary_verification.json` (with `--secondary`) |
| `numerical/` | Numerical summaries quoted in the text |

Figures 1, 4 and 5 show the experimental set-up, recorded traces and X-ray images.
They are not produced by code and are not included.

## Repository contents

| Folder | Contents |
|---|---|
| `data/` | `optocoupler_ttf_unit_level.csv`: 48 devices, six excursion levels, 29 failures and 19 run-outs (`event` = 1 failure, 0 run-out). Cycles = `ttf_hours` / 3. |
| `config/` | Priors, supports, chain lengths, initial states and seeds of the four prior settings (`final_posteriors.json`); secondary-analysis settings (`publication_analysis.json`); figure index (`figure_manifest.json`) |
| `draws/` | Retained posterior draws for the four prior settings and the two comparator models |
| `code/` | Analysis, figure and table scripts |
| `figures/` | Publication PDFs of the eight code-generated figures |
| `results/` | Retained results used by the checks, and the regeneration records |
| `reference_tables/` | Retained values used by the regeneration checks |

`code/run_final_posteriors.py` with `config/final_posteriors.json` is the single entry
point for the four prior settings reported in the manuscript. The shorter
sensitivity routine built into `baseline_analysis.py` is not used for them.

## Records

`VERIFICATION.md` summarises the regeneration runs and their checks. The run
records are in `results/final_posteriors/`, `results/publication/` and
`results/full_regeneration/`.
`SHA256SUMS.txt` lists the hash of every file. `CITATION.cff` gives the citation for this archive.
