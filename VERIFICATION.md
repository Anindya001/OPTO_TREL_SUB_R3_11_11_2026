# Verification record — R3, 11 November 2026

## Final from-scratch run

`python code/reproduce_all.py --secondary`, run from a clean copy of this repository.
Python 3.13.18; NumPy 2.3.5, SciPy 1.17.0, pandas 2.2.3, matplotlib 3.10.8.
Record: `results/full_regeneration/`.

| Check | Result |
|---|---|
| Four posterior settings (12 chains) | Regenerated; identical to the retained draws (maximum difference 0) |
| Statistical checks | 45 of 45 passed |
| Secondary analyses | Regenerated; 10 of 10 checks passed (comparator replay, group bootstrap, LOSO, Firth-type fit, Bartlett and delta routes) |
| Tables 1–12 | Agree with the published values (largest relative difference 5.5e-7) |
| Figures 2, 3, 6–11 | Rendered identically to `figures/` |

## Other regeneration records

| Record | Platform | Result |
|---|---|---|
| `results/final_posteriors/` | Python 3.12.14 | Four posterior settings regenerated exactly; 45 of 45 checks passed |
| `results/publication/` | Python 3.12.14, different machine | Regenerated within 3.535e-13 of the retained draws; 45 of 45 checks passed |

Each record holds the run record, the per-chain comparison with the retained draws and
the statistical checks; `results/final_posteriors/` also holds the MCMC diagnostics.
Matching package versions do not guarantee bit-identical floating-point results on every
platform. All reported summaries are unchanged at the stated precision.

## Firth-type fit

The Firth-type fit uses the analytic observed information and multi-start BFGS, with a
stationarity check and a Nelder–Mead confirmation. It gives k = 3.038, beta1 = 2.416,
beta0 = 17.561 and B10 at 35 °C = 3749.6 cycles.

## Scope

The repository reproduces the computation from the 48 recorded lifetimes. It does not
reconstruct exposure times from the raw electrical logs or reproduce the experiments
and X-ray images.
