"""Single entry point for the four final PICM-L posteriors (TR-2026-412).

Every setting that generated the retained draws in ``draws/`` is read from
``config/final_posteriors.json``: priors, support restrictions, chain
lengths, initial states and random seeds.  This script

1. regenerates the baseline, diffuse, conservative and relaxed-support chains
   from that configuration;
2. compares each regenerated chain with the retained draws, element by element;
3. recomputes the convergence diagnostics and the prior-sensitivity table of
   B10 at 20, 25 and 35 degC from the retained draws; and
4. writes all outputs to ``results/final_posteriors/``.

Usage (from the repository root)::

    python code/run_final_posteriors.py                # all four settings
    python code/run_final_posteriors.py --setting diffuse
    python code/run_final_posteriors.py --check-only   # no sampling
    python code/run_final_posteriors.py --save-chains  # keep regenerated .npy

The sampler below is the adaptive random-walk Metropolis routine of
``code/baseline_analysis.py`` (``run_adaptive_mh_picm_l``), restated without
plotting dependencies.  The arithmetic and the order of the random draws are
kept identical, so the chain paths are regenerated, not merely their
distribution.  Floating-point libraries differ between platforms; agreement
with the retained draws is therefore tested against a small absolute
tolerance and reported, not assumed.

No new sensitivity setting is introduced here.  The script reproduces the
analysis already reported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from analysis_core import (
    classical_rhat,
    effective_sample_size,
    load_dataset,
    weibull_b10,
    write_json,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "config" / "final_posteriors.json"
OUTPUT = ROOT / "results" / "final_posteriors"
SETTINGS = ("baseline", "diffuse", "conservative", "relaxed")
PARAMETERS = ("k", "beta1", "beta0")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def load_config(path: Path = CONFIG_FILE) -> dict:
    """Read the configuration and confirm that it refers to the archived data."""

    config = json.loads(path.read_text(encoding="utf-8"))
    data_file = ROOT / config["data"]["file"]
    digest = hashlib.sha256(data_file.read_bytes()).hexdigest()
    if digest != config["data"]["sha256"]:
        raise RuntimeError(
            f"{data_file} does not match the SHA-256 recorded in {path.name}"
        )
    for relative, expected in config["archive_sha256"].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise ValueError("Retained archive hash mismatch: " + relative)
    sampler = config["sampler"]
    if sampler["thinning"] != 1 or config["data"]["cycle_hours"] != 3.0:
        raise ValueError("The production implementation requires thinning=1 and a 3 h cycle")
    if sampler["burn_in"] + sampler["retained_per_chain"] != sampler["iterations_per_chain"]:
        raise ValueError("burn_in + retained_per_chain must equal iterations_per_chain")
    if len(sampler["seeds"]) != sampler["n_chains"]:
        raise ValueError("One seed is required per chain")
    if len(config["initialisation"]["initial_states"]) != sampler["n_chains"]:
        raise ValueError("One initial state is required per chain")
    return config


def initial_states_from_rule(mle: np.ndarray) -> np.ndarray:
    """Dispersed initial states defined in the configuration's initialisation rule."""

    k, beta1, beta0 = map(float, mle)
    return np.array(
        [
            [k, beta1, beta0],
            [0.8 * k, 1.2 * beta1, beta0 - 1.2],
            [1.2 * k, 0.8 * beta1, beta0 + 1.2],
        ]
    )


# ---------------------------------------------------------------------------
# Posterior density
# ---------------------------------------------------------------------------


class Posterior:
    """Log-posterior of PICM-L under one prior setting.

    ``theta = (k, beta1, beta0)``.  The likelihood is the exact right-censored
    Weibull likelihood with ``log(eta) = beta0 - beta1 log(delta_T)``.
    """

    def __init__(self, data: pd.DataFrame, prior: dict, likelihood_support: dict):
        self.x_log = np.log(data["delta_T_C"].to_numpy(dtype=float))
        self.y = data["cycles"].to_numpy(dtype=float)
        self.event = data["event"].to_numpy(dtype=int)
        self.k_shape = float(prior["k"]["shape"])
        self.k_rate = float(prior["k"]["rate"])
        self.k_lower = float(prior["k"]["support_gt"])
        self.b1_mean = float(prior["beta1"]["mean"])
        self.b1_sd = float(prior["beta1"]["sd"])
        self.b1_lower = float(prior["beta1"]["support_gt"])
        self.b0_mean = float(prior["beta0"]["mean"])
        self.b0_sd = float(prior["beta0"]["sd"])
        self.lik_k_lower = float(likelihood_support["k_gt"])
        self.lik_b1_lower = float(likelihood_support["beta1_gt"])

    def in_prior_support(self, theta: np.ndarray) -> bool:
        return bool(
            (theta[0] > self.k_lower)
            and (theta[1] > self.b1_lower)
            and np.isfinite(theta[2])
        )

    def log_prior(self, theta: np.ndarray) -> float:
        k, beta1, beta0 = (float(v) for v in theta)
        if (k <= self.k_lower) or (beta1 <= self.b1_lower):
            return -np.inf
        scale = 1.0 / self.k_rate
        tail = float(stats.gamma.sf(self.k_lower, a=self.k_shape, scale=scale))
        if tail <= 0.0 or not np.isfinite(tail):
            return -np.inf
        log_k = float(stats.gamma.logpdf(k, a=self.k_shape, scale=scale) - np.log(tail))
        if not np.isfinite(log_k):
            return -np.inf
        truncation = (self.b1_lower - self.b1_mean) / self.b1_sd
        mass = 1.0 - stats.norm.cdf(truncation)
        if mass <= 0.0:
            return -np.inf
        log_b1 = stats.norm.logpdf(beta1, loc=self.b1_mean, scale=self.b1_sd) - np.log(mass)
        log_b0 = stats.norm.logpdf(beta0, loc=self.b0_mean, scale=self.b0_sd)
        total = log_k + log_b1 + log_b0
        return float(total) if np.isfinite(total) else -np.inf

    def log_likelihood(self, theta: np.ndarray) -> float:
        k, beta1, beta0 = (float(v) for v in theta)
        if (k <= self.lik_k_lower) or (beta1 <= self.lik_b1_lower) or not np.isfinite(beta0):
            return -np.inf
        log_eta = beta0 - beta1 * self.x_log
        if not np.all(np.isfinite(log_eta)):
            return -np.inf
        log_ratio = np.log(self.y) - log_eta
        exponent = k * log_ratio
        if np.any(exponent > 700.0):
            return -np.inf
        hazard = np.exp(np.clip(exponent, -745.0, 700.0))
        terms = self.event * (np.log(k) + (k - 1.0) * log_ratio - log_eta) - hazard
        if not np.all(np.isfinite(terms)):
            return -np.inf
        return float(np.sum(terms))

    def __call__(self, theta: np.ndarray) -> float:
        log_prior = self.log_prior(theta)
        if not np.isfinite(log_prior):
            return -np.inf
        log_likelihood = self.log_likelihood(theta)
        if not np.isfinite(log_likelihood):
            return -np.inf
        return float(log_prior + log_likelihood)


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------


def run_chains(posterior: Posterior, starts: np.ndarray, sampler: dict) -> dict:
    """Adaptive random-walk Metropolis; adaptation stops at the end of burn-in."""

    d = 3
    burn_in = int(sampler["burn_in"])
    keep = int(sampler["retained_per_chain"])
    total = burn_in + keep
    block = int(sampler["covariance_update_block"])
    min_accepted = int(sampler["covariance_min_accepted_states"])
    target = float(sampler["target_acceptance"])
    exponent = float(sampler["scale_step_exponent"])
    clip = float(sampler["log_scale_clip"])
    initial_cov = np.diag(np.square(sampler["initial_proposal_covariance_diag_sd"]))
    seeds = [int(seed) for seed in sampler["seeds"]]

    chains = np.zeros((len(seeds), keep, d))
    acceptance = np.zeros(len(seeds))
    low_slope = np.zeros((len(seeds), 2), dtype=int)  # proposals with 0 < beta1 <= 0.5

    for c, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        theta = np.array(starts[c], dtype=float)
        log_post = posterior(theta)
        if not np.isfinite(log_post):
            raise RuntimeError(f"Initial state of chain {c + 1} lies outside the support")

        base_cov = initial_cov.copy()
        log_scale = 0.0
        proposal_cov = math.exp(2.0 * log_scale) * base_cov
        accepted_states: list[np.ndarray] = []
        accepted_after_burn_in = 0
        path = np.zeros((total, d))

        for t in range(total):
            accepted = 0
            proposal = rng.multivariate_normal(mean=theta, cov=proposal_cov)
            if 0.0 < proposal[1] <= 0.5:
                low_slope[c, 0 if t < burn_in else 1] += 1
            if posterior.in_prior_support(proposal):
                log_post_proposal = posterior(proposal)
                if np.isfinite(log_post_proposal):
                    if np.log(rng.uniform()) < log_post_proposal - log_post:
                        theta = proposal
                        log_post = log_post_proposal
                        accepted = 1
            path[t] = theta

            if t < burn_in:
                if accepted:
                    accepted_states.append(theta.copy())
                step = float((t + 1) ** -exponent)
                log_scale = float(np.clip(log_scale + step * (accepted - target), -clip, clip))
                if ((t + 1) % block == 0) and len(accepted_states) >= min_accepted:
                    empirical = np.cov(np.array(accepted_states).T, ddof=1)
                    empirical = 0.5 * (empirical + empirical.T)
                    smallest = float(np.min(np.linalg.eigvalsh(empirical)))
                    if smallest <= 0.0:
                        empirical = empirical + (abs(smallest) + 1e-8) * np.eye(d)
                    base_cov = (2.38**2 / d) * (empirical + 1e-8 * np.eye(d))
                    base_cov = 0.5 * (base_cov + base_cov.T)
                proposal_cov = math.exp(2.0 * log_scale) * base_cov
                proposal_cov = 0.5 * (proposal_cov + proposal_cov.T)
            else:
                accepted_after_burn_in += accepted

        chains[c] = path[burn_in:]
        acceptance[c] = accepted_after_burn_in / keep

    return {"chains": chains, "acceptance": acceptance, "low_slope_proposals": low_slope}


# ---------------------------------------------------------------------------
# Diagnostics and summaries
# ---------------------------------------------------------------------------


def rank_normalised_split_rhat(chains: np.ndarray) -> np.ndarray:
    """Rank-normalised split R-hat (bulk and folded; the larger is returned)."""

    def split_rhat(values: np.ndarray) -> float:
        half = values.shape[1] // 2
        parts = np.concatenate([values[:, :half], values[:, half : 2 * half]], axis=0)
        n = parts.shape[1]
        within = float(np.mean(np.var(parts, axis=1, ddof=1)))
        between = float(n * np.var(np.mean(parts, axis=1), ddof=1))
        return math.sqrt(((n - 1.0) / n * within + between / n) / within)

    def normal_scores(values: np.ndarray) -> np.ndarray:
        ranks = stats.rankdata(values.ravel(), method="average").reshape(values.shape)
        return stats.norm.ppf((ranks - 0.375) / (values.size + 0.25))

    output = np.zeros(chains.shape[2])
    for j in range(chains.shape[2]):
        values = chains[:, :, j]
        folded = np.abs(values - np.median(values))
        output[j] = max(split_rhat(normal_scores(values)), split_rhat(normal_scores(folded)))
    return output


def median_mcse(values: np.ndarray) -> float:
    """Monte Carlo standard error of a pooled median by batch means.

    Each chain is cut into 20 consecutive batches of equal length; the
    standard deviation of the batch medians, divided by the square root of
    the number of batches, estimates the standard error of the pooled median.
    """

    batches = np.median(values.reshape(values.shape[0], 20, -1), axis=2).ravel()
    return float(np.std(batches, ddof=1) / math.sqrt(batches.size))


def summarise(setting: str, chains: np.ndarray, excursions: list[float]) -> list[dict]:
    rows = []
    for excursion in excursions:
        b10 = weibull_b10(chains, excursion)
        q025, median, q975 = np.quantile(b10.reshape(-1), [0.025, 0.5, 0.975])
        rows.append(
            {
                "setting": setting,
                "delta_T": excursion,
                "n_draws": int(b10.size),
                "q025": float(q025),
                "median": float(median),
                "q975": float(q975),
                "median_mcse": median_mcse(b10),
                "beta1_at_or_below_0p5": int(np.count_nonzero(chains[:, :, 1] <= 0.5)),
            }
        )
    return rows


def diagnostics(setting: str, chains: np.ndarray) -> list[dict]:
    classical = classical_rhat(chains)
    split = rank_normalised_split_rhat(chains)
    return [
        {
            "setting": setting,
            "parameter": name,
            "rhat_classical": float(classical[j]),
            "rhat_rank_normalised_split": float(split[j]),
            "ess": effective_sample_size(chains[:, :, j]),
        }
        for j, name in enumerate(PARAMETERS)
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--setting", choices=SETTINGS, action="append",
                        help="prior setting to run (repeatable); default is all four")
    parser.add_argument("--check-only", action="store_true",
                        help="skip sampling; recompute summaries from the retained draws")
    parser.add_argument("--save-chains", action="store_true",
                        help="also write the regenerated chains as .npy files")
    parser.add_argument("--output", type=Path, help="separate output directory for this audit")
    arguments = parser.parse_args()
    global OUTPUT
    OUTPUT = arguments.output.resolve() if arguments.output else (
        ROOT / "results" / ("retained_summary_check" if arguments.check_only else "final_posteriors")
    )
    if OUTPUT == ROOT / "draws" or ROOT / "draws" in OUTPUT.parents:
        raise ValueError("Outputs must not overwrite the retained publication draws")

    config = load_config()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = load_dataset(ROOT / config["data"]["file"])
    sampler = config["sampler"]
    excursions = [float(v) for v in config["reporting"]["delta_T_degC"]]
    tolerance = config["tolerances"]
    starts = np.array(config["initialisation"]["initial_states"], dtype=float)
    failures: list[str] = []

    rule_gap = float(np.max(np.abs(
        starts - initial_states_from_rule(np.array(config["initialisation"]["mle"]))
    )))
    if rule_gap > 1e-12:
        failures.append(f"initial states differ from the stated rule by {rule_gap:.3g}")

    selected = tuple(arguments.setting) if arguments.setting else SETTINGS
    retained = {
        name: np.load(ROOT / config["prior_settings"][name]["retained_draws"])
        for name in SETTINGS
    }
    expected_shape = (sampler["n_chains"], sampler["retained_per_chain"], 3)
    for name, chains in retained.items():
        if chains.shape != tuple(expected_shape):
            raise ValueError(f"{name}: retained draws have shape {chains.shape}")
        prior = config["prior_settings"][name]
        if (not np.isfinite(chains).all() or
            not (chains[:, :, 0] > prior["k"]["support_gt"]).all() or
            not (chains[:, :, 1] > prior["beta1"]["support_gt"]).all()):
            raise ValueError(f"{name}: invalid retained draw or support violation")

    regeneration_rows = []
    if not arguments.check_only:
        for name in selected:
            started = time.time()
            posterior = Posterior(
                data, config["prior_settings"][name], config["model"]["likelihood_support"]
            )
            run = run_chains(posterior, starts, sampler)
            if arguments.save_chains:
                np.save(OUTPUT / f"{name}_chains_regenerated.npy", run["chains"])
            for c in range(sampler["n_chains"]):
                gap = float(np.max(np.abs(run["chains"][c] - retained[name][c])))
                reproduced = gap <= tolerance["chain_path_abs"]
                regeneration_rows.append(
                    {
                        "setting": name,
                        "chain": c + 1,
                        "seed": sampler["seeds"][c],
                        "acceptance_rate": float(run["acceptance"][c]),
                        "max_abs_difference_from_retained": gap,
                        "path_reproduced": bool(reproduced),
                        "exact_array_match": bool(np.array_equal(run["chains"][c], retained[name][c])),
                        "proposals_0_lt_beta1_le_0p5_burn_in": int(run["low_slope_proposals"][c, 0]),
                        "proposals_0_lt_beta1_le_0p5_retained": int(run["low_slope_proposals"][c, 1]),
                    }
                )
            gaps = [row["max_abs_difference_from_retained"] for row in regeneration_rows
                    if row["setting"] == name]
            print(f"{name:13s} regenerated in {time.time() - started:5.1f} s; "
                  f"max |regenerated - retained| = {max(gaps):.3e}")
            if max(gaps) > tolerance["chain_path_abs"]:
                failures.append(f"{name}: regenerated path differs beyond the configured tolerance")
                # A different floating-point library can change one accept/reject
                # decision, after which the path separates.  The posterior is then
                # compared through its summaries, within Monte Carlo error.
                for row, reference in zip(
                    summarise(name, run["chains"], excursions),
                    summarise(name, retained[name], excursions),
                ):
                    limit = tolerance["summary_fallback_mcse_multiple"] * reference["median_mcse"]
                    if abs(row["median"] - reference["median"]) > limit:
                        failures.append(
                            f"{name}: regenerated B10({row['delta_T']:g}) median outside "
                            f"{tolerance['summary_fallback_mcse_multiple']:g} MCSE of the retained value"
                        )
                print(f"{name:13s} path differs beyond {tolerance['chain_path_abs']:g}; "
                      "summaries compared within Monte Carlo error")
        pd.DataFrame(regeneration_rows).to_csv(OUTPUT / "regeneration_check.csv", index=False)

    summary_rows, diagnostic_rows = [], []
    for name in SETTINGS:
        summary_rows.extend(summarise(name, retained[name], excursions))
        diagnostic_rows.extend(diagnostics(name, retained[name]))
    summary = pd.DataFrame(summary_rows)
    baseline = summary[summary.setting == "baseline"].set_index("delta_T")
    summary["median_change_percent"] = [
        100.0 * (row.median / baseline.loc[row.delta_T, "median"] - 1.0)
        for row in summary.itertuples()
    ]
    summary["width_ratio"] = [
        (row.q975 - row.q025)
        / (baseline.loc[row.delta_T, "q975"] - baseline.loc[row.delta_T, "q025"])
        for row in summary.itertuples()
    ]
    summary["median_mcse_percent"] = 100.0 * summary["median_mcse"] / summary["median"]
    summary.to_csv(OUTPUT / "prior_sensitivity.csv", index=False)
    pd.DataFrame(diagnostic_rows).to_csv(OUTPUT / "mcmc_diagnostics.csv", index=False)

    for row in summary.itertuples():
        expected = config["expected_b10_cycles"][row.setting][f"{row.delta_T:g}"]
        gap = max(abs(row.q025 - expected[0]), abs(row.median - expected[1]),
                  abs(row.q975 - expected[2]))
        if gap > tolerance["summary_from_retained_abs_cycles"]:
            failures.append(f"{row.setting}: B10({row.delta_T:g}) differs from the reported value")

    shared = {
        f"chain_{c + 1}": bool(np.array_equal(retained["relaxed"][c], retained["baseline"][c]))
        for c in range(sampler["n_chains"])
    }
    write_json(
        OUTPUT / "run_record.json",
        {
            "configuration": str(CONFIG_FILE.relative_to(ROOT)),
            "configuration_sha256": hashlib.sha256(CONFIG_FILE.read_bytes()).hexdigest(),
            "mode": "check-only" if arguments.check_only else "regenerate",
            "settings_run": [] if arguments.check_only else list(selected),
            "exact_regeneration": None if arguments.check_only else all(row["exact_array_match"] for row in regeneration_rows),
            "regeneration_rows": regeneration_rows,
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
            "pandas": pd.__version__,
            "relaxed_chain_identical_to_baseline": shared,
            "failures": failures,
        },
    )

    table = summary.pivot(index="delta_T", columns="setting", values="median")[list(SETTINGS)]
    print("\nPosterior-median B10 (cycles) from the retained draws")
    print(table.round(0).astype(int).to_string())
    print("\nInterval-width ratio relative to baseline")
    print(summary.pivot(index="delta_T", columns="setting",
                        values="width_ratio")[list(SETTINGS)].round(3).to_string())
    worst = pd.DataFrame(diagnostic_rows)
    print(f"\nLargest classical R-hat {worst.rhat_classical.max():.4f}; "
          f"largest rank-normalised split R-hat {worst.rhat_rank_normalised_split.max():.4f}; "
          f"smallest ESS {worst.ess.min():.1f}")
    print(f"Outputs: {OUTPUT}")
    if failures:
        print("\nFAILED CHECKS")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
