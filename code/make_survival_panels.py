"""Stress-wise survival panels (manuscript Fig. 3) from the retained draws.

Each panel shows the Kaplan-Meier estimate for one excursion level together
with the PICM-L posterior median, the pointwise 95% credible band and the
PICM-L maximum-likelihood curve.

Kaplan-Meier convention.  For ordered distinct failure times t_(1) < ... with
d_j failures and n_j units at risk just before t_(j),

    S_KM(t) = prod_{j : t_(j) <= t} (1 - d_j / n_j).

The estimate is a right-continuous step function that is constant between
failure times.  It is defined up to the largest observation time in the
group, so the last step is drawn as a horizontal plateau to the
administrative censoring time when run-outs remain.  Censor marks are placed
on that plateau.  A group with no failure has S_KM(t) = 1 up to its
censoring time.

Usage (from the repository root)::

    python code/make_survival_panels.py

Outputs: results/figures/main_survival_panels_pub.png (600 dpi) and .pdf,
and results/figures/kaplan_meier_table.csv with the plotted step values.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from analysis_core import DRAWS_DIR, RESULTS_DIR, fit_primary_mle, load_dataset

OUTPUT = RESULTS_DIR / "figures"
STEM = "main_survival_panels_pub"
WIDTH_IN, HEIGHT_IN = 7.16, 4.20  # IEEE double-column width; height set for the 15-page layout

COLOURS = {
    "median": "#355c9a",
    "band": "#d6e4f5",
    "km": "#8f4b3e",
    "mle": "#3f4752",
    "ink": "#2f3742",
}


def kaplan_meier(cycles: np.ndarray, event: np.ndarray) -> dict[str, np.ndarray]:
    """Kaplan-Meier estimate as a complete step path.

    Returns ``step_t`` and ``step_s`` for plotting with ``where="post"``.  The
    path starts at (0, 1) and ends at the largest observation time, so the
    last survival level is carried to the censoring time.  ``censor_t`` and
    ``censor_s`` give the censor marks on the estimate.
    """

    cycles = np.asarray(cycles, dtype=float)
    event = np.asarray(event, dtype=int)
    at_risk = cycles.size
    survival = 1.0
    step_t, step_s = [0.0], [1.0]
    for time in np.unique(cycles):
        failed = int(np.sum((cycles == time) & (event == 1)))
        censored = int(np.sum((cycles == time) & (event == 0)))
        if failed > 0:
            survival *= 1.0 - failed / at_risk
            step_t.append(float(time))
            step_s.append(float(survival))
        at_risk -= failed + censored

    last_observation = float(np.max(cycles))
    if step_t[-1] < last_observation:
        step_t.append(last_observation)
        step_s.append(step_s[-1])

    step_t_array, step_s_array = np.array(step_t), np.array(step_s)
    censor_t = np.sort(cycles[event == 0])
    index = np.searchsorted(step_t_array, censor_t, side="right") - 1
    return {
        "step_t": step_t_array,
        "step_s": step_s_array,
        "censor_t": censor_t,
        "censor_s": step_s_array[index],
    }


def set_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "font.weight": "bold",
            "axes.labelsize": 9.0,
            "axes.titlesize": 8.9,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "axes.titlepad": 4.5,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "axes.grid": True,
            "axes.edgecolor": COLOURS["ink"],
            "axes.labelcolor": COLOURS["ink"],
            "xtick.color": COLOURS["ink"],
            "ytick.color": COLOURS["ink"],
            "text.color": COLOURS["ink"],
            "axes.axisbelow": True,
            "grid.alpha": 0.10,
            "grid.color": "#9ca3af",
            "grid.linewidth": 0.45,
            "axes.linewidth": 0.9,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    set_style()

    data = load_dataset()
    draws = np.load(DRAWS_DIR / "baseline_chains.npy").reshape(-1, 3)
    k_mle, beta1_mle, beta0_mle = fit_primary_mle(data)["theta"]

    figure, axes = plt.subplots(2, 3, figsize=(WIDTH_IN, HEIGHT_IN), sharey=True)
    table_rows = []

    for index, (excursion, group) in enumerate(data.groupby("delta_T_C", sort=True)):
        axis = axes.ravel()[index]
        cycles = group["cycles"].to_numpy(dtype=float)
        event = group["event"].to_numpy(dtype=int)
        x_max = 1.15 * float(np.max(cycles))
        grid = np.linspace(1.0e-3, x_max, 250)

        eta = np.exp(draws[:, 2] - draws[:, 1] * np.log(excursion))[:, None]
        survival = np.exp(-((grid[None, :] / eta) ** draws[:, 0][:, None]))
        lower, median, upper = np.quantile(survival, [0.025, 0.5, 0.975], axis=0)
        axis.fill_between(grid, lower, upper, color=COLOURS["band"], alpha=0.58, zorder=1)
        axis.plot(grid, median, color=COLOURS["median"], lw=2.1, zorder=2)

        eta_mle = np.exp(beta0_mle - beta1_mle * np.log(excursion))
        axis.plot(grid, np.exp(-((grid / eta_mle) ** k_mle)),
                  color=COLOURS["mle"], lw=1.6, ls="--", zorder=3)

        km = kaplan_meier(cycles, event)
        axis.step(km["step_t"], km["step_s"], where="post",
                  color=COLOURS["km"], lw=1.7, zorder=4)
        if km["censor_t"].size:
            axis.plot(km["censor_t"], km["censor_s"], linestyle="None", marker="+",
                      color=COLOURS["km"], ms=7.5, mew=1.4, zorder=5, clip_on=False)
        for time, level in zip(km["step_t"], km["step_s"]):
            table_rows.append({"delta_T_C": excursion, "cycles": time, "km_survival": level})

        failures = int(event.sum())
        censored = event.size - failures
        if failures == event.size:  # complete-failure panels: the upper right is clear
            axis.text(0.98, 0.96, f"{failures}/8 failed\n{censored}/8 censored",
                      transform=axis.transAxes, fontsize=8.5, ha="right", va="top",
                      linespacing=1.25)
        else:
            axis.text(0.04, 0.08, f"{failures}/8 failed, {censored}/8 censored",
                      transform=axis.transAxes, fontsize=8.5)
        axis.text(0.02, 0.95, f"({'abcdef'[index]})", transform=axis.transAxes,
                  fontsize=10, fontweight="bold", va="top")
        axis.set_title(f"Delta T = {int(excursion)} deg C")
        axis.set_xlabel("Cycles")
        if index % 3 == 0:
            axis.set_ylabel("Survival probability")
        axis.set_xlim(0.0, x_max)
        axis.set_ylim(0.0, 1.02)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        axis.tick_params(axis="both", which="major", length=3.8, width=0.85)
        for label in axis.get_xticklabels() + axis.get_yticklabels():
            label.set_fontweight("bold")

    handles = [
        Line2D([], [], color=COLOURS["km"], lw=1.7, label="Kaplan-Meier estimate"),
        Line2D([], [], color=COLOURS["km"], linestyle="None", marker="+", ms=7.5, mew=1.4,
               label="Censored unit (run-out)"),
        Line2D([], [], color=COLOURS["median"], lw=2.1, label="PICM-L posterior median"),
        Patch(facecolor=COLOURS["band"], alpha=0.58, label="PICM-L 95% CrI"),
        Line2D([], [], color=COLOURS["mle"], lw=1.6, ls="--", label="PICM-L MLE"),
    ]
    figure.tight_layout(rect=[0.035, 0.035, 0.985, 0.90], pad=0.45)
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.992),
                  ncol=3, fontsize=7.6, frameon=False, columnspacing=1.6,
                  handlelength=1.9, handletextpad=0.5, borderaxespad=0.2)

    figure.savefig(OUTPUT / f"{STEM}.png", dpi=600, bbox_inches="tight", pad_inches=0.08)
    figure.savefig(OUTPUT / f"{STEM}.pdf", bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)
    pd.DataFrame(table_rows).to_csv(OUTPUT / "kaplan_meier_table.csv", index=False)

    km75 = kaplan_meier(*(data.loc[data["delta_T_C"] == 75.0, column].to_numpy()
                          for column in ("cycles", "event")))
    assert abs(km75["step_t"][-1] - 3200.0 / 3.0) < 1e-9, "75 degC plateau must reach the censoring time"
    assert abs(km75["step_s"][-1] - 0.375) < 1e-12, "75 degC plateau level must be 3/8"
    assert np.allclose(km75["censor_s"], 0.375), "75 degC censor marks must lie on the plateau"
    print(f"Figure written to {OUTPUT / (STEM + '.png')}")
    print(f"75 degC: last failure at {km75['step_t'][-2]:.1f} cycles; plateau "
          f"S = {km75['step_s'][-1]:.3f} to {km75['step_t'][-1]:.1f} cycles")


if __name__ == "__main__":
    main()
