"""Manuscript Figs. 3-7 from the outputs of scripts/analysis.py.

Run:  python scripts/figs.py   ->  reports/figures/fig3_structural.png ... fig7_sensitivity.png (600 dpi)
Figs. 1 (map) and 2 (architecture diagram) are not data plots and are not produced here.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AUD, FIG = ROOT / "reports/audit", ROOT / "reports/figures"
FIG.mkdir(parents=True, exist_ok=True)
# manuscript labels; company sites are reported by facility type and district only
LABELS = {"S01": "Food plant A", "S02": "Food plant B", "S03": "Corporate office", "S04": "Tongi",
          "S05": "DEPZ Savar", "S06": "Ashulia", "S07": "Narayanganj town", "S08": "Chattogram",
          "S09": "Bogura", "S10": "Cumilla", "S11": "Sylhet", "S12": "Dinajpur"}
ORANGE, BLUE, GREY, LBLUE = "#E4572E", "#1F6FD1", "#999999", "#8FB8E8"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.titleweight": "bold",
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 600})

t5 = pd.read_csv(AUD / "table5_components_scores.csv")
t5["label"] = t5.site_id.map(LABELS)
mc = pd.read_csv(AUD / "table7_monte_carlo.csv")
mc["label"] = mc.site_id.map(LABELS)
order = t5.label.tolist()                     # sorted by revised score R


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)


# Fig. 3 structural diagnostics
corr = pd.read_csv(AUD / "correlations.csv", index_col=0)
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.4), gridspec_kw={"width_ratios": [1, 1, 1.1]})
ax[0].imshow(corr.values, cmap="Blues", vmin=-1, vmax=1)
for (i, j), v in np.ndenumerate(corr.values):
    ax[0].text(j, i, f"{v:.2f}", ha="center", va="center", color="white" if abs(v) > 0.6 else "black", fontsize=7)
ax[0].set_xticks(range(4), corr.columns, rotation=45)
ax[0].set_yticks(range(4), corr.index)
ax[0].set_title("(a) Correlations (Pearson upper, Spearman lower)")
ax[1].scatter(t5.facilities, t5.workers_000, color=BLUE, s=14)
for _, r in t5.iterrows():
    if r.facilities > 150:
        ax[1].annotate(r.label, (r.facilities, r.workers_000), fontsize=6, xytext=(3, 2), textcoords="offset points")
ax[1].set_xlabel("Facilities within 10 km")
ax[1].set_ylabel("Workers within 10 km (thousand)")
ax[1].set_title("(b) Redundant components")
acc = t5.sort_values("access_km")
ax[2].barh(acc.label, acc.access_km, color=[GREY if v == 0 else BLUE for v in acc.access_km])
for y, v in enumerate(acc.access_km):
    ax[2].text(v + 0.3, y, "0 (on gazetteer point)" if v == 0 else f"{v:.1f}", va="center", fontsize=6)
ax[2].set_xlabel("Gazetteer distance (km)")
ax[2].set_title("(c) Access proxy")
save(fig, "fig3_structural.png")

# Fig. 4 weighted contributions
w = np.array([0.30, 0.25, 0.25, 0.20])
mm = lambda s: 100 * (s - s.min()) / (s.max() - s.min())
orig = np.c_[mm(t5.pm25), mm(t5.facilities), mm(t5.workers_000), mm(t5.access_km)] * w
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
left = np.zeros(len(t5))
for k, (name, col) in enumerate(zip(["PM2.5", "Facilities", "Workers", "Access proxy"], [ORANGE, BLUE, LBLUE, GREY])):
    ax[0].barh(t5.label, orig[:, k], left=left, color=col, label=name)
    left += orig[:, k]
ax[0].set_title("(a) Original index")
ax[0].set_xlabel("Original CSHRI (weighted points)")
ax[1].barh(t5.label, 0.5 * t5.E, color=ORANGE, label="E: ambient PM2.5")
ax[1].barh(t5.label, 0.5 * t5.H, left=0.5 * t5.E, color=BLUE, label="H: industrial concentration")
ax[1].set_title("(b) Revised index")
ax[1].set_xlabel("Revised R (weighted points)")
ax[0].invert_yaxis()
for a in ax:
    a.legend(fontsize=6, loc="lower right", frameon=False)
save(fig, "fig4_contributions.png")

# Fig. 5 screening plane and score intervals
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
front = t5.pareto_front.astype(bool)
ax[0].scatter(t5.H[~front], t5.E[~front], color=ORANGE, s=16, label="Dominated")
ax[0].scatter(t5.H[front], t5.E[front], color="black", marker="D", s=22, label="Pareto front")
for _, r in t5.iterrows():
    if r.E != t5.E.mode()[0] or r.pareto_front:
        ax[0].annotate(r.label, (r.H, r.E), fontsize=6, xytext=(3, 3), textcoords="offset points")
ax[0].set_xlabel("H: industrial concentration score")
ax[0].set_ylabel("E: ambient PM2.5 score")
ax[0].set_title("(a) Screening plane")
ax[0].legend(fontsize=6, frameon=False)
m5 = mc.set_index("label").loc[order]
ax[1].errorbar(m5.R, range(len(m5)), xerr=[m5.R - m5.R_p05, m5.R_p95 - m5.R], fmt="o", color=ORANGE, ms=3, lw=1)
ax[1].set_yticks(range(len(m5)), m5.index)
ax[1].invert_yaxis()
ax[1].set_xlabel("Revised R (dot; bar = 5-95% Monte Carlo range)")
ax[1].set_title("(b) Revised scores and uncertainty")
save(fig, "fig5_screening.png")

# Fig. 6 rank intervals, original vs revised
m6 = mc.set_index("label").loc[order]
y = np.arange(len(m6))
fig, ax = plt.subplots(figsize=(5.0, 3.6))
ax.hlines(y - 0.15, m6.rank_p05_original, m6.rank_p95_original, color=BLUE, lw=2.5)
ax.plot(m6.rank_median_original, y - 0.15, "o", color=BLUE, label="Original CSHRI, Dirichlet weights")
ax.hlines(y + 0.15, m6.rank_p05_revised, m6.rank_p95_revised, color=ORANGE, lw=2.5)
ax.plot(m6.rank_median_revised, y + 0.15, "s", color=ORANGE, label="Revised R, structural Monte Carlo")
ax.set_yticks(y, m6.index)
ax.invert_yaxis()
ax.set_xlim(12.5, 0.5)
ax.set_xticks(range(12, 0, -1))
ax.set_xlabel("Rank (1 = highest priority): median and 5-95% range")
ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=2, frameon=False)
save(fig, "fig6_rank_intervals.png")

# Fig. 7 sensitivity diagnostics
sh = pd.read_csv(AUD / "first_order_shares.csv", index_col=0).first_order_share.sort_values()
sw = pd.read_csv(AUD / "pm_error_sweep_p_upper.csv", index_col=0)
sw.index = [LABELS[f"S{i:02d}"] for i in range(1, 13)]          # same row order as sites.csv
loo = pd.read_csv(AUD / "leave_one_out.csv")
loo["label"] = [LABELS[f"S{i:02d}"] for i in range(1, 13)]
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5))
ax[0].barh(sh.index, 100 * sh.values, color=[ORANGE if "pm" in i else GREY if "inter" in i else BLUE for i in sh.index])
for yy, v in enumerate(sh.values):
    ax[0].text(100 * v + 1, yy, f"{100 * v:.0f}%", va="center", fontsize=6)
ax[0].set_xlabel("Share of rank variance (%)")
ax[0].set_title("(a) Factor importance")
for name in ["Ashulia", "Tongi", "DEPZ Savar", "Food plant B", "Bogura", "Chattogram"]:
    ax[1].plot(sw.columns.astype(float), sw.loc[name], marker="o", ms=3,
               color=BLUE if name in ("Ashulia", "Tongi", "DEPZ Savar", "Food plant B") else ORANGE, label=name)
ax[1].set_xlabel("PM2.5 error, lognormal σ")
ax[1].set_ylabel("P(upper tier)")
ax[1].set_title("(b) Robustness to PM2.5 error")
ax[1].legend(fontsize=5, frameon=False)
lo = loo.sort_values("original_max_shift")
ax[2].barh(lo.label, lo.original_max_shift, color=BLUE)
ax[2].set_xlabel("Max score shift after dropping site")
ax[2].set_title("(c) Candidate-set dependence")
ax[2].text(0.98, 0.02, "revised index (fixed goalposts): 0 for all sites", transform=ax[2].transAxes,
           ha="right", fontsize=6, color=ORANGE)
save(fig, "fig7_sensitivity.png")
print("written:", sorted(p.name for p in FIG.glob("fig*.png")))
