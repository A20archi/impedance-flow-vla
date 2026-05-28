r"""Presentation figures from the Stage-1 results (sample efficiency + smoothness + compliance)."""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 13, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 150})

ROOT = "/home/user/Desktop/Saptarshi/impedance_flow_vla"
sweep = json.load(open(f"{ROOT}/out_gate/results_sweep.json"))
gate = json.load(open(f"{ROOT}/out_gate/results_gate.json"))
NS = [10, 25, 50, 100]
COL = {"vanilla": "#888888", "impedance": "#1f77b4"}
LAB = {"vanilla": "vanilla flow", "impedance": "impedance flow (ours)"}


def get(task, arm, n, metric):
    d = gate[f"{task}/{arm}"] if n == 100 else sweep[f"{task}/{arm}/n{n}"]
    return d["mean"][metric], d["sem"][metric]


def curve(ax, task, metric, scale=1.0, ylabel=""):
    for arm in ("vanilla", "impedance"):
        ys = [get(task, arm, n, metric)[0] * scale for n in NS]
        es = [get(task, arm, n, metric)[1] * scale for n in NS]
        ax.errorbar(NS, ys, yerr=es, marker="o", capsize=3, lw=2, color=COL[arm], label=LAB[arm])
    ax.set_xscale("log"); ax.set_xticks(NS); ax.set_xticklabels(NS)
    ax.set_xlabel("# demonstrations"); ax.set_ylabel(ylabel)


# ---- Figure 1: peg-insert-side (PRIMARY) ----
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
T = "peg-insert-side-v3"
curve(ax[0], T, "success", 100, "success rate (%)")
ax[0].set_title("Sample efficiency"); ax[0].legend(loc="lower right")
v10, i10 = get(T, "vanilla", 10, "success")[0]*100, get(T, "impedance", 10, "success")[0]*100
ax[0].annotate(f"+{i10-v10:.0f} pp\n@10 demos", xy=(10, i10), xytext=(13, (i10+v10)/2),
               fontsize=12, fontweight="bold", color=COL["impedance"],
               arrowprops=dict(arrowstyle="->", color=COL["impedance"]))
curve(ax[1], T, "jerk", 1.0, "action jerk  (lower = smoother)")
ax[1].set_title("Smoothness")
curve(ax[2], T, "peakF", 1.0, "peak contact force")
ax[2].set_title("Compliance")
v25, i25 = get(T, "vanilla", 25, "peakF")[0], get(T, "impedance", 25, "peakF")[0]
ax[2].annotate(f"-{100*(v25-i25)/v25:.0f}%\n@matched success", xy=(25, i25), xytext=(28, i25-60),
               fontsize=11, fontweight="bold", color=COL["impedance"])
fig.suptitle("Impedance-shaped flow on peg-insert-side (tight-tolerance contact)  —  matched params, 3 seeds, 30 eval eps", y=1.02, fontsize=13)
fig.tight_layout(); fig.savefig(f"{ROOT}/figs/fig1_peg_sample_efficiency.png", bbox_inches="tight")
print(f"wrote {ROOT}/figs/fig1_peg_sample_efficiency.png")

# ---- Figure 2: door-open (SECONDARY: saturates -> ties on success, smoother) ----
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
T = "door-open-v3"
curve(ax[0], T, "success", 100, "success rate (%)")
ax[0].set_title("door-open: success (saturates → ties)"); ax[0].legend(loc="lower right")
curve(ax[1], T, "jerk", 1.0, "action jerk")
ax[1].set_title("door-open: smoothness")
fig.suptitle("Secondary task (easy from state → no sample-efficiency headroom, mild smoothness edge)", y=1.02, fontsize=12)
fig.tight_layout(); fig.savefig(f"{ROOT}/figs/fig2_door_secondary.png", bbox_inches="tight")
print(f"wrote {ROOT}/figs/fig2_door_secondary.png")
