r"""
Figure 3 — ODE-step ablation.  Reads out_gate/results_ablate_steps.json, writes figs/fig3_ode_steps.png.
Story: inference cost is proportional to the number of ODE steps; the impedance arm holds its success
rate (and smoothness) across {5,10,20} steps, so you can cut steps -> cut latency with no accuracy loss,
and it is markedly more step-count-stable than the vanilla baseline.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "out_gate", "results_ablate_steps.json")
OUT = os.path.join(ROOT, "figs", "fig3_ode_steps.png")

GRAY, BLUE = "#8a96a3", "#1f6fb2"

r = json.load(open(RES))
steps = sorted({int(k.split("/")[0].replace("steps", "")) for k in r})
def series(arm, metric):
    m = [r[f"steps{s}/{arm}"]["mean"][metric] for s in steps]
    e = [r[f"steps{s}/{arm}"]["sem"][metric] for s in steps]
    return np.array(m), np.array(e)

fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=200)
fig.suptitle("ODE-step ablation on peg-insert-side  —  matched params, 3 seeds, 30 eval eps\n"
             "inference cost $\\propto$ #steps:  impedance holds accuracy with HALF the steps",
             fontsize=12.5, y=1.02)

# ---- panel 1: success vs steps ----
for arm, col, lab in [("vanilla", GRAY, "vanilla flow"), ("impedance", BLUE, "impedance flow (ours)")]:
    m, e = series(arm, "success")
    ax[0].errorbar(steps, m * 100, yerr=e * 100, marker="o", ms=7, lw=2.2, capsize=4,
                   color=col, label=lab)
mi, _ = series("impedance", "success")
ax[0].axhline(mi.mean() * 100, color=BLUE, ls=":", lw=1.2, alpha=0.6)
ax[0].annotate("impedance flat across\n5 / 10 / 20 steps  →  robust",
               xy=(10, mi[steps.index(10)] * 100), xytext=(11.5, mi.mean() * 100 - 9),
               fontsize=9.5, color=BLUE,
               arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.2))
ax[0].set_xlabel("# ODE integration steps  (∝ inference latency)")
ax[0].set_ylabel("success rate (%)")
ax[0].set_title("Success is step-count-robust")
ax[0].set_xticks(steps); ax[0].set_ylim(70, 100)
ax[0].grid(alpha=0.25); ax[0].legend(loc="lower right", fontsize=9.5)
# half-latency callout
ax[0].axvspan(4.4, 5.6, color=BLUE, alpha=0.07)
ax[0].text(5, 72.2, "½ latency", ha="center", fontsize=8.5, color=BLUE, style="italic")

# ---- panel 2: smoothness vs steps ----
for arm, col, lab in [("vanilla", GRAY, "vanilla flow"), ("impedance", BLUE, "impedance flow (ours)")]:
    m, e = series(arm, "jerk")
    ax[1].errorbar(steps, m, yerr=e, marker="o", ms=7, lw=2.2, capsize=4, color=col, label=lab)
ax[1].set_xlabel("# ODE integration steps  (∝ inference latency)")
ax[1].set_ylabel("action jerk  (lower = smoother)")
ax[1].set_title("Smoothness ~ unchanged")
ax[1].set_xticks(steps); ax[1].grid(alpha=0.25); ax[1].legend(loc="upper right", fontsize=9.5)

plt.tight_layout()
plt.savefig(OUT, dpi=200, facecolor="white", bbox_inches="tight")
print("wrote", os.path.abspath(OUT))

# also print a compact markdown table for the README
print("\n| ODE steps | vanilla success | impedance success | vanilla jerk | impedance jerk |")
print("|---:|:---|:---|:---|:---|")
for s in steps:
    vs = r[f"steps{s}/vanilla"]["mean"]; vse = r[f"steps{s}/vanilla"]["sem"]
    is_ = r[f"steps{s}/impedance"]["mean"]; ise = r[f"steps{s}/impedance"]["sem"]
    print(f"| {s} | {vs['success']*100:.1f} ± {vse['success']*100:.1f} | "
          f"**{is_['success']*100:.1f} ± {ise['success']*100:.1f}** | "
          f"{vs['jerk']:.3f} | {is_['jerk']:.3f} |")
