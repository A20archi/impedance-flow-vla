r"""
Architecture / methodology diagram for Impedance-Shaped Flow Matching.
Produces a single clean, publication-style figure:  figs/arch_methodology.png

Layout (3 horizontal zones, generous margins to avoid overlap):
  - title band
  - main pipeline:  obs -> encoder -> {K/D head, attractor head, residual field} -> impedance drift -> 2nd-order integrator -> action chunk
  - contrast band:  vanilla unstructured drift  vs  ours (drift IS the impedance law) + novelty one-liner
Faithful to gate/policy.py (M=I in Stage-1; K=cap(LLT), zeta>=1, D=2*zeta*sqrt(K); semi-implicit Euler, N=10).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

OUT = os.path.join(os.path.dirname(__file__), "..", "figs", "arch_methodology.png")

# palette
INK     = "#1b2430"
MUTED   = "#5b6b7b"
TEAL    = "#0d8a8a"   # impedance / ours
TEAL_BG = "#e3f4f3"
BLUE    = "#2f5d8c"
BLUE_BG = "#e6eef6"
GOLD    = "#c8801a"   # the key equation accent
GOLD_BG = "#fbf0db"
GRAY    = "#8a96a3"   # vanilla
GRAY_BG = "#eef1f4"

def box(ax, x, y, w, h, lines, fc="white", ec=INK, lw=1.6, fs=11,
        title=None, title_fs=12, title_color=None, round=0.018, text_color=INK):
    p = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                       boxstyle=f"round,pad=0.004,rounding_size={round}",
                       fc=fc, ec=ec, lw=lw, zorder=3)
    ax.add_patch(p)
    yc = y
    if title is not None:
        ax.text(x, y + h / 2 - 0.030, title, ha="center", va="top",
                fontsize=title_fs, fontweight="bold",
                color=title_color or ec, zorder=4)
        yc = y - 0.018
    body = "\n".join(lines) if isinstance(lines, (list, tuple)) else lines
    ax.text(x, yc, body, ha="center", va="center", fontsize=fs,
            color=text_color, zorder=4, linespacing=1.45)

def arrow(ax, x1, y1, x2, y2, color=INK, lw=2.0, ls="-", rad=0.0):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle="-|>", mutation_scale=18,
                        lw=lw, color=color, zorder=2,
                        connectionstyle=f"arc3,rad={rad}", ls=ls)
    ax.add_patch(a)

fig, ax = plt.subplots(figsize=(16, 9), dpi=200)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
fig.patch.set_facecolor("white")

# ---------------- title band ----------------
ax.text(0.5, 0.965, "Impedance-Shaped Flow Matching for VLA Policies",
        ha="center", va="center", fontsize=21, fontweight="bold", color=INK)
ax.text(0.5, 0.925,
        "The action-expert's flow-ODE drift IS a closed-loop, task-conditioned mass-spring-damper law",
        ha="center", va="center", fontsize=12.5, color=MUTED, style="italic")

# ---------------- main pipeline (y ~ 0.55 - 0.86) ----------------
yhead = 0.70
# 1. obs
box(ax, 0.085, yhead, 0.135, 0.16,
    ["state / image /", "language", "", r"$\mathit{MetaWorld\ 39\text{-}D}$", r"$\mathit{state\ (Stage\text{-}1)}$"],
    title="Observation", title_color=BLUE, ec=BLUE, fc=BLUE_BG, fs=9.5)
# 2. encoder
box(ax, 0.255, yhead, 0.125, 0.14,
    [r"$\phi(\mathrm{obs}) \rightarrow h$", r"$h \in \mathbb{R}^{256}$", "(MLP)"],
    title="Condition encoder", title_color=BLUE, ec=BLUE, fc=BLUE_BG, fs=10.5)
arrow(ax, 0.153, yhead, 0.192, yhead, color=BLUE)

# 3. three heads (vertical branch from h)
hx = 0.475
head_specs = [
    (0.855, ["$K = \\mathrm{cap}(LL^{\\!\\top}),\\ \\ \\zeta \\geq 1$", "$D = 2\\zeta\\sqrt{K}$   (SPD)"],
     "Stiffness / Damping head", TEAL, TEAL_BG),
    (0.700, ["$a_{goal}(\\mathrm{obs})$", "chunk attractor"],
     "Attractor head", TEAL, TEAL_BG),
    (0.545, ["$f_\\theta(a,\\ \\dot a,\\ \\tau,\\ \\mathrm{obs})$", "learned residual"],
     "Residual field", TEAL, TEAL_BG),
]
for hy, lines, ttl, ec, fc in head_specs:
    box(ax, hx, hy, 0.205, 0.115, lines, title=ttl, title_color=ec, ec=ec, fc=fc, fs=10.5)
    arrow(ax, 0.318, yhead, 0.372, hy, color=TEAL, rad=0.0 if abs(hy-yhead)<0.01 else (0.0))

# 4. impedance drift (the key equation) -- highlighted
dx = 0.745
box(ax, dx, yhead, 0.205, 0.20,
    ["", r"$v_\theta = M^{-1}\!\left(-D\,\dot a - K(a - a_{goal}) + f_\theta\right)$", "",
     r"$\mathit{M=I\ (Stage\text{-}1);\ M\text{-}shaping\rightarrow Stage\text{-}3}$"],
    title="Impedance drift", title_color=GOLD, ec=GOLD, fc=GOLD_BG, lw=2.4, fs=11.5)
for hy, *_ in head_specs:
    arrow(ax, 0.578, hy, 0.643, yhead, color=TEAL, rad=0.12 if hy > yhead else (-0.12 if hy < yhead else 0.0))

# 5. integrator
ix = 0.745
iy = 0.40
box(ax, ix, iy, 0.30, 0.15,
    [r"$w \leftarrow w + \Delta s\,(v_\theta)\,,\qquad a \leftarrow a + \Delta s\,w$",
     r"$\mathit{symplectic\ Euler\ \cdot\ N{=}10\ steps\ \cdot\ noise\rightarrow action}$"],
    title="2nd-order semi-implicit integrator", title_color=GOLD, ec=GOLD, fc=GOLD_BG, fs=11)
arrow(ax, dx, 0.60, ix, 0.475, color=GOLD)
# feedback loop arrow (a,w fed back into drift) -- dashed
arrow(ax, 0.595, 0.40, 0.62, 0.605, color=MUTED, lw=1.4, ls=(0, (4, 3)), rad=-0.35)
ax.text(0.55, 0.50, "state $(a,\\dot a)$\nfeedback", ha="center", va="center",
        fontsize=8.5, color=MUTED, style="italic")

# 6. output
ox = 0.93
box(ax, ox, iy, 0.105, 0.12,
    [r"$a \in \mathbb{R}^{16\times 4}$"],
    title="Action chunk", title_color=BLUE, ec=BLUE, fc=BLUE_BG, fs=11)
arrow(ax, 0.895, iy, 0.877, iy, color=BLUE)

# ---------------- contrast band (y ~ 0.05 - 0.26) ----------------
ax.add_line(Line2D([0.04, 0.96], [0.285, 0.285], color="#d7dde3", lw=1.2, zorder=1))
ax.text(0.5, 0.262, "What makes the drift structural", ha="center", va="center",
        fontsize=12.5, fontweight="bold", color=INK)

box(ax, 0.235, 0.135, 0.34, 0.155,
    [r"$v_\theta = f_\theta(a,\ \tau,\ \mathrm{obs})$",
     "unstructured neural drift", "(no notion of stiffness, damping,", "or an attractor)"],
    title="Vanilla flow  (param-matched baseline)", title_color=GRAY, ec=GRAY, fc=GRAY_BG, fs=10, title_fs=11)

box(ax, 0.655, 0.135, 0.34, 0.155,
    [r"$v_\theta = M^{-1}(-D\dot a - K(a-a_{goal}) + f_\theta)$",
     "the drift IS a closed-loop impedance law;", "K, D, attractor are conditioned on obs;",
     "end-to-end differentiable, no F/T sensor"],
    title="Ours  (impedance-shaped drift)", title_color=TEAL, ec=TEAL, fc=TEAL_BG, fs=10, title_fs=11)
arrow(ax, 0.41, 0.135, 0.48, 0.135, color=INK, lw=2.2)

ax.text(0.5, 0.038,
        "Novelty vs prior art:  CompliantVLA wraps an EXTERNAL impedance controller around a frozen VLA;  "
        "Flow-with-Force-Field adds a DOWNSTREAM passive controller.\n"
        "Ours embeds the K/D/M attractor law INSIDE the generative ODE drift itself.",
        ha="center", va="center", fontsize=9.8, color=MUTED, style="italic", linespacing=1.5)

plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
plt.savefig(OUT, dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.15)
print("wrote", os.path.abspath(OUT))
