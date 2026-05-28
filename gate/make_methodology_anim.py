r"""
Animated methodology walkthrough for Impedance-Shaped Flow Matching.
Produces  figs/methodology_anim.mp4  (and a still poster figs/methodology_poster.png).

Top band : the pipeline reveals stage-by-stage with a flowing pulse
           obs -> encoder -> {K/D, attractor, residual} heads -> impedance drift -> integrator -> action.
Bottom   : the generative ODE integration plays out in a 2-D action space --
           the impedance arm (teal) is a critically-damped spring-damper that lands softly on the
           attractor; the vanilla arm (gray) is an unstructured drift with no such guarantee.
Faithful to gate/policy.py (semi-implicit Euler, N=10, K=cap(LLT), zeta>=1 => D=2*zeta*sqrt(K)).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse
from matplotlib.animation import FuncAnimation, FFMpegWriter

FIGS = os.path.join(os.path.dirname(__file__), "..", "figs")
MP4  = os.path.join(FIGS, "methodology_anim.mp4")
POST = os.path.join(FIGS, "methodology_poster.png")

INK, MUTED = "#1b2430", "#5b6b7b"
TEAL, TEAL_BG = "#0d8a8a", "#e3f4f3"
BLUE, BLUE_BG = "#2f5d8c", "#e6eef6"
GOLD, GOLD_BG = "#c8801a", "#fbf0db"
GRAY, GRAY_BG = "#8a96a3", "#eef1f4"

FPS = 24
DUR = 13.0
N = int(FPS * DUR)

# ---- precompute the 2-D action-space integration (N_STEPS = 10, semi-implicit) ----
N_STEPS = 10
a_goal = np.array([0.0, 0.0])
a0 = np.array([1.70, 1.15])
# impedance: SPD K (soft regime), critically damped zeta=1  => D = 2*sqrt(K)
K = np.array([[9.0, 1.6], [1.6, 6.0]])
evals, evecs = np.linalg.eigh(K)
sqrtK = (evecs * np.sqrt(evals)) @ evecs.T
D = 2.0 * sqrtK
ds = 1.0 / N_STEPS
imp = [a0.copy()]; a = a0.copy(); w = np.zeros(2)
for _ in range(N_STEPS):
    acc = -D @ w - K @ (a - a_goal)
    w = w + ds * acc
    a = a + ds * w
    imp.append(a.copy())
imp = np.array(imp)
# vanilla: unstructured drift toward goal but uncontrolled -> overshoots then corrects (wiggly)
van = [a0.copy()]; a = a0.copy()
rng = np.random.default_rng(3)
dirn = (a_goal - a0)
for k in range(N_STEPS):
    step = dirn * (1.0 / (N_STEPS - k)) * 1.55          # overshooting gain
    step = step + rng.normal(0, 0.16, 2)                 # unstructured jitter
    a = a + step
    van.append(a.copy())
van = np.array(van)
# stiffness ellipse (low stiffness = wide ellipse along soft direction)
ell_w = 2.0 / np.sqrt(evals[0]); ell_h = 2.0 / np.sqrt(evals[1])
ell_ang = np.degrees(np.arctan2(evecs[1, 0], evecs[0, 0]))

# ---- phase schedule (fractions of timeline) ----
def ramp(t, t0, t1):
    return float(np.clip((t - t0) / max(t1 - t0, 1e-6), 0, 1))

def box(ax, x, y, w, h, lines, fc, ec, alpha, title=None, fs=10.5, lw=1.8, title_fs=11.5):
    if alpha <= 0.01:
        return
    p = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                       boxstyle="round,pad=0.004,rounding_size=0.02",
                       fc=fc, ec=ec, lw=lw, zorder=3, alpha=alpha)
    ax.add_patch(p)
    yc = y
    if title:
        ax.text(x, y + h / 2 - 0.022, title, ha="center", va="top", fontsize=title_fs,
                fontweight="bold", color=ec, zorder=4, alpha=alpha)
        yc = y - 0.014
    ax.text(x, yc, "\n".join(lines), ha="center", va="center", fontsize=fs,
            color=INK, zorder=4, alpha=alpha, linespacing=1.4)

def arrow(ax, x1, y1, x2, y2, color, alpha, lw=2.0, rad=0.0):
    if alpha <= 0.02:
        return
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=15, lw=lw, color=color, zorder=2, alpha=alpha,
                 connectionstyle=f"arc3,rad={rad}"))

fig = plt.figure(figsize=(16, 9), dpi=160)
fig.patch.set_facecolor("white")
axP = fig.add_axes([0.0, 0.46, 1.0, 0.50]); axP.set_xlim(0, 1); axP.set_ylim(0, 1); axP.axis("off")
axA = fig.add_axes([0.06, 0.06, 0.48, 0.40])      # action space
axE = fig.add_axes([0.58, 0.06, 0.40, 0.40]); axE.axis("off")   # equation / readout

def draw(frame):
    t = frame / (N - 1)
    for ax in (axP, axA, axE):
        ax.clear()
    axP.set_xlim(0, 1); axP.set_ylim(0, 1); axP.axis("off")
    axE.set_xlim(0, 1); axE.set_ylim(0, 1); axE.axis("off")

    # ---------- title ----------
    axP.text(0.5, 0.95, "Impedance-Shaped Flow Matching  -  how an action chunk is generated",
             ha="center", va="center", fontsize=18, fontweight="bold", color=INK)

    # reveal schedule
    aObs  = ramp(t, 0.02, 0.08)
    aEnc  = ramp(t, 0.08, 0.15)
    aHead = ramp(t, 0.16, 0.30)
    aDri  = ramp(t, 0.32, 0.44)
    aInt  = ramp(t, 0.45, 0.55)
    aOut  = ramp(t, 0.93, 0.99)
    pInt  = ramp(t, 0.56, 0.92)        # integration progress

    yc = 0.55
    box(axP, 0.085, yc, 0.135, 0.34, ["state / image /", "language"], BLUE_BG, BLUE, aObs,
        title="Observation", fs=10)
    arrow(axP, 0.155, yc, 0.205, yc, BLUE, min(aObs, aEnc))
    box(axP, 0.275, yc, 0.125, 0.30, [r"$\phi(\mathrm{obs})\!\rightarrow\! h$", r"$h\in\mathbb{R}^{256}$"],
        BLUE_BG, BLUE, aEnc, title="Encoder", fs=10)

    heads = [(0.80, ["$K=\\mathrm{cap}(LL^{\\top})$", "$D=2\\zeta\\sqrt{K},\\ \\zeta\\!\\geq\\!1$"], "Stiffness / Damping"),
             (0.55, ["$a_{goal}(\\mathrm{obs})$"], "Attractor"),
             (0.30, ["$f_\\theta(a,\\dot a,\\tau,\\mathrm{obs})$"], "Residual field")]
    for hy, lines, ttl in heads:
        box(axP, 0.49, hy, 0.205, 0.165, lines, TEAL_BG, TEAL, aHead, title=ttl, fs=9.5, title_fs=10.5)
        arrow(axP, 0.338, yc, 0.388, hy, TEAL, min(aEnc, aHead))
        arrow(axP, 0.595, hy, 0.66, yc, TEAL, min(aHead, aDri), rad=0.12 if hy > yc else (-0.12 if hy < yc else 0))

    box(axP, 0.76, yc, 0.205, 0.30,
        [r"$v_\theta=M^{-1}(-D\dot a$", r"$-K(a-a_{goal})+f_\theta)$"],
        GOLD_BG, GOLD, aDri, title="Impedance drift", fs=10.5, lw=2.4)
    arrow(axP, 0.865, yc, 0.915, yc, GOLD, min(aDri, aInt))
    box(axP, 0.95, yc, 0.085, 0.22, [r"$a$"], BLUE_BG, BLUE, aOut, title="Chunk", fs=11)

    # pulse dot travelling the spine during integration phase
    if 0.04 < t < 0.55:
        tp = ramp(t, 0.04, 0.5)
        px = 0.085 + tp * (0.76 - 0.085)
        axP.scatter([px], [yc], s=120, color=GOLD, zorder=6, alpha=0.8)

    # ---------- action space ----------
    axA.set_xlim(-0.6, 2.1); axA.set_ylim(-0.9, 1.7)
    axA.set_xticks([]); axA.set_yticks([])
    axA.set_title("Generative ODE in action space   (noise $\\rightarrow$ action)",
                  fontsize=12, color=INK, fontweight="bold", loc="left")
    for s in axA.spines.values():
        s.set_color("#cdd5dd")
    if aHead > 0.3:
        # attractor + stiffness ellipse
        el = Ellipse(a_goal, ell_w, ell_h, angle=ell_ang, fc="none", ec=TEAL,
                     lw=1.6, ls=(0, (5, 3)), alpha=0.5 * aHead)
        axA.add_patch(el)
        axA.scatter(*a_goal, marker="*", s=420, color=GOLD, edgecolor=INK,
                    lw=0.8, zorder=5, alpha=aHead)
        axA.text(a_goal[0] + 0.08, a_goal[1] - 0.16, "$a_{goal}$", fontsize=12,
                 color=GOLD, alpha=aHead, fontweight="bold")
        axA.text(0.9, 1.45, "stiffness $K$ (compliance ellipse)", fontsize=8.5,
                 color=TEAL, alpha=0.7 * aHead, style="italic")
    axA.scatter(*a0, s=70, color=MUTED, zorder=5)
    axA.text(a0[0] + 0.05, a0[1] + 0.06, "noise $a_0$", fontsize=10, color=MUTED)

    # progressive trajectories (10 discrete semi-implicit steps)
    fcont = pInt * N_STEPS
    kfull = int(np.floor(fcont))
    frac = fcont - kfull
    def partial(traj):
        if kfull <= 0 and frac <= 0:
            return traj[:1]
        pts = traj[:kfull + 1].tolist()
        if kfull < N_STEPS and frac > 0:
            pts.append((traj[kfull] + frac * (traj[kfull + 1] - traj[kfull])).tolist())
        return np.array(pts)
    if pInt > 0:
        vt = partial(van); it = partial(imp)
        axA.plot(vt[:, 0], vt[:, 1], "-o", color=GRAY, lw=2.0, ms=4, alpha=0.85,
                 label="vanilla (unstructured drift)")
        axA.plot(it[:, 0], it[:, 1], "-o", color=TEAL, lw=2.6, ms=5,
                 label="impedance (critically damped)")
        axA.scatter(it[-1, 0], it[-1, 1], s=90, color=TEAL, zorder=6,
                    edgecolor="white", lw=1.2)
        axA.scatter(vt[-1, 0], vt[-1, 1], s=70, color=GRAY, zorder=6,
                    edgecolor="white", lw=1.0)
        axA.legend(loc="lower left", fontsize=8.5, framealpha=0.9)
        axA.text(1.55, -0.75, f"step {min(kfull,N_STEPS)}/{N_STEPS}", fontsize=10,
                 color=INK, ha="right", fontweight="bold")

    # ---------- equation / readout panel ----------
    axE.text(0.0, 0.94, "The drift IS the impedance law", fontsize=13,
             fontweight="bold", color=INK, va="top")
    terms = [
        (r"$-K(a-a_{goal})$", "stiffness pulls toward the attractor", TEAL, 0.32),
        (r"$-D\,\dot a$",      "damping removes energy (no overshoot)", BLUE, 0.40),
        (r"$+f_\theta$",       "learned residual: task-specific correction", GOLD, 0.50),
    ]
    yy = 0.74
    for expr, desc, col, t0 in terms:
        on = ramp(t, t0, t0 + 0.06)
        axE.text(0.04, yy, expr, fontsize=15, color=col, va="center", alpha=0.25 + 0.75 * on)
        axE.text(0.34, yy, desc, fontsize=10.5, color=INK, va="center", alpha=0.25 + 0.75 * on)
        yy -= 0.155
    si = ramp(t, 0.50, 0.56)
    axE.text(0.0, 0.20, "Semi-implicit (symplectic) Euler,  $N=10$ steps:",
             fontsize=10.5, color=MUTED, va="center", alpha=si, style="italic")
    axE.text(0.04, 0.075, r"$w \leftarrow w + \Delta s\,v_\theta\,,\qquad a \leftarrow a + \Delta s\,w$",
             fontsize=12.5, color=INK, va="center", alpha=si)
    return []

print(f"rendering {N} frames -> {MP4}")
anim = FuncAnimation(fig, draw, frames=N, blit=False)
writer = FFMpegWriter(fps=FPS, bitrate=3200,
                      extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
anim.save(MP4, writer=writer, dpi=120)
draw(int(N * 0.85)); plt.savefig(POST, dpi=130, facecolor="white")
print("wrote", os.path.abspath(MP4))
print("wrote", os.path.abspath(POST))
