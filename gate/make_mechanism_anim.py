r"""
EXPLAINABLE mechanism animation: "generating one action through the flow ODE".
vanilla flow = unstructured (≈straight rectified path);  impedance flow = damped spring toward the
predicted attractor (target action).  Shows zeta=1 (ours, critical: smooth) vs zeta=0.3 (under-damped:
overshoots) -- which is exactly why we enforce zeta>=1.  Output MP4 for the talk.
"""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio

ROOT = "/home/user/Desktop/Saptarshi/impedance_flow_vla"
goal = np.array([1.0, 1.0]); x0 = np.array([-1.0, -0.5])
N = 48; dt = 1.0 / N


def imped(zeta, k=60.0):
    a = x0.copy(); w = np.zeros(2); traj = [a.copy()]; D = 2 * zeta * np.sqrt(k)
    for _ in range(N):
        acc = -D * w - k * (a - goal); w = w + dt * acc; a = a + dt * w; traj.append(a.copy())
    return np.array(traj)


van = np.stack([x0 + (goal - x0) * (i / N) for i in range(N + 1)])   # straight rectified flow
imp1, imp03 = imped(1.0), imped(0.3)


def speed(tr):
    d = np.linalg.norm(np.diff(tr, axis=0), axis=1) / dt
    return np.concatenate([[0.0], d])


sv, s1, s03 = speed(van), speed(imp1), speed(imp03)
smax = max(sv.max(), s1.max(), s03.max()) * 1.1
steps = np.arange(N + 1)
C = {"v": "#888888", "u": "#d62728", "c": "#1f77b4"}

frames = []
for t in range(N + 1):
    fig, (axp, axv) = plt.subplots(1, 2, figsize=(10.5, 5))
    # --- action-space path ---
    axp.plot(van[:t + 1, 0], van[:t + 1, 1], "--", color=C["v"], lw=2, label="vanilla flow (unstructured)")
    axp.plot(imp03[:t + 1, 0], imp03[:t + 1, 1], color=C["u"], lw=1.6, alpha=.75, label="impedance ζ=0.3 (under-damped)")
    axp.plot(imp1[:t + 1, 0], imp1[:t + 1, 1], color=C["c"], lw=3, label="impedance ζ=1  (ours)")
    for tr, c in [(van, C["v"]), (imp03, C["u"]), (imp1, C["c"])]:
        axp.scatter(tr[t, 0], tr[t, 1], color=c, s=70, zorder=5, edgecolor="k", linewidth=.5)
    axp.scatter(*goal, marker="*", s=340, color="gold", edgecolor="k", zorder=6, label="target action")
    axp.scatter(*x0, marker="o", s=90, facecolor="white", edgecolor="k", zorder=4)
    axp.annotate("noise", x0, (x0[0], x0[1] - 0.22), ha="center", fontsize=10)
    axp.set_xlim(-1.7, 1.9); axp.set_ylim(-1.3, 1.9); axp.set_xticks([]); axp.set_yticks([])
    axp.set_title("action-space path"); axp.legend(loc="lower right", fontsize=8.5, framealpha=.9)
    # --- speed profile (where the mechanism shows) ---
    axv.plot(steps[:t + 1], sv[:t + 1], "--", color=C["v"], lw=2)
    axv.plot(steps[:t + 1], s03[:t + 1], color=C["u"], lw=1.6, alpha=.75)
    axv.plot(steps[:t + 1], s1[:t + 1], color=C["c"], lw=3)
    axv.set_xlim(0, N); axv.set_ylim(0, smax); axv.grid(alpha=.3)
    axv.set_xlabel("ODE step"); axv.set_ylabel("|velocity|")
    axv.set_title("speed: impedance decelerates → soft landing\n(vanilla arrives at full speed)")
    fig.suptitle(f"Generating one action through the flow ODE   (step {t}/{N})   —   impedance drift = damped spring",
                 fontsize=12.5, y=1.02)
    fig.tight_layout(); fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (4,))
    frames.append(buf[:, :, :3].copy()); plt.close(fig)
frames += [frames[-1]] * 18                                          # hold on final frame
imageio.mimsave(f"{ROOT}/figs/mechanism_anim.mp4", frames, fps=15, quality=8)
print(f"wrote {ROOT}/figs/mechanism_anim.mp4  ({len(frames)} frames)")
