r"""
TOY UNIT TEST for the impedance-shaped flow-matching drift  (Week-1 gate).
==========================================================================
Core object: the flow ODE drift is a discrete closed-loop mass-spring-damper in action space

    v_k = M^{-1} ( -D (a_k - a_{k-1})/dt - K (a_k - a_goal) + f )
    a_{k+1} = a_k + dt * v_k          (Euler step over the flow, dt = 1/num_steps)

NO learning here. We verify the IMPLEMENTATION is correct before it goes into SmolVLA:
  (1) SPD parameterization  L -> M = L L^T + eps I  is symmetric positive-definite.
  (2) Attractor convergence: trajectories reach a_goal.
  (3) Damping-ratio control: zeta<1 oscillates, zeta=1 critical (no overshoot, fast),
      zeta>1 overdamped (no overshoot, slow) -- set via D = 2*zeta*sqrt(M K).
  (4) STABILITY at SmolVLA's ~10 Euler steps: explicit Euler on a stiff spring blows up;
      find the max stiffness K usable at num_steps=10  (constraint for the real head).
"""
import os
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
torch.set_default_dtype(torch.float64)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(HERE, "figs"); OUT = os.path.join(HERE, "out")
os.makedirs(FIGS, exist_ok=True); os.makedirs(OUT, exist_ok=True)


def spd_from_cholesky(L, eps=1e-3):
    """L (lower-tri) -> SPD M = L L^T + eps I. This is exactly the head's parameterization."""
    n = L.shape[-1]
    return L @ L.transpose(-1, -2) + eps * torch.eye(n)


def impedance_rollout(a0, a_goal, M, K, D, num_steps, f=None, w0=None):
    """2nd-order impedance ODE  M a'' + D a' + K (a-goal) = f  integrated SEMI-IMPLICITLY.
    Explicit velocity state w = a' (symplectic Euler): the damping -D w is dt-scaled, so it's
    stable for wn*dt < 2.  (The naive 1st-order finite-difference form -D (a-a_prev)/dt cancels
    the dt and is unstable for D >= 1 -- that is what the toy caught.)
    The network's structured field is the ACCELERATION a''; the generated trajectory a_tau is a
    damped spring toward a_goal."""
    dt = 1.0 / num_steps
    Minv = torch.linalg.inv(M)
    f = torch.zeros_like(a0) if f is None else f
    a = a0.clone(); w = torch.zeros_like(a0) if w0 is None else w0.clone()
    traj = [a.clone()]
    for _ in range(num_steps):
        acc = Minv @ (-D @ w - K @ (a - a_goal) + f)   # acceleration = the impedance law
        w = w + dt * acc                                # update velocity first (semi-implicit)
        a = a + dt * w                                  # then position
        traj.append(a.clone())
    return torch.stack(traj)                       # [num_steps+1, n]


def test_spd():
    print("=== (1) SPD parameterization ===")
    ok = True
    for n in (2, 7):
        L = torch.tril(torch.randn(n, n))
        for name, Mx in [("M", spd_from_cholesky(L)),
                         ("K", spd_from_cholesky(torch.tril(torch.randn(n, n)))),
                         ("D", spd_from_cholesky(torch.tril(torch.randn(n, n))))]:
            ev = torch.linalg.eigvalsh(Mx)
            sym = float((Mx - Mx.T).abs().max())
            good = bool(ev.min() > 0) and sym < 1e-9
            ok = ok and good
            print(f"  n={n} {name}: min eig={float(ev.min()):.4f}  symm_err={sym:.1e}  -> {'PSD ok' if good else 'FAIL'}")
    return ok


def test_damping():
    print("\n=== (2,3) attractor convergence + damping-ratio control (2-D) ===")
    n = 2; I = torch.eye(n)
    a0 = torch.tensor([-1.0, -0.5]); a_goal = torch.tensor([1.0, 1.0])
    k = 400.0; K = k * I; M = I; num_steps = 100
    wn = float(np.sqrt(k))                                   # natural freq (M=I)
    res = {}
    plt.figure(figsize=(11, 4))
    ax1 = plt.subplot(1, 2, 1); ax2 = plt.subplot(1, 2, 2)
    d0 = float((a0 - a_goal).norm())
    for zeta, col in [(0.3, "C3"), (1.0, "C0"), (2.0, "C2")]:
        D = 2 * zeta * np.sqrt(k) * I
        traj = impedance_rollout(a0, a_goal, M, K, D, num_steps)
        dist = (traj - a_goal).norm(dim=-1).numpy()
        ts = np.linspace(0, 1, num_steps + 1)
        ax1.plot(ts, dist, col, label=f"ζ={zeta}")
        ax2.plot(traj[:, 0].numpy(), traj[:, 1].numpy(), col, marker=".", ms=3, label=f"ζ={zeta}")
        # overshoot: first local-min (first approach to goal) -> next local-max peak, as % of d0.
        # (robust for decaying oscillations; monotonic convergence -> no valley -> 0%)
        overshoot = 0.0
        for i in range(1, len(dist) - 1):
            if dist[i] < dist[i - 1] and dist[i] <= dist[i + 1]:           # first valley
                for j in range(i + 1, len(dist) - 1):
                    if dist[j] > dist[j - 1] and dist[j] >= dist[j + 1]:   # next peak
                        overshoot = float(dist[j] / d0 * 100); break
                break
        res[zeta] = {"final_dist": float(dist[-1]), "min_dist": float(dist.min()),
                     "overshoot_%": overshoot, "converged": bool(dist[-1] < 0.05 * d0)}
        print(f"  ζ={zeta}: final_dist={dist[-1]:.4f}  overshoot={overshoot:5.1f}%  "
              f"{'converged' if res[zeta]['converged'] else 'NOT converged'}")
    ax1.axhline(0, color="k", lw=.5); ax1.set_xlabel("flow time τ"); ax1.set_ylabel("‖a−a_goal‖")
    ax1.set_title(f"convergence vs damping (ωn={wn:.0f})"); ax1.legend()
    ax2.scatter(*a_goal.numpy(), c="k", marker="*", s=120, zorder=5, label="goal")
    ax2.set_title("2-D action path"); ax2.set_xlabel("a₁"); ax2.set_ylabel("a₂"); ax2.legend()
    plt.tight_layout(); plt.savefig(f"{FIGS}/toy_impedance.png", dpi=140); plt.close()
    print(f"  wrote {FIGS}/toy_impedance.png")
    # damping-control signature: underdamped oscillates; critical clean+fast; overdamped monotonic (may be slower)
    ok = (res[0.3]["overshoot_%"] > 5                                  # underdamped overshoots
          and res[1.0]["overshoot_%"] < 1 and res[1.0]["converged"]   # critical: no overshoot, converges
          and res[2.0]["overshoot_%"] < 1)                            # overdamped: monotonic
    return ok, res


def test_stability():
    print("\n=== (4) explicit-Euler stability at num_steps=10 (SmolVLA regime), ζ=1 ===")
    n = 2; I = torch.eye(n); num_steps = 10
    a0 = torch.tensor([-1.0, -0.5]); a_goal = torch.tensor([1.0, 1.0])
    dt = 1.0 / num_steps
    kmax_undamped = (2.0 / dt) ** 2                          # undamped spring: wn*dt<2 => K<(2/dt)^2
    kmax_crit = (1.0 / dt) ** 2                              # critical damping D=2sqrt(K): dt*D<2 => K<(1/dt)^2 (binds tighter)
    theo_kmax = kmax_crit
    rows = []
    last_stable = None
    for k in [25, 100, 225, 400, 625, 900, 1600]:
        D = 2 * np.sqrt(k) * I
        traj = impedance_rollout(a0, a_goal, I, k * I, D, num_steps)  # M=I, K=k I, critical damping
        mx = float(traj.abs().max()); fin = float((traj[-1] - a_goal).norm())
        stable = bool(np.isfinite(mx) and mx < 10.0)
        if stable: last_stable = k
        rows.append((k, float(np.sqrt(k) * dt), stable, fin, mx))
        print(f"  K={k:5d}  ωn·dt={np.sqrt(k)*dt:4.2f}  {'STABLE' if stable else 'UNSTABLE'}  "
              f"final_dist={fin:8.3f}  max|a|={mx:8.2f}")
    print(f"  bound: undamped spring ωn·dt<2 => K<{kmax_undamped:.0f}; BUT critical damping binds tighter:")
    print(f"         dt·D<2 with D=2√K => K<(1/dt)^2 = {kmax_crit:.0f}. Empirical last-stable K={last_stable} (safe margin).")
    # plot stability
    ks = [r[0] for r in rows]; fins = [min(r[3], 1e3) for r in rows]
    plt.figure(figsize=(5, 4))
    plt.semilogy(ks, fins, "o-")
    plt.axvline(theo_kmax, color="C3", ls="--", label=f"theory K_max={theo_kmax:.0f}")
    plt.xlabel("stiffness K (eigenvalue)"); plt.ylabel("final ‖a−a_goal‖ (log)")
    plt.title("stability vs stiffness @ 10 Euler steps"); plt.legend()
    plt.tight_layout(); plt.savefig(f"{FIGS}/toy_stability.png", dpi=140); plt.close()
    print(f"  wrote {FIGS}/toy_stability.png")
    return last_stable is not None and last_stable <= theo_kmax * 1.1, theo_kmax, last_stable


def main():
    ok_spd = test_spd()
    ok_damp, _ = test_damping()
    ok_stab, kmax, last_stable = test_stability()
    print("\n================= TOY UNIT-TEST VERDICT =================")
    print(f"  (1) SPD head parameterization correct      : {'PASS' if ok_spd else 'FAIL'}")
    print(f"  (2,3) convergence + damping-ratio control  : {'PASS' if ok_damp else 'FAIL'}")
    print(f"  (4) stability boundary characterized       : {'PASS' if ok_stab else 'FAIL'}")
    print(f"  DESIGN CONSTRAINT (10 steps): keep K eigenvalues < ~{kmax:.0f} (critical-damping bound); empirical safe K≈{last_stable}.")
    print("  NOTE: soft/low stiffness is BOTH what 10-step inference permits AND what contact COMPLIANCE wants —")
    print("        the numerical limit is aligned with the method's goal. Stiffer => more ODE steps or an implicit integrator.")
    print("  => core impedance-flow object is sound; cleared to build the SmolVLA head." if (ok_spd and ok_damp and ok_stab)
          else "  => FIX before integrating into SmolVLA.")


if __name__ == "__main__":
    main()
