# Impedance-Shaped Flow Matching for VLA Policies

*New project, started 2026-05-28. Separate from the abandoned `adjoint_ml10` (kept intact, not a dependency).*
*Premise: prior-art check (in `docs/prior_art.md`) establishes the core intersection is unclaimed as of May 2026.*

## Thesis — the one new object
The first flow-matching VLA action expert whose ODE **drift is itself a closed-loop second-order impedance law**:

    v_θ(a_τ, τ, obs) = M⁻¹( −D·ȧ_τ − K·(a_τ − a_goal) + f_θ(a_τ, τ, obs) )

with task-conditioned **SPD** matrices (M, K, D) predicted from the VLM features (Cholesky-parameterized),
ȧ_τ is an **explicit velocity state** (2nd-order augmented flow, integrated **semi-implicitly** — see Toy gate;
the naive finite-difference form is unstable), a_goal an attractor (chunk target or a predicted attractor token).
End-to-end differentiable. **No downstream controller. No F/T sensor at inference.**

## Toy gate — CLEARED 2026-05-28  (`toy/toy_impedance_flow.py`, figs `toy_impedance.png`, `toy_stability.png`)
- SPD head (Cholesky `M=LLᵀ+εI`) verified PSD; damping-ratio control verified (ζ=0.3→36% overshoot, ζ=1→clean, ζ=2→monotonic).
- **DESIGN DECISION:** integrate as a 2nd-order ODE with an explicit velocity state, **semi-implicit (symplectic) Euler**; the network field is the *acceleration*. The literal `−D·(a−a_prev)/Δτ` drift is unstable for D≳1 (the Δτ cancels) — do NOT use it.
- **CONSTRAINT:** at 10 Euler steps, usable stiffness K ≲ 25–100 (critical-damping bound `dt·D<2`). Soft regime — which is what contact compliance wants anyway. Stiffer ⇒ more ODE steps or an implicit/exponential integrator.

## Why it's novel (and what it is NOT)
- vs **CompliantVLA-adaptor** (2601.15541): predicts K,D from a VLM but feeds an EXTERNAL variable-impedance
  controller around a *frozen* VLA. We embed the impedance law as the generative drift INSIDE the action expert.
- vs **Flow with the Force Field** (2510.02738): their flow velocity field is a standard unstructured neural drift;
  compliance is a downstream Passive Impedance Controller. Ours: the velocity field IS the impedance law. (Most
  easily-confused prior work — the abstract/figures must make this unmistakable.)
- vs **ACP / Diffusion-Impedance / MPD / FODMP**: stiffness lives on the OUTPUT (a head or a DMP decoder); none
  make the flow ODE drift a closed-loop K/D/M attractor conditioned on VLM features.
- Generalizes the Lyapunov/contractive DS-imitation tradition (SNDS, NDP) to large VLA flow experts.

## Expected wins (priors from adjacent ablations — treat as upper bounds)
Contact-rich: insertion +15–25pp, wiping +10–20pp, articulated +5–10pp; smoother actions (jerk / spectral arc
length); possibly fewer ODE steps; better sample efficiency. Honest non-wins: free-space pick-place (≈0).

## Plan (45 days) + kill-criteria
- **W1–2 Implement.** Impedance drift + SPD (M,K,D) head on SmolVLA. ← *toy unit test first (this repo, `toy/`).*
- **W3–4 Sim ablations.** LIBERO + MetaWorld contact subset (door-open, peg-insert-side, assembly, hammer).
  Ablations: vanilla / stiffness-output-only / full-impedance(ours) / fixed-vs-VLM-predicted K,D / ODE-steps {1,5,10,20} / horizon {16,32,50}.
- **W5 Real-arm demo (recommended).** Peg/USB insertion or wiping w/ wrist F/T — the credibility lever this subfield expects.
- **W6 Writing.** Pre-empt ablations (param-matched control, smoothness, sample-efficiency curve). arXiv within window.
- **GO** if W4 contact subset ≥ **+5pp** over vanilla SmolVLA at matched params. **NO-GO** if no contact-task gain
  AND no smoothness gain → reframe as a smoothness regularizer or merge with a force channel.

## Key risks
- LIBERO alone is NOT contact-rich → must add MetaWorld/RoboSuite force tasks or a real arm.
- Reviewer confusion with the "policy → external VIC" lineage → make the structural difference unmistakable.
- **Stiff K + 10-step Euler may be numerically unstable** → bound K eigenvalues / semi-implicit step (see `toy/` stability check).
- Concurrent work plausible in 3–6 months → move fast; open-source the (M,K,D) head early.

## Infra
Reuses the existing lerobot conda env (`/home/user/miniconda3/envs/lerobot`) + released SmolVLA checkpoint.

## Discipline (lesson from the prior project)
Validate the core object in isolation and clear each kill-gate before scaling: **toy ODE → SmolVLA single-task →
contact subset.** Fail fast; report numbers honestly; no tuning toward a predetermined win.
