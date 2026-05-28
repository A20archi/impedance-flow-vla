# From "Adjoint Attribution" to "Impedance-Shaped Flow" — the ideation log

*Archive of the research journey that produced this project. Written 2026-05-28.
Doubles as the script for the opening slides of the Saturday talk.*

---

## Act I — Where we started: Adjoint Attribution for Flow-Matching Policies

**The pitch.** *"Why Attention Explanations Are Provably Wrong."* Take a flow-matching VLA
(SmolVLA × MetaWorld ML10) and use the **adjoint sensitivity** of the action-generating ODE as a
principled attribution — then show that the attention maps everybody reads off transformers are a
misleading proxy for what actually drives the action.

Four bids for novelty, brainstormed in order:

1. **"Attention is provably wrong"** — claim attention rankings disagree with the true sensitivity.
2. **Second-order / interaction adjoint** — go beyond first-order gradients to curvature/interactions.
3. **Adjoint-KL fine-tuning** — use the adjoint signal to fine-tune the policy (the performance lever).
4. **Test-time goal-steering** — steer generation along adjoint directions at inference (the other lever).

## Act II — Why we abandoned it

Every model-internal claim dissolved the moment it was controlled for scale and token granularity:

| Idea | What happened |
|---|---|
| Attention "provably wrong" | **Confound.** Attention vs adjoint actually *agree* on rank (Spearman +0.52, top-1 100%) once token granularity is matched. |
| 2nd-order interaction adjoint | **Artifact.** The dramatic state-token curvature (90–122×) was an embedding-scale effect: state emb-norm ≈7 vs image ≈3700 (~518×), so unit perturbations weren't comparable. |
| Adjoint-KL fine-tuning | **Null.** +1.7 vs a proper control — within noise at n=12. |
| Test-time goal-steering | **Net-negative on ML10.** Helps free-space reach (37.5→75%) but *collapses* every contact task (push, window-open → 0%) at every steering strength. |

**The one clean result that survived** (analysis-only, not the method/performance paper we wanted):
an **interventional ground truth** — counterfactually replacing each input modality and measuring the
real Δaction — showed the grasp is causally driven by **state > language > vision**, and that
**attention's modality ranking is the *reverse* of causal truth** while the adjoint's matches.
Corollary: *causal importance ≠ control authority* (vision is the best steering channel by DOF despite
being the least causal modality).

**Verdict (2026-05-28): abandoned.** It never met the bar of *genuine novelty + an improving number*.

**The lesson worth keeping:** model-internal attributions (attention mass, gradient norm, curvature)
keep producing striking stories that evaporate under normalization. Only an **interventional** measure
is confound-free. So: *validate the core object in isolation, clear a kill-gate before scaling, report
numbers honestly.* That discipline — not any single result — is what carried into the next project.

---

## Act III — The pivot: make compliance *structural*, not an explanation

If steering a trained flow policy collapses contact tasks, the problem isn't the explanation — it's that
the policy has **no notion of physical compliance** baked into how it generates motion. So instead of
*attributing* behaviour after the fact, **build the right inductive bias into the generator itself.**

### The one new object
The flow-matching action expert's ODE **drift is itself a closed-loop second-order impedance law**:

> **v_θ(a_τ, τ, obs) = M⁻¹ ( −D·ȧ_τ − K·(a_τ − a_goal) + f_θ(a_τ, τ, obs) )**

- **M, K, D** are task-conditioned **SPD** matrices (mass / stiffness / damping), Cholesky-parameterized,
  predicted from features.
- **ȧ_τ** is an **explicit velocity state** — the flow is a 2nd-order augmented ODE, integrated
  **semi-implicitly (symplectic Euler)**. The naive finite-difference damping form is unstable (the Δτ
  cancels) — the toy gate proved this before we wrote a line of model code.
- End-to-end differentiable. **No downstream controller. No force/torque sensor at inference.**

### Why it's novel (and what it is *not*)
- vs **CompliantVLA-adaptor**: predicts K,D from a VLM but feeds an **external** impedance controller
  wrapped around a *frozen* VLA. We embed the law *inside* the generative drift.
- vs **Flow-with-Force-Field**: standard unstructured flow drift + a **downstream** passive controller.
  Ours: the velocity field **is** the impedance law.
- vs **ACP / Diffusion-Impedance / DMP decoders**: stiffness lives on the *output head*; none make the
  flow ODE drift a closed-loop K/D/M attractor.

---

## Act IV — Evidence so far (Stage-1, compact state-conditioned flow policy)

**Discipline applied:** toy ODE gate → compact param-matched policy on MetaWorld → (next) SmolVLA.

**Toy gate — CLEARED.** SPD heads verified PSD; damping-ratio control verified (ζ=0.3→36% overshoot,
ζ=1→clean, ζ=2→monotonic); stability bound established: at N Euler steps, usable stiffness
K ≲ (num_steps)² — the soft/compliant regime, which is exactly what contact wants.

**Stage-1 result (vanilla vs impedance, MATCHED ~400k params, 3 seeds, 30 eval eps):**

*peg-insert-side (tight-tolerance contact — the hard task):*
| demos | success (van→imp) | jerk (van→imp) | peak force (van→imp) |
|---|---|---|---|
| 10  | 30% → **52%  (+22pp)** | 0.99 → 0.57 (**−42%**) | 509 → 674 (↑ — more successful contact) |
| 25  | 74% → 73% (tie) | 0.41 → 0.24 (**−41%**) | 536 → 441 (**−18% @ matched success**) |
| 100 | 88% → 88% (tie — saturated) | 0.19 → 0.26 | 443 → 403 |

*door-open (easy-from-state — the control task):* saturates from few demos (vanilla 82% @10 demos) →
**ties on success**, mildly smoother at full data. Benefit concentrates on hard contact, as expected.

**Read:** a textbook **sample-efficiency** signature (big low-data gain that closes as data saturates),
**smoother** actions at low data, and **lower contact force at matched success** — at identical
parameter count and an identical training objective. The structural prior helps exactly where physics
matters and is redundant where the task is easy.

---

## Act V — Roadmap

- **Stage-2:** conditioning ablation — VLM-predicted M/K/D vs fixed; stiffness-output-only baseline.
- **Stage-3:** port the impedance drift into the **SmolVLA** action expert with the proper 2nd-order CFM.
- **Primary benchmark:** **ManiSkill2/3** contact-rich suite (PegInsertionSide, PlugCharger, TurnFaucet,
  AssemblyNut) — reviewer-accepted; MetaWorld peg-insert/door as cross-sim secondary. *Not LIBERO* (kinematic).
- **Efficiency angle:** ODE-step ablation (5 vs 10 steps) — same accuracy, fewer steps = lower latency.
- **Credibility lever:** real-arm peg/USB insertion with a wrist F/T sensor.
- **Kill-gate:** GO only if the contact subset shows ≥ +5pp over vanilla SmolVLA at matched params.
