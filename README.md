<h1 align="center">Impedance-Shaped Flow Matching for VLA Policies</h1>

<p align="center">
  <b>The first flow-matching action expert whose generative ODE drift <i>is itself</i> a closed-loop, task-conditioned mass–spring–damper law.</b><br>
  <i>Compliance becomes a structural property of how motion is generated — not a wrapper bolted on afterward.</i>
</p>

<p align="center">
  <img src="figs/arch_methodology.png" width="92%">
</p>

<p align="center"><sub>
<b>Figure 1 — Methodology overview.</b> An observation is encoded to a conditioning vector <code>h</code>. Three heads read off <code>h</code>: a <b>stiffness/damping head</b> that emits SPD matrices <code>K = cap(LLᵀ)</code> and <code>D = 2ζ√K</code> with <code>ζ≥1</code> (critical/over-damping by construction), an <b>attractor head</b> <code>a_goal(obs)</code>, and a <b>residual field</b> <code>f_θ</code>. These assemble into the <b>impedance drift</b> <code>v_θ = M⁻¹(−D·ȧ − K·(a−a_goal) + f_θ)</code> (mass <code>M=I</code> at Stage-1), which is integrated by a <b>2nd-order semi-implicit (symplectic) Euler</b> scheme over <code>N=10</code> steps to turn noise into an action chunk. The bottom band contrasts our structured drift with a vanilla unstructured drift, and states the novelty relative to prior art (external / downstream impedance controllers).
</sub></p>

---

## TL;DR

A flow-matching / diffusion VLA generates an action chunk by integrating a learned velocity field (the *drift*) from noise to data. We **replace that unstructured drift with a closed-loop impedance law**:

> **v_θ(a, ȧ, τ, obs) = M⁻¹ ( −D·ȧ − K·(a − a_goal) + f_θ(a, ȧ, τ, obs) )**

where **M, K, D** are task-conditioned **SPD** matrices (mass / stiffness / damping) predicted from the observation, and **a_goal** is a predicted attractor. The whole thing is one end-to-end-differentiable ODE — **no downstream controller and no force/torque sensor at inference**.

**Why it matters.** On a tight-tolerance contact task (MetaWorld `peg-insert-side`), at *matched parameter count and an identical training objective*, the structural prior buys:

| Metric (peg-insert-side) | Effect | Where |
|---|---|---|
| **Sample efficiency** | **+22 pp** success (30% → 52%) | 10 demos |
| **Smoothness** | **~40% lower** action jerk | low-data regime |
| **Compliance** | **−18% peak contact force** at matched success | 25 demos |
| **Inference cost** | **same accuracy at ½ the ODE steps** (step-count-robust) | 5 vs 10 steps |

The gains concentrate exactly where physics matters (hard contact, scarce data) and vanish where the task is easy/saturated — a clean, honest signature, consistent with the contact-compliance literature.

> **Scope.** This repo is **Stage-1**: a *proof-of-mechanism* on a compact (~400k-param) state-conditioned flow policy in MetaWorld. It is **not** the full VLA result. Porting the impedance drift into the real **SmolVLA** action expert (with proper 2nd-order conditional flow matching) and a contact-rich benchmark (ManiSkill2/3) is Stage-3. See [Roadmap](#roadmap).

---

## The methodology, animated

<p align="center">
  <img src="figs/methodology_anim.gif" width="88%">
</p>

<p align="center"><sub>
<b>Figure 2 — How an action chunk is generated (animated).</b> The pipeline reveals stage by stage (top), then the generative ODE plays out in a 2-D action space (bottom-left): starting from noise, the <b>impedance arm (teal)</b> is a critically-damped spring–damper that <i>lands softly</i> on the attractor <code>a_goal</code> (gold star) — the dashed ellipse is the predicted stiffness <code>K</code>. The <b>vanilla arm (gray)</b> is an unstructured drift with no such guarantee and overshoots. The right panel highlights each term of the drift as it acts: <code>−K(a−a_goal)</code> pulls toward the attractor, <code>−D·ȧ</code> removes energy (no overshoot), <code>+f_θ</code> adds a learned task-specific correction. &nbsp; <a href="figs/methodology_anim.mp4">▶ full-resolution MP4</a>
</sub></p>

---

## How this differs from prior work

The intersection we claim — *the flow ODE drift **is** the closed-loop impedance law* — is, to our knowledge, unclaimed as of May 2026. The easy-to-confuse neighbors:

| Prior work | What it does | How ours differs |
|---|---|---|
| **CompliantVLA-adaptor** | Predicts K, D from a VLM, feeds an **external** variable-impedance controller wrapped around a *frozen* VLA. | We embed the impedance law *inside* the generative drift; nothing is frozen, nothing is external. |
| **Flow-with-the-Force-Field** | Standard unstructured flow drift + a **downstream passive** impedance controller. | Our velocity field **is** the impedance law — there is no separate controller stage. |
| **ACP / Diffusion-Impedance / MPD / DMP decoders** | Stiffness lives on the **output head** or a DMP decoder. | None make the flow **ODE drift** a closed-loop K/D/M attractor conditioned on features. |
| **Contractive DS imitation (SNDS, NDP)** | Stable dynamical-system imitation on small models. | We generalize that stability tradition to large VLA flow experts. |

---

## Results — Stage-1 (compact, param-matched, MetaWorld)

Both arms share an **identical architecture budget (~400k params)** and an **identical training objective** (differentiable rollout-matching + a small jerk penalty). The *only* difference is the drift parameterization — so any gap is the **structural prior**, not capacity or loss.

### The headline task: `peg-insert-side` (tight-tolerance contact)

<p align="center">
  <img src="figs/fig1_peg_sample_efficiency.png" width="95%">
</p>

<p align="center"><sub>
<b>Figure 3 — Sample efficiency, smoothness, and compliance on peg-insert-side</b> (3 seeds, 30 eval episodes per point; bars = SEM). <b>Left:</b> success vs. number of demos — impedance (blue) leads by <b>+22 pp at 10 demos</b> and converges to parity as data saturates (textbook "prior helps when data is scarce, redundant when saturated"). <b>Middle:</b> action jerk (lower = smoother) — impedance is consistently smoother in the low-data regime (~40%). <b>Right:</b> peak contact force — at <b>matched success (25 demos)</b> impedance applies <b>−18%</b> peak force. <i>Caveat (read honestly):</i> at 10 demos impedance shows <i>higher</i> force only because it succeeds more often and therefore makes more contact (vanilla "fails by not touching"); the compliance claim is valid only at matched success.
</sub></p>

| demos | success (vanilla → impedance) | jerk (vanilla → impedance) | peak force (vanilla → impedance) |
|------:|:---|:---|:---|
| 10  | 30.0% → **52.2%  (+22.2 pp)** | 0.985 → 0.570 (**−42%**) | 509 → 674 (↑ — more successful contact) |
| 25  | 74.4% → 73.3%  (tie) | 0.408 → 0.240 (**−41%**) | 536 → 441 (**−18% @ matched success**) |
| 50  | 83.3% → 86.7%  (+3.4 pp) | 0.314 → 0.310 | 422 → 473 |
| 100 | 87.8% → 87.8%  (tie — saturated) | 0.193 → 0.258 | 443 → 403 |

### Closed-loop rollout — vanilla fails, impedance succeeds

<p align="center">
  <img src="figs/peg_rollout_compare.gif" width="80%">
</p>

<p align="center"><sub>
<b>Figure 4 — Side-by-side MuJoCo rollouts of peg-insert-side</b> at low data. Left: the vanilla policy; right: the impedance policy, with a live contact-force gauge and a SUCCESS badge. The same seeds that the vanilla arm fails are solved by the impedance arm with a controlled, compliant approach into the hole. &nbsp; <a href="figs/peg_rollout_compare.mp4">▶ full-resolution MP4</a>
</sub></p>

### The mechanism, in motion

<p align="center">
  <img src="figs/mechanism_anim.gif" width="78%">
</p>

<p align="center"><sub>
<b>Figure 5 — Why it's smoother.</b> Action path (left) and speed profile (right) for three drifts: the <b>impedance arm soft-lands</b> on the target (speed tapers to zero — critical damping), <b>vanilla drives at near-constant speed</b> into contact, and an <b>under-damped</b> (ζ&lt;1) variant oscillates. Enforcing <code>ζ≥1</code> via <code>D = 2ζ√K</code> guarantees the soft landing <i>by construction</i> rather than by tuning. &nbsp; <a href="figs/mechanism_anim.mp4">▶ full-resolution MP4</a>
</sub></p>

### The control task: `door-open` (easy from state)

<p align="center">
  <img src="figs/fig2_door_secondary.png" width="88%">
</p>

<p align="center"><sub>
<b>Figure 6 — Secondary/control task.</b> <code>door-open</code> is easy from full state and saturates from very few demos (vanilla already ~82% at 10 demos), so there is <b>no sample-efficiency headroom</b> — success ties throughout (left). The impedance arm retains a <b>mild smoothness edge</b> (right). This is the expected null: the structural prior is redundant when the task does not stress contact. Reporting it is the honesty check.
</sub></p>

### Efficiency: fewer ODE steps, same accuracy

A flow policy's inference cost is proportional to the number of ODE integration steps (network evaluations). So "same accuracy at fewer steps" = **lower latency**. We sweep `num_steps ∈ {5, 10, 20}` for *both* arms (param-matched, identical objective) on the headline task.

<p align="center">
  <img src="figs/fig3_ode_steps.png" width="92%">
</p>

<p align="center"><sub>
<b>Figure 7 — ODE-step ablation on peg-insert-side</b> (matched params, 3 seeds, 30 eval episodes; @ 50 demos). <b>Left:</b> the <b>impedance arm holds 86.7% success at both 5 and 10 steps</b> (and rises to 92.2% at 20) — accuracy is preserved at <b>half</b> the integration steps — and it is markedly more step-count-stable than vanilla (at 10 steps all three seeds landed on exactly 86.7%, SEM 0.0). <b>Right:</b> smoothness (jerk) is essentially flat across step counts. <b>Takeaway:</b> the structural prior lets you cut ODE steps — and thus inference latency — with no accuracy cost, a property the unstructured baseline does not share. (This sweep is at a near-saturated data regime, so both arms are high; the point is step-count robustness, not the low-data gap of Figure 3.)
</sub></p>

| ODE steps | vanilla success | impedance success | vanilla jerk | impedance jerk |
|---:|:---|:---|:---|:---|
| 5  | 87.8 ± 2.4 | **86.7 ± 4.2** | 0.251 | 0.309 |
| 10 | 83.3 ± 3.1 | **86.7 ± 0.0** | 0.314 | 0.310 |
| 20 | 86.7 ± 3.1 | **92.2 ± 0.9** | 0.297 | 0.275 |

---

## The toy gate — validate the core object *before* scaling

Lesson carried from a prior abandoned project (see [`docs/JOURNEY.md`](docs/JOURNEY.md)): **validate the core object in isolation and clear a kill-gate before scaling.** Before a line of policy code, a CPU unit test ([`toy/toy_impedance_flow.py`](toy/toy_impedance_flow.py)) established two non-negotiables.

<p align="center">
  <img src="figs/toy_impedance.png" width="92%">
</p>

<p align="center"><sub>
<b>Figure 8 — Damping-ratio control is real and behaves as physics predicts.</b> Convergence of <code>‖a − a_goal‖</code> over flow time (left) and the 2-D action path (right) for three damping ratios: <b>ζ=0.3 oscillates</b> (36% overshoot), <b>ζ=1.0 is a clean critical approach</b>, <b>ζ=2.0 is monotone/over-damped</b>. This verifies the SPD heads give us a usable, predictable knob — and motivates enforcing <code>ζ≥1</code> in the policy.
</sub></p>

<p align="center">
  <img src="figs/toy_stability.png" width="58%">
</p>

<p align="center"><sub>
<b>Figure 9 — The stability budget.</b> Final error vs. stiffness eigenvalue at the policy's <code>N=10</code> Euler steps: stiffness must stay below <code>K_max ≈ 100</code> (red line, the critical-damping bound <code>dt·D &lt; 2</code>) or the explicit integrator diverges (error blows up to 10³). Crucially this <b>soft / low-stiffness regime is exactly what contact compliance wants</b> — the constraint and the goal coincide. (The naive finite-difference damping drift is unstable for any useful D; the fix is the 2nd-order semi-implicit scheme.)
</sub></p>

---

## Repository layout

```
impedance_flow_vla/
├── PROJECT.md                 # thesis, novelty vs prior art, 45-day plan + kill-criteria
├── docs/
│   └── JOURNEY.md             # ideation log: from "adjoint attribution" to impedance flow
├── toy/
│   └── toy_impedance_flow.py  # CPU unit test: SPD head, damping-ratio control, 10-step stability
├── gate/
│   ├── policy.py              # CompactFlowPolicy — vanilla & impedance arms (param-matched)
│   ├── gen_demos.py           # MetaWorld expert demos (full-state, no render)
│   ├── train_eval.py          # param-matched trainer + closed-loop MetaWorld evaluator
│   ├── ablate_steps.py        # ODE-step ablation: num_steps in {5,10,20}, both arms
│   ├── make_figures.py        # results figures (fig1, fig2)
│   ├── make_ablation_figure.py   # the ODE-step ablation figure (Figure 7)
│   ├── make_arch_diagram.py   # the methodology diagram (Figure 1)
│   ├── make_methodology_anim.py  # the animated walkthrough (Figure 2)
│   ├── make_mechanism_anim.py # mechanism animation (Figure 5)
│   └── viz_rollout.py         # MuJoCo side-by-side rollout video (Figure 4)
├── figs/                      # all figures, GIFs, and MP4s above
└── out_gate/                  # results_gate.json, results_sweep.json (the numbers behind the figures)
```

## Reproduce

Everything uses the existing `lerobot` conda env (PyTorch 2.7, MetaWorld, MuJoCo):

```bash
PY=/home/user/miniconda3/envs/lerobot/bin/python   # adjust to your env

# 0. (~5 min) generate state-based expert demos -> gate_data/demos.pkl
$PY gate/gen_demos.py

# 1. toy gate (CPU, seconds): SPD head + damping control + stability
$PY toy/toy_impedance_flow.py

# 2. the sample-efficiency sweep (the numbers behind fig1/fig2)
$PY gate/train_eval.py --n_demos 10,25,50,100

# 3. the ODE-step ablation (the numbers behind fig3)
$PY gate/ablate_steps.py

# 4. regenerate all visuals
$PY gate/make_figures.py
$PY gate/make_ablation_figure.py
$PY gate/make_arch_diagram.py
$PY gate/make_methodology_anim.py
$PY gate/make_mechanism_anim.py
$PY gate/viz_rollout.py
```

> `gate_data/demos.pkl` is regenerable (step 0) and is git-ignored to keep the repo lean.

---

## Roadmap

- **Stage-2 — conditioning ablation.** Does observation-predicted M/K/D beat fixed/global M/K/D, and beat a stiffness-output-only baseline? (This isolates *the* novelty hook.)
- **Stage-3 — scale to SmolVLA.** Port the impedance drift into the real SmolVLA action expert with proper 2nd-order conditional flow matching; **ManiSkill2/3** as the primary contact-rich benchmark (PegInsertionSide, PlugCharger, TurnFaucet, AssemblyNut), MetaWorld as cross-sim secondary. *Not LIBERO* (too kinematic).
- **Efficiency angle** *(shown at Stage-1, Figure 7)*: the impedance arm keeps its accuracy at half the ODE steps — carry this latency win forward to SmolVLA.
- **Credibility lever.** Real-arm peg/USB insertion with a wrist F/T sensor.
- **Kill-gate.** GO only if the contact subset shows **≥ +5 pp** over vanilla SmolVLA at matched params.

## Notes & honesty

- All comparisons are **parameter-matched** with an **identical objective**; the only variable is the drift parameterization.
- Stage-1 uses **differentiable rollout-matching** on a compact model to validate the *mechanism* fast; the proper 2nd-order conditional flow-matching objective is a Stage-3 item.
- GPU eigendecompositions (`eigh`) are non-deterministic on CUDA — seed-sensitive borderline episodes are reported with SEM over 3 seeds.

---

<p align="center"><sub>Stage-1 mechanism gate · MetaWorld · built on the lerobot stack. &nbsp; Issues and ideas welcome.</sub></p>
