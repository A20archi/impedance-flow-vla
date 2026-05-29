# Critical Issues Audit — Impedance-Shaped Flow Matching (Stage-1/2)

**Date:** 2026-05-29  **Scope:** CP1 (MetaWorld MT10) + CP2 (LIBERO-Object), the compact ~0.4–0.7M from-scratch policies. **Purpose:** red-team the results — enumerate every reason the numbers look better than the contribution actually warrants, so claims can be scoped honestly before review.

---

## TL;DR (read this first)

The results are **valid only as a controlled, relative ablation** (impedance drift vs param-matched vanilla drift, identical data/objective). Their *apparent* strength is inflated by a stack of independent factors, none of which is fabrication, but all of which a naive reading over-credits:

1. The training objective **is not flow matching** — it's a deterministic rollout-MSE decoder. The headline framing is a stretch.
2. The mechanism **is not isolated** — "impedance" is confounded with "any smoothness/stability regularizer."
3. The LIBERO task is **trivialized** — behavior cloning, no language, and a canonical-placement shortcut.
4. MT10 is **saturated** — both arms sit at ceiling; the "win" is a near-tie.
5. Comparisons to SOTA (MOORE/Octo/SmolVLA) are **apples-to-oranges**; the param count was **mislabeled**.
6. The statistics are **fragile** — 3 seeds, bimodal collapse, GPU non-determinism.
7. Success metrics are **lenient** (proximity thresholds).

**Severity legend:** 🔴 CRITICAL (undermines the core claim) · 🟠 HIGH (materially inflates results) · 🟡 MEDIUM (caveat / presentation).

---

## 🔴 1. The objective is NOT flow matching — it's a deterministic rollout-MSE decoder

**Claim affected:** the entire thesis ("the flow-matching action expert's ODE drift *is* a closed-loop impedance law").

**The issue.** The Stage-1/2 training loss is **differentiable rollout-matching**: forward-integrate the full ODE over `num_steps` and take MSE between the produced action chunk and the expert chunk. This is **not** the conditional flow-matching (CFM) objective, which regresses a velocity field `v_θ(x_t, t)` toward `(x_1 − x_0)` at random interpolation times `t`. Consequences:
- The model is effectively a **structured deterministic decoder/integrator trained by output regression.** The `torch.randn` "noise" input is **near-vestigial** — with an MSE-to-single-target loss the network learns to wash it out, so there is no real generative/stochastic flow.
- "Impedance is the flow drift" is demonstrated **only under this surrogate objective**, not under the loss that real flow-matching VLAs (SmolVLA) actually use.

**Why it inflates.** Calling it "flow matching" borrows the credibility and generality of CFM while training something simpler and easier to fit. A reviewer who reads "flow matching" and then sees the loss will discount the whole framing.

**Honest fix.** Rename it for what it is (structured ODE decoder / rollout-matching). State explicitly that the **CFM-loss version is untested**. The Stage-3 SmolVLA transfer (impedance drift under genuine velocity-regression CFM) is an **assumption, not a result.**

---

## 🔴 2. The mechanism is not isolated — "impedance" vs "any regularizer" confound

**Claim affected:** that the *impedance structure specifically* (2nd-order SPD mass-spring-damper, ζ≥1, symplectic integration) is what helps.

**The issue.** The win could be explained by far simpler effects we never controlled for:
- An implicit **smoothness regularizer** (the spring-damper just biases toward smooth trajectories).
- Better-conditioned optimization / **stability regularization** (which is exactly why we describe it as a "seed-stability regularizer" on LIBERO — that's a *regularizer* description, not an impedance-specific one).
- We did **not** run the obvious controls: vanilla flow **+ jerk penalty**, vanilla **+ spectral-norm / weight constraints**, vanilla with a smaller LR or output low-pass. Any of these might close much of the +20pp gap.
- The **`imp_dims=6` gripper-exemption was hand-tuned** and *flipped* LIBERO from inconclusive to a win — evidence the benefit is **sensitive to design choices**, not a robust universal prior.

**Why it inflates.** Without these controls, "impedance helps" is not separable from "we added an inductive bias toward smooth, stable trajectories, and tuned it." The novelty hook (predicted vs fixed M/K/D — Stage-2) is **not validated**.

**Honest fix.** Run the regularizer-control baselines and the Stage-2 conditioning ablation before claiming the *structure* is the cause.

---

## 🔴 3. The LIBERO task is trivialized (BC + no language + canonical shortcut)

**Claim affected:** any reading of CP2 as "LIBERO-Object benchmark performance."

**The issues (compounding):**
- **(a) Behavior cloning, per object.** We train 10 *separate single-task* policies, not one generalist. Each is a specialist that always grabs the same object.
- **(b) No language / no text encoder.** The instruction is never fed to the network; it's metadata only.
- **(c) Canonical-placement shortcut (verified in BDDLs).** In every task the target object is relocated to the *same* `floor_target_object_region`; distractors fill the other regions. So "go to the canonical region and grasp" solves the task — **object identity is supplied by position, not by recognition or language.** This is a documented weakness of LIBERO-Object; we lean on it fully.
- **(d) Vision does localization, not recognition.** The blind-test (80→0) proves vision is load-bearing for *finding the randomized within-region position*, but the policy never has to tell one object from another.

**Why it inflates.** The benchmark is *designed* to test language grounding and object disambiguation; we removed both. Our task is "visually-servoed reach-grasp-place of one object in a known region from scratch." That is genuinely easier than what "LIBERO-Object" implies.

**Honest fix.** Relabel as **"per-object visuomotor BC on LIBERO-Object scenes (target identity via canonical placement; no language)."** Never present 48% as a benchmark number.

**What this does NOT break.** It is *not* trivial fixed-trajectory replay — the object position is randomized within the region and a frozen/open-loop policy scores 0%. It's closed-loop visuomotor BC. The triviality is in the *scope*, not in faked dynamics.

---

## 🟠 4. Statistical fragility — 3 seeds, bimodal collapse, GPU non-determinism

**Claim affected:** every per-task number and the "+20pp / wins 7/10" aggregate.

**The issues:**
- **Bimodal seeds.** From-scratch small models collapse to 0% or shoot to ~100% per seed. The reported means are dominated by *which seeds caught*: cream cheese ours = `[0.95, 0, 0]` → "31.7%"; salad dressing `[0, 1, 0.85]`; choc pudding `[1, 0, 0]`. These are not stable estimates.
- **n = 3 seeds** cannot resolve a signal inside 0–100% per-seed variance; SEMs are huge (often ±0.13–0.27).
- **GPU non-determinism.** Re-running the *same* seed-0 peg-insert policy gave **90%** vs the stored **80%/60%** — i.e. ±10–30% rollout-level noise even at fixed seed. Per-cell numbers are noisy point estimates, not reproducible constants.
- **Small eval n** (CP2: 20 eps/seed; CP1: 10 rollouts/seed).

**Why it inflates.** With this much variance and only 3 seeds, the aggregate confidence interval is wide; "wins 7/10" overstates a noisy mean. Favorable aggregates are partly luck of which seeds caught.

**Honest fix.** 5–10 seeds minimum; report bootstrap CIs; lead with the *low-SEM* wins (alphabet soup, tomato sauce, butter — all 3 seeds fire) and explicitly down-weight the one-seed numbers.

---

## 🟠 5. MT10 (CP1) is saturated and is BC-from-oracle

**Claim affected:** "96.0 → 97.7% on MT10."

**The issues:**
- **8/10 tasks at 100% for both arms** → no headroom, the "win" is a **near-tie at ceiling** (the difference is reach + peg, the only non-saturated tasks).
- **BC from MetaWorld's scripted oracle** (50 near-perfect demos/task) on kinematically easy tasks → ~95%+ is *expected and unimpressive*, not a benchmark achievement.
- Therefore the near-identical baseline-vs-ours **videos are honest (saturation), but visually unconvincing** and easy to misread as "cloned."

**Why it inflates.** A 97.7% headline reads as "near-perfect manipulation"; it actually means "imitation of an oracle on easy tasks, tied with the control."

**Honest fix.** Present MT10 as a **saturation control**, not a result. The real evidence is the low-data peg-insert ablation and CP2.

---

## 🟠 6. SOTA comparisons are apples-to-oranges; the param count was mislabeled

**Claim affected:** any side-by-side with MOORE / Octo / SmolVLA, and the "~0.5M param-matched" line.

**The issues:**
- **MOORE (88.7% MT10)** is multi-task **RL from scratch**; we do **BC from oracle**. Different problem. Our higher MT10 number does **not** beat it.
- **Octo-Small (~65%, 27M)** and **SmolVLA (97%, 450M)** are **pretrained on 800k+ trajectories** and **language-conditioned**; we are from-scratch, 0.5–0.7M, no language. Absolute comparison is meaningless.
- **Param-count mislabel (reporting bug).** `metrics.json` `param_counts` (479,974 / 499,248) is the **flow head only** — it omits the shared **238,640-param image encoder**. True totals: **ours 718,614 / baseline 737,888 (~0.72–0.74M)**. CP1 (state-based, no encoder) is unaffected.

**Why it inflates.** Quoting "0.5M" undersells size by ~1.5×; quoting our success next to SOTA invites a comparison we'd lose and don't intend.

**Honest fix.** Report **~0.72M total**; fix the label in code/metrics; never table absolute success against pretrained/RL models — only the matched-baseline delta.

> **Param label FIXED (2026-05-29).** `outputs/checkpoint_2_libero/metrics.json` `param_counts` now reports the flow-head, the 238,640 shared CNN encoder, and the **per-arm totals (ours 718,614 / baseline 737,888)** explicitly. The CFM pipeline (`pipelines/checkpoint_3_libero_cfm.py`) reports totals natively. The *SOTA-positioning* half of this issue is presentation discipline, not a code fix — still on the author. (#1 also now partially addressed: a genuine-CFM pipeline exists but is not yet run at scale.)

---

## 🟡 7. Smoothness / contact-force claims are regime-specific and conditional

**Claim affected:** "~40% lower jerk," "−18% peak force."

**The issues:**
- **Jerk reduction shows only at low data / contact.** At 50 demos (CP1) jerk is comparable; ours is *higher* on door-open. The "40%" is a **10-demo peg-insert** result.
- **Force claim is conditional:** "−18% **at matched success** (25 demos)." At 10 demos ours has *higher* peak force (it touches more because it succeeds more). Stated out of context it reverses.
- **On LIBERO, jerk "tracks success"** — failed seeds have high jerk — so part of the "smoothness win" is just "successful policies are smoother," not a pure impedance effect.

**Honest fix.** Always state the regime and the "at matched success" qualifier. Don't generalize the jerk/force numbers.

---

## 🟡 8. Success metrics are lenient (proximity thresholds)

**Claim affected:** the raw success rates, esp. peg-insert.

**The issue.** MetaWorld `success` for peg-insert fires at `obj_to_target < ~0.07 m` (a **proximity** criterion), not a snug mechanical insertion. LIBERO `is_success` = object in basket region. These are the benchmarks' own standards (fair to use) but **looser than a human's notion of "did it really do the task,"** which feeds the "feels forcibly simulated" reaction.

**Honest fix.** State the exact success definition wherever success rates appear.

---

## 🟡 9. Video artifacts undermine trust (cosmetic, but real)

- **SUCCESS badge hardcoded off** in the CP1 renderer → viewer never sees success.
- **Clip cuts at the success instant** → abrupt, looks unfinished.
- **Rendered seed (4242) is unlabeled and is not an eval seed** → footage doesn't substantiate the reported numbers.

**Honest fix.** Re-render with success/fail badge, goal-distance readout, hold past success, and pick non-saturated contrastive seeds.

---

## ✅ 10. Train/eval init-state overlap — VERIFIED DISJOINT (no leakage) *(resolved 2026-05-29)*

**The issue (as raised).** Eval uses `_init_state_id = 0..n-1` (LIBERO's stored init-states via `set_init_state(self._init_states[self._init_state_id])`). Were these held out from the 50 training-demo start states?

**Resolution — checked directly.** Parsed the 110-dim sim init-state; isolated the **22 dimensions that vary** across inits (object/robot placement). On those placement dims the eval init pool and the training-demo init pool are **disjoint: set overlap = 0/50**, nearest training placement to each eval init ≈ **2–3 cm** (never identical). (Two naive checks first gave false signals — a constant **dim-82** format offset of 0.98 with std=0, and a uniform ~2000 render-pipeline MSE floor — both were artifacts, excluded.) **There is no train/eval state leakage.**

**Residual caveat (→ #11).** Nearest training placement is only ~2–3 cm away, so eval is **held-out but in-distribution interpolation**, not out-of-distribution generalization. #10 is closed; OOD remains open under #11.

---

## 🟡 11. Generalization is entirely untested (open)

No novel objects, no clutter beyond the canonical layout, no off-canonical placement test, no language, no real robot, no larger model, no real CFM loss. Every transfer claim (Stage-3) is an assumption.

---

## What honestly survives all of the above

- **A controlled, param-matched ablation** in which the *only* difference between two arms is the ODE-drift structure.
- Under that control, the impedance-structured arm shows, **in an easy from-scratch BC setting**: a **+20pp** mean delta on contact-rich grasp-place (carried by 4 low-variance task wins), **seed-stabilization** (fewer collapses), and **lower jerk / contact force in the low-data, matched-success regime**.
- The policy is **genuinely closed-loop visuomotor** (blinding → 0%) and **not cloned** (arms emit different actions).

That is a legitimate **proof-of-mechanism for a structured ODE decoder**, *not* a benchmark result and *not* (yet) a flow-matching or language-grounded or scaled result.

---

## Required experiments to de-risk (priority order)

1. **Real CFM loss** — retrain the impedance arm under genuine velocity-regression flow matching; show the prior still helps. (De-risks #1.)
2. **Regularizer controls** — vanilla + jerk penalty, + spectral norm, + output low-pass. Show impedance beats *these*, not just unstructured vanilla. (De-risks #2.)
3. **Stage-2 conditioning ablation** — observation-predicted vs fixed/global M/K/D vs stiffness-only. Validate the actual novelty. (De-risks #2.)
4. **Off-canonical object test** — move the target out of the canonical region at eval; show the policy tracks it. (De-risks #3d.)
5. **More seeds (5–10) + bootstrap CIs.** (De-risks #4.)
6. **Frozen pretrained vision encoder** (DINOv2/SigLIP) — close the from-scratch gap, test if the prior stacks. (De-risks #3, #6.)
7. **Verify held-out eval init-states.** (De-risks #10.)
8. **A genuinely contact-rich, non-shortcut benchmark** (ManiSkill PegInsertion/PlugCharger). (De-risks #3, #11.)
