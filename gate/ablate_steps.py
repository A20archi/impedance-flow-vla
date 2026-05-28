r"""
ODE-step ablation:  how few integration steps can each arm tolerate?
Inference cost of a flow policy is proportional to the number of ODE steps (network evals), so
"same accuracy at fewer steps" = lower latency.  We sweep num_steps in {5,10,20} for BOTH arms
(param-matched, identical objective) on the headline contact task and report success/jerk/force.

  python gate/ablate_steps.py
Writes out_gate/results_ablate_steps.json and figs/fig3_ode_steps.png
"""
import os, sys, json, pickle, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from policy import CompactFlowPolicy, n_params
from train_eval import build_pairs, train, evaluate, matched_vanilla_hidden, ROOT, DEV

TASK = "peg-insert-side-v3"
N_DEMOS = 50
SEEDS = [0, 1, 2]
EPOCHS = 300
H = 16
EVAL_EPS = 30
MAX_STEPS = 200
STEP_GRID = [5, 10, 20]

def main():
    demos = pickle.load(open(f"{ROOT}/gate_data/demos.pkl", "rb"))
    cond_dim = demos[TASK][0][0].shape[1]
    imp_params = n_params(CompactFlowPolicy(cond_dim, 4, H, hidden=256, mode="impedance"))
    hv = matched_vanilla_hidden(cond_dim, H, imp_params)        # num_steps does not change param count
    hidden = {"impedance": 256, "vanilla": hv}

    X, Y = build_pairs(demos[TASK][:N_DEMOS], H)
    xm, xs = X.mean(0), X.std(0) + 1e-6
    ym, ys = Y.mean((0, 1)), Y.std((0, 1)) + 1e-6

    logf = open(f"{ROOT}/out_gate/progress_ablate_steps.log", "w")
    def log(m):
        print(m, flush=True); logf.write(m + "\n"); logf.flush(); os.fsync(logf.fileno())
    log(f"# ODE-step ablation on {TASK} @ {N_DEMOS} demos | DEV={DEV} | "
        f"param-match impedance={imp_params:,} vanilla(h={hv})")

    results = {}
    for ns in STEP_GRID:
        for arm in ("vanilla", "impedance"):
            per_seed = []
            for sd in SEEDS:
                t0 = time.time()
                pol = train(arm, cond_dim, H, X, Y, xm, xs, ym, ys, hidden[arm], EPOCHS, sd, num_steps=ns)
                m = evaluate(pol, TASK, H, xm, xs, ym, ys, EVAL_EPS, MAX_STEPS)
                per_seed.append(m)
                log(f"  steps={ns:<2d} {arm:9s} seed{sd}: succ={m['success']*100:5.1f}%  "
                    f"jerk={m['jerk']:.4f}  peakF={m['peakF']:8.1f}  ({time.time()-t0:.0f}s)")
            agg = {k: float(np.nanmean([s[k] for s in per_seed])) for k in per_seed[0]}
            sem = {k: float(np.nanstd([s[k] for s in per_seed]) / len(SEEDS) ** 0.5) for k in per_seed[0]}
            results[f"steps{ns}/{arm}"] = {"mean": agg, "sem": sem, "seeds": per_seed}
            log(f"==> steps={ns:<2d} {arm:9s} MEAN succ={agg['success']*100:5.1f}±{sem['success']*100:.1f}  "
                f"jerk={agg['jerk']:.4f}  peakF={agg['peakF']:.1f}")
        json.dump(results, open(f"{ROOT}/out_gate/results_ablate_steps.json", "w"), indent=2)  # checkpoint each grid pt
    log(f"\nwrote {ROOT}/out_gate/results_ablate_steps.json")
    logf.close()

if __name__ == "__main__":
    main()
