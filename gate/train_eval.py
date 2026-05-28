r"""
STAGE-1 mechanism gate: train vanilla vs impedance (PARAM-MATCHED) flow policies on MetaWorld
full-state demos, eval closed-loop.  Lead metrics (low variance): action JERK + peak CONTACT FORCE.
Success rate secondary.  Multiple seeds.  Same objective for both arms (rollout-matching) -- only the
drift parameterization differs.

  python train_eval.py --tasks peg-insert-side-v3 --arms vanilla,impedance --seeds 0,1,2 --epochs 300
  python train_eval.py --quick      # 1 task, 1 seed, tiny -- smoke test
"""
import os, sys, json, pickle, argparse, time
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from policy import CompactFlowPolicy, n_params
from lerobot.envs.metaworld import MetaworldEnv

DEV = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = "/home/user/Desktop/Saptarshi/impedance_flow_vla"
DEMOS = f"{ROOT}/gate_data/demos.pkl"


def build_pairs(eps, H):
    """(obs[t], action chunk[t:t+H] pad-by-repeat-last) over all episodes."""
    Xs, Ys = [], []
    for obs, act in eps:
        T = len(obs)
        for t in range(T):
            chunk = act[t:t + H]
            if len(chunk) < H:
                chunk = np.concatenate([chunk, np.repeat(chunk[-1:], H - len(chunk), 0)], 0)
            Xs.append(obs[t]); Ys.append(chunk)
    return np.stack(Xs).astype(np.float32), np.stack(Ys).astype(np.float32)


def matched_vanilla_hidden(cond_dim, H, target_params):
    best = (256, 1e18)
    for hv in range(192, 640, 16):
        p = n_params(CompactFlowPolicy(cond_dim, 4, H, hidden=hv, mode="vanilla"))
        if abs(p - target_params) < best[1]:
            best = (hv, abs(p - target_params))
    return best[0]


def train(mode, cond_dim, H, X, Y, xm, xs, ym, ys, hidden, epochs, seed, bs=256, lr=1e-3):
    torch.manual_seed(seed); np.random.seed(seed)
    pol = CompactFlowPolicy(cond_dim, 4, H, hidden=hidden, mode=mode).to(DEV)
    opt = torch.optim.Adam(pol.parameters(), lr=lr)
    Xn = torch.tensor((X - xm) / xs, device=DEV)
    Yn = torch.tensor((Y - ym) / ys, device=DEV)
    N = len(Xn)
    for ep in range(epochs):
        perm = torch.randperm(N, device=DEV)
        for i in range(0, N, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            L, _ = pol.loss(Xn[idx], Yn[idx])
            L.backward(); opt.step()
    return pol


def contact_force(env):
    """Total external contact-force magnitude over bodies (MuJoCo cfrc_ext). Defensive."""
    try:
        d = env._env.unwrapped.data
        return float(np.linalg.norm(np.asarray(d.cfrc_ext)[:, :3], axis=1).sum())
    except Exception:
        return float("nan")


@torch.no_grad()
def evaluate(pol, task, H, xm, xs, ym, ys, eval_eps, max_steps, exec_h=4, seed0=5000):
    env = MetaworldEnv(task=task, obs_type="pixels_agent_pos", camera_name="corner2")
    succ = 0; jerks = []; peakF = []
    for ep in range(eval_eps):
        raw_obs, _ = env._env.reset(seed=seed0 + ep)
        chunk = None; sip = 0; ok = False; acts = []; fs = []
        for t in range(max_steps):
            if chunk is None or sip >= exec_h:
                cond = torch.tensor(((np.asarray(raw_obs, np.float32) - xm) / xs)[None], device=DEV)
                noise = torch.randn(1, H, 4, device=DEV)
                gen = pol(cond, noise)[0].cpu().numpy()
                chunk = (gen * ys + ym).astype(np.float32); sip = 0
            a = np.clip(chunk[sip], -1, 1); sip += 1
            acts.append(a)
            raw_obs, r, done, trunc, info = env._env.step(a)
            fs.append(contact_force(env))
            if info.get("success", 0):
                ok = True; break
        succ += int(ok)
        acts = np.stack(acts)
        if len(acts) >= 3:
            jerks.append(float(np.mean(np.sum((acts[2:] - 2 * acts[1:-1] + acts[:-2]) ** 2, -1))))
        fs = [f for f in fs if np.isfinite(f)]
        if fs:
            peakF.append(float(np.percentile(fs, 95)))      # 95th-pct contact force (robust peak)
    env.close()
    return {"success": succ / eval_eps, "jerk": float(np.mean(jerks)) if jerks else float("nan"),
            "peakF": float(np.mean(peakF)) if peakF else float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="peg-insert-side-v3")
    ap.add_argument("--arms", default="vanilla,impedance")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--H", type=int, default=16)
    ap.add_argument("--eval_eps", type=int, default=30)
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--n_demos", default="100")      # comma list -> sample-efficiency sweep (first N demos/task)
    ap.add_argument("--tag", default="gate")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.tasks = "peg-insert-side-v3"; args.seeds = "0"; args.epochs = 30; args.eval_eps = 6; args.n_demos = "25"
    tasks = args.tasks.split(","); arms = args.arms.split(","); seeds = [int(s) for s in args.seeds.split(",")]

    demos = pickle.load(open(DEMOS, "rb"))
    cond_dim = demos[tasks[0]][0][0].shape[1]
    imp_params = n_params(CompactFlowPolicy(cond_dim, 4, args.H, hidden=args.hidden, mode="impedance"))
    hv = matched_vanilla_hidden(cond_dim, args.H, imp_params)
    hidden = {"impedance": args.hidden, "vanilla": hv}
    progf = open(f"{ROOT}/out_gate/progress_{args.tag}.log", "w") if os.path.isdir(f"{ROOT}/out_gate") or os.makedirs(f"{ROOT}/out_gate", exist_ok=True) is None else None
    def log(m):
        print(m, flush=True); progf.write(m + "\n"); progf.flush(); os.fsync(progf.fileno())
    log(f"# param-match: impedance(h={args.hidden})={imp_params:,}  vanilla(h={hv})="
        f"{n_params(CompactFlowPolicy(cond_dim,4,args.H,hidden=hv,mode='vanilla')):,}  | DEV={DEV}")

    ndemos = [int(x) for x in args.n_demos.split(",")]
    results = {}
    for task in tasks:
        for nd in ndemos:
            X, Y = build_pairs(demos[task][:nd], args.H)
            xm, xs = X.mean(0), X.std(0) + 1e-6
            ym, ys = Y.mean((0, 1)), Y.std((0, 1)) + 1e-6
            for arm in arms:
                per_seed = []
                for sd in seeds:
                    t0 = time.time()
                    pol = train(arm, cond_dim, args.H, X, Y, xm, xs, ym, ys, hidden[arm], args.epochs, sd)
                    m = evaluate(pol, task, args.H, xm, xs, ym, ys, args.eval_eps, args.max_steps)
                    per_seed.append(m)
                    log(f"  {task:18s} n{nd:<3d} {arm:9s} seed{sd}: succ={m['success']*100:5.1f}%  "
                        f"jerk={m['jerk']:.4f}  peakF={m['peakF']:8.1f}  ({time.time()-t0:.0f}s)")
                agg = {k: float(np.nanmean([s[k] for s in per_seed])) for k in per_seed[0]}
                sem = {k: float(np.nanstd([s[k] for s in per_seed]) / max(1, len(seeds) ** 0.5)) for k in per_seed[0]}
                results[f"{task}/{arm}/n{nd}"] = {"mean": agg, "sem": sem, "seeds": per_seed}
                log(f"==> {task:18s} n{nd:<3d} {arm:9s} MEAN succ={agg['success']*100:5.1f}±{sem['success']*100:.1f}  "
                    f"jerk={agg['jerk']:.4f}±{sem['jerk']:.4f}  peakF={agg['peakF']:.1f}±{sem['peakF']:.1f}")
    json.dump(results, open(f"{ROOT}/out_gate/results_{args.tag}.json", "w"), indent=2)
    log(f"\nwrote {ROOT}/out_gate/results_{args.tag}.json")
    for task in tasks:
        for nd in ndemos:
            v = results.get(f"{task}/vanilla/n{nd}", {}).get("mean"); im = results.get(f"{task}/impedance/n{nd}", {}).get("mean")
            if v and im:
                log(f"VERDICT {task} n{nd}: succ {v['success']*100:.0f}->{im['success']*100:.0f}  "
                    f"jerk {v['jerk']:.3f}->{im['jerk']:.3f}  peakF {v['peakF']:.0f}->{im['peakF']:.0f}")
    progf.close()


if __name__ == "__main__":
    main()
