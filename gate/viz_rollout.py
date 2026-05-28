r"""
EXPLAINABLE SIM for the presentation: side-by-side MuJoCo peg-insert rollouts,
vanilla flow (left) vs impedance flow (right), same scene/seed, with a live contact-force gauge
and SUCCESS badge.  Trains both arms on 10 demos (the +22pp regime), picks the most contrasting
seeds (impedance succeeds where vanilla fails), renders them into one MP4.
"""
import os, sys, pickle
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, cv2, imageio
from policy import CompactFlowPolicy, n_params
from train_eval import build_pairs, train, matched_vanilla_hidden, contact_force, DEMOS, ROOT, DEV
from lerobot.envs.metaworld import MetaworldEnv

TASK = "peg-insert-side-v3"; H = 16; NDEMOS = 10; EPOCHS = 300; MAXSTEPS = 130; EXEC = 4


def render256(env):
    img = env._env.render()
    if env.camera_name == "corner2":
        img = np.flip(img, (0, 1))
    return cv2.resize(np.ascontiguousarray(img), (256, 256), interpolation=cv2.INTER_AREA)


def rollout(pol, env, seed, xm, xs, ym, ys, render=False):
    raw, _ = env._env.reset(seed=seed)
    torch.manual_seed(7000 + seed)                # deterministic noise -> selection matches the rendered run
    frames, forces = [], []; chunk = None; sip = 0; ok = False
    for t in range(MAXSTEPS):
        if render:
            frames.append(render256(env))
        if chunk is None or sip >= EXEC:
            cond = torch.tensor(((np.asarray(raw, np.float32) - xm) / xs)[None], device=DEV)
            with torch.no_grad():
                gen = pol(cond, torch.randn(1, H, 4, device=DEV))[0].cpu().numpy()
            chunk = (gen * ys + ym).astype(np.float32); sip = 0
        a = np.clip(chunk[sip], -1, 1); sip += 1
        raw, r, done, trunc, info = env._env.step(a)
        forces.append(contact_force(env))
        if info.get("success", 0):
            ok = True
            if render:
                frames.append(render256(env))
            break
    return frames, forces, ok


def panel(frame, title, force, fmax, ok, color):
    im = cv2.resize(frame, (384, 384)).copy()
    cv2.rectangle(im, (0, 0), (384, 30), (0, 0, 0), -1)
    cv2.putText(im, title, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
    fmax = max(fmax, 1.0)
    h = int(280 * min(1.0, float(force) / fmax))
    cv2.rectangle(im, (356, 350), (376, 70), (255, 255, 255), 1)
    cv2.rectangle(im, (357, 350), (375, 350 - h), (60, 170, 255), -1)
    cv2.putText(im, f"contact F", (300, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(im, f"{force:.0f}", (340, 368), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    if ok:
        cv2.putText(im, "SUCCESS", (110, 205), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 230, 0), 3, cv2.LINE_AA)
    return im


def main():
    demos = pickle.load(open(DEMOS, "rb"))
    X, Y = build_pairs(demos[TASK][:NDEMOS], H)
    xm, xs = X.mean(0), X.std(0) + 1e-6
    ym, ys = Y.mean((0, 1)), Y.std((0, 1)) + 1e-6
    cond_dim = X.shape[1]
    imp_p = n_params(CompactFlowPolicy(cond_dim, 4, H, mode="impedance"))
    hv = matched_vanilla_hidden(cond_dim, H, imp_p)
    print(f"training peg-insert @{NDEMOS} demos: vanilla(h={hv}) + impedance ...", flush=True)
    van = train("vanilla", cond_dim, H, X, Y, xm, xs, ym, ys, hv, EPOCHS, 0)
    imp = train("impedance", cond_dim, H, X, Y, xm, xs, ym, ys, 256, EPOCHS, 0)

    env = MetaworldEnv(task=TASK, obs_type="pixels_agent_pos", camera_name="corner2")
    # Select episodes from the ACTUAL RENDERED rollouts (GPU eigh is non-deterministic, so a separate
    # no-render selection can't be trusted). Keep ones where the rendered result is van-fail / imped-OK.
    kept = []
    for sd in range(40, 110):
        fv, Fv, okv = rollout(van, env, sd, xm, xs, ym, ys, render=True)
        fi, Fi, oki = rollout(imp, env, sd, xm, xs, ym, ys, render=True)
        if oki and not okv:
            kept.append((sd, fv, Fv, fi, Fi, okv, oki))
            print(f"  seed {sd}: vanilla FAIL / impedance OK  -> kept {len(kept)}/3  "
                  f"(peakF v={max(Fv):.0f} i={max(Fi):.0f})", flush=True)
        if len(kept) >= 3:
            break
    env.close()
    if not kept:
        print("no clean contrast found"); return
    video = []
    for sd, fv, Fv, fi, Fi, okv, oki in kept:
        L = max(len(fv), len(fi)); fmax = max(max(Fv + [1]), max(Fi + [1]))
        for t in range(L + 10):                                  # +10 hold on last frame
            tv, ti = min(t, len(fv) - 1), min(t, len(fi) - 1)
            pv = panel(fv[tv], "vanilla flow", Fv[min(tv, len(Fv) - 1)], fmax, okv and tv >= len(fv) - 2, (200, 200, 200))
            pi = panel(fi[ti], "impedance flow (ours)", Fi[min(ti, len(Fi) - 1)], fmax, oki and ti >= len(fi) - 2, (90, 200, 255))
            sep = np.full((384, 4, 3), 255, np.uint8)
            video.append(np.concatenate([pv, sep, pi], axis=1))
    out = f"{ROOT}/figs/peg_rollout_compare.mp4"
    imageio.mimsave(out, video, fps=20, quality=8)
    print(f"\nwrote {out}  ({len(video)} frames, {len(kept)} episodes: seeds {[k[0] for k in kept]})", flush=True)


if __name__ == "__main__":
    main()
