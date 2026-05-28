r"""
STAGE-1 data: full-STATE MetaWorld expert demos (no rendering -> fast) for the mechanism gate.
Conditioning = full raw MetaWorld observation (~39-D: ee, gripper, object(s), goal). Action = 4-D.
Tasks: peg-insert-side (horizontal insertion, primary contact), push, door-open (contact),
       reach (free-space CONTROL -- expect no impedance gain here).
Saves a pickle: {task: [(obs[T,od] float32, act[T,4] float32), ...successful episodes...]}.
"""
import os, sys, pickle, argparse
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
from lerobot.envs.metaworld import MetaworldEnv, TASK_DESCRIPTIONS

TASKS = ["peg-insert-side-v3", "push-v3", "door-open-v3", "reach-v3"]
OUT = "/home/user/Desktop/Saptarshi/impedance_flow_vla/gate_data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps_per_task", type=int, default=100)
    ap.add_argument("--max_steps", type=int, default=200)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    data = {}
    for tid, task in enumerate(TASKS):
        env = MetaworldEnv(task=task, obs_type="pixels_agent_pos", camera_name="corner2")
        eps = []; attempts = 0
        while len(eps) < args.eps_per_task and attempts < args.eps_per_task * 5:
            attempts += 1
            raw_obs, _ = env._env.reset(seed=10000 * tid + attempts)
            expert = env.expert_policy
            obs_seq, act_seq = [], []
            ok = False
            for t in range(args.max_steps):
                a = np.clip(expert.get_action(raw_obs), -1, 1).astype(np.float32)
                obs_seq.append(np.asarray(raw_obs, dtype=np.float32))
                act_seq.append(a)
                raw_obs, r, done, trunc, info = env._env.step(a)
                if info.get("success", 0):
                    ok = True; break
            if ok and len(obs_seq) >= 8:
                eps.append((np.stack(obs_seq), np.stack(act_seq)))
        env.close()
        od = eps[0][0].shape[1] if eps else -1
        data[task] = eps
        print(f"[{tid}] {task:20s} {len(eps):3d} demos  (obs_dim={od}, {attempts} attempts, "
              f"mean_len={np.mean([len(e[0]) for e in eps]):.0f})", flush=True)

    with open(f"{OUT}/demos.pkl", "wb") as f:
        pickle.dump(data, f)
    tot = sum(len(v) for v in data.values())
    print(f"\nDONE: {tot} demos across {len(TASKS)} tasks -> {OUT}/demos.pkl", flush=True)


if __name__ == "__main__":
    main()
