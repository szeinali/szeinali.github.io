"""
Week 1: MuJoCo Reacher (Gymnasium Reacher-v5) trained with SAC or PPO
(Stable-Baselines3). Writes the same JSON files as the Day 0 baseline, so the
site renders the results with no code changes.

Setup (on your machine):
    pip install "gymnasium[mujoco]" stable-baselines3 torch

Run (about 10-30 min per seed on a laptop CPU for 200k steps with SAC):
    python train_mujoco_reacher.py --algo sac --steps 200000 --seeds 0 1 2 \
        --out ../../projects/sim2real/data/week1-mujoco

Then add an entry to projects/sim2real/entries.json pointing at that folder
(copy the day0 entry and change "src" paths, title, date and text).

Reacher-v5 has link lengths 0.1 m and 0.11 m, the same as the Day 0 arm, so
the arm_replay visual works as-is. Success = final fingertip-target
distance <= 2 cm, the same threshold as Day 0.
"""
import argparse, json, os, time
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback

SUCCESS_DIST = 0.02
EVAL_SEED0 = 10_000          # fixed evaluation seeds: EVAL_SEED0 .. EVAL_SEED0+N-1


def run_episode(env, act_fn, seed, record=False):
    obs, _ = env.reset(seed=seed)
    data = env.unwrapped.data
    target = data.qpos[2:4].copy()
    q0 = data.qpos[:2].copy()
    ret, dists, qs, done = 0.0, [], [], False
    while not done:
        obs, r, term, trunc, _ = env.step(act_fn(obs))
        ret += r
        dists.append(float(np.linalg.norm(obs[-3:-1])))     # fingertip - target (x, y)
        if record:
            qs.append(data.qpos[:2].copy())
        done = term or trunc
    return ret, np.array(dists), target, q0, np.array(qs)


def evaluate(env, act_fn, n, record_first=0):
    rets, finals, dist_t, rec = [], [], [], []
    for i in range(n):
        ret, d, tg, q0, qs = run_episode(env, act_fn, EVAL_SEED0 + i, record=i < record_first)
        rets.append(ret); finals.append(d[-1]); dist_t.append(d)
        if i < record_first:
            rec.append((tg, q0, qs, d[-1]))
    finals = np.array(finals)
    return {"mean_return": float(np.mean(rets)),
            "success_rate": float((finals <= SUCCESS_DIST).mean()),
            "mean_final_dist_cm": float(100 * finals.mean())}, finals, np.stack(dist_t), rec


class EvalLog(BaseCallback):
    def __init__(self, eval_env, every, n_eval, snap_steps, snap_dir):
        super().__init__()
        self.eval_env, self.every, self.n_eval = eval_env, every, n_eval
        self.snap_steps, self.snap_dir = sorted(snap_steps), snap_dir
        self.log, self.start, self.eval_time, self.train_time = [], None, 0.0, 0.0

    def _act(self, o):
        return self.model.predict(o, deterministic=True)[0]

    def _do_eval(self):
        t = time.perf_counter()
        m, *_ = evaluate(self.eval_env, self._act, self.n_eval)
        steps = self.num_timesteps
        m.update(iter=len(self.log), env_steps=steps,
                 steps_per_sec=steps / self.train_time if self.train_time > 0 else None)
        self.log.append(m)
        print(f"  {steps:>8} steps  success={m['success_rate']:.2f}  dist={m['mean_final_dist_cm']:.2f}cm")
        return time.perf_counter() - t

    def _on_training_start(self):
        self.start = time.perf_counter()
        self.model.save(os.path.join(self.snap_dir, "snap_0"))
        self.eval_time += self._do_eval()

    def _on_step(self):
        n = self.num_timesteps
        if n in self.snap_steps:
            self.model.save(os.path.join(self.snap_dir, f"snap_{n}"))
        if n % self.every == 0:
            self.train_time = time.perf_counter() - self.start - self.eval_time   # excludes eval
            self.eval_time += self._do_eval()
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["sac", "ppo"], default="sac")
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--eval-every", type=int, default=5_000)
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--out", default="out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    Algo = {"sac": SAC, "ppo": PPO}[a.algo]
    eval_env = gym.make("Reacher-v5")
    snap_steps = [int(a.steps * f) // a.eval_every * a.eval_every for f in (0.05, 0.2, 0.5)] + [a.steps]

    # random-policy reference on the same evaluation seeds
    rng = np.random.default_rng(7)
    rand_m, rand_f, rand_dt, _ = evaluate(eval_env, lambda o: rng.uniform(-1, 1, 2), a.n_eval)

    runs, best = [], None
    for s in a.seeds:
        print(f"seed {s}")
        env = gym.make("Reacher-v5")
        model = Algo("MlpPolicy", env, seed=s, verbose=0)
        snap_dir = os.path.join(a.out, f"_ckpt_seed{s}"); os.makedirs(snap_dir, exist_ok=True)
        cb = EvalLog(eval_env, a.eval_every, a.n_eval, snap_steps, snap_dir)
        t0 = time.perf_counter()
        model.learn(total_timesteps=a.steps, callback=cb)
        sps = a.steps / (time.perf_counter() - t0)
        runs.append({"seed": s, "log": cb.log, "steps_per_sec": sps})
        if best is None or cb.log[-1]["mean_return"] > best[0]:
            best = (cb.log[-1]["mean_return"], s, snap_dir)

    _, best_seed, best_dir = best
    final_model = Algo.load(os.path.join(best_dir, f"snap_{a.steps}"))
    act = lambda o: final_model.predict(o, deterministic=True)[0]
    _, fin_f, fin_dt, _ = evaluate(eval_env, act, a.n_eval)

    metrics = {"schema": "sim2real.metrics.v1", "task": "Gymnasium Reacher-v5 (MuJoCo)",
               "algo": f"{a.algo.upper()} (Stable-Baselines3 defaults)", "success_threshold_cm": SUCCESS_DIST * 100,
               "eval_episodes": a.n_eval, "runs": runs, "reference": {"random_policy": rand_m}}
    bins = np.linspace(0, 30, 31)
    hist = lambda f: np.histogram(np.clip(100 * f, 0, 29.99), bins)[0].tolist()
    T = fin_dt.shape[1]
    dt = eval_env.unwrapped.dt
    band = lambda D: {"mean": (100 * D.mean(0)).round(3).tolist(),
                      "p10": (100 * np.percentile(D, 10, 0)).round(3).tolist(),
                      "p90": (100 * np.percentile(D, 90, 0)).round(3).tolist()}
    dist = {"schema": "sim2real.distributions.v1", "bins_cm": bins.tolist(),
            "final_dist_hist": {"random_policy": hist(rand_f), "trained": hist(fin_f)},
            "time_s": (dt * np.arange(1, T + 1)).round(3).tolist(),
            "dist_over_time_cm": {"random_policy": band(rand_dt), "trained": band(fin_dt)}}

    stages, targets, q0s = {}, None, None
    for st in [0] + snap_steps:
        m = Algo.load(os.path.join(best_dir, f"snap_{st}"))
        _, _, _, rec = evaluate(eval_env, lambda o: m.predict(o, deterministic=True)[0], 6, record_first=6)
        targets = [r[0].round(4).tolist() for r in rec]; q0s = [r[1].round(4).tolist() for r in rec]
        stages[str(st)] = {"q": [r[2].round(4).tolist() for r in rec],
                           "final_dist_cm": [round(100 * r[3], 2) for r in rec]}
    rollouts = {"schema": "sim2real.rollouts.v1", "seed": best_seed, "dt": dt, "links": [0.1, 0.11],
                "targets": targets, "q0": q0s, "stages": stages}

    for name, obj in [("metrics", metrics), ("distributions", dist), ("rollouts", rollouts)]:
        with open(os.path.join(a.out, f"{name}.json"), "w") as f:
            json.dump(obj, f, separators=(",", ":"))
    print("wrote", a.out, "(checkpoints in _ckpt_seed*/ are not needed on the site)")


if __name__ == "__main__":
    main()
