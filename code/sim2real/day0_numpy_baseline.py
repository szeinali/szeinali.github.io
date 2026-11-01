"""
Day 0 sanity baseline: planar 2-link arm "reacher" in pure numpy, trained with
OpenAI-style Evolution Strategies (antithetic sampling, rank shaping, Adam).

Why this exists: it gives the site real data on day 0 and fixes the metrics
schema that the MuJoCo/RL runs (train_mujoco_reacher.py) will also write.
The physics here is deliberately simplified (decoupled joints, linear damping,
no gravity) and is NOT a substitute for MuJoCo. It's a pipeline check.

Usage:
    python day0_numpy_baseline.py --out ../../projects/sim2real/data/day0
"""
import argparse, json, os, time
import numpy as np

# ---------------- environment ----------------
L1, L2 = 0.1, 0.11          # link lengths (m), similar to Gymnasium Reacher
DT, HORIZON = 0.02, 50      # 1 s episodes
GAIN, DAMP = 40.0, 8.0      # qdd = GAIN*a - DAMP*qd
SUCCESS_DIST = 0.02         # m, final-step distance counted as success
CTRL_COST = 0.05


def fk(q):
    """Forward kinematics -> elbow (N,2), fingertip (N,2)."""
    e = np.stack([L1 * np.cos(q[:, 0]), L1 * np.sin(q[:, 0])], 1)
    t = e + np.stack([L2 * np.cos(q[:, 0] + q[:, 1]), L2 * np.sin(q[:, 0] + q[:, 1])], 1)
    return e, t


def sample_targets(rng, n):
    r = 0.2 * np.sqrt(rng.uniform(0.05, 1.0, n))   # uniform in disk, avoid the base
    th = rng.uniform(-np.pi, np.pi, n)
    return np.stack([r * np.cos(th), r * np.sin(th)], 1)


def obs(q, qd, tgt):
    _, tip = fk(q)
    return np.concatenate([np.cos(q), np.sin(q), qd / 10.0, tgt / 0.2, (tip - tgt) / 0.2], 1)


OBS_DIM, ACT_DIM, HID = 10, 2, 32


def rollout(policy_fn, targets, q0, record=False):
    """Batched rollout. policy_fn(obs)->actions in [-1,1]. Returns dict of metrics."""
    q, qd = q0.copy(), np.zeros_like(q0)
    ret = np.zeros(len(targets))
    dists, qs = [], []
    for _ in range(HORIZON):
        a = np.clip(policy_fn(obs(q, qd, targets)), -1, 1)
        qd = qd + DT * (GAIN * a - DAMP * qd)
        q = q + DT * qd
        _, tip = fk(q)
        d = np.linalg.norm(tip - targets, axis=1)
        ret += -d - CTRL_COST * (a ** 2).sum(1)
        dists.append(d)
        if record:
            qs.append(q.copy())
    out = {"ret": ret, "final_dist": dists[-1], "dist_t": np.stack(dists, 1)}
    if record:
        out["q"] = np.stack(qs, 1)   # (N, T, 2)
    return out


def jacobian_transpose_controller(o, kp=800.0, kv=0.3):
    """Hand-written classical reference: torque = kp * J^T (target - tip) - kv * qd."""
    q = np.arctan2(o[:, 2:4], o[:, 0:2]); qd = o[:, 4:6] * 10.0; err = -o[:, 8:10] * 0.2
    s1, c1, s12, c12 = np.sin(q[:, 0]), np.cos(q[:, 0]), np.sin(q.sum(1)), np.cos(q.sum(1))
    J = np.zeros((len(o), 2, 2))
    J[:, 0, 0] = -L1 * s1 - L2 * s12; J[:, 0, 1] = -L2 * s12
    J[:, 1, 0] = L1 * c1 + L2 * c12; J[:, 1, 1] = L2 * c12
    return kp * np.einsum("nji,nj->ni", J, err) - kv * qd


# ---------------- policy (tiny MLP, batched over population) ----------------
def n_params():
    return OBS_DIM * HID + HID + HID * ACT_DIM + ACT_DIM


def unpack(theta):
    """theta: (P, n) -> weights with leading pop dim."""
    i = 0
    def take(*shape):
        nonlocal i
        k = int(np.prod(shape))
        w = theta[:, i:i + k].reshape(theta.shape[0], *shape)
        i += k
        return w
    return take(OBS_DIM, HID), take(HID), take(HID, ACT_DIM), take(ACT_DIM)


def make_pop_policy(theta, per_member):
    """Policy over a batch laid out as P members x per_member envs."""
    W1, b1, W2, b2 = unpack(theta)
    P = theta.shape[0]
    def f(o):
        o = o.reshape(P, per_member, OBS_DIM)
        h = np.tanh(np.einsum("pno,poh->pnh", o, W1) + b1[:, None])
        return np.tanh(np.einsum("pnh,pha->pna", h, W2) + b2[:, None]).reshape(-1, ACT_DIM)
    return f


def single_policy(theta, n):
    return make_pop_policy(theta[None], n)


# ---------------- ES training ----------------
def train(seed, iters, pop_pairs, eps_per_member, sigma, lr, eval_every, snap_iters, eval_set):
    rng = np.random.default_rng(seed)
    theta = rng.normal(0, 0.1, n_params())
    m, v = np.zeros_like(theta), np.zeros_like(theta)
    b1, b2 = 0.9, 0.999
    P = 2 * pop_pairs
    tgt_eval, q0_eval = eval_set
    log, snaps = [], {}
    env_steps, t_train = 0, 0.0

    def evaluate(th):
        r = rollout(single_policy(th, len(tgt_eval)), tgt_eval, q0_eval)
        return {
            "mean_return": float(r["ret"].mean()),
            "success_rate": float((r["final_dist"] < SUCCESS_DIST).mean()),
            "mean_final_dist_cm": float(100 * r["final_dist"].mean()),
        }

    for it in range(iters + 1):
        if it % eval_every == 0 or it == iters:
            e = evaluate(theta)
            e.update(iter=it, env_steps=env_steps,
                     steps_per_sec=(env_steps / t_train) if t_train > 0 else None)
            log.append(e)
        if it in snap_iters:
            snaps[it] = theta.copy()
        if it == iters:
            break
        t0 = time.perf_counter()
        eps = rng.normal(size=(pop_pairs, theta.size))
        noise = np.concatenate([eps, -eps])
        pop = theta + sigma * noise
        tg = sample_targets(rng, eps_per_member)            # common random numbers
        q0 = rng.uniform(-np.pi, np.pi, (eps_per_member, 2))
        r = rollout(make_pop_policy(pop, eps_per_member), np.tile(tg, (P, 1)), np.tile(q0, (P, 1)))
        fit = r["ret"].reshape(P, eps_per_member).mean(1)
        ranks = np.empty(P); ranks[np.argsort(fit)] = np.arange(P)
        shaped = ranks / (P - 1) - 0.5
        g = -(shaped @ noise) / (P * sigma) + 0.005 * theta   # minimise -fitness, L2
        m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g
        mh, vh = m / (1 - b1 ** (it + 1)), v / (1 - b2 ** (it + 1))
        theta = theta - lr * mh / (np.sqrt(vh) + 1e-8)
        env_steps += P * eps_per_member * HORIZON
        t_train += time.perf_counter() - t0
    return theta, log, snaps, env_steps / t_train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--iters", type=int, default=1200)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    erng = np.random.default_rng(12345)
    eval_set = (sample_targets(erng, 200), erng.uniform(-np.pi, np.pi, (200, 2)))
    demo_tg, demo_q0 = eval_set[0][:6], eval_set[1][:6]
    snap_iters = [0, 50, 200, 500, a.iters]

    # random-policy reference
    rrng = np.random.default_rng(7)
    rand = rollout(lambda o: rrng.uniform(-1, 1, (len(o), 2)), *eval_set)
    zero = rollout(lambda o: np.zeros((len(o), 2)), *eval_set)
    jt = rollout(jacobian_transpose_controller, *eval_set)

    runs, finals, best = [], [], None
    for s in a.seeds:
        t0 = time.time()
        theta, log, snaps, sps = train(s, a.iters, pop_pairs=48, eps_per_member=16, sigma=0.1,
                                       lr=0.05, eval_every=10, snap_iters=snap_iters, eval_set=eval_set)
        print(f"seed {s}: success={log[-1]['success_rate']:.2f} "
              f"final_dist={log[-1]['mean_final_dist_cm']:.2f}cm  {sps:,.0f} steps/s  ({time.time()-t0:.1f}s)")
        runs.append({"seed": s, "log": log, "steps_per_sec": sps})
        finals.append((log[-1]["mean_return"], s, theta, snaps))

    _, best_seed, best_theta, best_snaps = max(finals, key=lambda x: x[0])
    final = rollout(single_policy(best_theta, 200), *eval_set)

    # ---- metrics.json: learning curves (all seeds) + baselines ----
    metrics = {
        "schema": "sim2real.metrics.v1",
        "task": "planar 2-link reach (numpy, simplified dynamics)",
        "algo": "Evolution Strategies (antithetic, rank-shaped, Adam), MLP 10-32-2",
        "success_threshold_cm": SUCCESS_DIST * 100,
        "eval_episodes": 200,
        "runs": runs,
        "reference": {
            "random_policy": {"mean_return": float(rand["ret"].mean()),
                              "success_rate": float((rand["final_dist"] < SUCCESS_DIST).mean()),
                              "mean_final_dist_cm": float(100 * rand["final_dist"].mean())},
            "classical_controller": {"mean_return": float(jt["ret"].mean()),
                                     "success_rate": float((jt["final_dist"] < SUCCESS_DIST).mean()),
                                     "mean_final_dist_cm": float(100 * jt["final_dist"].mean())},
            "zero_action": {"mean_return": float(zero["ret"].mean()),
                            "success_rate": float((zero["final_dist"] < SUCCESS_DIST).mean()),
                            "mean_final_dist_cm": float(100 * zero["final_dist"].mean())},
        },
    }
    # ---- distributions.json: final-distance histograms + distance-over-time ----
    bins = np.linspace(0, 30, 31)
    dist = {
        "schema": "sim2real.distributions.v1",
        "bins_cm": bins.tolist(),
        "final_dist_hist": {
            "random_policy": np.histogram(np.clip(100 * rand["final_dist"], 0, 29.99), bins)[0].tolist(),
            "trained": np.histogram(np.clip(100 * final["final_dist"], 0, 29.99), bins)[0].tolist(),
            "classical_controller": np.histogram(np.clip(100 * jt["final_dist"], 0, 29.99), bins)[0].tolist(),
        },
        "time_s": (DT * np.arange(1, HORIZON + 1)).round(3).tolist(),
        "dist_over_time_cm": {
            k: {"mean": (100 * r["dist_t"].mean(0)).round(3).tolist(),
                "p10": (100 * np.percentile(r["dist_t"], 10, 0)).round(3).tolist(),
                "p90": (100 * np.percentile(r["dist_t"], 90, 0)).round(3).tolist()}
            for k, r in [("random_policy", rand), ("trained", final), ("classical_controller", jt)]
        },
    }
    # ---- rollouts.json: joint trajectories at training snapshots for arm replay ----
    stages = {}
    for it in snap_iters:
        r = rollout(single_policy(best_snaps[it], len(demo_tg)), demo_tg, demo_q0, record=True)
        stages[str(it)] = {"q": r["q"].round(4).tolist(),
                           "final_dist_cm": (100 * r["final_dist"]).round(2).tolist()}
    rollouts = {
        "schema": "sim2real.rollouts.v1", "seed": best_seed, "dt": DT,
        "links": [L1, L2], "targets": demo_tg.round(4).tolist(),
        "q0": demo_q0.round(4).tolist(), "stages": stages,
    }
    for name, obj in [("metrics", metrics), ("distributions", dist), ("rollouts", rollouts)]:
        with open(os.path.join(a.out, f"{name}.json"), "w") as f:
            json.dump(obj, f, separators=(",", ":"))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
