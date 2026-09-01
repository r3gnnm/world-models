"""Сбор переходов с OpenWorldEnv — формат совместим с train.py/train_ensemble.py
(тот же .npz со полями obs/actions/next_obs/states), так что весь существующий
пайплайн обучения JEPA переиспользуется без изменений.

Запуск:  python collect_openworld.py --transitions 20000 --out data/openworld.npz
"""
import argparse
import os
import numpy as np
from env_openworld import OpenWorldEnv


def collect(n_transitions, episode_len=150, seed=0):
    env = OpenWorldEnv(seed=seed)
    rng = np.random.default_rng(seed)
    obs, actions, next_obs, states = [], [], [], []
    while len(obs) < n_transitions:
        o = env.reset()
        a = rng.uniform(-1, 1, size=2)
        for _ in range(episode_len):
            a = np.clip(0.7 * a + 0.5 * rng.normal(size=2), -1, 1)
            s = env.global_pos.copy()
            o1 = env.step(a)
            obs.append(o); actions.append(a.astype(np.float32))
            next_obs.append(o1); states.append(s)
            o = o1
            if len(obs) >= n_transitions:
                break
    return (np.stack(obs), np.stack(actions), np.stack(next_obs), np.stack(states))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--transitions", type=int, default=20_000)
    p.add_argument("--out", type=str, default="data/openworld.npz")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    o, a, o1, s = collect(args.transitions, seed=args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, obs=o, actions=a, next_obs=o1, states=s)
    print(f"Сохранено {len(o)} переходов в {args.out}")
