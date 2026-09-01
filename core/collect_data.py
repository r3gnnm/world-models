"""Сбор датасета переходов (o_t, a_t, o_t+1, state_t) случайной политикой.

Запуск:  python collect_data.py --transitions 50000 --out data/transitions.npz
Коррелированные действия (инерция) дают более разнообразное покрытие карты,
чем чисто белый шум.
"""
import argparse
import os
import numpy as np
from env import TwoRoomsEnv


def collect(n_transitions: int, episode_len: int = 200, seed: int = 0):
    env = TwoRoomsEnv(seed=seed)
    rng = np.random.default_rng(seed)
    obs, actions, next_obs, states = [], [], [], []
    while len(obs) < n_transitions:
        o = env.reset()
        a = rng.uniform(-1, 1, size=2)
        for _ in range(episode_len):
            # инерционная случайная политика: новое действие = сглаженное старое + шум
            a = np.clip(0.7 * a + 0.5 * rng.normal(size=2), -1, 1)
            s = env.state.copy()
            o_next = env.step(a)
            obs.append(o)
            actions.append(a.astype(np.float32))
            next_obs.append(o_next)
            states.append(s)
            o = o_next
            if len(obs) >= n_transitions:
                break
    return (np.stack(obs), np.stack(actions), np.stack(next_obs), np.stack(states))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--transitions", type=int, default=50_000)
    p.add_argument("--out", type=str, default="data/transitions.npz")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    o, a, o1, s = collect(args.transitions, seed=args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, obs=o, actions=a, next_obs=o1, states=s)
    print(f"Сохранено {len(o)} переходов в {args.out}"
          f" | obs {o.shape} actions {a.shape} states {s.shape}")
