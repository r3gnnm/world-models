"""Сбор последовательностей с Quad2DEnv для обучения с multi-step loss.

Отличие от collect_openworld.py: сохраняем данные как ПОСЛЕДОВАТЕЛЬНОСТИ
(эпизоды), а не как отдельные переходы — multi-step loss требует
последовательных кадров. Формат: (n_episodes, ep_len+1, 1, 64, 64).

Политика — каскадный PD (см. env_quad2d.hover_policy) со случайными целями
висения и шумом. Чисто случайная политика на недоуправляемой системе
вырождается в столкновения и не даёт полезного датасета.

Запуск:  python collect_quad2d.py --episodes 300 --ep-len 64
"""
import argparse
import os
import numpy as np
from env_quad2d import Quad2DEnv, hover_policy, SIZE


def collect(n_episodes, ep_len, seed=0, noise=0.15, retarget_every=24,
           maneuver_every=15, maneuver_len=4, maneuver_scale=0.85,
           egocentric=True, fixed_map=False, dual_view=False):
    """maneuver_*: периодически впрыскиваем несколько шагов почти предельных
    команд поверх PD-коррекции. Без этого политика сбора данных (сошедшийся
    PD-контроллер) выдаёт почти всегда маленькие корректирующие действия —
    action_gap ablation (перемешивание действий внутри батча) не видит
    контраста между "правильным" и "случайным" действием, если оба похожи.
    Манёвры дают датасету диапазон, сравнимый с инерционным случайным
    блужданием в комнатных средах, где gap хорошо считывался."""
    rng = np.random.default_rng(seed)
    O, A, S = [], [], []
    for ep in range(n_episodes):
        # fixed_map: одна и та же карта во всех эпизодах. Нужно для
        # глобального вида: при разных картах дисперсия между эпизодами
        # в ~5 раз больше дисперсии движения дрона, и модель кодирует
        # статичный фон вместо агента (измерено — shortcut learning:
        # loss падает, action_gap ноль, probe отрицательный).
        if fixed_map:
            env = Quad2DEnv(seed=seed, egocentric=egocentric, dual_view=dual_view)
            # ВАЖНО: seed фиксирует и карту, И стартовые позиции. Карту
            # оставляем, а rng переуседиваем — иначе все эпизоды стартуют
            # из одной точки и покрытие мира схлопывается (измерено:
            # x охватывал лишь 70..98 из 128 вместо 3..125).
            env.rng = np.random.default_rng(seed * 1000 + ep)
            env.reset()
        else:
            env = Quad2DEnv(seed=seed * 1000 + ep, egocentric=egocentric, dual_view=dual_view)
        o = env.reset()
        tx, tz = rng.uniform(30, SIZE - 30, 2)
        obs_ep, act_ep, st_ep = [o], [], [env.state.copy()]
        maneuver_left = 0
        maneuver_dir = np.zeros(2, dtype=np.float32)
        for t in range(ep_len):
            if t % retarget_every == 0 and t > 0:
                tx, tz = rng.uniform(30, SIZE - 30, 2)
            if t % maneuver_every == 0:
                maneuver_left = maneuver_len
                maneuver_dir = rng.uniform(-1, 1, 2).astype(np.float32)
                maneuver_dir /= (np.linalg.norm(maneuver_dir) + 1e-6)

            a = hover_policy(env, rng, target_z=tz, target_x=tx, noise=noise)
            if maneuver_left > 0:
                a = np.clip(a + maneuver_dir * maneuver_scale, -1, 1)
                maneuver_left -= 1

            o = env.step(a)
            obs_ep.append(o)
            act_ep.append(a.astype(np.float32))
            st_ep.append(env.state.copy())
        O.append(np.stack(obs_ep))
        A.append(np.stack(act_ep))
        S.append(np.stack(st_ep))
    return np.stack(O), np.stack(A), np.stack(S)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--ep-len", type=int, default=64)
    p.add_argument("--noise", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--global-view", action="store_true",
                   help="весь мир в кадре вместо эгоцентрического кропа")
    p.add_argument("--fixed-map", action="store_true",
                   help="одна карта препятствий на все эпизоды (обязательно "
                        "для глобального вида, иначе модель кодирует фон)")
    p.add_argument("--dual-view", action="store_true",
                   help="2 канала: глобальный + эгоцентрический вместе")
    p.add_argument("--out", type=str, default="data/quad2d.npz")
    args = p.parse_args()

    O, A, S = collect(args.episodes, args.ep_len, seed=args.seed,
                      noise=args.noise, egocentric=not args.global_view,
                      fixed_map=args.fixed_map, dual_view=args.dual_view)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, obs=O, actions=A, states=S)
    print(f"Сохранено: {O.shape[0]} эпизодов × {O.shape[1]} кадров -> {args.out}")
    print(f"  obs {O.shape}, actions {A.shape}, states {S.shape}")
    print(f"  покрытие x: {S[..., 0].min()*SIZE:.0f}..{S[..., 0].max()*SIZE:.0f}, "
          f"z: {S[..., 1].min()*SIZE:.0f}..{S[..., 1].max()*SIZE:.0f}")
    print(f"  диапазон наклона (норм.): {S[..., 4].min():+.2f}..{S[..., 4].max():+.2f}")
