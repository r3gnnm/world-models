"""Развёртка "разрыв обученный/случайный абстрактор" по числу ориентиров,
с усреднением по нескольким seed — чтобы отделить реальный тренд от шума
обучения (см. обсуждение: одиночные прогоны на 0/3/6 ориентиров дали
неоднозначную картину при добавлении промежуточных точек).

Всё в ОДНОМ процессе: без накладных расходов на перезапуск Python
15+ раз, результаты собираются и усредняются автоматически, строится
график с доверительным интервалом.

Запуск:
  python run_landmark_sweep.py --landmarks 0 3 6 10 15 --seeds 0 1 2 \
      --episodes 200 --ep-len 48 --epochs 40 --k 8

Быстрая проверка перед полным прогоном (несколько минут):
  python run_landmark_sweep.py --landmarks 0 6 15 --seeds 0 1 \
      --episodes 40 --ep-len 20 --epochs 8 --k 4 --quick-check
"""
import argparse
import json
import time
import numpy as np
import torch

from train_hier import collect, train, encode_all, room_accuracy, ridge_r2
from models_hier import Abstractor
from env_openworld import ABSTRACT_GRID


def run_one(n_landmarks, seed, episodes, ep_len, epochs, k, device,
            latent1=128, latent2=32):
    obs, acts, local_pos, rooms = collect(
        episodes, ep_len, seed=seed, env_name="open_landmark",
        n_landmarks=n_landmarks)

    enc, abst, p1, p2, gap = train(
        obs, acts, epochs, k, device, flat=False, seed=seed,
        latent1=latent1, latent2=latent2)
    enc.eval(); abst.eval(); p1.eval(); p2.eval()

    Z1, Z2 = encode_all(enc, abst, obs, device)
    rnd_abst = Abstractor(latent1, latent2).to(device).eval()
    _, Z2rnd = encode_all(enc, rnd_abst, obs, device)

    Z2f = Z2.reshape(-1, Z2.shape[-1])
    Z2r = Z2rnd.reshape(-1, Z2rnd.shape[-1])
    RID = torch.from_numpy(rooms.reshape(-1)).long()
    n_cls = ABSTRACT_GRID * ABSTRACT_GRID

    z2_acc = room_accuracy(Z2f, RID, device, n_cls)
    z2rand_acc = room_accuracy(Z2r, RID, device, n_cls)
    return {
        "n_landmarks": n_landmarks, "seed": seed,
        "z2_room_acc": round(float(z2_acc), 4),
        "z2rand_room_acc": round(float(z2rand_acc), 4),
        "gap_pp": round(100 * (z2_acc - z2rand_acc), 2),
        "level_use_gap": round(float(gap), 4),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--landmarks", type=int, nargs="+", default=[0, 3, 6, 10, 15])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--ep-len", type=int, default=48)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--quick-check", action="store_true",
                   help="просто пометка в имени файла для тестовых прогонов")
    p.add_argument("--out", type=str, default="landmark_sweep.json")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device} | ориентиры: {args.landmarks} | seeds: {args.seeds}")
    print(f"всего прогонов: {len(args.landmarks) * len(args.seeds)}\n")

    all_runs = []
    total = len(args.landmarks) * len(args.seeds)
    done = 0
    t0 = time.time()
    for n in args.landmarks:
        for s in args.seeds:
            r = run_one(n, s, args.episodes, args.ep_len, args.epochs,
                       args.k, device)
            all_runs.append(r)
            done += 1
            elapsed = time.time() - t0
            eta = elapsed / done * (total - done)
            print(f"[{done}/{total}] n_landmarks={n} seed={s} -> "
                  f"gap={r['gap_pp']:+.1f}pp  "
                  f"(прошло {elapsed/60:.1f} мин, осталось ~{eta/60:.1f} мин)",
                  flush=True)

    # --- агрегация: среднее и std по seed для каждого n_landmarks ---
    summary = []
    for n in args.landmarks:
        gaps = [r["gap_pp"] for r in all_runs if r["n_landmarks"] == n]
        summary.append({
            "n_landmarks": n,
            "gap_mean_pp": round(float(np.mean(gaps)), 2),
            "gap_std_pp": round(float(np.std(gaps)), 2),
            "gap_all": gaps,
        })

    print("\n" + "=" * 60)
    print(f"{'n_landmarks':>12} {'gap mean (pp)':>15} {'gap std (pp)':>14}")
    print("-" * 60)
    for s in summary:
        print(f"{s['n_landmarks']:>12} {s['gap_mean_pp']:>15.2f} "
              f"{s['gap_std_pp']:>14.2f}")
    print("=" * 60)

    with open(args.out, "w") as f:
        json.dump({"runs": all_runs, "summary": summary}, f, indent=2)
    print(f"\nСохранено: {args.out}")

    # --- график с доверительным интервалом ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = [s["n_landmarks"] for s in summary]
        ys = [s["gap_mean_pp"] for s in summary]
        es = [s["gap_std_pp"] for s in summary]
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=4, color="tab:purple")
        ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.set_xlabel("число постоянных ориентиров")
        ax.set_ylabel("разрыв обученный/случайный z2 (п.п.)")
        ax.set_title(f"Польза иерархии vs плотность якорей "
                     f"({len(args.seeds)} seeds/точка)")
        fig.tight_layout()
        fig.savefig("landmark_sweep_curve.png", dpi=150)
        print("Сохранено: landmark_sweep_curve.png")
    except Exception as e:
        print(f"График не построен: {e}")


if __name__ == "__main__":
    main()
