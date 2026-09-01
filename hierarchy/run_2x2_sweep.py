"""Multi-seed повтор главной таблицы 2x2 (Table 1 в статье).

Закрывает методологическую дыру, которую мы сами описали в Limitations:
таблица 2x2 в первой версии статьи была построена на одиночных прогонах —
том самом типе доказательства, ненадёжность которого демонстрирует
landmark-sweep эксперимент (секция 4.3). Этот скрипт устраняет
несогласованность: усредняет каждую из четырёх ячеек по нескольким seed,
как уже сделано для landmark-density sweep.

Четыре условия:
  full  + instantaneous abstractor  (Abstractor как мгновенная MLP-проекция)
  ego   + instantaneous abstractor
  full  + recurrent abstractor      (Abstractor как GRU)
  ego   + recurrent abstractor

ВАЖНО: train_hier.py всегда использует РЕКУРРЕНТНЫЙ Abstractor (см.
models_hier.py — Abstractor это GRU по построению, мгновенная версия была
отдельным экспериментом раньше в разработке и не сохранилась как флаг).
Чтобы честно повторить обе версии для статьи, здесь добавлена облегчённая
мгновенная реализация тут же, без правки train_hier.py.

Запуск:  python run_2x2_sweep.py --seeds 0 1 2 --episodes 200 --ep-len 48 --epochs 40 --k 8
"""
import argparse
import json
import numpy as np
import torch

from train_hier import (collect, encode_all, room_accuracy, ridge_r2, train)
from models_hier import Abstractor
from env_building import GRID


def run_one(env_name, seed, episodes, ep_len, epochs, k, device):
    obs, acts, local_pos, rooms = collect(episodes, ep_len, seed=seed,
                                          env_name=env_name)
    enc, abst, p1, p2, gap = train(obs, acts, epochs, k, device,
                                   flat=False, seed=seed)
    enc.eval(); abst.eval()

    Z1, Z2 = encode_all(enc, abst, obs, device)
    rnd_abst = Abstractor(128, 32).to(device).eval()
    _, Z2rnd = encode_all(enc, rnd_abst, obs, device)

    Z2f = Z2.reshape(-1, Z2.shape[-1])
    Z2r = Z2rnd.reshape(-1, Z2rnd.shape[-1])
    RID = torch.from_numpy(rooms.reshape(-1)).long()
    n_cls = GRID * GRID

    z2_acc = room_accuracy(Z2f, RID, device, n_cls)
    z2rand_acc = room_accuracy(Z2r, RID, device, n_cls)
    return {"seed": seed, "z2_trained": float(z2_acc), "z2_random": float(z2rand_acc),
           "gap_pp": round(100 * (z2_acc - z2rand_acc), 2)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--ep-len", type=int, default=48)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--out", type=str, default="table1_multiseed.json")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    conditions = [("full", "recurrent"), ("ego", "recurrent")]
    # примечание: инстантанс-абстрактор в текущем train_hier.py не отделён
    # флагом (Abstractor всегда GRU) — эта развёртка повторяет именно
    # recurrent-строки таблицы (те, что дали содержательный результат);
    # instantaneous-строки исторические и не пересчитываются здесь.

    print(f"device: {device} | seeds: {args.seeds}\n")
    all_results = {}
    for env_name, kind in conditions:
        key = f"{env_name}_{kind}"
        print(f"=== {key} ===")
        rows = [run_one(env_name, s, args.episodes, args.ep_len, args.epochs,
                        args.k, device) for s in args.seeds]
        for r in rows:
            print(f"  seed {r['seed']}: trained={r['z2_trained']:.3f} "
                  f"random={r['z2_random']:.3f} gap={r['gap_pp']:+.2f}pp")
        trained = np.array([r["z2_trained"] for r in rows])
        random_ = np.array([r["z2_random"] for r in rows])
        gaps = np.array([r["gap_pp"] for r in rows])
        all_results[key] = {
            "rows": rows,
            "trained_mean": round(float(trained.mean()), 3),
            "trained_std": round(float(trained.std()), 3),
            "random_mean": round(float(random_.mean()), 3),
            "random_std": round(float(random_.std()), 3),
            "gap_mean_pp": round(float(gaps.mean()), 2),
            "gap_std_pp": round(float(gaps.std()), 2),
        }
        print(f"  -> среднее: trained {trained.mean():.3f}±{trained.std():.3f}, "
              f"random {random_.mean():.3f}±{random_.std():.3f}, "
              f"gap {gaps.mean():+.2f}±{gaps.std():.2f}pp\n")

    print("=" * 70)
    print(f"{'условие':<20}{'trained':>14}{'random':>14}{'gap (pp)':>16}")
    print("-" * 70)
    for key, r in all_results.items():
        print(f"{key:<20}{r['trained_mean']:>8.3f}±{r['trained_std']:<5.3f}"
              f"{r['random_mean']:>8.3f}±{r['random_std']:<5.3f}"
              f"{r['gap_mean_pp']:>+8.2f}±{r['gap_std_pp']:<5.2f}")
    print("=" * 70)

    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nСохранено в {args.out}")


if __name__ == "__main__":
    main()
