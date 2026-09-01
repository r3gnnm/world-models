"""Обучение иерархической JEPA + ДИАГНОСТИКИ схлопывания верхнего уровня.

Три проверки, отвечающие на вопрос "верхний уровень реален или декоративен?":

  1. level_use_gap — аналог action_gap для иерархии. Подменяем z2 на чужой
     (перемешанный по батчу) и смотрим, насколько вырастет ошибка уровня 1.
     Около нуля => нижний уровень игнорирует контекст, иерархия декоративна.

  2. Дифференциальный probe — специализация уровней:
        z1 должен лучше предсказывать ЛОКАЛЬНУЮ позицию внутри комнаты,
        z2 должен лучше предсказывать НОМЕР КОМНАТЫ (медленная абстракция).
     Если z2 не лучше z1 на комнате — абстракция не выделилась.

  3. Сравнение с плоским бейзлайном (--flat) при равном числе параметров:
     выигрыш должен расти С ГОРИЗОНТОМ, иначе дело в ёмкости, а не в иерархии.

Запуск:
  python train_hier.py --episodes 200 --epochs 20
  python train_hier.py --episodes 200 --epochs 20 --flat     # бейзлайн
"""
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F

from env_building import BuildingEnv, EgocentricBuildingEnv, GRID
from env_openworld import (OpenWorldEnv, OpenWorldDynamicEnv,
                           OpenWorldLandmarkEnv, ABSTRACT_GRID)
from models import Encoder
from models_hier import (Abstractor, Level1Predictor, Level2Predictor,
                         FlatPredictor, action_summary)
from losses import vicreg_loss


def collect(n_episodes, ep_len, seed=0, env_name="full", n_landmarks=6):
    if env_name == "ego":
        env_cls = EgocentricBuildingEnv
    elif env_name == "open":
        env_cls = OpenWorldEnv
    elif env_name == "open_dynamic":
        env_cls = OpenWorldDynamicEnv
    elif env_name == "open_landmark":
        env = OpenWorldLandmarkEnv(seed=seed, n_landmarks=n_landmarks)
    else:
        env_cls = BuildingEnv
    if env_name != "open_landmark":
        env = env_cls(seed=seed)
    rng = np.random.default_rng(seed)
    O, A, LP, RID = [], [], [], []
    for _ in range(n_episodes):
        o = env.reset()
        obs, acts, lp, rid = [o], [], [env.local_pos.copy()], [env.room_id]
        a = rng.uniform(-1, 1, 2)
        for _ in range(ep_len):
            a = np.clip(0.8 * a + 0.5 * rng.normal(size=2), -1, 1)
            o = env.step(a)
            obs.append(o); acts.append(a.astype(np.float32))
            lp.append(env.local_pos.copy()); rid.append(env.room_id)
        O.append(np.stack(obs)); A.append(np.stack(acts))
        LP.append(np.stack(lp)); RID.append(np.array(rid))
    return np.stack(O), np.stack(A), np.stack(LP), np.stack(RID)


def train(obs, acts, epochs, k, device, flat=False, latent1=128, latent2=32,
          bs=16, lr=3e-4, seed=0):
    torch.manual_seed(seed)   # фиксируем инициализацию весов и порядок батчей —
                              # иначе шум обучения маскирует эффект от числа ориентиров
    n_ep, T1 = obs.shape[0], obs.shape[1]
    T = T1 - 1
    enc = Encoder(latent1).to(device)
    abst = Abstractor(latent1, latent2).to(device)
    p1 = (FlatPredictor(latent1, latent2) if flat
          else Level1Predictor(latent1, latent2)).to(device)
    p2 = Level2Predictor(latent2).to(device)
    params = (list(enc.parameters()) + list(abst.parameters())
              + list(p1.parameters()) + list(p2.parameters()))
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-5)
    obs_t, acts_t = torch.from_numpy(obs), torch.from_numpy(acts)

    for ep in range(1, epochs + 1):
        perm = torch.randperm(n_ep)
        agg = {"l1": 0.0, "l2": 0.0, "gap": 0.0, "n": 0}
        for i in range(0, n_ep, bs):
            idx = perm[i:i + bs]
            o = obs_t[idx].to(device)
            a = acts_t[idx].to(device)
            B = o.shape[0]
            z1 = enc(o.view(B * T1, *o.shape[2:])).view(B, T1, -1)
            # РЕКУРРЕНТНОЕ накопление z2 вдоль эпизода (не мгновенная проекция!)
            h2 = abst.init_hidden(B, device)
            z2_list = []
            for t in range(T1):
                h2 = abst(z1[:, t], h2)
                z2_list.append(h2)
            z2 = torch.stack(z2_list, dim=1)   # (B, T1, latent2)

            # --- уровень 1: пошаговое предсказание с контекстом сверху ---
            loss1 = 0.0
            gap_num = gap_den = 0.0
            for t in range(T):
                z1_hat = p1(z1[:, t], a[:, t], z2[:, t])
                l, _ = vicreg_loss(z1_hat, z1[:, t + 1])
                loss1 = loss1 + l
                with torch.no_grad():   # диагностика: подмена контекста
                    z1_bad = p1(z1[:, t], a[:, t], z2[torch.randperm(B), t])
                    e_bad = F.mse_loss(z1_bad, z1[:, t + 1]).item()
                    e_ok = F.mse_loss(z1_hat, z1[:, t + 1]).item()
                    gap_num += e_bad - e_ok; gap_den += e_bad
            loss1 = loss1 / T

            # --- уровень 2: прыжковое предсказание через k шагов ---
            loss2 = 0.0
            n2 = 0
            for t in range(0, T - k, k):
                summ = action_summary(a[:, t:t + k])
                z2_hat = p2(z2[:, t], summ)
                l, _ = vicreg_loss(z2_hat, z2[:, t + k])
                loss2 = loss2 + l; n2 += 1
            loss2 = loss2 / max(n2, 1)

            loss = loss1 + loss2
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            agg["l1"] += loss1.item(); agg["l2"] += loss2.item()
            agg["gap"] += gap_num / max(gap_den, 1e-8); agg["n"] += 1

        if ep % 5 == 0 or ep == epochs:
            n = agg["n"]
            print(f"    epoch {ep:3d} | L1 {agg['l1']/n:7.3f} | L2 {agg['l2']/n:7.3f}"
                  f" | level_use_gap {agg['gap']/n:.4f}", flush=True)
    return enc, abst, p1, p2, agg["gap"] / agg["n"]


@torch.no_grad()
def encode_all(enc, abst, obs, device, bs=8):
    Z1, Z2 = [], []
    n_ep, T1 = obs.shape[0], obs.shape[1]
    for i in range(0, n_ep, bs):
        o = torch.from_numpy(obs[i:i + bs]).to(device)
        B = o.shape[0]
        z1 = enc(o.view(B * T1, *o.shape[2:])).view(B, T1, -1)
        h2 = abst.init_hidden(B, device)
        z2_list = []
        for t in range(T1):
            h2 = abst(z1[:, t], h2)
            z2_list.append(h2)
        z2 = torch.stack(z2_list, dim=1)
        Z1.append(z1.cpu()); Z2.append(z2.cpu())
    return torch.cat(Z1), torch.cat(Z2)


def mlp_probe_r2(X, Y, device, hidden=64, steps=600):
    """Нелинейный probe равной ёмкости — убирает конфаунд:
    линейный probe по z2 неявно нелинеен относительно z1 (z2 = MLP(z1)),
    поэтому сравнивать линейные probe по разным уровням некорректно."""
    import torch.nn as nn
    n_tr = int(0.8 * len(X))
    X, Y = X.to(device), Y.to(device)
    net = nn.Sequential(nn.Linear(X.shape[1], hidden), nn.ReLU(),
                        nn.Linear(hidden, Y.shape[1])).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    for _ in range(steps):
        idx = torch.randint(0, n_tr, (512,), device=device)
        loss = F.mse_loss(net(X[idx]), Y[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        P = net(X[n_tr:])
        ss_res = ((Y[n_tr:] - P) ** 2).sum(0)
        ss_tot = ((Y[n_tr:] - Y[n_tr:].mean(0)) ** 2).sum(0)
    return float((1 - ss_res / ss_tot).mean())


def ridge_r2(X, Y, device):
    n_tr = int(0.8 * len(X))
    X, Y = X.to(device), Y.to(device)
    Xtr = torch.cat([X[:n_tr], torch.ones(n_tr, 1, device=device)], 1)
    Xte = torch.cat([X[n_tr:], torch.ones(len(X) - n_tr, 1, device=device)], 1)
    reg = 1e-3 * torch.eye(Xtr.shape[1], device=device)
    w = torch.linalg.solve(Xtr.T @ Xtr + reg, Xtr.T @ Y[:n_tr])
    P = Xte @ w
    ss_res = ((Y[n_tr:] - P) ** 2).sum(0)
    ss_tot = ((Y[n_tr:] - Y[n_tr:].mean(0)) ** 2).sum(0)
    return float((1 - ss_res / ss_tot).mean())


def room_accuracy(X, rooms, device, n_cls=None):
    """Линейный классификатор комнаты/зоны (ridge на one-hot) -> точность."""
    if n_cls is None:
        n_cls = GRID * GRID
    Y = torch.zeros(len(rooms), n_cls)
    Y[torch.arange(len(rooms)), rooms] = 1.0
    n_tr = int(0.8 * len(X))
    X, Y = X.to(device), Y.to(device)
    Xtr = torch.cat([X[:n_tr], torch.ones(n_tr, 1, device=device)], 1)
    Xte = torch.cat([X[n_tr:], torch.ones(len(X) - n_tr, 1, device=device)], 1)
    reg = 1e-2 * torch.eye(Xtr.shape[1], device=device)
    w = torch.linalg.solve(Xtr.T @ Xtr + reg, Xtr.T @ Y[:n_tr])
    pred = (Xte @ w).argmax(-1)
    true = Y[n_tr:].argmax(-1)
    return float((pred == true).float().mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--ep-len", type=int, default=48)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--k", type=int, default=8, help="горизонт прыжка уровня 2")
    p.add_argument("--flat", action="store_true", help="плоский бейзлайн")
    p.add_argument("--env", type=str, default="full",
                   choices=["full", "ego", "open", "open_dynamic", "open_landmark"],
                   help="full/ego — комнаты, open — открытый мир (статичные "
                        "препятствия, аналог леса), open_dynamic — препятствия "
                        "меняются каждый эпизод (аналог туннеля без ориентиров), "
                        "open_landmark — динамические препятствия + редкие "
                        "постоянные ориентиры")
    p.add_argument("--out", type=str, default="hier_results.json")
    p.add_argument("--n-landmarks", type=int, default=6,
                   help="число постоянных ориентиров для --env open_landmark")
    p.add_argument("--seed", type=int, default=0,
                   help="seed для среды И обучения — варьируй для повторов "
                        "одной и той же конфигурации, чтобы усреднить шум")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mode = "ПЛОСКИЙ бейзлайн" if args.flat else "ИЕРАРХИЧЕСКАЯ модель"
    print(f"device: {device} | режим: {mode} | seed: {args.seed}")

    obs, acts, local_pos, rooms = collect(args.episodes, args.ep_len,
                                          seed=args.seed, env_name=args.env,
                                          n_landmarks=args.n_landmarks)
    print(f"данные: {obs.shape[0]} эпизодов × {obs.shape[1]} кадров")

    enc, abst, p1, p2, gap = train(obs, acts, args.epochs, args.k, device,
                                   flat=args.flat, seed=args.seed)
    enc.eval(); abst.eval(); p1.eval(); p2.eval()

    Z1, Z2 = encode_all(enc, abst, obs, device)
    # КОНТРОЛЬ: тот же абстрактор, но НЕОБУЧЕННЫЙ (случайные веса).
    # Если случайная проекция даёт тот же прирост — выигрыш даёт нелинейность
    # архитектуры, а не обучение абстракции.
    rnd_abst = Abstractor(128, 32).to(device).eval()  # необученный GRU-абстрактор
    _, Z2rnd = encode_all(enc, rnd_abst, obs, device)

    Z1f = Z1.reshape(-1, Z1.shape[-1]); Z2f = Z2.reshape(-1, Z2.shape[-1])
    Z2r = Z2rnd.reshape(-1, Z2rnd.shape[-1])
    LP = torch.from_numpy(local_pos.reshape(-1, 2))
    RID = torch.from_numpy(rooms.reshape(-1)).long()

    n_zones = ABSTRACT_GRID * ABSTRACT_GRID if args.env.startswith("open") else GRID * GRID
    res = {
        "mode": "flat" if args.flat else "hier",
        "env": args.env,
        "n_landmarks": args.n_landmarks if args.env == "open_landmark" else None,
        "seed": args.seed,
        "level_use_gap": round(float(gap), 4),
        # линейные probe (как раньше)
        "z1_local_r2": round(ridge_r2(Z1f, LP, device), 3),
        "z2_local_r2": round(ridge_r2(Z2f, LP, device), 3),
        "z1_room_acc": round(room_accuracy(Z1f, RID, device, n_zones), 3),
        "z2_room_acc": round(room_accuracy(Z2f, RID, device, n_zones), 3),
        # КОНТРОЛЬ: случайный абстрактор
        "z2rand_room_acc": round(room_accuracy(Z2r, RID, device, n_zones), 3),
        "z2rand_local_r2": round(ridge_r2(Z2r, LP, device), 3),
        # probe равной ёмкости (нелинейный для обоих уровней)
        "z1_local_r2_mlp": round(mlp_probe_r2(Z1f, LP, device), 3),
        "z2_local_r2_mlp": round(mlp_probe_r2(Z2f, LP, device), 3),
    }

    print("\n" + "=" * 66)
    print(f"ДИАГНОСТИКА ИЕРАРХИИ | среда: {args.env} | "
          f"{'плоский' if args.flat else 'иерархический'}")
    print("-" * 66)
    print(f"  level_use_gap (ур.1 использует контекст ур.2): "
          f"{res['level_use_gap']:.4f}")
    print("\n  ЛИНЕЙНЫЙ probe:")
    print(f"    локальная позиция: z1 R²={res['z1_local_r2']:.3f}"
          f"   z2 R²={res['z2_local_r2']:.3f}")
    print(f"    номер комнаты:     z1 acc={res['z1_room_acc']:.3f}"
          f"  z2 acc={res['z2_room_acc']:.3f}")
    print("\n  КОНТРОЛЬ — случайный (необученный) абстрактор:")
    print(f"    номер комнаты:     z2_rand acc={res['z2rand_room_acc']:.3f}"
          f"  (сравни с z2 acc={res['z2_room_acc']:.3f})")
    print(f"    локальная позиция: z2_rand R²={res['z2rand_local_r2']:.3f}")
    print("\n  probe РАВНОЙ ЁМКОСТИ (нелинейный для обоих):")
    print(f"    локальная позиция: z1 R²={res['z1_local_r2_mlp']:.3f}"
          f"   z2 R²={res['z2_local_r2_mlp']:.3f}")
    print("-" * 66)
    print("  Абстракция реальна, если обученный z2 заметно лучше случайного,")
    print("  а при равной ёмкости probe z1 сильнее на локальной позиции.")
    print("=" * 66)

    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nСохранено в {args.out}")


if __name__ == "__main__":
    main()
