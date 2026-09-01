"""2.5D-среда: планарный квадрокоптер с недоуправляемой динамикой.

КЛЮЧЕВОЕ ОТЛИЧИЕ от всех предыдущих сред проекта. Раньше действие задавало
скорость напрямую (pos += action * STEP) — эффект действия на следующий кадр
был мгновенным и сильным. Здесь агент управляет ТЯГОЙ и УГЛОВЫМ УСКОРЕНИЕМ,
а горизонтальное движение возникает только как следствие наклона корпуса:

    ax = (T / m) * sin(theta)          # горизонт — только через наклон
    az = (T / m) * cos(theta) - g      # вертикаль — тяга против гравитации

Это делает систему НЕДОУПРАВЛЯЕМОЙ (2 входа, 3 степени свободы: x, z, theta),
как настоящий квадрокоптер. Практическое следствие для world model:
одношаговый action_gap становится слабым сигналом, потому что действие
влияет на УСКОРЕНИЕ, а видимое смещение появляется лишь через несколько
шагов интегрирования. Ожидается, что потребуется multi-step loss при
обучении предиктора — не как поздняя доработка, а с самого начала.

Состояние: (x, z, vx, vz, theta, omega)
Действие:  (thrust_cmd, pitch_rate_cmd), оба в [-1, 1]
Наблюдение: 64×64 grayscale — тот же формат, что во всех прошлых средах,
            чтобы энкодер/VICReg/ансамбль/QUBO переиспользовались без правок.
"""
import numpy as np

SIZE = 128                  # мир 128×128 условных единиц
VIEW = 64                   # эгоцентрическое окно наблюдения (пикселей)

# --- физика ---
DT = 0.08                   # шаг интегрирования, с
GRAVITY = 9.81
MASS = 1.0
THRUST_HOVER = MASS * GRAVITY          # тяга, компенсирующая вес
THRUST_RANGE = 0.6                     # ±60% от висения
MAX_PITCH_RATE = 2.5                   # рад/с, максимальная угловая скорость
MAX_PITCH = 0.7                        # рад (~40°), ограничение наклона
DRAG = 0.25                            # линейное сопротивление
ANG_DRAG = 2.0                         # угловое демпфирование

AGENT_R = 2.5
N_OBSTACLES = 14
OBSTACLE_R_RANGE = (3, 6)
GROUND_Z = 4.0                         # уровень земли (крушение ниже)
CEILING_Z = SIZE - 4.0


class Quad2DEnv:
    """Планарный квадрокоптер: вертикальная плоскость (x, z)."""

    def __init__(self, seed=None, obstacles=True, egocentric=True, dual_view=False):
        self.rng = np.random.default_rng(seed)
        self.use_obstacles = obstacles
        self.egocentric = egocentric
        self.dual_view = dual_view
        self._scatter_obstacles()
        self.reset()

    def _scatter_obstacles(self):
        if not self.use_obstacles:
            self.obs_pos = np.zeros((0, 2))
            self.obs_r = np.zeros(0)
            return
        self.obs_pos = self.rng.uniform(12, SIZE - 12, size=(N_OBSTACLES, 2))
        self.obs_r = self.rng.uniform(*OBSTACLE_R_RANGE, size=N_OBSTACLES)

    def _collides(self, pos):
        x, z = pos
        if z < GROUND_Z or z > CEILING_Z or x < AGENT_R or x > SIZE - AGENT_R:
            return True
        if len(self.obs_pos) == 0:
            return False
        d = np.linalg.norm(self.obs_pos - pos, axis=1)
        return bool(np.any(d < self.obs_r + AGENT_R))

    def reset(self):
        while True:
            p = self.rng.uniform([15, GROUND_Z + 10],
                                 [SIZE - 15, CEILING_Z - 10]).astype(np.float32)
            if not self._collides(p):
                break
        self.pos = p                                    # (x, z)
        self.vel = self.rng.normal(0, 0.5, 2).astype(np.float32)
        self.theta = float(self.rng.normal(0, 0.05))    # наклон, рад
        self.omega = 0.0                                # угловая скорость
        self.crashed = False
        return self.render()

    def step(self, action):
        """action = (thrust_cmd, pitch_rate_cmd), оба в [-1, 1]."""
        a = np.clip(np.asarray(action, dtype=np.float32), -1, 1)
        thrust = THRUST_HOVER * (1.0 + THRUST_RANGE * a[0])
        pitch_cmd = MAX_PITCH_RATE * a[1]

        # угловая динамика: команда задаёт угловую скорость с демпфированием
        self.omega += (pitch_cmd - ANG_DRAG * self.omega) * DT
        self.theta = float(np.clip(self.theta + self.omega * DT,
                                   -MAX_PITCH, MAX_PITCH))

        # линейная динамика: горизонталь ТОЛЬКО через наклон
        ax = (thrust / MASS) * np.sin(self.theta) - DRAG * self.vel[0]
        az = (thrust / MASS) * np.cos(self.theta) - GRAVITY - DRAG * self.vel[1]

        self.vel = self.vel + np.array([ax, az], dtype=np.float32) * DT
        new_pos = self.pos + self.vel * DT * 10.0        # масштаб мира

        if self._collides(new_pos):
            # мягкое столкновение: гасим скорость и отталкиваемся, но эпизод
            # НЕ прерывается. Для сбора обучающих данных важно продолжать —
            # обход препятствий это задача планировщика, а не политики сбора;
            # прерывание эпизода на каждом касании давало вырожденные,
            # обрубленные траектории. Флаг crashed остаётся для диагностики.
            self.vel *= -0.4
            self.crashed = True
        else:
            self.pos = new_pos.astype(np.float32)
            self.crashed = False

        return self.render()

    def render(self):
        """Наблюдение 64×64. Два режима, задаваемых egocentric в конструкторе:

        egocentric=True (по умолчанию) — кроп вокруг дрона, дрон всегда в
            центре. Абсолютная позиция выводима лишь частично, по окружающим
            препятствиям. ИЗМЕРЕНО: probe R² по позиции ~0.53/0.69, что
            ставит потолок ~0.43 на корреляцию любой латентной метрики с
            реальной дистанцией — из-за чего планирование по цели-изображению
            структурно не работает (0/10 против 10/10 на истинной физике).

        egocentric=False — весь мир в кадре, дрон виден на своём месте.
            Абсолютная позиция видна напрямую. Прямой аналог условия `full`
            из комнатных сред. Проверка гипотезы: если планирование чинится
            сменой только этого флага — диагноз подтверждён.

        Наклон корпуса виден как ориентация отметки дрона в обоих режимах.
        """
        world = np.zeros((SIZE, SIZE), dtype=np.float32)
        zz, xx = np.mgrid[0:SIZE, 0:SIZE]

        # земля и потолок
        world[zz < GROUND_Z] = 0.35
        world[zz > CEILING_Z] = 0.35

        for (ox, oz), r in zip(self.obs_pos, self.obs_r):
            world[(xx - ox) ** 2 + (zz - oz) ** 2 <= r ** 2] = 0.5

        # дрон: короткая «планка», повёрнутая на угол theta — наклон виден
        cx, cz = self.pos
        length = 4.0 if self.egocentric else 6.0
        dx, dz = np.cos(self.theta) * length, np.sin(self.theta) * length
        radius = 1.4 if self.egocentric else 2.0
        for t in np.linspace(-1, 1, 11):
            px, pz = cx + dx * t, cz + dz * t
            mask = (xx - px) ** 2 + (zz - pz) ** 2 <= radius ** 2
            world[mask] = 1.0

        if self.dual_view:
            # ДВА КАНАЛА: глобальный + эгоцентрический одновременно.
            # Обоснование (измерено): глобальный вид даёт позицию (probe
            # x/z ~0.99) но теряет наклон (theta 0.34); эгоцентрический даёт
            # наклон (0.92) но теряет позицию (0.53/0.69). Для недоуправляемой
            # динамики нужны ОБА: позиция — для cost-функции планировщика,
            # наклон — потому что горизонтальное движение идёт только через него.
            step = SIZE // VIEW
            glob = world[::step, ::step][::-1]
            h = VIEW // 2
            ix, iz = int(round(cx)), int(round(cz))
            padded = np.pad(world, h, mode="constant", constant_values=0.35)
            ego = padded[iz: iz + 2 * h, ix: ix + 2 * h][::-1]
            return np.stack([glob, ego]).astype(np.float32)

        if not self.egocentric:
            # весь мир, сжатый до 64×64 (SIZE=128 -> берём каждый второй пиксель)
            step = SIZE // VIEW
            out = world[::step, ::step]
            return out[::-1][None].astype(np.float32)

        h = VIEW // 2
        ix, iz = int(round(cx)), int(round(cz))
        padded = np.pad(world, h, mode="constant", constant_values=0.35)
        crop = padded[iz: iz + 2 * h, ix: ix + 2 * h]
        # переворачиваем по вертикали: z вверх — как принято в физике
        return crop[::-1][None].astype(np.float32)

    # --- ground truth для probe ---
    @property
    def state(self) -> np.ndarray:
        """Полное истинное состояние, нормированное — для диагностики."""
        return np.array([self.pos[0] / SIZE, self.pos[1] / SIZE,
                         self.vel[0] / 10.0, self.vel[1] / 10.0,
                         self.theta / MAX_PITCH, self.omega / MAX_PITCH_RATE],
                        dtype=np.float32)

    @property
    def global_pos(self) -> np.ndarray:
        return (self.pos / SIZE).astype(np.float32)


def hover_policy(env, rng, target_z=None, target_x=None, noise=0.3):
    """PD-контроллер висения + шум, для сбора осмысленных данных.

    Чисто случайная политика на недоуправляемой системе роняет дрон за
    ~18 шагов — данные вырождаются в сплошные столкновения. Этот контроллер
    удерживает высоту И горизонтальную позицию, а шум поверх обеспечивает
    покрытие пространства состояний, не разрушая эпизод.

    Каскадное управление по x — важная деталь недоуправляемой системы:
    нельзя задать горизонтальную силу напрямую, поэтому внешний контур
    вычисляет ЖЕЛАЕМЫЙ УГОЛ наклона из ошибки по x, а внутренний контур
    отрабатывает этот угол через угловое ускорение. Именно эта каскадная
    структура (позиция → угол → момент) и делает квадрокоптер сложнее
    для world model, чем все предыдущие среды проекта.
    """
    if target_z is None:
        target_z = SIZE * 0.5
    if target_x is None:
        target_x = SIZE * 0.5

    # внешний контур: желаемый наклон из ошибки по горизонтали
    x_err = target_x - env.pos[0]

    # простое отталкивание от ближайшего препятствия — политика сбора данных
    # не должна залипать у стен (иначе датасет забит столкновениями), но и
    # не должна быть полноценным планировщиком: это задача CEM/QUBO выше
    if len(env.obs_pos) > 0:
        rel = env.pos - env.obs_pos
        d = np.linalg.norm(rel, axis=1) - env.obs_r
        i = int(np.argmin(d))
        if d[i] < 14.0:
            push = rel[i] / (np.linalg.norm(rel[i]) + 1e-6)
            x_err += push[0] * (14.0 - d[i]) * 2.5
            target_z = env.pos[1] + push[1] * (14.0 - d[i]) * 2.5

    z_err = target_z - env.pos[1]
    thrust_cmd = np.clip(0.05 * z_err - 0.30 * env.vel[1], -1, 1)

    theta_des = np.clip(0.012 * x_err - 0.10 * env.vel[0], -MAX_PITCH, MAX_PITCH)

    # внутренний контур: отработка желаемого угла
    pitch_cmd = np.clip(3.0 * (theta_des - env.theta) - 0.8 * env.omega, -1, 1)

    a = np.array([thrust_cmd, pitch_cmd], dtype=np.float32)
    a += rng.normal(0, noise, 2).astype(np.float32)
    return np.clip(a, -1, 1)
