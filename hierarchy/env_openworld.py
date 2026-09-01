"""Открытый мир: большое пространство со случайными препятствиями, БЕЗ комнат.

Ключевое отличие от env_building.py: нет дискретных зон с чистым ground truth
(room_id). Вместо этого — непрерывное пространство, разбитое на грубую
регулярную сетку (occupancy-подобную) ТОЛЬКО для диагностики, а не как
структура среды. Это честная проверка: держатся ли выводы про память и
иерархию, когда "медленная переменная" не задана архитектурой мира, а
выделяется искусственно поверх непрерывного пространства.

SIZE увеличен вчетверо по площади относительно building.py (128x128 против
63x63) — открытость проверяем и через масштаб, не только через отсутствие
структуры. Препятствия разбросаны случайно и не образуют регулярных стен,
как в лесу или на поле: их плотность и расположение — единственный источник
структуры, которую может выучить модель.
"""
import numpy as np

SIZE = 128
AGENT_R = 2
STEP = 4.0
N_OBSTACLES = 40
OBSTACLE_R_RANGE = (2, 6)
ABSTRACT_GRID = 4          # 4x4 грубых зоны — ТОЛЬКО для диагностики probe


class OpenWorldEnv:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.pos = np.zeros(2, dtype=np.float32)
        self._scatter_obstacles()

    def _scatter_obstacles(self):
        """Случайные круглые препятствия — без регулярной структуры (не стены)."""
        self.obs_pos = self.rng.uniform(10, SIZE - 10, size=(N_OBSTACLES, 2))
        self.obs_r = self.rng.uniform(*OBSTACLE_R_RANGE, size=N_OBSTACLES)

    def _collides(self, pos):
        x, y = pos
        if x < AGENT_R or y < AGENT_R or x > SIZE - AGENT_R or y > SIZE - AGENT_R:
            return True
        d = np.linalg.norm(self.obs_pos - pos, axis=1)
        return bool(np.any(d < self.obs_r + AGENT_R))

    def reset(self):
        while True:
            p = self.rng.uniform(15, SIZE - 15, size=2).astype(np.float32)
            if not self._collides(p):
                self.pos = p
                return self.render()

    def step(self, action):
        d = np.clip(np.asarray(action, dtype=np.float32), -1, 1) * STEP
        for cand in (self.pos + d,
                     self.pos + np.array([d[0], 0.0], dtype=np.float32),
                     self.pos + np.array([0.0, d[1]], dtype=np.float32)):
            if not self._collides(cand):
                self.pos = cand.astype(np.float32)
                break
        return self.render()

    def render(self):
        """Локальное окно 64x64 вокруг агента — эгоцентрический вид по умолчанию,
        т.к. в открытом мире вид сверху на всю карту нереалистичен (мир большой)."""
        img = np.zeros((SIZE, SIZE), dtype=np.float32)
        yy, xx = np.mgrid[0:SIZE, 0:SIZE]
        for (ox, oy), r in zip(self.obs_pos, self.obs_r):
            mask = (xx - ox) ** 2 + (yy - oy) ** 2 <= r ** 2
            img[mask] = 0.5
        mask = (xx - self.pos[0]) ** 2 + (yy - self.pos[1]) ** 2 <= AGENT_R ** 2
        img[mask] = 1.0

        h = 32
        cx, cy = int(round(self.pos[0])), int(round(self.pos[1]))
        padded = np.pad(img, h, mode="constant", constant_values=0.0)
        crop = padded[cy: cy + 2 * h, cx: cx + 2 * h]
        return crop[None].astype(np.float32)

    # --- ground truth для диагностики (не видна модели) ---
    @property
    def zone_id(self) -> int:
        """Грубая 4x4 зона по абсолютной позиции — честная замена room_id.
        Не архитектурная структура мира, а чисто диагностическая величина."""
        gx = min(int(self.pos[0] / SIZE * ABSTRACT_GRID), ABSTRACT_GRID - 1)
        gy = min(int(self.pos[1] / SIZE * ABSTRACT_GRID), ABSTRACT_GRID - 1)
        return gy * ABSTRACT_GRID + gx

    @property
    def local_pos(self) -> np.ndarray:
        """Позиция внутри грубой зоны, нормированная в [0,1] — аналог
        локальной позиции внутри комнаты в building.py."""
        cell = SIZE / ABSTRACT_GRID
        return np.array([(self.pos[0] % cell) / cell,
                         (self.pos[1] % cell) / cell], dtype=np.float32)

    @property
    def room_id(self) -> int:
        """Алиас zone_id для совместимости с общим кодом train_hier.py."""
        return self.zone_id

    @property
    def global_pos(self) -> np.ndarray:
        return (self.pos / SIZE).astype(np.float32)


class OpenWorldDynamicEnv(OpenWorldEnv):
    """Открытый мир, но препятствия перемешиваются КАЖДЫЙ эпизод.

    Зачем: в базовом OpenWorldEnv карта препятствий фиксирована на весь
    датасет, и каждая зона получает уникальный, но ПОСТОЯННЫЙ узор соседних
    препятствий — модель может научиться узнавать зону как "отпечаток
    пальца" по одному кадру, без всякой памяти (аналог леса с уникальными
    визуальными ориентирами, где локализация по одному кадру реалистична).

    Здесь узор препятствий каждый раз новый, поэтому landmark-узнавание
    невозможно в принципе — зону можно понять только интегрируя историю
    движения. Это честный аналог туннеля: самоповторяющаяся, лишённая
    уникальных ориентиров структура, где классическая геометрия (и
    безпамятная модель) ломается по построению, а не случайно.
    """

    def reset(self):
        self._scatter_obstacles()      # новая карта препятствий на каждый эпизод
        return super().reset()


N_LANDMARKS_DEFAULT = 6
LANDMARK_R = 7
LANDMARK_VALUE = 0.85          # отличается от обычных препятствий (0.5) и агента (1.0)


class OpenWorldLandmarkEnv(OpenWorldEnv):
    """Открытый мир: препятствия перемешиваются каждый эпизод, КРОМЕ нескольких
    постоянных ориентиров, которые видны во все эпизоды на тех же местах.

    n_landmarks — параметр: снимаем полную зависимость "разрыв обученный/
    случайный абстрактор" от плотности якорей (0, 1, 2, 3, 6, 10, 15...),
    чтобы увидеть форму кривой — линейный рост, плато, порог.

    Гипотеза (уточнение правила из трёх условий): рекуррентной абстракции
    нужен не только механизм накопления истории и нехватка информации, но и
    РЕДКИЙ, РАЗЛИЧИМЫЙ якорный сигнал, за который можно "зацепиться" при
    обновлении — аналог прохода через дверь в building-среде. Чистое
    счисление пути (dead reckoning) без единого визуального якоря не
    обучается (см. OpenWorldDynamicEnv, n_landmarks=0).
    """

    def __init__(self, seed=None, n_landmarks=N_LANDMARKS_DEFAULT):
        self.n_landmarks = n_landmarks
        super().__init__(seed=seed)
        if n_landmarks > 0:
            self.landmark_pos = self.rng.uniform(20, SIZE - 20,
                                                  size=(n_landmarks, 2))
        else:
            self.landmark_pos = np.zeros((0, 2))

    def reset(self):
        self._scatter_obstacles()      # обычные препятствия — каждый раз новые
        # landmark_pos НЕ трогаем — они постоянны на протяжении всего датасета
        return OpenWorldEnv.reset(self)

    def _collides(self, pos):
        if super()._collides(pos):
            return True
        d = np.linalg.norm(self.landmark_pos - pos, axis=1)
        return bool(np.any(d < LANDMARK_R + AGENT_R))

    def render(self):
        img = super().render()   # обычные препятствия + агент, уже эгоцентрический кроп
        # рисуем ориентиры отдельно поверх того же кропа
        h = 32
        cx, cy = int(round(self.pos[0])), int(round(self.pos[1]))
        yy, xx = np.mgrid[0:2 * h, 0:2 * h]
        for lx, ly in self.landmark_pos:
            local_x, local_y = lx - cx + h, ly - cy + h
            mask = (xx - local_x) ** 2 + (yy - local_y) ** 2 <= LANDMARK_R ** 2
            valid = (mask) & (yy >= 0) & (yy < 2 * h) & (xx >= 0) & (xx < 2 * h)
            img[0][valid] = LANDMARK_VALUE
        return img
