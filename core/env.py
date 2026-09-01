"""Среда "две комнаты": агент-точка движется в боксе 64x64 с перегородкой.

Никаких зависимостей кроме numpy. Наблюдение — картинка (1, 64, 64) float32 в [0, 1],
действие — вектор (dx, dy) в [-1, 1]. Истинное состояние (x, y) доступно для probe.
"""
import numpy as np

SIZE = 64          # размер кадра в пикселях
WALL_X = 32        # x-координата перегородки
GAP = (26, 38)     # проём в перегородке (y от и до)
AGENT_R = 3        # радиус агента
STEP = 8.0         # максимальный сдвиг за шаг, пикселей


class TwoRoomsEnv:
    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        self.pos = np.zeros(2, dtype=np.float32)
        self._build_walls()

    def _build_walls(self):
        w = np.zeros((SIZE, SIZE), dtype=bool)
        w[0:2, :] = w[-2:, :] = w[:, 0:2] = w[:, -2:] = True   # рамка
        w[:, WALL_X - 1:WALL_X + 1] = True                     # перегородка
        w[GAP[0]:GAP[1], WALL_X - 1:WALL_X + 1] = False        # проём
        self.walls = w

    def _collides(self, pos) -> bool:
        x, y = pos
        lo_x, hi_x = int(x - AGENT_R), int(x + AGENT_R) + 1
        lo_y, hi_y = int(y - AGENT_R), int(y + AGENT_R) + 1
        if lo_x < 0 or lo_y < 0 or hi_x > SIZE or hi_y > SIZE:
            return True
        return bool(self.walls[lo_y:hi_y, lo_x:hi_x].any())

    def reset(self) -> np.ndarray:
        while True:
            pos = self.rng.uniform(AGENT_R + 3, SIZE - AGENT_R - 3, size=2).astype(np.float32)
            if not self._collides(pos):
                self.pos = pos
                return self.render()

    def step(self, action: np.ndarray) -> np.ndarray:
        """action: (dx, dy) в [-1, 1]. Возвращает следующее наблюдение."""
        delta = np.clip(np.asarray(action, dtype=np.float32), -1, 1) * STEP
        # пробуем полное движение, затем по осям (скольжение вдоль стен)
        for cand in (self.pos + delta,
                     self.pos + np.array([delta[0], 0.0]),
                     self.pos + np.array([0.0, delta[1]])):
            if not self._collides(cand):
                self.pos = cand.astype(np.float32)
                break
        return self.render()

    def render(self) -> np.ndarray:
        img = np.zeros((SIZE, SIZE), dtype=np.float32)
        img[self.walls] = 0.5
        yy, xx = np.mgrid[0:SIZE, 0:SIZE]
        mask = (xx - self.pos[0]) ** 2 + (yy - self.pos[1]) ** 2 <= AGENT_R ** 2
        img[mask] = 1.0
        return img[None]  # (1, 64, 64)

    @property
    def state(self) -> np.ndarray:
        """Истинная позиция агента, нормированная в [0, 1] — для linear probe."""
        return self.pos / SIZE
