"""Варианты среды для стресс-тестирования мировой модели.

Каждый вариант ломает своё предположение базовой среды:

  ThreeRooms      — усложнённая топология (два проёма, три комнаты).
                    Проверяет: масштабируется ли выученная геометрия латента.

  MovingDistractor — второй объект движется НЕЗАВИСИМО от действий агента.
                    Проверяет ключевой тезис JEPA: модель должна игнорировать
                    непредсказуемое и не тратить ёмкость на его моделирование.
                    (Автоэнкодер здесь обязан деградировать, JEPA — нет.)

  FogOfWar        — видно только окно вокруг агента, остальное скрыто.
                    Ломает марковость: одного кадра больше НЕ достаточно.
                    Проверяет предел безпамятного предиктора z_t -> z_{t+1}.

Все варианты наследуют интерфейс TwoRoomsEnv: reset() / step(a) / render() / state.
"""
import numpy as np
from env import TwoRoomsEnv, SIZE, AGENT_R, STEP


class ThreeRoomsEnv(TwoRoomsEnv):
    """Три комнаты: две вертикальные перегородки, проёмы в разных местах."""

    def _build_walls(self):
        w = np.zeros((SIZE, SIZE), dtype=bool)
        w[0:2, :] = w[-2:, :] = w[:, 0:2] = w[:, -2:] = True
        w[:, 21:23] = True                 # первая перегородка
        w[8:20, 21:23] = False             # проём вверху
        w[:, 42:44] = True                 # вторая перегородка
        w[44:56, 42:44] = False            # проём внизу
        self.walls = w


class MovingDistractorEnv(TwoRoomsEnv):
    """Базовая среда + объект, движущийся независимо от действий агента.

    Тезис JEPA: этот объект непредсказуем по (z, a), поэтому хорошее
    представление должно его игнорировать, а не пытаться моделировать.
    """

    def __init__(self, seed=None, speed=2.5):
        super().__init__(seed=seed)
        self.speed = speed
        self.d_pos = np.array([SIZE * 0.75, SIZE * 0.25], dtype=np.float32)
        self.d_vel = self.rng.normal(size=2).astype(np.float32)

    def reset(self):
        obs = super().reset()
        while True:
            p = self.rng.uniform(AGENT_R + 4, SIZE - AGENT_R - 4, 2).astype(np.float32)
            if not self._collides(p):
                self.d_pos = p
                break
        self.d_vel = self.rng.normal(size=2).astype(np.float32)
        return self.render()

    def _move_distractor(self):
        # случайное блуждание с отражением от препятствий
        self.d_vel = 0.85 * self.d_vel + 0.4 * self.rng.normal(size=2)
        nxt = self.d_pos + np.clip(self.d_vel, -1, 1) * self.speed
        if self._collides(nxt):
            self.d_vel = -self.d_vel
            nxt = self.d_pos
        self.d_pos = nxt.astype(np.float32)

    def step(self, action):
        super().step(action)
        self._move_distractor()
        return self.render()

    def render(self):
        img = super().render()
        yy, xx = np.mgrid[0:SIZE, 0:SIZE]
        mask = (xx - self.d_pos[0]) ** 2 + (yy - self.d_pos[1]) ** 2 <= (AGENT_R - 1) ** 2
        img[0][mask] = 0.75                # отвлекающий объект другой яркости
        return img


class EgocentricEnv(TwoRoomsEnv):
    """Эгоцентрический вид: кадр ВЫРЕЗАН вокруг агента, агент всегда в центре.

    ВАЖНО (урок отладки): наивный "туман войны" — маска поверх глобального
    кадра — НЕ даёт частичной наблюдаемости: агент остаётся нарисован на своей
    настоящей позиции, и она считывается напрямую. Более того, задача
    становится ЛЕГЧЕ (меньше лишних пикселей стен).

    Здесь вид действительно эгоцентрический: агент всегда в центре кадра,
    поэтому глобальная позиция по одному наблюдению принципиально не
    восстановима — только по конфигурации видимых стен. Это ломает марковость
    и должно обрушить probe R^2 у безпамятного предиктора z_t -> z_{t+1}.
    """

    def __init__(self, seed=None, window=24):
        self.window = window
        super().__init__(seed=seed)

    def render(self):
        full = super().render()[0]
        h = self.window // 2
        cx, cy = int(round(self.pos[0])), int(round(self.pos[1]))
        # паддим мир, чтобы вырезка у границ не выходила за пределы
        padded = np.pad(full, h, mode="constant", constant_values=0.5)
        crop = padded[cy: cy + 2 * h, cx: cx + 2 * h]
        # растягиваем до 64x64 повтором пикселей (без внешних зависимостей)
        k = SIZE // (2 * h)
        out = np.kron(crop, np.ones((k, k), dtype=np.float32))
        if out.shape[0] != SIZE:                    # добиваем до нужного размера
            out = np.pad(out, ((0, SIZE - out.shape[0]),
                               (0, SIZE - out.shape[1])), mode="edge")
        return out[None].astype(np.float32)


VARIANTS = {
    "base": TwoRoomsEnv,
    "three_rooms": ThreeRoomsEnv,
    "distractor": MovingDistractorEnv,
    "egocentric": EgocentricEnv,
}
