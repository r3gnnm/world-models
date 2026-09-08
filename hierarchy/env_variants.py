import numpy as np
from env import TwoRoomsEnv, SIZE, AGENT_R, STEP


class ThreeRoomsEnv(TwoRoomsEnv):

    def _build_walls(self):
        w = np.zeros((SIZE, SIZE), dtype=bool)
        w[0:2, :] = w[-2:, :] = w[:, 0:2] = w[:, -2:] = True
        w[:, 21:23] = True                 
        w[8:20, 21:23] = False             
        w[:, 42:44] = True                 
        w[44:56, 42:44] = False            
        self.walls = w


class MovingDistractorEnv(TwoRoomsEnv):

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
        img[0][mask] = 0.75                
        return img


class EgocentricEnv(TwoRoomsEnv):


    def __init__(self, seed=None, window=24):
        self.window = window
        super().__init__(seed=seed)

    def render(self):
        full = super().render()[0]
        h = self.window // 2
        cx, cy = int(round(self.pos[0])), int(round(self.pos[1]))
        padded = np.pad(full, h, mode="constant", constant_values=0.5)
        crop = padded[cy: cy + 2 * h, cx: cx + 2 * h]
        k = SIZE // (2 * h)
        out = np.kron(crop, np.ones((k, k), dtype=np.float32))
        if out.shape[0] != SIZE:                    
            out = np.pad(out, ((0, SIZE - out.shape[0]),
                               (0, SIZE - out.shape[1])), mode="edge")
        return out[None].astype(np.float32)


VARIANTS = {
    "base": TwoRoomsEnv,
    "three_rooms": ThreeRoomsEnv,
    "distractor": MovingDistractorEnv,
    "egocentric": EgocentricEnv,
}
