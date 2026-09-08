import numpy as np

SIZE = 63           
ROOM = 21
GRID = 3
AGENT_R = 2
STEP = 2.5


class BuildingEnv:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.pos = np.zeros(2, dtype=np.float32)
        self._build_walls()

    def _build_walls(self):
        w = np.zeros((SIZE, SIZE), dtype=bool)
        w[0, :] = w[-1, :] = w[:, 0] = w[:, -1] = True
        # внутренние стены между комнатами
        for i in (1, 2):
            c = i * ROOM
            w[:, c - 1:c + 1] = True          
            w[c - 1:c + 1, :] = True          
        for i in (1, 2):
            c = i * ROOM
            for j in range(GRID):
                mid = j * ROOM + ROOM // 2
                w[mid - 2:mid + 3, c - 1:c + 1] = False   
                w[c - 1:c + 1, mid - 2:mid + 3] = False   
        self.walls = w

    def _collides(self, pos):
        x, y = pos
        lo_x, hi_x = int(x - AGENT_R), int(x + AGENT_R) + 1
        lo_y, hi_y = int(y - AGENT_R), int(y + AGENT_R) + 1
        if lo_x < 0 or lo_y < 0 or hi_x > SIZE or hi_y > SIZE:
            return True
        return bool(self.walls[lo_y:hi_y, lo_x:hi_x].any())

    def reset(self):
        while True:
            p = self.rng.uniform(AGENT_R + 2, SIZE - AGENT_R - 2, 2).astype(np.float32)
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
        img = np.zeros((SIZE, SIZE), dtype=np.float32)
        img[self.walls] = 0.5
        yy, xx = np.mgrid[0:SIZE, 0:SIZE]
        m = (xx - self.pos[0]) ** 2 + (yy - self.pos[1]) ** 2 <= AGENT_R ** 2
        img[m] = 1.0
        img = np.pad(img, ((0, 64 - SIZE), (0, 64 - SIZE)), constant_values=0.5)
        return img[None]

    @property
    def room_id(self) -> int:
        cx = min(int(self.pos[0] // ROOM), GRID - 1)
        cy = min(int(self.pos[1] // ROOM), GRID - 1)
        return cy * GRID + cx

    @property
    def local_pos(self) -> np.ndarray:
        return np.array([(self.pos[0] % ROOM) / ROOM,
                         (self.pos[1] % ROOM) / ROOM], dtype=np.float32)

    @property
    def global_pos(self) -> np.ndarray:
        return (self.pos / SIZE).astype(np.float32)


class EgocentricBuildingEnv(BuildingEnv):


    def __init__(self, seed=None, window=20):
        self.window = window
        super().__init__(seed=seed)

    def render(self):
        full = super().render()[0][:SIZE, :SIZE]     
        h = self.window // 2
        cx, cy = int(round(self.pos[0])), int(round(self.pos[1]))
        padded = np.pad(full, h, mode="constant", constant_values=0.5)
        crop = padded[cy: cy + 2 * h, cx: cx + 2 * h]
        k = 64 // (2 * h)
        out = np.kron(crop, np.ones((k, k), dtype=np.float32))
        if out.shape[0] != 64:
            out = np.pad(out, ((0, 64 - out.shape[0]), (0, 64 - out.shape[1])),
                         mode="edge")
        return out[None].astype(np.float32)
