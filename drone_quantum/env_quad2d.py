import numpy as np

SIZE = 128                  
VIEW = 64                   

DT = 0.08                  
GRAVITY = 9.81
MASS = 1.0
THRUST_HOVER = MASS * GRAVITY          
THRUST_RANGE = 0.6                     
MAX_PITCH_RATE = 2.5                   
MAX_PITCH = 0.7                        
DRAG = 0.25                            
ANG_DRAG = 2.0                         

AGENT_R = 2.5
N_OBSTACLES = 14
OBSTACLE_R_RANGE = (3, 6)
GROUND_Z = 4.0                         
CEILING_Z = SIZE - 4.0


class Quad2DEnv:

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
        self.theta = float(self.rng.normal(0, 0.05))    
        self.omega = 0.0                                
        self.crashed = False
        return self.render()

    def step(self, action):
        """action = (thrust_cmd, pitch_rate_cmd), оба в [-1, 1]."""
        a = np.clip(np.asarray(action, dtype=np.float32), -1, 1)
        thrust = THRUST_HOVER * (1.0 + THRUST_RANGE * a[0])
        pitch_cmd = MAX_PITCH_RATE * a[1]

        self.omega += (pitch_cmd - ANG_DRAG * self.omega) * DT
        self.theta = float(np.clip(self.theta + self.omega * DT,
                                   -MAX_PITCH, MAX_PITCH))

        ax = (thrust / MASS) * np.sin(self.theta) - DRAG * self.vel[0]
        az = (thrust / MASS) * np.cos(self.theta) - GRAVITY - DRAG * self.vel[1]

        self.vel = self.vel + np.array([ax, az], dtype=np.float32) * DT
        new_pos = self.pos + self.vel * DT * 10.0        

        if self._collides(new_pos):.
            self.vel *= -0.4
            self.crashed = True
        else:
            self.pos = new_pos.astype(np.float32)
            self.crashed = False

        return self.render()

    def render(self):
        world = np.zeros((SIZE, SIZE), dtype=np.float32)
        zz, xx = np.mgrid[0:SIZE, 0:SIZE]

        world[zz < GROUND_Z] = 0.35
        world[zz > CEILING_Z] = 0.35

        for (ox, oz), r in zip(self.obs_pos, self.obs_r):
            world[(xx - ox) ** 2 + (zz - oz) ** 2 <= r ** 2] = 0.5

        cx, cz = self.pos
        length = 4.0 if self.egocentric else 6.0
        dx, dz = np.cos(self.theta) * length, np.sin(self.theta) * length
        radius = 1.4 if self.egocentric else 2.0
        for t in np.linspace(-1, 1, 11):
            px, pz = cx + dx * t, cz + dz * t
            mask = (xx - px) ** 2 + (zz - pz) ** 2 <= radius ** 2
            world[mask] = 1.0

        if self.dual_view:
            step = SIZE // VIEW
            glob = world[::step, ::step][::-1]
            h = VIEW // 2
            ix, iz = int(round(cx)), int(round(cz))
            padded = np.pad(world, h, mode="constant", constant_values=0.35)
            ego = padded[iz: iz + 2 * h, ix: ix + 2 * h][::-1]
            return np.stack([glob, ego]).astype(np.float32)

        if not self.egocentric:
            step = SIZE // VIEW
            out = world[::step, ::step]
            return out[::-1][None].astype(np.float32)

        h = VIEW // 2
        ix, iz = int(round(cx)), int(round(cz))
        padded = np.pad(world, h, mode="constant", constant_values=0.35)
        crop = padded[iz: iz + 2 * h, ix: ix + 2 * h]
        return crop[::-1][None].astype(np.float32)

    @property
    def state(self) -> np.ndarray:
        return np.array([self.pos[0] / SIZE, self.pos[1] / SIZE,
                         self.vel[0] / 10.0, self.vel[1] / 10.0,
                         self.theta / MAX_PITCH, self.omega / MAX_PITCH_RATE],
                        dtype=np.float32)

    @property
    def global_pos(self) -> np.ndarray:
        return (self.pos / SIZE).astype(np.float32)


def hover_policy(env, rng, target_z=None, target_x=None, noise=0.3):
    if target_z is None:
        target_z = SIZE * 0.5
    if target_x is None:
        target_x = SIZE * 0.5

    x_err = target_x - env.pos[0]

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

    pitch_cmd = np.clip(3.0 * (theta_des - env.theta) - 0.8 * env.omega, -1, 1)

    a = np.array([thrust_cmd, pitch_cmd], dtype=np.float32)
    a += rng.normal(0, noise, 2).astype(np.float32)
    return np.clip(a, -1, 1)
