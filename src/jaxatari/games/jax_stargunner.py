from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import chex
from flax import struct

import jaxatari.spaces as spaces
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action, ObjectObservation

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.patheffects as pe
import numpy as np


class StarGunnerConstants(struct.PyTreeNode):
    WIDTH: int = struct.field(pytree_node=False, default=160)
    HEIGHT: int = struct.field(pytree_node=False, default=210)

    PLAYER_WIDTH: int = struct.field(pytree_node=False, default=10)
    PLAYER_HEIGHT: int = struct.field(pytree_node=False, default=6)
    PLAYER_SPEED: int = struct.field(pytree_node=False, default=3)

    PLAYER_START_X: int = struct.field(pytree_node=False, default=20)
    PLAYER_START_Y: int = struct.field(pytree_node=False, default=100)


class StarGunnerState(struct.PyTreeNode):
    player_x: chex.Array
    player_y: chex.Array
    step_counter: chex.Array
    key: chex.PRNGKey

    #  grass
    grass_offset: chex.Array

class StarGunnerObservation(struct.PyTreeNode):
    player: ObjectObservation


class StarGunnerInfo(struct.PyTreeNode):
    time: jnp.ndarray


def draw_rect(img, x, y, w, h, color):
    H, W, _ = img.shape

    yy = jnp.arange(H)[:, None]
    xx = jnp.arange(W)[None, :]

    mask = (
        (xx >= x) & (xx < x + w) &
        (yy >= y) & (yy < y + h)
    )

    return jnp.where(mask[:, :, None], color, img)

# -----------------------------
# ENVIRONMENT
# -----------------------------

class JaxStarGunner(JaxEnvironment[
    StarGunnerState,
    StarGunnerObservation,
    StarGunnerInfo,
    StarGunnerConstants
]):
    ACTION_SET = jnp.array([
        Action.NOOP,
        Action.UP,
        Action.DOWN,
        Action.LEFT,
        Action.RIGHT,
    ], dtype=jnp.int32)

    def __init__(self, consts: StarGunnerConstants = None):
        consts = consts or StarGunnerConstants()
        super().__init__(consts)
        self.renderer = True

        # -------------------------
        # STARS INIT
        # -------------------------
    def _init_stars(self, key, n=120):
        k1, k2, k3, k4 = jax.random.split(key, 4)

        stars_x = jax.random.randint(k1, (n,), 0, self.consts.WIDTH)
        stars_y = jax.random.randint(k2, (n,), 0, self.consts.HEIGHT)

        stars_speed = jax.random.uniform(k3, (n,), minval=0.3, maxval=1.8)
        stars_brightness = jax.random.uniform(k4, (n,), minval=0.4, maxval=1.0)

        return stars_x, stars_y, stars_speed, stars_brightness


    def reset(self, key: chex.PRNGKey = jax.random.PRNGKey(0)):

        stars_x, stars_y, stars_speed, stars_brightness = self._init_stars(key)

        state = StarGunnerState(
            player_x=jnp.array(self.consts.PLAYER_START_X, dtype=jnp.int32),
            player_y=jnp.array(self.consts.PLAYER_START_Y, dtype=jnp.int32),
            step_counter=jnp.array(0, dtype=jnp.int32),
            key=key,

            grass_offset=jnp.array(0.0)
        )

        return self._get_observation(state), state

    # -------------------------
    # PLAYER MOVE
    # -------------------------
    def _player_step(self, state: StarGunnerState, action: chex.Array):
        dx = jnp.where(action == Action.LEFT, -self.consts.PLAYER_SPEED, 0)
        dx = jnp.where(action == Action.RIGHT, self.consts.PLAYER_SPEED, dx)

        dy = jnp.where(action == Action.UP, -self.consts.PLAYER_SPEED, 0)
        dy = jnp.where(action == Action.DOWN, self.consts.PLAYER_SPEED, dy)

        new_x = jnp.clip(
            state.player_x + dx,
            0,
            self.consts.WIDTH - self.consts.PLAYER_WIDTH,
        )

        new_y = jnp.clip(
            state.player_y + dy,
            0,
            self.consts.HEIGHT - self.consts.PLAYER_HEIGHT,
        )

        return state.replace(
            player_x=new_x,
            player_y=new_y,
        )

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: StarGunnerState, action: chex.Array):
        atari_action = jnp.take(self.ACTION_SET, action.astype(jnp.int32))

        state = self._player_step(state, atari_action)

        #  move grass
        new_offset = (state.grass_offset + 1.2) % 24

        state = state.replace(
            grass_offset=new_offset,
            step_counter=state.step_counter + 1,
        )

        obs = self._get_observation(state)
        reward = jnp.array(0.0)
        done = jnp.array(False)
        info = self._get_info(state)

        return obs, state, reward, done, info

    # -------------------------
    # BACKGROUND
    # -------------------------
    def draw_background(self, img, state):
        H = self.consts.HEIGHT
        W = self.consts.WIDTH

        img = jnp.zeros((H, W, 3), dtype=jnp.uint8)

        yy = jnp.arange(H)[:, None]
        xx = jnp.arange(W)[None, :]

        # stabile Grundhöhe
        base_horizon = 165

        offset = state.grass_offset
        # ----------------------------------
        # Wellige Oberkante des Grases
        # ----------------------------------

        horizon = (
                base_horizon
                + 4 * jnp.sin((xx + offset * 2) * 0.05)
                + 1.5 * jnp.sin((xx + offset * 3) * 0.11)
        )

        grass_mask = yy >= horizon

        # ----------------------------------
        # Grasmuster
        # ----------------------------------

        y_rel = yy - horizon
        y_scroll = (y_rel + offset) % 24

        band_wave = 2 * jnp.sin(xx * 0.08)

        bright = y_scroll < (3 + band_wave)

        medium = (
                (y_scroll >= (3 + band_wave))
                & (y_scroll < (8 + band_wave))
        )

        dark_green = jnp.array([20, 90, 20], dtype=jnp.uint8)
        medium_green = jnp.array([40, 150, 40], dtype=jnp.uint8)
        bright_green = jnp.array([90, 220, 90], dtype=jnp.uint8)

        img = jnp.where(
            grass_mask[..., None],
            dark_green,
            img
        )

        img = jnp.where(
            (grass_mask & medium)[..., None],
            medium_green,
            img
        )

        img = jnp.where(
            (grass_mask & bright)[..., None],
            bright_green,
            img
        )

        return img

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: StarGunnerState):
        img = jnp.zeros(
            (self.consts.HEIGHT, self.consts.WIDTH, 3),
            dtype=jnp.uint8,
        )

        # Hintergrund zeichnen
        img = self.draw_background(img, state)

        player_color = jnp.array([0, 255, 0], dtype=jnp.uint8)

        img = draw_rect(
            img,
            state.player_x,
            state.player_y,
            self.consts.PLAYER_WIDTH,
            self.consts.PLAYER_HEIGHT,
            player_color,
        )

        return img

    def _get_observation(self, state: StarGunnerState):
        player = ObjectObservation.create(
            x=state.player_x,
            y=state.player_y,
            width=jnp.array(self.consts.PLAYER_WIDTH),
            height=jnp.array(self.consts.PLAYER_HEIGHT),
        )

        return StarGunnerObservation(player=player)

    def action_space(self):
        return spaces.Discrete(len(self.ACTION_SET))

    def observation_space(self):
        object_space = spaces.get_object_space(
            n=None,
            screen_size=(self.consts.HEIGHT, self.consts.WIDTH),
        )

        return spaces.Dict({
            "player": object_space,
        })

    def image_space(self):
        return spaces.Box(
            low=0,
            high=255,
            shape=(210, 160, 3),
            dtype=jnp.uint8,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state):
        return StarGunnerInfo(time=state.step_counter)


class StarGunnerRenderer:
    PIXEL_FONT = {
        '0': [0b01110, 0b10011, 0b10101, 0b11001, 0b01110, 0, 0],
        '1': [0b00100, 0b01100, 0b00100, 0b00100, 0b01110, 0, 0],
        '2': [0b01110, 0b10001, 0b00110, 0b01000, 0b11111, 0, 0],
        '3': [0b11110, 0b00001, 0b00110, 0b00001, 0b11110, 0, 0],
        '4': [0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0, 0],
        '5': [0b11111, 0b10000, 0b11110, 0b00001, 0b11110, 0, 0],
        '6': [0b00110, 0b01000, 0b11110, 0b10001, 0b01110, 0, 0],
        '7': [0b11111, 0b00001, 0b00010, 0b00100, 0b00100, 0, 0],
        '8': [0b01110, 0b10001, 0b01110, 0b10001, 0b01110, 0, 0],
        '9': [0b01110, 0b10001, 0b01111, 0b00001, 0b01110, 0, 0],
        'A': [0b01110, 0b10001, 0b11111, 0b10001, 0b10001, 0, 0],
        'B': [0b11110, 0b10001, 0b11110, 0b10010, 0b11110, 0, 0],
        'C': [0b01111, 0b10000, 0b10000, 0b10000, 0b01111, 0, 0],
        'D': [0b11110, 0b10001, 0b10001, 0b10001, 0b11110, 0, 0],
        'E': [0b11111, 0b10000, 0b11110, 0b10000, 0b11111, 0, 0],
        'F': [0b11111, 0b10000, 0b11110, 0b10000, 0b10000, 0, 0],
        'G': [0b01111, 0b10000, 0b10111, 0b10001, 0b01111, 0, 0],
        'H': [0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0, 0],
        'I': [0b01110, 0b00100, 0b00100, 0b00100, 0b01110, 0, 0],
        'J': [0b00111, 0b00010, 0b00010, 0b10010, 0b01100, 0, 0],
        'K': [0b10001, 0b10010, 0b11100, 0b10010, 0b10001, 0, 0],
        'L': [0b10000, 0b10000, 0b10000, 0b10000, 0b11111, 0, 0],
        'M': [0b10001, 0b11011, 0b10101, 0b10001, 0b10001, 0, 0],
        'N': [0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0, 0],
        'O': [0b01110, 0b10001, 0b10001, 0b10001, 0b01110, 0, 0],
        'P': [0b11110, 0b10001, 0b11110, 0b10000, 0b10000, 0, 0],
        'Q': [0b01110, 0b10001, 0b10101, 0b10010, 0b01101, 0, 0],
        'R': [0b11110, 0b10001, 0b11110, 0b10010, 0b10001, 0, 0],
        'S': [0b01111, 0b10000, 0b01110, 0b00001, 0b11110, 0, 0],
        'T': [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0, 0],
        'U': [0b10001, 0b10001, 0b10001, 0b10001, 0b01110, 0, 0],
        'V': [0b10001, 0b10001, 0b10001, 0b01010, 0b00100, 0, 0],
        'W': [0b10001, 0b10001, 0b10101, 0b11011, 0b10001, 0, 0],
        'X': [0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0, 0],
        'Y': [0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0, 0],
        'Z': [0b11111, 0b00010, 0b00100, 0b01000, 0b11111, 0, 0],
        ':': [0, 0b00100, 0, 0b00100, 0, 0, 0],
        ' ': [0, 0, 0, 0, 0, 0, 0],
        '-': [0, 0, 0b11111, 0, 0, 0, 0],
        '©': [0b01110, 0b10101, 0b10111, 0b10101, 0b01110, 0, 0],
        'U_G': [0b10001, 0b10001, 0b10001, 0b10011, 0b01111, 0, 0],
        'N_G': [0b00000, 0b11010, 0b10110, 0b10010, 0b10010, 0, 0],
        'E_G': [0b00000, 0b01110, 0b11000, 0b11110, 0b01111, 0, 0],
        'R_G': [0b00000, 0b10110, 0b11000, 0b10000, 0b10000, 0, 0],
    }

    def draw_pixel_text(self, ax, text, x, y, color, scale=1.0):
        cx = x
        for ch in text:
            glyph_key = ch.upper()
            if ch == 'u':
                glyph_key = 'U_G'
            elif ch == 'n':
                glyph_key = 'N_G'
            elif ch == 'e':
                glyph_key = 'E_G'
            elif ch == 'r':
                glyph_key = 'R_G'

            glyph = self.PIXEL_FONT.get(glyph_key, self.PIXEL_FONT[' '])
            for row_i, row_bits in enumerate(glyph):
                for col_i in range(5):
                    if row_bits & (1 << (4 - col_i)):
                        ax.add_patch(self.patches.Rectangle(
                            (cx + col_i * scale, y + row_i * scale),
                            scale, scale,
                            linewidth=0, facecolor=color, zorder=9))
            cx += 6 * scale

    def __init__(self, env):
        self.env = env
        self.plt = plt
        self.anim = animation
        self.patches = patches
        self.np = np

        self.game_ctx = {
            "screen": "start",
            "state": None,
            "frame": 0,
            "countdown": 3,
            "countdown_frame": 0,
            "high_score": 0,
            "score": 0,
        }

        # Parallax star layers
        np.random.seed(42)
        self.layers = []
        for speed, n, bri_range in [
            (0.3, 40, (0.2, 0.45)),
            (0.8, 50, (0.45, 0.70)),
            (1.6, 30, (0.70, 1.00)),
        ]:
            self.layers.append({
                "x": np.random.randint(0, 160, n).astype(float),
                "y": np.random.randint(0, 210, n).astype(float),
                "spd": np.full(n, speed) + np.random.uniform(-0.1, 0.1, n),
                "bri": np.random.uniform(*bri_range, n),
                "sz": np.where(np.random.uniform(size=n) > 0.85, 2.2, 1.0),
            })

        self.demo_ship = {"x": -20.0, "y": 60.0, "vy": 0.0, "target_y": 60.0}
        self.current_action = jnp.array(0)
        self.bolts = []

        np.random.seed(99)
        self.drones = []
        for i in range(5):
            self.drones.append({
                "x": 160 + i * 45,
                "y": 30 + i * 18,
                "vx": -(0.9 + i * 0.15),
                "alive": True,
                "explode": 0,
            })

        self.particles = []
        self._title1 = list("STAR")
        self._title2 = list("GUNNER")

        rng = np.random.default_rng(7)
        self._stars = []
        for spd, n, bri_lo, bri_hi in [(0.4, 55, 0.30, 0.60), (1.1, 35, 0.65, 1.00)]:
            self._stars.append({
                "x": rng.integers(0, 160, n).astype(float),
                "y": rng.uniform(0, 168, n),
                "spd": rng.uniform(spd * 0.8, spd * 1.2, n),
                "bri": rng.uniform(bri_lo, bri_hi, n),
                "big": rng.uniform(size=n) > 0.80,
            })
        self._ship = {"x": 12.0, "y": 90.0, "vy": 0.0, "target_y": 90.0}
        self._bolts = []
        self._enemies = []
        for i in range(4):
            self._enemies.append({
                "type": "saucer",
                "x": float(160 + 20 + i * 38),
                "y": float(rng.uniform(22, 154)),
                "vx": -(0.8 + i * 0.12), "vy": 0.0,
                "amp": rng.uniform(0.3, 0.8),
                "phase": rng.uniform(0, 6.28),
                "alive": True, "explode": 0,
            })
        for i in range(3):
            self._enemies.append({
                "type": "buzzie",
                "x": float(160 + 55 + i * 50),
                "y": float(rng.uniform(30, 150)),
                "vx": -(1.2 + i * 0.15),
                "vy": rng.choice([-0.4, 0.4]),
                "amp": 0.0, "phase": 0.0,
                "alive": True, "explode": 0,
            })
        self._sparks = []
        self._gnd_offset = 0.0
        self._demo_score = 0
        self._rng = rng

        self.action_map = {
            "up": jnp.array(1),
            "down": jnp.array(2),
            "left": jnp.array(3),
            "right": jnp.array(4),
        }

    def _draw_player_ship(self, ax, x, y):
        P = self.patches.Rectangle
        ax.add_patch(P((x, y + 1), 12, 4, linewidth=0, facecolor="#00CC00", zorder=5))
        ax.add_patch(P((x + 12, y + 2), 3, 2, linewidth=0, facecolor="#00CC00", zorder=5))
        ax.add_patch(P((x + 2, y - 2), 6, 2, linewidth=0, facecolor="#004400", zorder=5))
        ax.add_patch(P((x + 2, y + 5), 6, 3, linewidth=0, facecolor="#004400", zorder=5))
        ax.add_patch(P((x + 4, y + 1), 3, 2, linewidth=0, facecolor="#88FF88", zorder=6))
        ax.add_patch(P((x - 2, y + 2), 2, 2, linewidth=0, facecolor="#FF8800", zorder=5))

    def _draw_saucer(self, ax, x, y):
        P = self.patches.Rectangle
        ax.add_patch(P((x + 2, y), 6, 3, linewidth=0, facecolor="#FF4400", zorder=4))
        ax.add_patch(P((x, y + 3), 10, 3, linewidth=0, facecolor="#CC2200", zorder=4))
        ax.add_patch(P((x + 4, y + 1), 2, 2, linewidth=0, facecolor="#FFFF00", zorder=5))

    def _draw_buzzie(self, ax, x, y):
        ax.add_patch(self.patches.Polygon(
            [[x + 4, y], [x + 8, y + 4], [x + 4, y + 8], [x, y + 4]],
            closed=True, facecolor="#FF9900", edgecolor="#FFCC00",
            linewidth=0.5, zorder=4))
        ax.add_patch(self.patches.Rectangle(
            (x + 3, y + 3), 2, 2, linewidth=0, facecolor="#FFFF00", zorder=5))

    def _draw_hud(self, ax, score, hi_score, lives):
        ax.add_patch(self.patches.Rectangle(
            (0, 0), 160, 16, linewidth=0, facecolor="black", zorder=8))
        ax.add_patch(self.patches.Rectangle(
            (0, 16), 160, 1, linewidth=0, facecolor="#444444", zorder=8))
        self.draw_pixel_text(ax, f"{score:06d}", 2, 4, "#FF6600", scale=1)
        self.draw_pixel_text(ax, f"{hi_score:06d}", 58, 4, "#FF6600", scale=1)
        for i in range(lives):
            lx = 160 - 4 - i * 12
            ax.add_patch(self.patches.Rectangle(
                (lx, 5), 8, 3, linewidth=0, facecolor="#00CC00", zorder=9))
            ax.add_patch(self.patches.Rectangle(
                (lx + 2, 3), 3, 2, linewidth=0, facecolor="#006600", zorder=9))

    def _explode(self, x, y):
        for _ in range(18):
            angle = np.random.uniform(0, 2 * np.pi)
            speed = np.random.uniform(0.5, 2.5)
            self.particles.append({
                "x": x, "y": y,
                "vx": np.cos(angle) * speed,
                "vy": np.sin(angle) * speed,
                "life": np.random.randint(8, 20),
                "color": np.random.choice(["#FF6600", "#FFCC00", "#FF2200", "#FFFFFF"]),
            })

    def _setup_figure(self):
        fig, ax = self.plt.subplots(figsize=(4, 5.25))
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")
        ax.axis("off")
        self.plt.tight_layout(pad=0)
        self.fig = fig
        self.ax = ax

    _EGG_COLORS = ["#CCAA00", "#AA44CC", "#4499CC", "#CC4488"]
    _SHIP_COLORS = ["#CC8800", "#CCCC00", "#AA44CC", "#4499CC", "#CC4488"]

    def _draw_robot_ship(self, ax, x, y, color):
        P = self.patches.Rectangle
        ax.add_patch(P((x, y), 10, 7, linewidth=0, facecolor=color, zorder=5))
        ax.add_patch(P((x + 1, y - 2), 2, 2, linewidth=0, facecolor=color, zorder=5))
        ax.add_patch(P((x + 7, y - 2), 2, 2, linewidth=0, facecolor=color, zorder=5))
        ax.add_patch(P((x + 2, y + 1), 6, 4, linewidth=0, facecolor="black", zorder=6))
        ax.add_patch(P((x + 3, y + 2), 1, 1, linewidth=0, facecolor=color, zorder=7))
        ax.add_patch(P((x + 6, y + 2), 1, 1, linewidth=0, facecolor=color, zorder=7))
        ax.add_patch(P((x + 2, y + 7), 2, 4, linewidth=0, facecolor=color, zorder=5))
        ax.add_patch(P((x + 6, y + 7), 2, 4, linewidth=0, facecolor=color, zorder=5))

    def _draw_egg(self, ax, x, y, color):
        ax.add_patch(self.patches.Ellipse(
            (x + 4, y + 6), 8, 13, facecolor=color, linewidth=0, zorder=4))
        ax.add_patch(self.patches.Ellipse(
            (x + 3, y + 4), 3, 4, facecolor="white", alpha=0.15, linewidth=0, zorder=5))

    def _draw_wing(self, ax, x, y):
        ax.add_patch(self.patches.Polygon(
            [[x + 12, y + 1], [x + 6, y], [x, y + 3], [x + 6, y + 5], [x + 12, y + 3]],
            closed=True, facecolor="#CC5566", linewidth=0, zorder=4))
        ax.add_patch(self.patches.Polygon(
            [[x, y + 3], [x - 6, y + 5], [x - 4, y + 2]],
            closed=True, facecolor="#AA3344", linewidth=0, zorder=4))

    def _draw_start_screen(self):
        ax = self.ax
        f = self.game_ctx["frame"]
        W, H = 160, 210
        GY = 178

        ax.clear()
        ax.set_facecolor("black")
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.axis("off")

        # 1. Wavy green landscape floor
        self._gnd_offset = (self._gnd_offset + 0.5) % (2 * np.pi)
        xx = self.np.linspace(0, W, 300)
        wave = (GY
                + 8 * self.np.sin(xx * 0.040 + self._gnd_offset)
                + 3.5 * self.np.sin(xx * 0.088 + self._gnd_offset * 1.3)
                + 1.5 * self.np.sin(xx * 0.160 + self._gnd_offset * 0.8))
        wave_stepped = self.np.floor(wave / 2) * 2

        ax.fill_between(xx, wave_stepped, H, color=(0.09, 0.38, 0.09), zorder=2, step="mid")
        ax.fill_between(xx, wave_stepped, wave_stepped + 2, color=(0.30, 0.70, 0.15), zorder=3, step="mid")

        # 2. Score text header
        score = self.game_ctx.get("score", 0)
        score_str = f"{score:03d}"
        sx = W // 2 - 14
        for i, ch in enumerate(score_str):
            self.draw_pixel_text(ax, ch, sx + i * 16, 12, "#4477EE", scale=2)

        # 3. Static helper ships under score
        self._draw_wing(ax, W // 2 - 24, 25)
        self._draw_wing(ax, W // 2 - 4, 25)
        self._draw_wing(ax, W // 2 + 16, 25)

        # 4. Alternating Page states (Text Title vs Active Attract Mode)
        is_static_logo_page = (f // 220) % 2 == 1

        if is_static_logo_page:
            # TEXT DISPLAY STATE
            self.draw_pixel_text(ax, "STAR", W // 2 - 28, 80, "#D040A0", scale=2.5)
            self.draw_pixel_text(ax, "Gunner", W // 2 - 20, 105, "#40A090", scale=1.3)
            self.draw_pixel_text(ax, "©1982", W // 2 - 18, 122, "#CC6633", scale=1.1)
            self.draw_pixel_text(ax, "Telesys", W // 2 - 22, 137, "#CC6633", scale=1.3)
            self.draw_pixel_text(ax, "1", W // 2 - 3, 162, "#CCCC44", scale=1.5)
        else:
            # LIVE ATTRACT DEMO GAMEPLAY STATE
            wing_enemies = [e for e in self._enemies if e["type"] == "saucer"]
            for e in wing_enemies:
                e["x"] += e["vx"]
                if e["x"] < -20:
                    e["x"] = float(W + self._rng.integers(5, 40))
                    e["y"] = float(self._rng.uniform(35, 65))
                self._draw_wing(ax, e["x"], e["y"])

            egg_enemies = [e for e in self._enemies if e["type"] == "buzzie"]
            egg_col = self._EGG_COLORS[(f // 60) % len(self._EGG_COLORS)]
            for e in egg_enemies:
                e["y"] += e["vy"]
                if e["y"] < 60:      e["vy"] = abs(e["vy"]) + 0.3
                if e["y"] > GY - 30: e["vy"] = -(abs(e["vy"]) + 0.3)
                e["x"] = float(self.np.clip(e["x"], 18, 50))
                self._draw_egg(ax, e["x"], e["y"], egg_col)

            ship = self._ship
            ship_col = self._SHIP_COLORS[(f // 60) % len(self._SHIP_COLORS)]
            if f % 70 == 0:
                ship["target_y"] = float(self._rng.uniform(55, GY - 35))
            ship["vy"] += (ship["target_y"] - ship["y"]) * 0.015
            ship["vy"] *= 0.90
            ship["y"] += ship["vy"]
            ship["x"] = 22.0
            self._draw_robot_ship(ax, ship["x"], ship["y"], ship_col)

            if f % 18 == 0:
                self._bolts.append({"x": float(ship["x"] + 11), "y": float(ship["y"] + 3), "alive": True, "col": "#88FF00"})
            for b in self._bolts:
                b["x"] += 5.5
                b["alive"] = b["x"] < W
                if b["alive"]:
                    ax.add_patch(self.patches.Rectangle((b["x"], b["y"]), 8, 2, linewidth=0, facecolor=b["col"], zorder=6))
            self._bolts = [b for b in self._bolts if b["alive"]]

        self.game_ctx["frame"] += 1

    def _draw_countdown(self):
        ax = self.ax
        cf = self.game_ctx["countdown_frame"]

        ax.clear()
        ax.set_facecolor("black")
        ax.set_xlim(0, 160)
        ax.set_ylim(210, 0)
        ax.axis("off")

        for layer in self.layers:
            layer["y"] = (layer["y"] + layer["spd"]) % 210
            for i in range(len(layer["x"])):
                b = layer["bri"][i]
                ax.plot(layer["x"][i], layer["y"][i], "o",
                        color=(b, b, b), markersize=layer["sz"][i], zorder=1)

        total = 4 * 30
        beat = cf // 30
        phase = (cf % 30) / 30

        labels = ["3", "2", "1", "GO!"]
        colors = ["#FF4444", "#FF8800", "#FFFF00", "#00FF88"]

        if beat < 4:
            label = labels[beat]
            color = colors[beat]
            scale = 1.0 + 0.4 * (1 - phase)
            alpha = 1.0 - 0.3 * phase
            fontsize = int(50 * scale)
            ax.text(80, 105, label,
                    fontsize=fontsize, fontweight="bold",
                    color=color, alpha=alpha,
                    ha="center", va="center",
                    fontfamily="monospace", zorder=8)
            ring_r = 20 + 50 * phase
            ring_alpha = 0.6 * (1 - phase)
            ring = self.patches.Circle((80, 105), ring_r,
                                       facecolor="none", edgecolor=color,
                                       linewidth=1.5, alpha=ring_alpha, zorder=7)
            ax.add_patch(ring)

        self.game_ctx["countdown_frame"] += 1
        if cf >= total:
            self.game_ctx["screen"] = "playing"

    def _draw_game_screen(self):
        ax = self.ax
        frame = self.np.array(self.env.render(self.game_ctx["state"]))
        ax.clear()
        ax.axis("off")
        ax.imshow(frame, origin="upper", aspect="auto", extent=[0, 160, 210, 0])
        ax.set_xlim(0, 160)
        ax.set_ylim(210, 0)

    def _update(self, _frame):
        screen = self.game_ctx["screen"]
        if screen == "start":
            self._draw_start_screen()
        elif screen == "countdown":
            self._draw_countdown()
        elif screen == "playing":
            _, new_state, _, _, _ = self.env.step(
                self.game_ctx["state"],
                self.current_action
            )
            self.game_ctx["state"] = new_state
            self._draw_game_screen()
        return []

    def _on_key(self, event):
        screen = self.game_ctx["screen"]

        if screen == "start":
            if event.key == "enter":
                k = jax.random.PRNGKey(self.game_ctx["frame"])
                _, fresh_state = self.env.reset(k)
                self.game_ctx["state"] = fresh_state
                self.game_ctx["countdown_frame"] = 0
                self.game_ctx["screen"] = "countdown"
            return

        if screen == "countdown":
            return

        if screen == "playing":
            if event.key == "escape":
                self.game_ctx["screen"] = "start"
                return
            self.current_action = self.action_map.get(
                event.key,
                jnp.array(0)
            )

    def _on_key_release(self, event):
        if self.game_ctx["screen"] == "playing":
            self.current_action = jnp.array(0)  # NOOP

    # RUN
    def run(self):
        self._setup_figure()
        self.fig.canvas.mpl_connect(
            "key_press_event",
            self._on_key
        )

        self.fig.canvas.mpl_connect(
            "key_release_event",
            self._on_key_release
        )

        self._ani = self.anim.FuncAnimation(
            self.fig,
            self._update,
            interval=33,
            blit=False,
            cache_frame_data=False
        )
        self.plt.show()


# MAIN

if __name__ == "__main__":
    env = JaxStarGunner()
    renderer = StarGunnerRenderer(env)
    renderer.run()
