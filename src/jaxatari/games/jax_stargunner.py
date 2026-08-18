from functools import partial
import jax
import jax.lax as lax
import jax.numpy as jnp
import chex
import numpy as np
from flax import struct

import jaxatari.spaces as spaces
from jaxatari.environment import (
    JaxEnvironment,
    JAXAtariAction as Action,
    ObjectObservation,
)

# Screen modes and enemy types

ATTRACT = 0
PLAY = 1
GAME_OVER = 2

SAUCER = 0
BUZZIE = 1
SQUEEZER = 2

# Enemy properties (must match extracted sprites)

ENEMY_POINTS = jnp.array([50, 100, 150], dtype=jnp.int32)
ENEMY_W = jnp.array([8, 8, 8], dtype=jnp.float32)  # width of each enemy sprite
ENEMY_H = jnp.array([4, 4, 4], dtype=jnp.float32)  # height of each enemy sprite
ENEMY_COLOR = jnp.array([[255, 90, 0], [255, 180, 40], [200, 60, 200]], jnp.uint8)

# SPRITE DEFINITIONS – placeholder until we extract exact data from ALE
# To get perfect sprites, run capture_sprites_from_ale()
# and replace the arrays below with the cropped binary data.

PLAYER_SPRITE = jnp.array(
    [
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 0, 0],
    ],
    dtype=jnp.bool_,
)
PLAYER_SPRITE_LEFT = jnp.flip(PLAYER_SPRITE, axis=1)

SAUCER_SPRITE = jnp.array(
    [
        [0, 0, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 0, 1, 1, 0, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 0],
    ],
    dtype=jnp.bool_,
)

BUZZIE_SPRITE = jnp.array(
    [
        [0, 1, 1, 0, 0, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 0, 1, 1, 0, 1, 0],
    ],
    dtype=jnp.bool_,
)

SQUEEZER_SPRITE = jnp.array(
    [
        [0, 1, 0, 0, 0, 0, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 0, 1, 1, 0, 1, 0],
    ],
    dtype=jnp.bool_,
)

BOBO_SPRITE = jnp.array(
    [
        [0, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 0, 1, 1, 0, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 0, 1, 1, 0, 1, 0],
    ],
    dtype=jnp.bool_,
)


# Constants


class StarGunnerConstants(struct.PyTreeNode):
    WIDTH: int = struct.field(pytree_node=False, default=160)
    HEIGHT: int = struct.field(pytree_node=False, default=210)
    DIFFICULTY: int = struct.field(pytree_node=False, default=0)

    FRAMESKIP: int = struct.field(pytree_node=False, default=4)
    STICKY_ACTION_PROB: float = struct.field(
        pytree_node=False,
        default=0.25,
    )

    PLAY_TOP: int = struct.field(pytree_node=False, default=24)
    PLAY_BOTTOM: int = struct.field(pytree_node=False, default=170)
    HILL_Y: int = struct.field(pytree_node=False, default=193)

    PLAYER_WIDTH: int = struct.field(pytree_node=False, default=8)
    PLAYER_HEIGHT: int = struct.field(pytree_node=False, default=6)
    PLAYER_SPEED: int = struct.field(pytree_node=False, default=2)
    PLAYER_START_X: int = struct.field(pytree_node=False, default=20)
    PLAYER_START_Y: int = struct.field(pytree_node=False, default=100)
    PLAYER_LIVES_START: int = struct.field(pytree_node=False, default=3)
    INVULN_FRAMES: int = struct.field(pytree_node=False, default=60)

    BULLET_WIDTH: int = struct.field(pytree_node=False, default=4)
    BULLET_HEIGHT: int = struct.field(pytree_node=False, default=2)
    BULLET_SPEED: float = struct.field(pytree_node=False, default=5.0)
    MAX_BULLETS: int = struct.field(pytree_node=False, default=2)
    FIRE_COOLDOWN: int = struct.field(pytree_node=False, default=8)

    NUM_ENEMIES: int = struct.field(pytree_node=False, default=6)
    ENEMY_SPEED: float = struct.field(pytree_node=False, default=1.0)
    ENEMY_SPEED_INCREMENT: float = struct.field(pytree_node=False, default=0.10)
    MAX_ENEMY_SPEED_MULTIPLIER: float = struct.field(pytree_node=False, default=2.0)
    ENEMY_AMP: float = struct.field(pytree_node=False, default=8.0)
    WAVE_BONUS: int = struct.field(pytree_node=False, default=1000)
    # Time between spawning a new wave and starting movement
    WAVE_START_DELAY: int = struct.field(pytree_node=False, default=30)

    ENEMY_SPAWN_Y: int = struct.field(pytree_node=False, default=60)
    ENEMY_SPAWN_SPACING: int = struct.field(pytree_node=False, default=20)
    HIT_TOP_MARGIN: int = struct.field(pytree_node=False, default=0)

    BOBO_WIDTH: int = struct.field(pytree_node=False, default=8)
    BOBO_HEIGHT: int = struct.field(pytree_node=False, default=4)
    BOBO_Y: int = struct.field(pytree_node=False, default=10)
    BOBO_SPEED: float = struct.field(pytree_node=False, default=3.0)
    BOBO_BOMB_PERIOD: int = struct.field(pytree_node=False, default=45)

    MAX_BOMBS: int = struct.field(pytree_node=False, default=4)
    BOMB_WIDTH: int = struct.field(pytree_node=False, default=2)
    BOMB_HEIGHT: int = struct.field(pytree_node=False, default=4)
    BOMB_SPEED: float = struct.field(pytree_node=False, default=2.0)

    HILL_SCROLL_SPEED: float = struct.field(pytree_node=False, default=0.85)

    EXPLOSION_SIZE: int = struct.field(pytree_node=False, default=10)
    EXPLOSION_DURATION: int = struct.field(pytree_node=False, default=10)

    DEATH_PENALTY: float = struct.field(pytree_node=False, default=0.0)
    PLAYER_EXPLOSION_SIZE: int = struct.field(pytree_node=False, default=14)
    PLAYER_EXPLOSION_DURATION: int = struct.field(pytree_node=False, default=16)


# State structure


class StarGunnerState(struct.PyTreeNode):
    mode: chex.Array
    player_x: chex.Array
    player_y: chex.Array
    player_facing: chex.Array
    prev_player_x: chex.Array
    prev_player_y: chex.Array
    step_counter: chex.Array

    # ALE-style action handling
    previous_action: chex.Array
    key: chex.PRNGKey

    bullet_x: chex.Array
    bullet_y: chex.Array
    bullet_dir: chex.Array
    bullet_vx: chex.Array
    bullet_vy: chex.Array
    bullet_active: chex.Array
    fire_cooldown: chex.Array

    enemy_x: chex.Array
    enemy_y: chex.Array
    enemy_type: chex.Array
    enemy_vy: chex.Array
    enemy_phase: chex.Array
    enemy_alive: chex.Array

    bobo_x: chex.Array
    bobo_vx: chex.Array
    bomb_x: chex.Array
    bomb_y: chex.Array
    bomb_active: chex.Array

    explosion_x: chex.Array
    explosion_y: chex.Array
    explosion_timer: chex.Array
    explosion_active: chex.Array

    player_explosion_x: chex.Array
    player_explosion_y: chex.Array
    player_explosion_timer: chex.Array
    player_explosion_active: chex.Array

    score: chex.Array
    lives: chex.Array
    wave: chex.Array
    wave_timer: chex.Array
    invuln_timer: chex.Array


class StarGunnerObservation(struct.PyTreeNode):
    player: ObjectObservation
    enemies: ObjectObservation
    bullets: ObjectObservation
    bombs: ObjectObservation
    bobo: ObjectObservation


class StarGunnerInfo(struct.PyTreeNode):
    time: jnp.ndarray
    lives: jnp.ndarray
    wave: jnp.ndarray


# Raster drawing helpers


def draw_rect(img, x, y, w, h, color):
    H, W, _ = img.shape
    yy = jnp.arange(H)[:, None]
    xx = jnp.arange(W)[None, :]
    mask = (xx >= x) & (xx < x + w) & (yy >= y) & (yy < y + h)
    return jnp.where(mask[:, :, None], color, img)


def draw_sprite(img, x, y, sprite, color):
    H, W = sprite.shape
    x = jnp.asarray(x).astype(jnp.int32)
    y = jnp.asarray(y).astype(jnp.int32)
    yy = jnp.arange(H)[:, None]
    xx = jnp.arange(W)[None, :]
    img_h, img_w, _ = img.shape
    y_pos = y + yy
    x_pos = x + xx
    valid_y = (y_pos >= 0) & (y_pos < img_h)
    valid_x = (x_pos >= 0) & (x_pos < img_w)
    valid = valid_y & valid_x & sprite
    valid_expanded = valid[:, :, None]
    color_expanded = color[None, None, :]
    img_region = img[y_pos, x_pos]
    masked_region = jnp.where(valid_expanded, color_expanded, img_region)
    img = img.at[y_pos, x_pos].set(masked_region)
    return img


def _aabb_overlap(ax, ay, aw, ah, bx, by, bw, bh):
    return (ax < bx + bw) & (ax + aw > bx) & (ay < by + bh) & (ay + ah > by)


# 5x7 pixel font (used for HUD and attract screen)

_FONT = {
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
    "6": ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00010", "01100"],
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01110", "10001", "10000", "10111", "10001", "10001", "01111"],
    "I": ["01110", "00100", "00100", "00100", "00100", "00100", "01110"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
    "(": ["01110", "10001", "10110", "10100", "10110", "10001", "01110"],
}


def _glyph_np(ch):
    rows = _FONT.get(ch, _FONT[" "])
    return np.array([[1 if b == "1" else 0 for b in r] for r in rows], np.uint8)


def _bake_text(canvas, text, x, y, scale):
    cx = x
    for ch in text:
        g = _glyph_np(ch)
        ys, xs = np.where(g == 1)
        for gy, gx in zip(ys, xs):
            canvas[
                y + gy * scale : y + gy * scale + scale,
                cx + gx * scale : cx + gx * scale + scale,
            ] = True
        cx += 6 * scale
    return canvas


def _centered_x(text, scale, width):
    return (width - (len(text) * 6 * scale - scale)) // 2


# Main Environment class


class JaxStarGunner(
    JaxEnvironment[
        StarGunnerState, StarGunnerObservation, StarGunnerInfo, StarGunnerConstants
    ]
):
    ACTION_SET = jnp.array(
        [
            Action.NOOP,  # 0
            Action.FIRE,  # 1
            Action.UP,  # 2
            Action.RIGHT,  # 3
            Action.LEFT,  # 4
            Action.DOWN,  # 5
            Action.UPRIGHT,  # 6
            Action.UPLEFT,  # 7
            Action.DOWNRIGHT,  # 8
            Action.DOWNLEFT,  # 9
            Action.UPFIRE,  # 10
            Action.RIGHTFIRE,  # 11
            Action.LEFTFIRE,  # 12
            Action.DOWNFIRE,  # 13
            Action.UPRIGHTFIRE,  # 14
            Action.UPLEFTFIRE,  # 15
            Action.DOWNRIGHTFIRE,  # 16
            Action.DOWNLEFTFIRE,  # 17
        ],
        dtype=jnp.int32,
    )

    def __init__(self, consts: StarGunnerConstants = None, start_in_play: bool = False):
        consts = consts or StarGunnerConstants()
        super().__init__(consts)
        self.start_in_play = start_in_play
        self.renderer = StarGunnerRenderer(consts)

    def _is_fire(self, a):
        return (
            (a == Action.FIRE)
            | (a == Action.UPFIRE)
            | (a == Action.RIGHTFIRE)
            | (a == Action.LEFTFIRE)
            | (a == Action.DOWNFIRE)
            | (a == Action.UPRIGHTFIRE)
            | (a == Action.UPLEFTFIRE)
            | (a == Action.DOWNRIGHTFIRE)
            | (a == Action.DOWNLEFTFIRE)
        )

    def _spawn_wave(self, key, wave):
        """
        Create a deterministic enemy formation.

        Enemy positions are based on the wave number instead of
        being completely random.
        """

        n = self.consts.NUM_ENEMIES
        idx = jnp.arange(n)

        # ---------------------------------------------------------
        # Horizontal spawn positions
        # ---------------------------------------------------------

        spacing = self.consts.WIDTH / (n + 1)

        x = (idx.astype(jnp.float32) + 1.0) * spacing

        # ---------------------------------------------------------
        # Vertical formation
        # ---------------------------------------------------------

        row = idx % 3

        y = (
            self.consts.ENEMY_SPAWN_Y
            + row.astype(jnp.float32) * self.consts.ENEMY_SPAWN_SPACING
        )

        # Keep enemies inside play area
        y = jnp.clip(
            y,
            self.consts.PLAY_TOP + 4,
            self.consts.PLAY_BOTTOM - 12,
        )

        # ---------------------------------------------------------
        # Enemy type
        # ---------------------------------------------------------
        type_pattern = jnp.array(
            [
                SAUCER,
                BUZZIE,
                SQUEEZER,
                SAUCER,
                BUZZIE,
                SQUEEZER,
            ],
            dtype=jnp.int32,
        )

        type_index = jnp.mod(
            idx + wave - 1,
            3,
        )

        enemy_type = type_index.astype(jnp.int32)

        # ---------------------------------------------------------
        # Vertical movement
        # ---------------------------------------------------------

        direction = jnp.where(
            idx % 2 == 0,
            1.0,
            -1.0,
        )

        enemy_vy = jnp.where(
            enemy_type == BUZZIE,
            direction * 0.6,
            0.0,
        )

        # ---------------------------------------------------------
        # Animation phase
        # ---------------------------------------------------------

        phase = idx.astype(jnp.float32) * (2.0 * jnp.pi / n)

        return (
            x,
            y,
            enemy_type,
            enemy_vy,
            phase,
        )

    def _fresh_game_fields(self, key):
        n = self.consts.NUM_ENEMIES
        key, wkey = jax.random.split(key)
        ex, ey, et, evy, eph = self._spawn_wave(wkey, jnp.array(1, jnp.int32))
        start_x = jnp.array(self.consts.PLAYER_START_X, jnp.float32)
        start_y = jnp.array(self.consts.PLAYER_START_Y, jnp.float32)
        return dict(
            key=key,
            player_x=start_x,
            player_y=start_y,
            player_facing=jnp.array(1, jnp.int32),
            prev_player_x=start_x,
            prev_player_y=start_y,
            previous_action=jnp.array(0, jnp.int32),
            bullet_x=jnp.zeros((self.consts.MAX_BULLETS,), jnp.float32),
            bullet_y=jnp.zeros((self.consts.MAX_BULLETS,), jnp.float32),
            bullet_dir=jnp.ones((self.consts.MAX_BULLETS,), jnp.float32),
            bullet_vx=jnp.zeros((self.consts.MAX_BULLETS,), jnp.float32),
            bullet_vy=jnp.zeros((self.consts.MAX_BULLETS,), jnp.float32),
            bullet_active=jnp.zeros((self.consts.MAX_BULLETS,), jnp.bool_),
            fire_cooldown=jnp.array(0, jnp.int32),
            enemy_x=ex,
            enemy_y=ey,
            enemy_type=et,
            enemy_vy=evy,
            enemy_phase=eph,
            enemy_alive=jnp.ones((n,), jnp.bool_),
            bobo_x=jnp.array(self.consts.WIDTH / 2, jnp.float32),
            bobo_vx=jnp.array(self.consts.BOBO_SPEED, jnp.float32),
            bomb_x=jnp.zeros((self.consts.MAX_BOMBS,), jnp.float32),
            bomb_y=jnp.zeros((self.consts.MAX_BOMBS,), jnp.float32),
            bomb_active=jnp.zeros((self.consts.MAX_BOMBS,), jnp.bool_),
            explosion_x=jnp.zeros((n,), jnp.float32),
            explosion_y=jnp.zeros((n,), jnp.float32),
            explosion_timer=jnp.zeros((n,), jnp.int32),
            explosion_active=jnp.zeros((n,), jnp.bool_),
            player_explosion_x=jnp.array(0.0, jnp.float32),
            player_explosion_y=jnp.array(0.0, jnp.float32),
            player_explosion_timer=jnp.array(0, jnp.int32),
            player_explosion_active=jnp.array(False, jnp.bool_),
            score=jnp.array(0, jnp.int32),
            lives=jnp.array(self.consts.PLAYER_LIVES_START, jnp.int32),
            wave=jnp.array(1, jnp.int32),
            wave_timer=jnp.array(0, jnp.int32),
            invuln_timer=jnp.array(0, jnp.int32),
        )

    def _apply_sticky_action(self, state, action):
        """
        ALE-style sticky action.

        With probability 25%, the previous action is repeated.
        Otherwise the newly selected action is used.
        """

        key, sticky_key = jax.random.split(state.key)

        repeat_previous = (
            jax.random.uniform(sticky_key) < self.consts.STICKY_ACTION_PROB
        )

        effective_action = jnp.where(
            repeat_previous,
            state.previous_action,
            action,
        )

        state = state.replace(
            key=key,
            previous_action=effective_action,
        )

        return state, effective_action

    def reset(self, key: chex.PRNGKey = jax.random.PRNGKey(0)):
        fields = self._fresh_game_fields(key)
        mode = jnp.array(PLAY if self.start_in_play else ATTRACT, jnp.int32)
        state = StarGunnerState(
            mode=mode, step_counter=jnp.array(0, jnp.int32), **fields
        )
        return self._get_observation(state), state

    # Game logic steps

    def _player_step(self, state, a):
        # ---------------------------------------------------------
        # Direction detection
        # ---------------------------------------------------------

        left = (
            (a == Action.LEFT)
            | (a == Action.LEFTFIRE)
            | (a == Action.UPLEFT)
            | (a == Action.DOWNLEFT)
            | (a == Action.UPLEFTFIRE)
            | (a == Action.DOWNLEFTFIRE)
        )

        right = (
            (a == Action.RIGHT)
            | (a == Action.RIGHTFIRE)
            | (a == Action.UPRIGHT)
            | (a == Action.DOWNRIGHT)
            | (a == Action.UPRIGHTFIRE)
            | (a == Action.DOWNRIGHTFIRE)
        )

        up = (
            (a == Action.UP)
            | (a == Action.UPFIRE)
            | (a == Action.UPRIGHT)
            | (a == Action.UPLEFT)
            | (a == Action.UPRIGHTFIRE)
            | (a == Action.UPLEFTFIRE)
        )

        down = (
            (a == Action.DOWN)
            | (a == Action.DOWNFIRE)
            | (a == Action.DOWNRIGHT)
            | (a == Action.DOWNLEFT)
            | (a == Action.DOWNRIGHTFIRE)
            | (a == Action.DOWNLEFTFIRE)
        )

        # ---------------------------------------------------------
        # Convert direction into movement
        # ---------------------------------------------------------

        mx = right.astype(jnp.float32) - left.astype(jnp.float32)
        my = down.astype(jnp.float32) - up.astype(jnp.float32)

        # ---------------------------------------------------------
        # Save previous position
        # ---------------------------------------------------------

        prev_x = state.player_x
        prev_y = state.player_y

        # ---------------------------------------------------------
        # Player movement
        # ---------------------------------------------------------

        nx = state.player_x + mx * self.consts.PLAYER_SPEED
        ny = state.player_y + my * self.consts.PLAYER_SPEED

        # ---------------------------------------------------------
        # Horizontal wraparound
        # ---------------------------------------------------------

        nx = jnp.mod(nx, self.consts.WIDTH)

        # ---------------------------------------------------------
        # Vertical boundaries
        # ---------------------------------------------------------

        ny = jnp.clip(
            ny,
            float(self.consts.PLAY_TOP),
            float(self.consts.PLAY_BOTTOM - self.consts.PLAYER_HEIGHT),
        )

        # ---------------------------------------------------------
        # Facing direction
        # ---------------------------------------------------------

        facing = jnp.where(
            right,
            jnp.array(1, dtype=jnp.int32),
            jnp.where(
                left,
                jnp.array(-1, dtype=jnp.int32),
                state.player_facing,
            ),
        )

        return state.replace(
            player_x=nx,
            player_y=ny,
            player_facing=facing,
            prev_player_x=prev_x,
            prev_player_y=prev_y,
        )

    def _bullet_step(self, state, a):
        """
        Update player bullets.

        The firing direction follows the directional fire action.
        """

        fire = self._is_fire(a)

        # ---------------------------------------------------------
        # Determine firing direction
        # ---------------------------------------------------------

        normal_direction = jnp.where(
            state.player_facing > 0,
            1.0,
            -1.0,
        )

        # Horizontal direction
        vx_dir = jnp.where(
            (a == Action.LEFTFIRE)
            | (a == Action.UPLEFTFIRE)
            | (a == Action.DOWNLEFTFIRE),
            -1.0,
            jnp.where(
                (a == Action.RIGHTFIRE)
                | (a == Action.UPRIGHTFIRE)
                | (a == Action.DOWNRIGHTFIRE),
                1.0,
                jnp.where(
                    a == Action.FIRE,
                    normal_direction,
                    0.0,
                ),
            ),
        )

        # Vertical direction
        vy_dir = jnp.where(
            (a == Action.UPFIRE)
            | (a == Action.UPLEFTFIRE)
            | (a == Action.UPRIGHTFIRE),
            -1.0,
            jnp.where(
                (a == Action.DOWNFIRE)
                | (a == Action.DOWNLEFTFIRE)
                | (a == Action.DOWNRIGHTFIRE),
                1.0,
                0.0,
            ),
        )

        diag = (vx_dir != 0.0) & (vy_dir != 0.0)

        speed_factor = jnp.where(
            diag,
            1.0 / jnp.sqrt(2.0),
            1.0,
        )

        vx = vx_dir * self.consts.BULLET_SPEED * speed_factor
        vy = vy_dir * self.consts.BULLET_SPEED * speed_factor

        # ---------------------------------------------------------
        # Find free bullet
        # ---------------------------------------------------------

        free = jnp.argmax(~state.bullet_active)

        ready = state.fire_cooldown <= 0

        can_fire = fire & ready & (~jnp.all(state.bullet_active))

        # ---------------------------------------------------------
        # Spawn position
        # ---------------------------------------------------------

        spawn_x = (
            state.player_x + self.consts.PLAYER_WIDTH / 2 - self.consts.BULLET_WIDTH / 2
        )

        spawn_y = (
            state.player_y
            + self.consts.PLAYER_HEIGHT / 2
            - self.consts.BULLET_HEIGHT / 2
        )

        # ---------------------------------------------------------
        # Insert bullet
        # ---------------------------------------------------------

        bx = state.bullet_x.at[free].set(spawn_x)

        by = state.bullet_y.at[free].set(spawn_y)

        bvx = state.bullet_vx.at[free].set(vx)

        bvy = state.bullet_vy.at[free].set(vy)

        bd = state.bullet_dir.at[free].set(
            jnp.where(
                vx >= 0,
                1.0,
                -1.0,
            )
        )

        ba = state.bullet_active.at[free].set(True)

        bullet_x = jnp.where(
            can_fire,
            bx,
            state.bullet_x,
        )

        bullet_y = jnp.where(
            can_fire,
            by,
            state.bullet_y,
        )

        bullet_vx = jnp.where(
            can_fire,
            bvx,
            state.bullet_vx,
        )

        bullet_vy = jnp.where(
            can_fire,
            bvy,
            state.bullet_vy,
        )

        bullet_dir = jnp.where(
            can_fire,
            bd,
            state.bullet_dir,
        )

        bullet_active = jnp.where(
            can_fire,
            ba,
            state.bullet_active,
        )

        # ---------------------------------------------------------
        # Move bullets
        # ---------------------------------------------------------

        bullet_x = bullet_x + bullet_vx

        bullet_y = bullet_y + bullet_vy

        # ---------------------------------------------------------
        # Remove bullets outside screen
        # ---------------------------------------------------------

        bullet_active = (
            bullet_active
            & (bullet_x >= 0)
            & (bullet_x < self.consts.WIDTH)
            & (bullet_y >= 0)
            & (bullet_y < self.consts.HEIGHT)
        )

        # ---------------------------------------------------------
        # Cooldown
        # ---------------------------------------------------------

        cooldown = jnp.where(
            can_fire,
            self.consts.FIRE_COOLDOWN,
            jnp.maximum(
                state.fire_cooldown - 1,
                0,
            ),
        )

        return state.replace(
            bullet_x=bullet_x,
            bullet_y=bullet_y,
            bullet_dir=bullet_dir,
            bullet_vx=bullet_vx,
            bullet_vy=bullet_vy,
            bullet_active=bullet_active,
            fire_cooldown=cooldown,
        )

    def _enemy_display_y(self, state):
        bob = self.consts.ENEMY_AMP * jnp.sin(
            state.enemy_phase + state.step_counter.astype(jnp.float32) * 0.1
        )
        return state.enemy_y + jnp.where(state.enemy_type == SAUCER, bob, 0.0)

    def _enemy_step(self, state):
        wave_multiplier = 1.0 + (
            self.consts.ENEMY_SPEED_INCREMENT * (state.wave - 1).astype(jnp.float32)
        )

        wave_multiplier = jnp.minimum(
            wave_multiplier,
            self.consts.MAX_ENEMY_SPEED_MULTIPLIER,
        )

        speed = self.consts.ENEMY_SPEED * wave_multiplier

        nx = (state.enemy_x - speed) % self.consts.WIDTH

        ny = state.enemy_y + state.enemy_vy

        lo = float(self.consts.PLAY_TOP + 4)

        hi = float(self.consts.PLAY_BOTTOM - 12)

        bounce = (ny < lo) | (ny > hi)

        nvy = jnp.where(
            bounce,
            -state.enemy_vy,
            state.enemy_vy,
        )

        return state.replace(
            enemy_x=nx,
            enemy_y=jnp.clip(
                ny,
                lo,
                hi,
            ),
            enemy_vy=nvy,
        )

    def _bobo_step(self, state):
        nx = state.bobo_x + state.bobo_vx
        flip = (nx < 4) | (nx > self.consts.WIDTH - self.consts.BOBO_WIDTH - 4)
        nvx = jnp.where(flip, -state.bobo_vx, state.bobo_vx)
        return state.replace(
            bobo_x=jnp.clip(nx, 4, self.consts.WIDTH - self.consts.BOBO_WIDTH - 4),
            bobo_vx=nvx,
        )

    def _bomb_step(self, state):
        # ---------------------------------------------------------
        # Bomb spawning
        # ---------------------------------------------------------

        drop = (state.step_counter > 0) & (
            state.step_counter % self.consts.BOBO_BOMB_PERIOD == 0
        )

        free = jnp.argmax(~state.bomb_active)

        can_drop = drop & (~jnp.all(state.bomb_active))

        # ---------------------------------------------------------
        # Spawn bomb below Bobo
        # ---------------------------------------------------------

        spawn_x = state.bobo_x + self.consts.BOBO_WIDTH / 2 - self.consts.BOMB_WIDTH / 2

        spawn_y = self.consts.BOBO_Y + self.consts.BOBO_HEIGHT

        bx = state.bomb_x.at[free].set(spawn_x)

        by = state.bomb_y.at[free].set(spawn_y)

        ba = state.bomb_active.at[free].set(True)

        bomb_x = jnp.where(
            can_drop,
            bx,
            state.bomb_x,
        )

        bomb_y = jnp.where(
            can_drop,
            by,
            state.bomb_y,
        )

        bomb_active = jnp.where(
            can_drop,
            ba,
            state.bomb_active,
        )

        # ---------------------------------------------------------
        # Move bombs
        # ---------------------------------------------------------

        bomb_y = bomb_y + self.consts.BOMB_SPEED

        # ---------------------------------------------------------
        # Bomb disappears when reaching ground
        # ---------------------------------------------------------

        bomb_active = bomb_active & (bomb_y < self.consts.HILL_Y)

        return state.replace(
            bomb_x=bomb_x,
            bomb_y=bomb_y,
            bomb_active=bomb_active,
        )

    def _resolve_collisions(self, state):
        edy = self._enemy_display_y(state)
        ew, eh = ENEMY_W[state.enemy_type], ENEMY_H[state.enemy_type]

        def row(bx, by, active):
            return (
                _aabb_overlap(
                    bx,
                    by,
                    self.consts.BULLET_WIDTH,
                    self.consts.BULLET_HEIGHT,
                    state.enemy_x,
                    edy,
                    ew,
                    eh,
                )
                & active
                & state.enemy_alive
            )

        hits = jax.vmap(row)(state.bullet_x, state.bullet_y, state.bullet_active)
        enemy_hit = jnp.any(hits, axis=0)
        bullet_used = jnp.any(hits, axis=1)
        gained = jnp.sum(jnp.where(enemy_hit, ENEMY_POINTS[state.enemy_type], 0))

        exp_active = state.explosion_active | enemy_hit
        exp_timer = jnp.where(
            enemy_hit, self.consts.EXPLOSION_DURATION, state.explosion_timer
        )
        exp_x = jnp.where(enemy_hit, state.enemy_x, state.explosion_x)
        exp_y = jnp.where(enemy_hit, edy, state.explosion_y)

        new_alive = state.enemy_alive & (~enemy_hit)
        vulnerable = state.invuln_timer <= 0
        enemy_touch = jnp.any(
            _aabb_overlap(
                state.player_x,
                state.player_y,
                self.consts.PLAYER_WIDTH,
                self.consts.PLAYER_HEIGHT,
                state.enemy_x,
                edy,
                ew,
                eh,
            )
            & new_alive
        )
        bomb_each = (
            _aabb_overlap(
                state.player_x,
                state.player_y,
                self.consts.PLAYER_WIDTH,
                self.consts.PLAYER_HEIGHT,
                state.bomb_x,
                state.bomb_y,
                self.consts.BOMB_WIDTH,
                self.consts.BOMB_HEIGHT,
            )
            & state.bomb_active
        )
        damaged = vulnerable & (enemy_touch | jnp.any(bomb_each))

        p_exp_active = state.player_explosion_active | damaged
        p_exp_timer = jnp.where(
            damaged,
            self.consts.PLAYER_EXPLOSION_DURATION,
            jnp.maximum(state.player_explosion_timer - 1, 0),
        )
        p_exp_active = p_exp_active & (p_exp_timer > 0)
        p_exp_x = jnp.where(damaged, state.player_x, state.player_explosion_x)
        p_exp_y = jnp.where(damaged, state.player_y, state.player_explosion_y)

        px = jnp.where(damaged, jnp.float32(self.consts.PLAYER_START_X), state.player_x)
        py = jnp.where(damaged, jnp.float32(self.consts.PLAYER_START_Y), state.player_y)

        return (
            state.replace(
                score=state.score + gained,
                lives=jnp.maximum(0, state.lives - damaged.astype(jnp.int32)),
                invuln_timer=jnp.where(
                    damaged, self.consts.INVULN_FRAMES, state.invuln_timer
                ),
                enemy_alive=new_alive,
                bullet_active=state.bullet_active & (~bullet_used),
                bomb_active=state.bomb_active & (~(bomb_each & damaged)),
                player_x=px,
                player_y=py,
                explosion_active=exp_active,
                explosion_timer=exp_timer,
                explosion_x=exp_x,
                explosion_y=exp_y,
                player_explosion_active=p_exp_active,
                player_explosion_timer=p_exp_timer,
                player_explosion_x=p_exp_x,
                player_explosion_y=p_exp_y,
            ),
            damaged,
        )

    def _wave_step(self, state):
        """
        Start a new wave after all enemies are destroyed.

        A short delay is inserted between waves.
        """

        all_dead = jnp.all(~state.enemy_alive)

        # ---------------------------------------------------------
        # Wave timer
        # ---------------------------------------------------------
        timer_running = state.wave_timer > 0

        new_timer = jnp.maximum(
            state.wave_timer - 1,
            0,
        )

        # ---------------------------------------------------------
        # Start next wave
        # ---------------------------------------------------------
        start_next_wave = all_dead & ~timer_running & (state.lives > 0)

        next_wave = state.wave + start_next_wave.astype(jnp.int32)
        # ---------------------------------------------------------
        # Generate next formation
        # ---------------------------------------------------------
        key, spawn_key = jax.random.split(state.key)
        ex, ey, et, evy, eph = self._spawn_wave(
            spawn_key,
            next_wave,
        )
        # ---------------------------------------------------------
        # First detect wave completion
        # ---------------------------------------------------------
        new_wave_timer = jnp.where(
            start_next_wave,
            self.consts.WAVE_START_DELAY,
            new_timer,
        )
        # ---------------------------------------------------------
        # Spawn enemies only after the delay
        # ---------------------------------------------------------
        spawn_enemies = all_dead & (state.wave_timer == 1)

        # ---------------------------------------------------------
        # Wave bonus
        # ---------------------------------------------------------
        wave_bonus = jnp.where(
            start_next_wave,
            self.consts.WAVE_BONUS,
            0,
        )

        return state.replace(
            key=key,
            enemy_x=jnp.where(
                spawn_enemies,
                ex,
                state.enemy_x,
            ),
            enemy_y=jnp.where(
                spawn_enemies,
                ey,
                state.enemy_y,
            ),
            enemy_type=jnp.where(
                spawn_enemies,
                et,
                state.enemy_type,
            ),
            enemy_vy=jnp.where(
                spawn_enemies,
                evy,
                state.enemy_vy,
            ),
            enemy_phase=jnp.where(
                spawn_enemies,
                eph,
                state.enemy_phase,
            ),
            enemy_alive=jnp.where(
                spawn_enemies,
                jnp.ones(
                    (self.consts.NUM_ENEMIES,),
                    dtype=jnp.bool_,
                ),
                state.enemy_alive,
            ),
            wave=next_wave,
            wave_timer=new_wave_timer,
            score=state.score + wave_bonus,
        )

    def _explosion_step(self, state):
        t = jnp.maximum(state.explosion_timer - 1, 0)
        return state.replace(explosion_timer=t, explosion_active=t > 0)

    def _play_frame(self, state, action):
        """
        Execute exactly one internal game frame.
        """

        old_score = state.score

        state = self._player_step(state, action)
        state = self._bullet_step(state, action)

        state = self._enemy_step(state)
        state = self._bobo_step(state)
        state = self._bomb_step(state)

        state, damaged = self._resolve_collisions(state)

        state = self._wave_step(state)
        state = self._explosion_step(state)

        state = state.replace(
            step_counter=state.step_counter + 1,
            invuln_timer=jnp.maximum(
                state.invuln_timer - 1,
                0,
            ),
        )

        done = state.lives <= 0

        reward = (state.score - old_score).astype(jnp.float32) - damaged.astype(
            jnp.float32
        ) * self.consts.DEATH_PENALTY

        state = state.replace(
            mode=jnp.where(
                done,
                GAME_OVER,
                PLAY,
            )
        )

        return state, reward, done

    # Mode branches
    def _play_branch(self, state, action):
        """
        Execute one ALE-style environment step.

        One external step consists of FRAMESKIP internal frames.
        """

        def body_fn(_, carry):
            state, total_reward, done = carry

            # Do not advance the game after terminal state.
            def run_frame(_):
                new_state, reward, frame_done = self._play_frame(
                    state,
                    action,
                )

                return (
                    new_state,
                    total_reward + reward,
                    done | frame_done,
                )

            def skip_frame(_):
                return state, total_reward, done

            return jax.lax.cond(
                done,
                skip_frame,
                run_frame,
                operand=None,
            )

        state, reward, done = jax.lax.fori_loop(
            0,
            self.consts.FRAMESKIP,
            body_fn,
            (
                state,
                jnp.float32(0.0),
                jnp.bool_(False),
            ),
        )

        return state, reward, done

    def _attract_branch(self, state, a):
        fire = self._is_fire(a)

        def start(_):
            fields = self._fresh_game_fields(state.key)
            return StarGunnerState(
                mode=jnp.array(PLAY, jnp.int32),
                step_counter=jnp.array(0, jnp.int32),
                **fields,
            )

        def wait(_):
            return state.replace(step_counter=state.step_counter + 1)

        new = jax.lax.cond(fire, start, wait, operand=None)
        return new, jnp.float32(0.0), jnp.bool_(False)

    def _gameover_branch(self, state, a):
        fire = self._is_fire(a)
        new_mode = jnp.where(fire, ATTRACT, GAME_OVER)
        new = state.replace(mode=new_mode, step_counter=state.step_counter + 1)
        return new, jnp.float32(0.0), jnp.bool_(False)

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: StarGunnerState, action: chex.Array):
        a = jnp.take(self.ACTION_SET, action.astype(jnp.int32))

        # ---------------------------------------------------------
        # ALE-style sticky action
        # ---------------------------------------------------------

        state, effective_action = self._apply_sticky_action(
            state,
            a,
        )
        state, reward, done = jax.lax.switch(
            state.mode,
            [self._attract_branch, self._play_branch, self._gameover_branch],
            state,
            effective_action,
        )
        return self._get_observation(state), state, reward, done, self._get_info(state)

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: StarGunnerState):
        return self.renderer.render(state)

    def _get_observation(self, state):
        edy = self._enemy_display_y(state)
        player = ObjectObservation.create(
            x=state.player_x,
            y=state.player_y,
            width=jnp.array(self.consts.PLAYER_WIDTH),
            height=jnp.array(self.consts.PLAYER_HEIGHT),
        )
        enemies = ObjectObservation.create(
            x=jnp.where(state.enemy_alive, state.enemy_x, -1.0),
            y=jnp.where(state.enemy_alive, edy, -1.0),
            width=ENEMY_W[state.enemy_type],
            height=ENEMY_H[state.enemy_type],
        )
        bullets = ObjectObservation.create(
            x=jnp.where(state.bullet_active, state.bullet_x, -1.0),
            y=jnp.where(state.bullet_active, state.bullet_y, -1.0),
            width=jnp.full(
                (self.consts.MAX_BULLETS,), self.consts.BULLET_WIDTH, jnp.float32
            ),
            height=jnp.full(
                (self.consts.MAX_BULLETS,), self.consts.BULLET_HEIGHT, jnp.float32
            ),
        )
        bombs = ObjectObservation.create(
            x=jnp.where(state.bomb_active, state.bomb_x, -1.0),
            y=jnp.where(state.bomb_active, state.bomb_y, -1.0),
            width=jnp.full(
                (self.consts.MAX_BOMBS,), self.consts.BOMB_WIDTH, jnp.float32
            ),
            height=jnp.full(
                (self.consts.MAX_BOMBS,), self.consts.BOMB_HEIGHT, jnp.float32
            ),
        )
        bobo = ObjectObservation.create(
            x=state.bobo_x,
            y=jnp.array(float(self.consts.BOBO_Y)),
            width=jnp.array(self.consts.BOBO_WIDTH),
            height=jnp.array(self.consts.BOBO_HEIGHT),
        )
        return StarGunnerObservation(player, enemies, bullets, bombs, bobo)

    def action_space(self):
        return spaces.Discrete(len(self.ACTION_SET))

    def observation_space(self):
        s = (self.consts.HEIGHT, self.consts.WIDTH)
        return spaces.Dict(
            {
                "player": spaces.get_object_space(n=None, screen_size=s),
                "enemies": spaces.get_object_space(
                    n=self.consts.NUM_ENEMIES, screen_size=s
                ),
                "bullets": spaces.get_object_space(
                    n=self.consts.MAX_BULLETS, screen_size=s
                ),
                "bombs": spaces.get_object_space(
                    n=self.consts.MAX_BOMBS, screen_size=s
                ),
                "bobo": spaces.get_object_space(n=None, screen_size=s),
            }
        )

    def image_space(self):
        return spaces.Box(
            low=0,
            high=255,
            shape=(self.consts.HEIGHT, self.consts.WIDTH, 3),
            dtype=jnp.uint8,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state):
        return StarGunnerInfo(
            time=state.step_counter, lives=state.lives, wave=state.wave
        )


# Renderer – produces the final RGB image


class StarGunnerRenderer:
    def __init__(self, consts: StarGunnerConstants):
        self.consts = consts
        W, H = consts.WIDTH, consts.HEIGHT

        # Atari 2600 colour palette
        C_GREEN = (120, 196, 110)
        C_RED = (150, 60, 55)
        C_BLUE = (90, 140, 200)
        C_YELLOW = (150, 150, 70)

        # Attract screen text (copied from original)
        _ATTRACT = [
            (
                [
                    "111100001111",
                    "100100001001",
                    "100100001001",
                    "000000000000",
                    "100100001001",
                    "100100001001",
                    "111100001111",
                ],
                81,
                16,
                C_BLUE,
            ),
            (
                [
                    "1000000000000000100000000000000010000000",
                    "1100000000000000110000000000000011000000",
                    "1111000000000000111100000000000011110000",
                    "1111100000000000111110000000000011111000",
                    "1111111100000000111111110000000011111111",
                ],
                55,
                24,
                C_RED,
            ),
            (
                [
                    "00011111101111100111100011110000",
                    "00011000100011001100010011001000",
                    "00110001100010001000010010001100",
                    "00100000000010001000010010000010",
                    "00011100000010001111110010111100",
                    "00000010000110001100001001111000",
                    "00000010000100001000001001001000",
                    "01000100000100001000001001000100",
                    "10000100000100001000001001000010",
                    "11111000000100001000001001000001",
                ],
                60,
                84,
                C_GREEN,
            ),
            (
                [
                    "0110000000000000011000",
                    "1100000000000000000000",
                    "1011010101110111011011",
                    "1001010101110111000011",
                    "0110011101010101011010",
                ],
                65,
                97,
                C_GREEN,
            ),
            (
                [
                    "011110000000000000000",
                    "100001001011101110111",
                    "101101001011101110001",
                    "101101001011101110111",
                    "101101001000101110110",
                    "110001001000101110110",
                    "011110001000101110111",
                ],
                65,
                106,
                C_RED,
            ),
            (
                [
                    "11111100010000000000000000",
                    "00000000010000000000000000",
                    "11111100010000000000000000",
                    "00110000010000000000000000",
                    "00000111010111011101010111",
                    "00110111010111011001110110",
                    "00110111010111011100100111",
                    "00110110010110000100100001",
                    "00110111010111011101000111",
                ],
                63,
                115,
                C_RED,
            ),
            (["1", "1", "0", "0", "1", "1"], 71, 142, C_YELLOW),
        ]

        attract_rgb = np.zeros((H, W, 3), np.uint8)
        attract_mask = np.zeros((H, W), bool)

        temp_masks = []
        temp_rgbs = []

        for rows, ox, oy, col in _ATTRACT:
            for gy, r in enumerate(rows):
                for gx, ch in enumerate(r):
                    if ch == "1":
                        attract_rgb[oy + gy, ox + gx] = col
                        attract_mask[oy + gy, ox + gx] = True

            if col == C_RED and oy == 24:
                curr_mask = np.zeros((H, W), bool)
                curr_rgb = np.zeros((H, W, 3), np.uint8)
                for gy, r in enumerate(rows):
                    for gx, ch in enumerate(r):
                        if ch == "1":
                            curr_mask[oy + gy, ox + gx] = True
                            curr_rgb[oy + gy, ox + gx] = col
                temp_masks.append(jnp.array(curr_mask))
                temp_rgbs.append(jnp.array(curr_rgb))

        self.attract_rgb = jnp.array(attract_rgb)
        self.attract_mask = jnp.array(attract_mask)
        self.red_icon_masks = jnp.stack(temp_masks)
        self.red_icon_rgbs = jnp.stack(temp_rgbs)

        go = np.zeros((H, W), bool)
        _bake_text(go, "GAME OVER", _centered_x("GAME OVER", 2, W), 95, 2)
        self.gameover_mask = jnp.array(go)

        self.digit_font = jnp.array(np.stack([_glyph_np(str(d)) for d in range(10)]))

    # -------------------------------------------------------------------------
    # HUD number display
    # -------------------------------------------------------------------------
    def _blit_number_spaced(self, img, value, x0, y0, color, ndigits=4, spacing=10):
        clear_width = ndigits * spacing
        img = img.at[y0 : y0 + 7, x0 : x0 + clear_width].set(
            jnp.array([0, 0, 0], jnp.uint8)
        )
        for i in range(ndigits):
            place = 10 ** (ndigits - 1 - i)
            d = (value // place) % 10
            glyph = self.digit_font[d]
            xi = x0 + i * spacing
            img = img.at[y0 : y0 + 7, xi : xi + 5].set(
                jnp.where(glyph[..., None] > 0, color, img[y0 : y0 + 7, xi : xi + 5])
            )
        return img

    # Background: pure black + scrolling hills (authentic 2600)

    def _background(self, step):
        c = self.consts
        img = jnp.zeros((c.HEIGHT, c.WIDTH, 3), jnp.uint8)

        yy = jnp.arange(c.HEIGHT)[:, None]
        xx = jnp.arange(c.WIDTH)[None, :]
        scroll = step.astype(jnp.float32) * c.HILL_SCROLL_SPEED
        period = 75.0
        u = jnp.mod(xx + scroll, period) / period * 2 * jnp.pi
        wave = jnp.sin(u)
        amp = 5.0
        offset = wave * amp
        horizon = c.HILL_Y + offset.astype(jnp.int32)
        grass = yy >= horizon

        depth = jnp.clip((yy - horizon).astype(jnp.float32) / 6.0, 0.0, 1.0)
        top = jnp.array([80, 160, 80], jnp.float32)
        base = jnp.array([40, 80, 30], jnp.float32)
        shade = (
            top[None, None, :] * (1.0 - depth[..., None])
            + base[None, None, :] * depth[..., None]
        )
        shade = shade.astype(jnp.uint8)

        img = jnp.where(grass[..., None], shade, img)
        return img

    # Draw all moving entities

    def _draw_entities(self, img, state):
        c = self.consts

        # Bobo
        img = draw_sprite(
            img,
            state.bobo_x,
            c.BOBO_Y,
            BOBO_SPRITE,
            jnp.array([180, 60, 220], jnp.uint8),
        )

        # Bombs
        for i in range(c.MAX_BOMBS):

            def draw_bomb(active):
                def draw(_):
                    bomb_sprite = jnp.array([[1], [1], [1], [1]], dtype=jnp.bool_)
                    return draw_sprite(
                        img,
                        state.bomb_x[i],
                        state.bomb_y[i],
                        bomb_sprite,
                        jnp.array([255, 255, 255], jnp.uint8),
                    )

                def skip(_):
                    return img

                return jax.lax.cond(active, draw, skip, operand=None)

            img = draw_bomb(state.bomb_active[i])

        # Enemies
        edy = state.enemy_y + c.ENEMY_AMP * jnp.sin(
            state.enemy_phase + state.step_counter.astype(jnp.float32) * 0.1
        ) * (state.enemy_type == SAUCER)

        for i in range(c.NUM_ENEMIES):

            def draw_enemy(alive):
                def alive_fn(_):
                    etype = state.enemy_type[i]
                    color = ENEMY_COLOR[etype]
                    sprite = jax.lax.switch(
                        etype,
                        [
                            lambda: SAUCER_SPRITE,
                            lambda: BUZZIE_SPRITE,
                            lambda: SQUEEZER_SPRITE,
                        ],
                    )
                    return draw_sprite(img, state.enemy_x[i], edy[i], sprite, color)

                def dead_fn(_):
                    return img

                return jax.lax.cond(alive, alive_fn, dead_fn, operand=None)

            img = draw_enemy(state.enemy_alive[i])

        # Bullets
        for i in range(c.MAX_BULLETS):
            w = jnp.where(state.bullet_active[i], c.BULLET_WIDTH, 0)
            h = jnp.where(state.bullet_active[i], c.BULLET_HEIGHT, 0)
            img = draw_rect(
                img,
                state.bullet_x[i],
                state.bullet_y[i],
                w,
                h,
                jnp.array([255, 255, 0], jnp.uint8),
            )

        # Enemy explosions
        for i in range(c.NUM_ENEMIES):
            size = jnp.where(
                state.explosion_active[i],
                c.EXPLOSION_SIZE + (c.EXPLOSION_DURATION - state.explosion_timer[i]),
                0,
            )
            col = jnp.where(
                state.explosion_timer[i] > 6,
                jnp.array([255, 220, 0], jnp.uint8),
                jnp.array([255, 80, 0], jnp.uint8),
            )
            img = draw_rect(
                img, state.explosion_x[i], state.explosion_y[i], size, size, col
            )

        # Player explosion
        p_size = jnp.where(
            state.player_explosion_active,
            c.PLAYER_EXPLOSION_SIZE
            + (c.PLAYER_EXPLOSION_DURATION - state.player_explosion_timer),
            0,
        )
        p_col = jnp.where(
            state.player_explosion_timer > (c.PLAYER_EXPLOSION_DURATION // 2),
            jnp.array([255, 240, 80], jnp.uint8),
            jnp.array([255, 60, 20], jnp.uint8),
        )
        img = draw_rect(
            img,
            state.player_explosion_x,
            state.player_explosion_y,
            p_size,
            p_size,
            p_col,
        )

        # Player ship
        show = (state.invuln_timer <= 0) | ((state.step_counter // 4) % 2 == 0)

        def draw_player(should_show):
            def show_fn(_):
                sprite = jnp.where(
                    state.player_facing > 0, PLAYER_SPRITE, PLAYER_SPRITE_LEFT
                )
                is_moving = (state.player_x != state.prev_player_x) | (
                    state.player_y != state.prev_player_y
                )
                is_firing = state.fire_cooldown > 0
                color = jax.lax.select(
                    is_firing,
                    jnp.array([255, 0, 0], jnp.uint8),
                    jax.lax.select(
                        is_moving,
                        jnp.array([255, 255, 0], jnp.uint8),
                        jnp.array([0, 255, 0], jnp.uint8),
                    ),
                )
                return draw_sprite(img, state.player_x, state.player_y, sprite, color)

            def hide_fn(_):
                return img

            return jax.lax.cond(should_show, show_fn, hide_fn, operand=None)

        img = draw_player(show)
        return img

    # Mode renderers
    def _render_attract(self, state):
        img = self._background(state.step_counter)
        img = jnp.where(self.attract_mask[..., None], self.attract_rgb, img)
        return img

    def _render_play(self, state):
        img = self._background(state.step_counter)

        # Score area
        clear_color = jnp.array([0, 0, 0], jnp.uint8)
        img = draw_rect(img, 80, 15, 30, 10, clear_color)

        color_blue = jnp.array([90, 140, 200], jnp.uint8)
        img = draw_rect(img, 81, 16, 11, 7, color_blue)
        img = draw_rect(img, 96, 16, 11, 7, color_blue)
        img = draw_rect(img, 82, 17, 9, 5, jnp.array([0, 0, 0], jnp.uint8))
        img = draw_rect(img, 97, 17, 9, 5, jnp.array([0, 0, 0], jnp.uint8))

        img = self._blit_number_spaced(
            img, state.score, 81, 16, color_blue, ndigits=4, spacing=10
        )

        # Lives icons
        color_red = jnp.array([150, 60, 55], jnp.uint8)
        lives = state.lives
        mask0 = self.red_icon_masks[0]
        mask1 = self.red_icon_masks[1]
        mask2 = self.red_icon_masks[2]
        img = jnp.where((lives > 0) & mask0[..., None], self.red_icon_rgbs[0], img)
        img = jnp.where((lives > 1) & mask1[..., None], self.red_icon_rgbs[1], img)
        img = jnp.where((lives > 2) & mask2[..., None], self.red_icon_rgbs[2], img)

        img = self._draw_entities(img, state)
        return img

    def _render_gameover(self, state):
        # Freeze the playfield exactly as it was when the game ended
        img = self._render_play(state)
        img = jnp.where(
            self.gameover_mask[..., None], jnp.array([255, 50, 50], jnp.uint8), img
        )
        return img

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: StarGunnerState):
        return jax.lax.switch(
            state.mode,
            [self._render_attract, self._render_play, self._render_gameover],
            state,
        )


# ALE sprite capture helper – run this once to extract exact sprites


def capture_sprites_from_ale(num_episodes=5, save_path="star_gunner_sprites.pkl"):
    """
    Runs the Gymnasium ALE environment, captures frames, and saves them.
    You can then manually crop each sprite and replace the arrays above.
    """
    import gymnasium as gym
    import pickle

    env = gym.make("ALE/StarGunner-v5", render_mode="rgb_array")
    frames = []
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            frames.append(obs)
        env.reset()
    env.close()

    with open(save_path, "wb") as f:
        pickle.dump(frames, f)
    print(f"Captured {len(frames)} frames. Saved to {save_path}")
    print(
        "Manually inspect each frame, crop the sprite regions, and replace the placeholder arrays."
    )


# Local preview (human play with keyboard)

if __name__ == "__main__":
    # capture_sprites_from_ale()

    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    env = JaxStarGunner()
    _, g_state = env.reset(jax.random.PRNGKey(0))
    held = {"up": False, "down": False, "left": False, "right": False, "fire": False}

    def action_idx():
        u, d, l, r, f = (
            held["up"],
            held["down"],
            held["left"],
            held["right"],
            held["fire"],
        )
        # ---------------------------------------------------------
        # Fire + diagonal
        # ---------------------------------------------------------
        if f and u and r:
            return 14  # UPRIGHTFIRE
        if f and u and l:
            return 15  # UPLEFTFIRE
        if f and d and r:
            return 16  # DOWNRIGHTFIRE
        if f and d and l:
            return 17  # DOWNLEFTFIRE
        # ---------------------------------------------------------
        # Fire + cardinal direction
        # ---------------------------------------------------------
        if f and u:
            return 10  # UPFIRE
        if f and r:
            return 11  # RIGHTFIRE
        if f and l:
            return 12  # LEFTFIRE
        if f and d:
            return 13  # DOWNFIRE
        # ---------------------------------------------------------
        # Fire only
        # ---------------------------------------------------------
        if f:
            return 1  # FIRE
        # ---------------------------------------------------------
        # Diagonal movement
        # ---------------------------------------------------------
        if u and r:
            return 6  # UPRIGHT
        if u and l:
            return 7  # UPLEFT
        if d and r:
            return 8  # DOWNRIGHT
        if d and l:
            return 9  # DOWNLEFT
        # ---------------------------------------------------------
        # Cardinal movement
        # ---------------------------------------------------------
        if u:
            return 2
        if r:
            return 3
        if l:
            return 4
        if d:
            return 5
        return 0  # NOOP

    fig, ax = plt.subplots(figsize=(4, 5.25))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.set_position([0, 0, 1, 1])
    ax.axis("off")
    im = ax.imshow(np.asarray(env.render(g_state)).astype(np.uint8), aspect="auto")

    def on_press(e):
        if e.key in ("up", "down", "left", "right"):
            held[e.key] = True
        elif e.key == " ":
            held["fire"] = True

    def on_release(e):
        if e.key in ("up", "down", "left", "right"):
            held[e.key] = False
        elif e.key == " ":
            held["fire"] = False

    fig.canvas.mpl_connect("key_press_event", on_press)
    fig.canvas.mpl_connect("key_release_event", on_release)

    def update(_):
        global g_state
        _, g_state, _, _, _ = env.step(g_state, jnp.array(action_idx(), jnp.int32))
        im.set_data(np.asarray(env.render(g_state)).astype(np.uint8))
        return [im]

    print("Arrow keys move, SPACE fires. Press SPACE on the title to start.")
    _ani = animation.FuncAnimation(
        fig, update, interval=33, blit=False, cache_frame_data=False
    )
    plt.show()
