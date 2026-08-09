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
ENEMY_W = jnp.array([8, 8, 8], dtype=jnp.float32)   # width of each enemy sprite
ENEMY_H = jnp.array([4, 4, 4], dtype=jnp.float32)   # height of each enemy sprite
ENEMY_COLOR = jnp.array([[255, 90, 0], [255, 180, 40], [200, 60, 200]], jnp.uint8)

# SPRITE DEFINITIONS – placeholder until we extract exact data from ALE
# To get perfect sprites, run capture_sprites_from_ale()
# and replace the arrays below with the cropped binary data.

PLAYER_SPRITE = jnp.array([
    [0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 1, 1, 1, 1, 0, 0],
], dtype=jnp.bool_)
PLAYER_SPRITE_LEFT = jnp.flip(PLAYER_SPRITE, axis=1)

SAUCER_SPRITE = jnp.array([
    [0, 0, 1, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [1, 1, 0, 1, 1, 0, 1, 1],
    [0, 1, 1, 1, 1, 1, 1, 0],
], dtype=jnp.bool_)

BUZZIE_SPRITE = jnp.array([
    [0, 1, 1, 0, 0, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [0, 1, 0, 1, 1, 0, 1, 0],
], dtype=jnp.bool_)

SQUEEZER_SPRITE = jnp.array([
    [0, 1, 0, 0, 0, 0, 1, 0],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [0, 1, 0, 1, 1, 0, 1, 0],
], dtype=jnp.bool_)

BOBO_SPRITE = jnp.array([
    [0, 1, 1, 1, 1, 1, 1, 0],
    [1, 1, 0, 1, 1, 0, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [0, 1, 0, 1, 1, 0, 1, 0],
], dtype=jnp.bool_)




# Constants

class StarGunnerConstants(struct.PyTreeNode):
    WIDTH: int = struct.field(pytree_node=False, default=160)
    HEIGHT: int = struct.field(pytree_node=False, default=210)
    DIFFICULTY: int = struct.field(pytree_node=False, default=0)

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
    ENEMY_AMP: float = struct.field(pytree_node=False, default=8.0)
    WAVE_BONUS: int = struct.field(pytree_node=False, default=1000)
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
    key: chex.PRNGKey

    bullet_x: chex.Array
    bullet_y: chex.Array
    bullet_dir: chex.Array
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
    '0': ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    '1': ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    '2': ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    '3': ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    '4': ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    '5': ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
    '6': ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
    '7': ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    '8': ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    '9': ["01110", "10001", "10001", "01111", "00001", "00010", "01100"],
    'A': ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    'E': ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    'F': ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    'G': ["01110", "10001", "10000", "10111", "10001", "10001", "01111"],
    'I': ["01110", "00100", "00100", "00100", "00100", "00100", "01110"],
    'L': ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    'M': ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    'N': ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    'O': ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    'P': ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    'R': ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    'S': ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    'T': ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    'U': ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    'V': ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    'Y': ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    ' ': ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
    '(': ["01110", "10001", "10110", "10100", "10110", "10001", "01110"],
}


def _glyph_np(ch):
    rows = _FONT.get(ch, _FONT[' '])
    return np.array([[1 if b == '1' else 0 for b in r] for r in rows], np.uint8)


def _bake_text(canvas, text, x, y, scale):
    cx = x
    for ch in text:
        g = _glyph_np(ch)
        ys, xs = np.where(g == 1)
        for gy, gx in zip(ys, xs):
            canvas[y + gy * scale: y + gy * scale + scale,
                   cx + gx * scale: cx + gx * scale + scale] = True
        cx += 6 * scale
    return canvas


def _centered_x(text, scale, width):
    return (width - (len(text) * 6 * scale - scale)) // 2



# Main Environment class

class JaxStarGunner(JaxEnvironment[
                        StarGunnerState, StarGunnerObservation, StarGunnerInfo, StarGunnerConstants
                    ]):
    ACTION_SET = jnp.array([
        Action.NOOP, Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT,
        Action.FIRE, Action.UPFIRE, Action.DOWNFIRE, Action.LEFTFIRE, Action.RIGHTFIRE,
    ], dtype=jnp.int32)

    def __init__(self, consts: StarGunnerConstants = None, start_in_play: bool = False):
        consts = consts or StarGunnerConstants()
        super().__init__(consts)
        self.start_in_play = start_in_play
        self.renderer = StarGunnerRenderer(consts)

    def _is_fire(self, a):
        return ((a == Action.FIRE) | (a == Action.UPFIRE) | (a == Action.DOWNFIRE) |
                (a == Action.LEFTFIRE) | (a == Action.RIGHTFIRE))

    def _spawn_wave(self, key):
        n = self.consts.NUM_ENEMIES
        keys = jax.random.split(key, n)
        idxs = jnp.arange(n)

        def one(k, i):
            k1, k2, k3 = jax.random.split(k, 3)
            x = (i.astype(jnp.float32) + 0.5) * (self.consts.WIDTH / n)
            y = jax.random.uniform(k2, (), minval=float(self.consts.PLAY_TOP + 4),
                                   maxval=float(self.consts.PLAY_BOTTOM - 12))
            etype = jax.random.randint(k1, (), 0, 3)
            phase = jax.random.uniform(k3, (), minval=0.0, maxval=2 * jnp.pi)
            vy_sign = jnp.where(jax.random.normal(k1, ()) >= 0, 1.0, -1.0)
            vy = jnp.where(etype == BUZZIE, vy_sign * 0.6, 0.0)
            return x, y, etype.astype(jnp.int32), vy.astype(jnp.float32), phase

        return jax.vmap(one)(keys, idxs)

    def _fresh_game_fields(self, key):
        n = self.consts.NUM_ENEMIES
        key, wkey = jax.random.split(key)
        ex, ey, et, evy, eph = self._spawn_wave(wkey)
        start_x = jnp.array(self.consts.PLAYER_START_X, jnp.float32)
        start_y = jnp.array(self.consts.PLAYER_START_Y, jnp.float32)
        return dict(
            key=key,
            player_x=start_x,
            player_y=start_y,
            player_facing=jnp.array(1, jnp.int32),
            prev_player_x=start_x,
            prev_player_y=start_y,
            bullet_x=jnp.zeros((self.consts.MAX_BULLETS,), jnp.float32),
            bullet_y=jnp.zeros((self.consts.MAX_BULLETS,), jnp.float32),
            bullet_dir=jnp.ones((self.consts.MAX_BULLETS,), jnp.float32),
            bullet_active=jnp.zeros((self.consts.MAX_BULLETS,), jnp.bool_),
            fire_cooldown=jnp.array(0, jnp.int32),
            enemy_x=ex, enemy_y=ey, enemy_type=et, enemy_vy=evy, enemy_phase=eph,
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
            invuln_timer=jnp.array(0, jnp.int32),
        )

    def reset(self, key: chex.PRNGKey = jax.random.PRNGKey(0)):
        fields = self._fresh_game_fields(key)
        mode = jnp.array(PLAY if self.start_in_play else ATTRACT, jnp.int32)
        state = StarGunnerState(mode=mode, step_counter=jnp.array(0, jnp.int32), **fields)
        return self._get_observation(state), state

    # Game logic steps

    def _player_step(self, state, a):
        left = (a == Action.LEFT) | (a == Action.LEFTFIRE)
        right = (a == Action.RIGHT) | (a == Action.RIGHTFIRE)
        up = (a == Action.UP) | (a == Action.UPFIRE)
        down = (a == Action.DOWN) | (a == Action.DOWNFIRE)

        if self.consts.DIFFICULTY == 1:
            left = jnp.zeros_like(left)

        mx = right.astype(jnp.float32) - left.astype(jnp.float32)
        my = down.astype(jnp.float32) - up.astype(jnp.float32)

        prev_x = state.player_x
        prev_y = state.player_y

        nx = jnp.clip(state.player_x + mx * self.consts.PLAYER_SPEED,
                      0.0, self.consts.WIDTH - self.consts.PLAYER_WIDTH)
        ny = jnp.clip(state.player_y + my * self.consts.PLAYER_SPEED,
                      self.consts.PLAY_TOP, self.consts.PLAY_BOTTOM - self.consts.PLAYER_HEIGHT)
        facing = jnp.where(right, 1, jnp.where(left, -1, state.player_facing))
        return state.replace(
            player_x=nx, player_y=ny, player_facing=facing,
            prev_player_x=prev_x, prev_player_y=prev_y
        )

    def _bullet_step(self, state, a):
        fire = self._is_fire(a)
        fdir = 1.0 if self.consts.DIFFICULTY == 1 else state.player_facing.astype(jnp.float32)
        sx = jnp.where(fdir > 0, state.player_x + self.consts.PLAYER_WIDTH,
                       state.player_x - self.consts.BULLET_WIDTH)
        sy = state.player_y + self.consts.PLAYER_HEIGHT / 2 - self.consts.BULLET_HEIGHT / 2

        ready = state.fire_cooldown <= 0
        free = jnp.argmax(~state.bullet_active)
        can = (~jnp.all(state.bullet_active)) & fire & ready
        bx = jnp.where(can, state.bullet_x.at[free].set(sx), state.bullet_x)
        by = jnp.where(can, state.bullet_y.at[free].set(sy), state.bullet_y)
        bd = jnp.where(can, state.bullet_dir.at[free].set(fdir), state.bullet_dir)
        ba = jnp.where(can, state.bullet_active.at[free].set(True), state.bullet_active)
        bx = jnp.where(ba, bx + bd * self.consts.BULLET_SPEED, bx)
        ba = ba & (bx >= 0) & (bx < self.consts.WIDTH)

        cooldown = jnp.where(can, self.consts.FIRE_COOLDOWN,
                             jnp.maximum(state.fire_cooldown - 1, 0))
        return state.replace(bullet_x=bx, bullet_y=by, bullet_dir=bd, bullet_active=ba,
                             fire_cooldown=cooldown)

    def _enemy_display_y(self, state):
        bob = self.consts.ENEMY_AMP * jnp.sin(
            state.enemy_phase + state.step_counter.astype(jnp.float32) * 0.1)
        return state.enemy_y + jnp.where(state.enemy_type == SAUCER, bob, 0.0)

    def _enemy_step(self, state):
        speed = self.consts.ENEMY_SPEED * (1.0 + 0.1 * (state.wave - 1).astype(jnp.float32))
        nx = (state.enemy_x - speed) % self.consts.WIDTH
        ny = state.enemy_y + state.enemy_vy
        lo, hi = float(self.consts.PLAY_TOP + 4), float(self.consts.PLAY_BOTTOM - 12)
        bounce = (ny < lo) | (ny > hi)
        nvy = jnp.where(bounce, -state.enemy_vy, state.enemy_vy)
        return state.replace(enemy_x=nx, enemy_y=jnp.clip(ny, lo, hi), enemy_vy=nvy)

    def _bobo_step(self, state):
        nx = state.bobo_x + state.bobo_vx
        flip = (nx < 4) | (nx > self.consts.WIDTH - self.consts.BOBO_WIDTH - 4)
        nvx = jnp.where(flip, -state.bobo_vx, state.bobo_vx)
        return state.replace(bobo_x=jnp.clip(nx, 4, self.consts.WIDTH - self.consts.BOBO_WIDTH - 4),
                             bobo_vx=nvx)

    def _bomb_step(self, state):
        drop = (state.step_counter % self.consts.BOBO_BOMB_PERIOD) == 0
        free = jnp.argmax(~state.bomb_active)
        can = (~jnp.all(state.bomb_active)) & drop
        bx = jnp.where(can, state.bomb_x.at[free].set(state.bobo_x + self.consts.BOBO_WIDTH / 2), state.bomb_x)
        by = jnp.where(can, state.bomb_y.at[free].set(self.consts.BOBO_Y + self.consts.BOBO_HEIGHT), state.bomb_y)
        ba = jnp.where(can, state.bomb_active.at[free].set(True), state.bomb_active)
        by = jnp.where(ba, by + self.consts.BOMB_SPEED, by)
        ba = ba & (by < self.consts.HILL_Y)
        return state.replace(bomb_x=bx, bomb_y=by, bomb_active=ba)

    def _resolve_collisions(self, state):
        edy = self._enemy_display_y(state)
        ew, eh = ENEMY_W[state.enemy_type], ENEMY_H[state.enemy_type]
        hit_y = edy + self.consts.HIT_TOP_MARGIN
        hit_h = eh - self.consts.HIT_TOP_MARGIN

        def row(bx, by, active):
            return _aabb_overlap(bx, by, self.consts.BULLET_WIDTH, self.consts.BULLET_HEIGHT,
                                 state.enemy_x, hit_y, ew, hit_h) & active & state.enemy_alive

        hits = jax.vmap(row)(state.bullet_x, state.bullet_y, state.bullet_active)
        enemy_hit = jnp.any(hits, axis=0)
        bullet_used = jnp.any(hits, axis=1)
        gained = jnp.sum(jnp.where(enemy_hit, ENEMY_POINTS[state.enemy_type], 0))

        exp_active = state.explosion_active | enemy_hit
        exp_timer = jnp.where(enemy_hit, self.consts.EXPLOSION_DURATION, state.explosion_timer)
        exp_x = jnp.where(enemy_hit, state.enemy_x, state.explosion_x)
        exp_y = jnp.where(enemy_hit, edy, state.explosion_y)

        new_alive = state.enemy_alive & (~enemy_hit)
        vulnerable = state.invuln_timer <= 0
        enemy_touch = jnp.any(_aabb_overlap(
            state.player_x, state.player_y, self.consts.PLAYER_WIDTH, self.consts.PLAYER_HEIGHT,
            state.enemy_x, edy, ew, eh) & new_alive)
        bomb_each = _aabb_overlap(
            state.player_x, state.player_y, self.consts.PLAYER_WIDTH, self.consts.PLAYER_HEIGHT,
            state.bomb_x, state.bomb_y, self.consts.BOMB_WIDTH, self.consts.BOMB_HEIGHT) & state.bomb_active
        damaged = vulnerable & (enemy_touch | jnp.any(bomb_each))

        p_exp_active = state.player_explosion_active | damaged
        p_exp_timer = jnp.where(damaged, self.consts.PLAYER_EXPLOSION_DURATION,
                                jnp.maximum(state.player_explosion_timer - 1, 0))
        p_exp_active = p_exp_active & (p_exp_timer > 0)
        p_exp_x = jnp.where(damaged, state.player_x, state.player_explosion_x)
        p_exp_y = jnp.where(damaged, state.player_y, state.player_explosion_y)

        px = jnp.where(damaged, jnp.float32(self.consts.PLAYER_START_X), state.player_x)
        py = jnp.where(damaged, jnp.float32(self.consts.PLAYER_START_Y), state.player_y)

        return state.replace(
            score=state.score + gained,
            lives=jnp.maximum(0, state.lives - damaged.astype(jnp.int32)),
            invuln_timer=jnp.where(damaged, self.consts.INVULN_FRAMES, state.invuln_timer),
            enemy_alive=new_alive,
            bullet_active=state.bullet_active & (~bullet_used),
            bomb_active=state.bomb_active & (~(bomb_each & damaged)),
            player_x=px, player_y=py,
            explosion_active=exp_active, explosion_timer=exp_timer,
            explosion_x=exp_x, explosion_y=exp_y,
            player_explosion_active=p_exp_active, player_explosion_timer=p_exp_timer,
            player_explosion_x=p_exp_x, player_explosion_y=p_exp_y,
        ), damaged

    def _wave_step(self, state):
        all_dead = jnp.all(~state.enemy_alive)
        key, wkey = jax.random.split(state.key)
        ex, ey, et, evy, eph = self._spawn_wave(wkey)
        n = self.consts.NUM_ENEMIES
        return state.replace(
            key=key,
            enemy_x=jnp.where(all_dead, ex, state.enemy_x),
            enemy_y=jnp.where(all_dead, ey, state.enemy_y),
            enemy_type=jnp.where(all_dead, et, state.enemy_type),
            enemy_vy=jnp.where(all_dead, evy, state.enemy_vy),
            enemy_phase=jnp.where(all_dead, eph, state.enemy_phase),
            enemy_alive=jnp.where(all_dead, jnp.ones((n,), jnp.bool_), state.enemy_alive),
            wave=state.wave + all_dead.astype(jnp.int32),
            score=state.score + all_dead.astype(jnp.int32) * self.consts.WAVE_BONUS,
        )

    def _explosion_step(self, state):
        t = jnp.maximum(state.explosion_timer - 1, 0)
        return state.replace(explosion_timer=t, explosion_active=t > 0)


    # Mode branches

    def _play_branch(self, state, a):
        old = state.score
        state = self._player_step(state, a)
        state = self._bullet_step(state, a)
        state = self._enemy_step(state)
        state = self._bobo_step(state)
        state = self._bomb_step(state)
        state, damaged = self._resolve_collisions(state)
        state = self._wave_step(state)
        state = self._explosion_step(state)
        state = state.replace(step_counter=state.step_counter + 1,
                              invuln_timer=jnp.maximum(state.invuln_timer - 1, 0))
        done = state.lives <= 0
        reward = (state.score - old).astype(jnp.float32) - damaged.astype(jnp.float32) * self.consts.DEATH_PENALTY
        state = state.replace(mode=jnp.where(done, GAME_OVER, PLAY))
        return state, reward, done

    def _attract_branch(self, state, a):
        fire = self._is_fire(a)

        def start(_):
            fields = self._fresh_game_fields(state.key)
            return StarGunnerState(mode=jnp.array(PLAY, jnp.int32),
                                   step_counter=jnp.array(0, jnp.int32), **fields)

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
        state, reward, done = jax.lax.switch(
            state.mode,
            [self._attract_branch, self._play_branch, self._gameover_branch],
            state, a,
        )
        return self._get_observation(state), state, reward, done, self._get_info(state)

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: StarGunnerState):
        return self.renderer.render(state)

    def _get_observation(self, state):
        edy = self._enemy_display_y(state)
        player = ObjectObservation.create(
            x=state.player_x, y=state.player_y,
            width=jnp.array(self.consts.PLAYER_WIDTH), height=jnp.array(self.consts.PLAYER_HEIGHT))
        enemies = ObjectObservation.create(
            x=jnp.where(state.enemy_alive, state.enemy_x, -1.0),
            y=jnp.where(state.enemy_alive, edy, -1.0),
            width=ENEMY_W[state.enemy_type], height=ENEMY_H[state.enemy_type])
        bullets = ObjectObservation.create(
            x=jnp.where(state.bullet_active, state.bullet_x, -1.0),
            y=jnp.where(state.bullet_active, state.bullet_y, -1.0),
            width=jnp.full((self.consts.MAX_BULLETS,), self.consts.BULLET_WIDTH, jnp.float32),
            height=jnp.full((self.consts.MAX_BULLETS,), self.consts.BULLET_HEIGHT, jnp.float32))
        bombs = ObjectObservation.create(
            x=jnp.where(state.bomb_active, state.bomb_x, -1.0),
            y=jnp.where(state.bomb_active, state.bomb_y, -1.0),
            width=jnp.full((self.consts.MAX_BOMBS,), self.consts.BOMB_WIDTH, jnp.float32),
            height=jnp.full((self.consts.MAX_BOMBS,), self.consts.BOMB_HEIGHT, jnp.float32))
        bobo = ObjectObservation.create(
            x=state.bobo_x, y=jnp.array(float(self.consts.BOBO_Y)),
            width=jnp.array(self.consts.BOBO_WIDTH), height=jnp.array(self.consts.BOBO_HEIGHT))
        return StarGunnerObservation(player, enemies, bullets, bombs, bobo)

    def action_space(self):
        return spaces.Discrete(len(self.ACTION_SET))

    def observation_space(self):
        s = (self.consts.HEIGHT, self.consts.WIDTH)
        return spaces.Dict({
            "player": spaces.get_object_space(n=None, screen_size=s),
            "enemies": spaces.get_object_space(n=self.consts.NUM_ENEMIES, screen_size=s),
            "bullets": spaces.get_object_space(n=self.consts.MAX_BULLETS, screen_size=s),
            "bombs": spaces.get_object_space(n=self.consts.MAX_BOMBS, screen_size=s),
            "bobo": spaces.get_object_space(n=None, screen_size=s),
        })

    def image_space(self):
        return spaces.Box(low=0, high=255,
                          shape=(self.consts.HEIGHT, self.consts.WIDTH, 3), dtype=jnp.uint8)

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state):
        return StarGunnerInfo(time=state.step_counter, lives=state.lives, wave=state.wave)



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
            (["111100001111",
              "100100001001",
              "100100001001",
              "000000000000",
              "100100001001",
              "100100001001",
              "111100001111"], 81, 16, C_BLUE),
            (["1000000000000000100000000000000010000000",
              "1100000000000000110000000000000011000000",
              "1111000000000000111100000000000011110000",
              "1111100000000000111110000000000011111000",
              "1111111100000000111111110000000011111111"], 55, 24, C_RED),
            (["00011111101111100111100011110000",
              "00011000100011001100010011001000",
              "00110001100010001000010010001100",
              "00100000000010001000010010000010",
              "00011100000010001111110010111100",
              "00000010000110001100001001111000",
              "00000010000100001000001001001000",
              "01000100000100001000001001000100",
              "10000100000100001000001001000010",
              "11111000000100001000001001000001"], 60, 84, C_GREEN),
            (["0110000000000000011000",
              "1100000000000000000000",
              "1011010101110111011011",
              "1001010101110111000011",
              "0110011101010101011010"], 65, 97, C_GREEN),
            (["011110000000000000000",
              "100001001011101110111",
              "101101001011101110001",
              "101101001011101110111",
              "101101001000101110110",
              "110001001000101110110",
              "011110001000101110111"], 65, 106, C_RED),
            (["11111100010000000000000000",
              "00000000010000000000000000",
              "11111100010000000000000000",
              "00110000010000000000000000",
              "00000111010111011101010111",
              "00110111010111011001110110",
              "00110111010111011100100111",
              "00110110010110000100100001",
              "00110111010111011101000111"], 63, 115, C_RED),
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
        img = img.at[y0:y0 + 7, x0:x0 + clear_width].set(jnp.array([0, 0, 0], jnp.uint8))
        for i in range(ndigits):
            place = 10 ** (ndigits - 1 - i)
            d = (value // place) % 10
            glyph = self.digit_font[d]
            xi = x0 + i * spacing
            img = img.at[y0:y0 + 7, xi:xi + 5].set(
                jnp.where(glyph[..., None] > 0, color, img[y0:y0 + 7, xi:xi + 5]))
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
        grass = (yy >= horizon)

        depth = jnp.clip((yy - horizon).astype(jnp.float32) / 6.0, 0.0, 1.0)
        top = jnp.array([80, 160, 80], jnp.float32)
        base = jnp.array([40, 80, 30], jnp.float32)
        shade = (top[None, None, :] * (1.0 - depth[..., None]) +
                 base[None, None, :] * depth[..., None])
        shade = shade.astype(jnp.uint8)

        img = jnp.where(grass[..., None], shade, img)
        return img


    # Draw all moving entities

    def _draw_entities(self, img, state):
        c = self.consts

        # Bobo
        img = draw_sprite(img, state.bobo_x, c.BOBO_Y, BOBO_SPRITE,
                          jnp.array([180, 60, 220], jnp.uint8))

        # Bombs
        for i in range(c.MAX_BOMBS):
            def draw_bomb(active):
                def draw(_):
                    bomb_sprite = jnp.array([[1], [1], [1], [1]], dtype=jnp.bool_)
                    return draw_sprite(img, state.bomb_x[i], state.bomb_y[i],
                                       bomb_sprite, jnp.array([255, 255, 255], jnp.uint8))
                def skip(_):
                    return img
                return jax.lax.cond(active, draw, skip, operand=None)
            img = draw_bomb(state.bomb_active[i])

        # Enemies
        edy = state.enemy_y + c.ENEMY_AMP * jnp.sin(
            state.enemy_phase + state.step_counter.astype(jnp.float32) * 0.1) * (state.enemy_type == SAUCER)

        for i in range(c.NUM_ENEMIES):
            def draw_enemy(alive):
                def alive_fn(_):
                    etype = state.enemy_type[i]
                    color = ENEMY_COLOR[etype]
                    sprite = jax.lax.switch(
                        etype,
                        [lambda: SAUCER_SPRITE, lambda: BUZZIE_SPRITE, lambda: SQUEEZER_SPRITE]
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
            img = draw_rect(img, state.bullet_x[i], state.bullet_y[i], w, h,
                            jnp.array([255, 255, 0], jnp.uint8))

        # Enemy explosions
        for i in range(c.NUM_ENEMIES):
            size = jnp.where(state.explosion_active[i],
                             c.EXPLOSION_SIZE + (c.EXPLOSION_DURATION - state.explosion_timer[i]), 0)
            col = jnp.where(state.explosion_timer[i] > 6,
                            jnp.array([255, 220, 0], jnp.uint8),
                            jnp.array([255, 80, 0], jnp.uint8))
            img = draw_rect(img, state.explosion_x[i], state.explosion_y[i], size, size, col)

        # Player explosion
        p_size = jnp.where(state.player_explosion_active,
                           c.PLAYER_EXPLOSION_SIZE + (c.PLAYER_EXPLOSION_DURATION - state.player_explosion_timer), 0)
        p_col = jnp.where(state.player_explosion_timer > (c.PLAYER_EXPLOSION_DURATION // 2),
                          jnp.array([255, 240, 80], jnp.uint8),
                          jnp.array([255, 60, 20], jnp.uint8))
        img = draw_rect(img, state.player_explosion_x, state.player_explosion_y, p_size, p_size, p_col)

        # Player ship
        show = (state.invuln_timer <= 0) | ((state.step_counter // 4) % 2 == 0)

        def draw_player(should_show):
            def show_fn(_):
                sprite = jnp.where(state.player_facing > 0, PLAYER_SPRITE, PLAYER_SPRITE_LEFT)
                is_moving = (state.player_x != state.prev_player_x) | (state.player_y != state.prev_player_y)
                is_firing = state.fire_cooldown > 0
                color = jax.lax.select(
                    is_firing,
                    jnp.array([255, 0, 0], jnp.uint8),
                    jax.lax.select(
                        is_moving,
                        jnp.array([255, 255, 0], jnp.uint8),
                        jnp.array([0, 255, 0], jnp.uint8)
                    )
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

        img = self._blit_number_spaced(img, state.score, 81, 16, color_blue, ndigits=4, spacing=10)

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
        img = jnp.where(self.gameover_mask[..., None], jnp.array([255, 50, 50], jnp.uint8), img)
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
    print("Manually inspect each frame, crop the sprite regions, and replace the placeholder arrays.")


# Local preview (human play with keyboard)

if __name__ == "__main__":
    # capture_sprites_from_ale()

    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    env = JaxStarGunner()
    _, g_state = env.reset(jax.random.PRNGKey(0))
    held = {"up": False, "down": False, "left": False, "right": False, "fire": False}

    def action_idx():
        u, d, l, r, f = (held["up"], held["down"], held["left"], held["right"], held["fire"])
        if f and u: return 6
        if f and d: return 7
        if f and l: return 8
        if f and r: return 9
        if u: return 1
        if d: return 2
        if l: return 3
        if r: return 4
        if f: return 5
        return 0

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
    _ani = animation.FuncAnimation(fig, update, interval=33, blit=False, cache_frame_data=False)
    plt.show()