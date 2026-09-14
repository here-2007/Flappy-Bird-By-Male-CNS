"""
game/flappy.py — Headless Flappy Bird physics + numpy frame renderer.

No pygame; no display dependency. Renders to numpy arrays for both
the sensory encoder (grayscale) and the visualisation dashboard (RGB).

Physics constants approximate the canonical Flappy Bird game.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple
import config as CFG


# ─── Physics constants ────────────────────────────────────────────────────────
GRAVITY      =  0.35     # px / frame²  (downward acceleration)
FLAP_VEL     = -5.5      # px / frame   (upward velocity on flap)
PIPE_SPEED   =  2.5      # px / frame   (pipe moves left)
PIPE_GAP     =  70       # px           (vertical gap height)
PIPE_SPACING = 130       # px           (horizontal distance between pipes)
PIPE_WIDTH   =  28       # px
BIRD_X       =  36       # px           (fixed horizontal position)
BIRD_RADIUS  =   8       # px
MAX_VEL      =  8.0      # px / frame   (terminal velocity cap)

# Canvas size (matches CFG.GAME_RENDER_W / H)
W = CFG.GAME_RENDER_W
H = CFG.GAME_RENDER_H

# Colour palette (RGB)
COL_BG     = np.array([135, 206, 235], dtype=np.uint8)  # sky blue
COL_BIRD   = np.array([255, 200,  30], dtype=np.uint8)  # yellow
COL_PIPE   = np.array([ 92, 164,  30], dtype=np.uint8)  # green
COL_GROUND = np.array([222, 184, 135], dtype=np.uint8)  # tan
GROUND_H   = 20   # px height of ground strip


@dataclass
class GameState:
    bird_y:    float = H // 2
    bird_vy:   float = 0.0
    pipes:     list  = field(default_factory=list)   # list of {x, gap_top}
    score:     int   = 0
    alive:     bool  = True
    frame_idx: int   = 0

    # Normalised features (computed each frame for structured encoder)
    bird_y_norm:   float = 0.5
    bird_vy_norm:  float = 0.0
    gap_center_norm: float = 0.5
    pipe_dist_norm: float = 1.0
    passed_pipe:    bool = False


class FlappyBird:
    """
    Headless Flappy Bird engine.

    Usage
    -----
    game = FlappyBird()
    game.reset()
    while game.state.alive:
        frame_gray, frame_rgb = game.render()
        reward = game.step(flap=True/False)
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.default_rng(seed)
        self.state = GameState()

    def reset(self, seed: int | None = None) -> None:
        """Reset to initial game state."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.state = GameState(
            bird_y  = H // 2,
            bird_vy = 0.0,
            pipes   = [],
            score   = 0,
            alive   = True,
            frame_idx = 0,
        )
        # Spawn first pipe off-screen right
        self._spawn_pipe(x=W + 30)

    # ── Core loop step ───────────────────────────────────────────────────

    def step(self, flap: bool) -> float:
        """
        Advance physics by one game frame.

        Returns
        -------
        reward : float
            +REWARD_PASS_PIPE if passed a pipe this frame
            +REWARD_ALIVE each frame alive
            PUNISH_DEATH on collision (game.state.alive becomes False)
        """
        s = self.state
        if not s.alive:
            return 0.0

        s.frame_idx  += 1
        s.passed_pipe = False
        reward = CFG.REWARD_ALIVE

        # ── Flap ──────────────────────────────────────────────────────────
        if flap:
            s.bird_vy = FLAP_VEL

        # ── Gravity ───────────────────────────────────────────────────────
        s.bird_vy = min(s.bird_vy + GRAVITY, MAX_VEL)
        s.bird_y  = s.bird_y + s.bird_vy

        # ── Move pipes ────────────────────────────────────────────────────
        for pipe in s.pipes:
            pipe["x"] -= PIPE_SPEED

        # Remove off-screen pipes and count score
        new_pipes = []
        for pipe in s.pipes:
            if pipe["x"] + PIPE_WIDTH < 0:
                continue
            if not pipe.get("scored") and pipe["x"] + PIPE_WIDTH < BIRD_X - BIRD_RADIUS:
                pipe["scored"] = True
                s.score += 1
                s.passed_pipe = True
                reward += CFG.REWARD_PASS_PIPE
            new_pipes.append(pipe)
        s.pipes = new_pipes

        # Spawn new pipe if needed
        if not s.pipes or s.pipes[-1]["x"] < W - PIPE_SPACING:
            self._spawn_pipe(x=W)

        # ── Collision detection ───────────────────────────────────────────
        dead = self._check_collision()
        if dead:
            s.alive = False
            reward  = CFG.PUNISH_DEATH

        # ── Normalised features ───────────────────────────────────────────
        self._update_normalised_features()

        return reward

    # ── Rendering ────────────────────────────────────────────────────────

    def render(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Render current game state.

        Returns
        -------
        gray : [H, W] float32 in [0, 1]   — for sensory encoder
        rgb  : [H, W, 3] uint8             — for dashboard visualisation
        """
        s   = self.state
        rgb = np.full((H, W, 3), COL_BG, dtype=np.uint8)

        # Ground strip
        rgb[H - GROUND_H:, :] = COL_GROUND

        # Pipes
        for pipe in s.pipes:
            px     = int(pipe["x"])
            gap_t  = int(pipe["gap_top"])
            gap_b  = gap_t + PIPE_GAP
            pw     = PIPE_WIDTH
            # Upper pipe
            if gap_t > 0:
                _fill_rect(rgb, px, 0, pw, gap_t, COL_PIPE)
            # Lower pipe
            if gap_b < H - GROUND_H:
                _fill_rect(rgb, px, gap_b, pw, H - GROUND_H - gap_b, COL_PIPE)

        # Bird (circle approximation with square)
        bx  = BIRD_X - BIRD_RADIUS
        by  = int(s.bird_y) - BIRD_RADIUS
        bsz = BIRD_RADIUS * 2
        _fill_rect(rgb, bx, by, bsz, bsz, COL_BIRD)

        # Grayscale (Rec.601 weights)
        gray = (
            0.299 * rgb[:, :, 0].astype(np.float32) +
            0.587 * rgb[:, :, 1].astype(np.float32) +
            0.114 * rgb[:, :, 2].astype(np.float32)
        ) / 255.0

        return gray, rgb

    # ── Private ──────────────────────────────────────────────────────────

    def _spawn_pipe(self, x: float) -> None:
        gap_top = float(self._rng.integers(
            int(H * 0.15),
            int(H * 0.70 - PIPE_GAP),
        ))
        self.state.pipes.append({"x": float(x), "gap_top": gap_top, "scored": False})

    def _check_collision(self) -> bool:
        s = self.state
        bx_l = BIRD_X - BIRD_RADIUS
        bx_r = BIRD_X + BIRD_RADIUS
        by_t = s.bird_y - BIRD_RADIUS
        by_b = s.bird_y + BIRD_RADIUS

        # Ground / ceiling
        if by_b >= H - GROUND_H or by_t <= 0:
            return True

        # Pipes
        for pipe in s.pipes:
            px_l  = pipe["x"]
            px_r  = pipe["x"] + PIPE_WIDTH
            gap_t = pipe["gap_top"]
            gap_b = gap_t + PIPE_GAP

            if bx_r > px_l and bx_l < px_r:          # horizontal overlap
                if by_t < gap_t or by_b > gap_b:      # outside the gap
                    return True
        return False

    def _update_normalised_features(self) -> None:
        s = self.state
        s.bird_y_norm  = float(s.bird_y) / H
        s.bird_vy_norm = float(s.bird_vy) / MAX_VEL

        # Nearest upcoming pipe
        upcoming = [p for p in s.pipes if p["x"] + PIPE_WIDTH >= BIRD_X]
        if upcoming:
            np_pipe = min(upcoming, key=lambda p: p["x"])
            gap_center = np_pipe["gap_top"] + PIPE_GAP / 2
            s.gap_center_norm = gap_center / H
            s.pipe_dist_norm  = max(0.0, (np_pipe["x"] - BIRD_X)) / W
        else:
            s.gap_center_norm = 0.5
            s.pipe_dist_norm  = 1.0


# ─── Utility ─────────────────────────────────────────────────────────────────

def _fill_rect(
    img: np.ndarray,
    x: int, y: int,
    w: int, h: int,
    color: np.ndarray,
) -> None:
    """Draw a filled rectangle onto img (in-place). Clips to canvas."""
    x0 = max(0, x);     x1 = min(img.shape[1], x + w)
    y0 = max(0, y);     y1 = min(img.shape[0], y + h)
    if x0 < x1 and y0 < y1:
        img[y0:y1, x0:x1] = color
