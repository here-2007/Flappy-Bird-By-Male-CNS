"""
run.py — FlyFlappyBird main training loop.

Connects:
  MaleCNS connectome (Berg et al. 2026)
  → LIF simulation (Dayan & Abbott 2001; Lappalainen et al. 2024)
  → Sensory encoding (Berg et al. 2026 frontal VPN circuit)
  → Motor decoding (TN1A / vPR9 / dPR1 wing pre-motor neurons)
  → Flappy Bird game physics
  → PPL101 reward-modulated Hebbian plasticity (Aso et al. 2014)

Kaggle usage:
    !pip install -q neuprint-python torch numpy scipy matplotlib Pillow tqdm
    %run run.py

Local usage:
    python run.py [--no-video] [--episodes N]
"""

import os
import sys
import time
import argparse
import numpy as np
import scipy.sparse as sp
from typing import Optional, Union
import torch

# ── Project imports ──────────────────────────────────────────────────────────
import config as CFG
from connectome.builder   import load_connectome
from simulation.lif       import LIFEngine
from simulation.sensory   import SensoryEncoder
from simulation.motor     import MotorDecoder
from simulation.plasticity import PPL101Plasticity
from game.flappy          import FlappyBird
from viz.dashboard        import Dashboard


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes",  type=int, default=CFG.MAX_EPISODES)
    p.add_argument("--mock",      action="store_true",
                   help="Use mock connectome instead of real data")
    p.add_argument("--no-video",  action="store_true",
                   help="Do not save simulation video")
    p.add_argument("--no-viz",    action="store_true",
                   help="Skip dashboard and video generation (faster; for benchmark runs)")
    p.add_argument("--viz-interval", type=int, default=CFG.VIZ_EVERY_N_FRAMES,
                   help="Interval (in frames) to update dashboard and capture video frames")
    p.add_argument("--save-weights", action="store_true",
                   help="Save updated weight matrix each episode")
    p.add_argument("--structured-encoder", action="store_true",
                   help="Use structured (feature-based) sensory encoder "
                        "instead of full frame rendering")
    p.add_argument("--device", type=str, default=None,
                   help="Execution device ('cuda', 'mps', 'cpu', or auto-detect)")
    return p.parse_args()


def kaggle_bar(ep: int, total_ep: int, score: int, best: int, fps: float) -> None:
    """Simple single-line progress for Kaggle notebook."""
    pct = int(30 * ep / max(total_ep, 1))
    bar = "█" * pct + "░" * (30 - pct)
    print(
        f"\r[{bar}] Ep {ep:4d}/{total_ep}  "
        f"Score {score:4d}  Best {best:4d}  "
        f"{fps:.1f} fps",
        end="", flush=True
    )


def run_simulation(
    episodes: int = CFG.MAX_EPISODES,
    mock: bool = False,
    no_video: bool = False,
    no_viz: bool = False,
    viz_interval: int = CFG.VIZ_EVERY_N_FRAMES,
    save_weights: bool = False,
    structured_encoder: bool = False,
    device: Optional[Union[str, torch.device]] = None,
) -> dict:
    """
    Run FlyFlappyBird simulation programmatically.
    Works seamlessly both in CLI scripts and Kaggle / Jupyter notebooks.
    """
    active_device = CFG.resolve_device(device)

    print("=" * 60)
    print("  FlyFlappyBird — MaleCNS connectome simulation")
    print("  Berg et al. (2026) Cell 189, 5504–5526")
    print(f"  Target compute device: {active_device}")
    print("=" * 60)

    # ── 1. Load connectome ───────────────────────────────────────────────
    if mock:
        print("\n[1/5] Loading MOCK connectome …")
        from data.mock_data import get_or_create_mock_connectome
        W_t, W_scipy, neuron_df, pop, coords = get_or_create_mock_connectome(device=active_device)
        N = len(neuron_df)
    else:
        print("\n[1/5] Loading MaleCNS connectome …")
        W_t, neuron_df, pop, coords = load_connectome(device=active_device)
        N = len(neuron_df)

        # Keep a scipy CSR copy for in-place plasticity updates
        W_scipy = sp.load_npz(CFG.CONN_NPZ)
        # Apply NT sign + scale (same as builder.py, needed for plasticity)
        from connectome.builder import _nt_sign_vector
        nt_sign = _nt_sign_vector(neuron_df)
        W_coo = W_scipy.tocoo().astype(np.float32)
        W_coo.data *= nt_sign[W_coo.col] * CFG.WEIGHT_SCALE
        np.clip(W_coo.data, CFG.W_MIN, CFG.W_MAX, out=W_coo.data)
        W_scipy = W_coo.tocsr()

    # ── 2. Initialise simulation components ──────────────────────────────
    print("\n[2/5] Initialising simulation components …")

    engine = LIFEngine(W=W_t, N=N, device=active_device)

    encoder = SensoryEncoder(
        photo_idx   = pop.photoreceptors_R1_R6,
        soma_coords = coords,
        N           = N,
        device      = active_device,
    )

    decoder = MotorDecoder(
        wing_motor_idx = pop.wing_motor,
        N              = N,
        device         = active_device,
    )

    plasticity = PPL101Plasticity(
        W_scipy    = W_scipy,
        ppl101_idx = pop.ppl101,
        kc_idx     = pop.kenyon_cells,
        N          = N,
        device     = active_device,
    )

    game = FlappyBird()

    # ── 3. Dashboard ─────────────────────────────────────────────────────
    print("\n[3/5] Setting up dashboard …")
    video_path = None if (no_video or no_viz) else CFG.VIDEO_OUT
    dashboard  = None if no_viz else Dashboard(
        coords     = coords,
        pop        = pop,
        video_path = video_path,
    )

    # ── 4. Motor calibration (warm-up with random input) ─────────────────
    print("\n[4/5] Motor calibration (3 warm-up steps) …")
    observed_rates: list[float] = []
    for _ in range(3):
        engine.reset()
        engine.clear_spike_accumulator()
        for _ in range(CFG.LIF_STEPS_PER_FRAME):
            I_noise = torch.randn(N, device=active_device) * 0.05
            engine.step(I_noise)
        rates = engine.spike_rates(pop.wing_motor)
        observed_rates.extend(rates.tolist())
    decoder.calibrate_threshold(observed_rates)

    # ── 5. Main training loop ─────────────────────────────────────────────
    print(f"\n[5/5] Training loop — {episodes} episodes …\n")
    best_score  = 0
    episode     = 0
    da_signal   = 0.0

    for episode in range(1, episodes + 1):
        game.reset(seed=episode)
        engine.reset()
        total_reward  = 0.0
        frame_count   = 0
        t_ep_start    = time.time()

        while game.state.alive and frame_count < CFG.MAX_FRAMES_PER_EP:
            frame_count += 1

            # ── Render ──────────────────────────────────────────────────
            frame_gray, frame_rgb = game.render()

            # ── Sensory encoding → I_ext ─────────────────────────────────
            if structured_encoder:
                s = game.state
                I_ext = encoder.encode_structured(
                    bird_y     = s.bird_y_norm,
                    gap_center = s.gap_center_norm,
                    pipe_dist  = s.pipe_dist_norm,
                    bird_vy    = s.bird_vy_norm,
                )
            else:
                I_ext = encoder.encode(frame_gray)

            # ── LIF steps ────────────────────────────────────────────────
            engine.clear_spike_accumulator()
            for _ in range(CFG.LIF_STEPS_PER_FRAME):
                spikes = engine.step(I_ext)
                # Plasticity per-step update
                plasticity.step(spikes, da_signal)

            # ── Motor decode → game action ────────────────────────────────
            flap = decoder.decode(
                engine.spike_history_sum,
                engine._step_count,
            )

            # ── Game step → reward ────────────────────────────────────────
            reward    = game.step(flap)
            da_signal = reward          # DA signal = reward this frame
            total_reward += reward

            # ── Dashboard update ──────────────────────────────────────────
            should_update_viz = dashboard and (
                frame_count % viz_interval == 0 or not game.state.alive or frame_count == 1
            )
            if should_update_viz:
                dashboard.update(
                    rgb_frame   = frame_rgb,
                    spike_rates = engine.all_spike_rates(),
                    episode     = episode,
                    score       = game.state.score,
                    best_score  = best_score,
                    wing_rate   = decoder.recent_mean_rate(),
                    da_signal   = da_signal,
                )
                dashboard.show_inline()

        # Capture death frame if it occurred after the last interval
        if dashboard and not game.state.alive and not should_update_viz:
            frame_gray, frame_rgb = game.render()
            dashboard.update(
                rgb_frame   = frame_rgb,
                spike_rates = engine.all_spike_rates(),
                episode     = episode,
                score       = game.state.score,
                best_score  = best_score,
                wing_rate   = decoder.recent_mean_rate(),
                da_signal   = da_signal,
            )
            dashboard.show_inline()

        # ── End of episode ────────────────────────────────────────────────
        score = game.state.score
        if score > best_score:
            best_score = score

        # Apply plasticity weight updates
        W_t = plasticity.apply_updates(W_t)
        engine.W = W_t                 # hot-swap updated weights

        if dashboard:
            dashboard.add_episode_result(score, total_reward)

        # Save weights checkpoint
        if save_weights:
            torch.save(W_scipy.data, CFG.WEIGHTS_PATH)

        # Progress
        fps = frame_count / max(time.time() - t_ep_start, 1e-6)
        kaggle_bar(episode, episodes, score, best_score, fps)

        if episode % 10 == 0:
            dw = plasticity.episode_dw_norm[-10:] if len(plasticity.episode_dw_norm) >= 10 else []
            avg_dw = float(np.mean(dw)) if dw else 0.0
            print(
                f"\n  [ep {episode:4d}]  score={score:3d}  best={best_score:3d}"
                f"  reward={total_reward:+7.2f}  |dW|_mean={avg_dw:.2e}"
                f"  frames={frame_count:5d}  {fps:.1f} fps"
            )

    print(f"\n\nTraining complete. Best score: {best_score}")

    if dashboard:
        dashboard.close()

    final_video = dashboard.video_path if (dashboard and getattr(dashboard, "video_saved", False)) else None
    if final_video:
        print(f"[done] Video: {final_video}")
    elif no_viz:
        print("[done] Video: not saved (disabled by --no-viz)")
    elif no_video:
        print("[done] Video: not saved (disabled by --no-video)")
    else:
        print(f"[done] Video: not saved")
    print(f"[done] Weights: {CFG.WEIGHTS_PATH if save_weights else 'not saved'}")

    return {
        "best_score": best_score,
        "episodes": episode,
        "video_path": final_video,
        "weights_path": CFG.WEIGHTS_PATH if save_weights else None,
    }


def main():
    args = parse_args()
    run_simulation(
        episodes=args.episodes,
        mock=args.mock,
        no_video=args.no_video,
        no_viz=args.no_viz,
        viz_interval=args.viz_interval,
        save_weights=args.save_weights,
        structured_encoder=args.structured_encoder,
        device=args.device,
    )


if __name__ == "__main__":
    main()
