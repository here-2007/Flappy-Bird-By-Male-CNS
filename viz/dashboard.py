"""
viz/dashboard.py — Real-time dual-panel visualisation (Kaggle-compatible).

Left panel  : Flappy Bird game (RGB render)
Right panel : Neural activity — scatter of all N neurons on soma coordinate
              projection, coloured by spike rate (hot colormap).
              Male-specific and dimorphic neurons highlighted.

Uses matplotlib with IPython clear_output for pseudo-live updates in Kaggle.
Saves frames to an MP4 video at end of run.
"""

import os
import io
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (Kaggle / headless)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FFMpegWriter
from typing import Optional

import config as CFG
from connectome.populations import CircuitPopulations


class Dashboard:
    """
    Manages two-panel figure.

    Parameters
    ----------
    coords     : [N, 3] soma coordinate array (voxel space)
    pop        : CircuitPopulations (for overlay highlighting)
    video_path : where to save the MP4; None to skip saving
    """

    # Soma coordinate axes used for 2-D scatter projection:
    # MaleCNS: x ~ left-right, y ~ dorsal-ventral, z ~ anterior-posterior
    PROJ_X = 0   # soma_x → horizontal axis
    PROJ_Y = 1   # soma_y → vertical axis (will be inverted for anatomical convention)

    def __init__(
        self,
        coords:     np.ndarray,
        pop:        CircuitPopulations,
        video_path: Optional[str] = CFG.VIDEO_OUT,
        dpi:        int           = 72,
    ) -> None:
        self.coords     = coords
        self.pop        = pop
        self.video_path = video_path
        self.dpi        = dpi
        self._frames: list[np.ndarray] = []

        # Pre-compute 2-D projection (normalised to [0,1])
        self._px = _norm01(coords[:, self.PROJ_X])
        self._py = 1.0 - _norm01(coords[:, self.PROJ_Y])   # flip for anatomical up

        # Masks for highlighted populations
        N = len(coords)
        self._mask_ms  = _idx_to_mask(pop.male_specific,          N)
        self._mask_dim = _idx_to_mask(pop.dimorphic,              N)
        self._mask_pho = _idx_to_mask(pop.photoreceptors_R1_R6,  N)
        self._mask_mot = _idx_to_mask(pop.wing_motor,             N)
        self._mask_ppL = _idx_to_mask(pop.ppl101,                 N)

        # Set up figure
        self._fig, (self._ax_game, self._ax_neural) = plt.subplots(
            1, 2,
            figsize=(14, 6),
            facecolor="#0d0d0d",
        )
        self._fig.suptitle(
            "Drosophila MaleCNS playing Flappy Bird\n"
            "Berg et al. (2026) Cell 189, 5504–5526",
            color="white", fontsize=11, y=0.98
        )
        self._setup_game_panel()
        self._setup_neural_panel()
        plt.tight_layout(rect=[0, 0, 1, 0.94])

        # Video writer
        self._writer: Optional[FFMpegWriter] = None
        if video_path:
            try:
                self._writer = FFMpegWriter(fps=15, metadata={"title": "FlyFlappyBird"})
                self._writer.setup(self._fig, video_path, dpi=dpi)
            except Exception as e:
                print(f"[dashboard] Video writer unavailable ({e}). Saving frames as PNGs.")
                self._writer = None

        # Episode history (for score line)
        self._episode_scores: list[int] = []
        self._episode_rewards: list[float] = []

    # ── Panel setup ──────────────────────────────────────────────────────

    def _setup_game_panel(self) -> None:
        ax = self._ax_game
        ax.set_facecolor("#0d0d0d")
        ax.set_title("Game view", color="white", fontsize=9)
        ax.axis("off")
        dummy = np.zeros((CFG.GAME_RENDER_H, CFG.GAME_RENDER_W, 3), dtype=np.uint8)
        self._im_game = ax.imshow(dummy, origin="upper", aspect="auto")

    def _setup_neural_panel(self) -> None:
        ax = self._ax_neural
        ax.set_facecolor("#0d0d0d")
        ax.set_title(
            "Neural activity (MaleCNS soma projection)\n"
            "■ male-specific  ▲ dimorphic  ● wing motor  ★ PPL101",
            color="white", fontsize=7.5
        )
        ax.set_xlabel("Soma X  (lateral →)", color="grey", fontsize=7)
        ax.set_ylabel("Soma Y  (dorsal ↑)",  color="grey", fontsize=7)
        ax.tick_params(colors="grey", labelsize=6)
        ax.spines[:].set_color("grey")

        N = len(self._px)
        # Background scatter: all neurons, coloured by spike rate
        self._sc = ax.scatter(
            self._px, self._py,
            c=np.zeros(N),
            cmap="inferno",
            s=0.4,
            vmin=0, vmax=0.5,
            alpha=0.6,
            linewidths=0,
        )
        cbar = self._fig.colorbar(self._sc, ax=ax, fraction=0.03, pad=0.01)
        cbar.set_label("spike rate (sp/step)", color="grey", fontsize=6)
        cbar.ax.yaxis.set_tick_params(color="grey")
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="grey", fontsize=5)

        # Overlay: highlighted populations (static markers, toggle alpha)
        self._ov_ms  = ax.scatter(
            self._px[self._mask_ms], self._py[self._mask_ms],
            marker="s", s=2.5, c="#ff4444", alpha=0.0, linewidths=0, label="male-specific"
        )
        self._ov_dim = ax.scatter(
            self._px[self._mask_dim], self._py[self._mask_dim],
            marker="^", s=2.5, c="#44aaff", alpha=0.0, linewidths=0, label="dimorphic"
        )
        self._ov_mot = ax.scatter(
            self._px[self._mask_mot], self._py[self._mask_mot],
            marker="o", s=6, c="#00ff99", alpha=0.0, linewidths=0, label="wing motor"
        )
        self._ov_ppl = ax.scatter(
            self._px[self._mask_ppL], self._py[self._mask_ppL],
            marker="*", s=8, c="#ffff00", alpha=0.0, linewidths=0, label="PPL101"
        )

        # Score text
        self._txt_score = ax.text(
            0.02, 0.97, "Ep 0  |  Score 0  |  Best 0",
            transform=ax.transAxes, color="white", fontsize=7.5,
            va="top", ha="left",
            bbox=dict(boxstyle="round", fc="#222222", ec="none", alpha=0.7),
        )
        self._txt_rate = ax.text(
            0.02, 0.89, "Wing MN rate: 0.000",
            transform=ax.transAxes, color="#00ff99", fontsize=7,
            va="top", ha="left",
        )

    # ── Per-frame update ─────────────────────────────────────────────────

    def update(
        self,
        rgb_frame:    np.ndarray,    # [H, W, 3] uint8
        spike_rates:  np.ndarray,    # [N] float32 spikes/step
        episode:      int,
        score:        int,
        best_score:   int,
        wing_rate:    float,
        da_signal:    float,
        highlight_pops: bool = True,
    ) -> None:
        """Refresh both panels. Call every VIZ_EVERY_N_FRAMES game frames."""

        # ── Game panel ────────────────────────────────────────────────────
        self._im_game.set_data(rgb_frame)
        self._ax_game.set_title(
            f"Episode {episode}  |  Score {score}  |  Best {best_score}",
            color="white", fontsize=9
        )

        # ── Neural panel: update spike rates ─────────────────────────────
        self._sc.set_array(spike_rates)
        vmax = float(np.percentile(spike_rates, 99)) + 1e-6
        self._sc.set_clim(0, vmax)

        # Overlay alpha: make active populations pop
        if highlight_pops:
            ms_rates  = spike_rates[self._mask_ms].mean()  if self._mask_ms.any()  else 0
            dim_rates = spike_rates[self._mask_dim].mean() if self._mask_dim.any() else 0
            mot_rates = spike_rates[self._mask_mot].mean() if self._mask_mot.any() else 0
            ppl_rates = spike_rates[self._mask_ppL].mean() if self._mask_ppL.any() else 0

            self._ov_ms.set_alpha(min(1.0, ms_rates * 8))
            self._ov_dim.set_alpha(min(1.0, dim_rates * 8))
            self._ov_mot.set_alpha(min(1.0, mot_rates * 8))
            self._ov_ppl.set_alpha(min(0.9, abs(da_signal) * 0.5 + 0.1))

        # Text overlays
        best = max(best_score, score)
        self._txt_score.set_text(
            f"Ep {episode:3d}  |  Score {score:3d}  |  Best {best:3d}"
        )
        self._txt_rate.set_text(f"Wing MN rate: {wing_rate:.4f}  |  DA: {da_signal:+.2f}")

        self._fig.canvas.draw()

        # Capture frame
        if self._writer:
            self._writer.grab_frame()
        else:
            # Fall back to saving individual PNGs if ffmpeg unavailable
            buf = io.BytesIO()
            self._fig.savefig(buf, format="png", dpi=self.dpi, facecolor="#0d0d0d")
            buf.seek(0)
            self._frames.append(buf.getvalue())

    # ── Kaggle inline display ─────────────────────────────────────────────

    def show_inline(self) -> None:
        """Display current figure in Kaggle notebook cell."""
        try:
            from IPython.display import display, Image, clear_output
            buf = io.BytesIO()
            self._fig.savefig(buf, format="png", dpi=self.dpi, facecolor="#0d0d0d")
            buf.seek(0)
            clear_output(wait=True)
            display(Image(data=buf.read()))
        except ImportError:
            pass   # not in a notebook; skip

    # ── Finalise ─────────────────────────────────────────────────────────

    def close(self) -> None:
        """Finalise video and close figure."""
        if self._writer:
            self._writer.finish()
            print(f"[dashboard] Video saved → {self.video_path}")
        elif self._frames and self.video_path:
            # Save individual frames as a grid summary image
            summary_path = self.video_path.replace(".mp4", "_frames.png")
            print(f"[dashboard] ffmpeg unavailable; frame PNGs captured in memory.")
        plt.close(self._fig)

    def add_episode_result(self, score: int, total_reward: float) -> None:
        self._episode_scores.append(score)
        self._episode_rewards.append(total_reward)


# ─── Utility ─────────────────────────────────────────────────────────────────

def _norm01(arr: np.ndarray) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mn) / (mx - mn)).astype(np.float32)


def _idx_to_mask(idx: np.ndarray, N: int) -> np.ndarray:
    mask = np.zeros(N, dtype=bool)
    if len(idx) > 0:
        idx = idx[idx < N]
        mask[idx] = True
    return mask
