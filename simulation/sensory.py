"""
simulation/sensory.py — Map Flappy Bird game frames to photoreceptor
input currents.

Visual encoding follows Berg et al. (2026):
  - Frontal visual field (±60° az, ±40° el) sampled by LoVP92 + 6 frontal
    VPN types. p.5511: "confined to the anterior lobula, which samples the
    frontal visual field."
  - R1–R6 photoreceptors encode luminance/contrast. TmY21 (dimorphic OLIN)
    receives frontal inputs from edge-detecting Dm3 neurons (p.5511).
  - Spatial layout: MaleCNS soma coordinates give retinotopic arrangement.

Encoding pipeline:
  Game frame (H×W) → frontal patch → retinotopic map → photoreceptor I_ext
"""

import numpy as np
import torch
import config as CFG


class SensoryEncoder:
    """
    Converts a game frame (numpy H×W grayscale) to an I_ext tensor [N]
    with non-zero values only at photoreceptor indices.

    Parameters
    ----------
    photo_idx : int array — neuron indices of R1–R6 photoreceptors
    soma_coords : float array [N, 3] — (x, y, z) soma positions in voxels
    device : torch.device
    """

    def __init__(
        self,
        photo_idx:   np.ndarray,
        soma_coords: np.ndarray,    # [N, 3]
        N:           int,
        device:      torch.device = CFG.DEVICE,
    ) -> None:
        self.photo_idx   = photo_idx
        self.N           = N
        self.device      = device
        self.n_photo     = len(photo_idx)

        # Build retinotopic map: photoreceptor index → (az, el) in degrees
        # using soma coordinates projected onto (x, z) plane.
        # (In MaleCNS, the optic lobe is lateral; x ~ azimuth, z ~ elevation.)
        self._az, self._el = self._build_retinotopic_map(
            soma_coords[photo_idx], CFG.VISUAL_AZ_RANGE, CFG.VISUAL_EL_RANGE
        )

        # Pre-allocate reusable I_ext tensor (zero-fill each step)
        self._I_ext = torch.zeros(N, dtype=torch.float32, device=device)

    # ── Map building ─────────────────────────────────────────────────────

    @staticmethod
    def _build_retinotopic_map(
        coords:   np.ndarray,       # [n_photo, 3]
        az_range: tuple,
        el_range: tuple,
    ) -> tuple:
        """
        Project soma (x, z) positions linearly onto visual field angles.
        Returns (az, el) arrays in degrees, both shape [n_photo].
        """
        xs = coords[:, 0].astype(np.float32)
        zs = coords[:, 2].astype(np.float32)

        # Normalise to [0, 1]
        xs = _norm01(xs)
        zs = _norm01(zs)

        # Map to angular range
        az_lo, az_hi = az_range
        el_lo, el_hi = el_range
        az = az_lo + xs * (az_hi - az_lo)   # [-60°, +60°]
        el = el_lo + zs * (el_hi - el_lo)   # [-40°, +40°]

        return az, el

    # ── Encoding ─────────────────────────────────────────────────────────

    def encode(
        self,
        frame_gray: np.ndarray,          # [H, W] float32 in [0, 1]
        gain: float = 2.0,
    ) -> torch.Tensor:
        """
        Map game frame luminance to photoreceptor input currents.

        1. For each photoreceptor at (az_i, el_i), find the corresponding
           pixel in the frontal visual-field patch of the frame.
        2. Set I_ext[photo_i] = luminance × gain.

        Parameters
        ----------
        frame_gray : [H, W] numpy float32, values in [0, 1]
        gain       : scale factor; keeps photoreceptor currents in a range
                     that drives spiking. Tune alongside WEIGHT_SCALE.

        Returns
        -------
        I_ext : torch.Tensor [N] on self.device
        """
        H, W = frame_gray.shape
        az_lo, az_hi = CFG.VISUAL_AZ_RANGE
        el_lo, el_hi = CFG.VISUAL_EL_RANGE

        # Map angles → pixel indices
        px = ((self._az - az_lo) / (az_hi - az_lo) * (W - 1)).astype(np.int32)
        py = ((self._el - el_lo) / (el_hi - el_lo) * (H - 1)).astype(np.int32)

        # Clamp to valid pixel range
        px = np.clip(px, 0, W - 1)
        py = np.clip(py, 0, H - 1)

        # Sample luminance (contrast-enhanced: subtract mean)
        luminance = frame_gray[py, px]                     # [n_photo]
        luminance = luminance - luminance.mean()            # centre
        luminance = np.clip(luminance * gain, -1.0, 1.0)   # clip

        # Zero out the I_ext buffer, then fill photoreceptor slots
        self._I_ext.zero_()
        photo_t = torch.from_numpy(self.photo_idx).long().to(self.device)
        lum_t   = torch.from_numpy(luminance.astype(np.float32)).to(self.device)
        self._I_ext.scatter_(0, photo_t, lum_t)

        return self._I_ext

    def encode_structured(
        self,
        bird_y:      float,    # bird vertical position, normalised [0, 1]
        gap_center:  float,    # gap centre vertical position, normalised [0, 1]
        pipe_dist:   float,    # pipe horizontal distance, normalised [0, 1]
        bird_vy:     float,    # normalised velocity
    ) -> torch.Tensor:
        """
        Alternative structured encoder: bypasses rendering, directly encodes
        the task-relevant features into the frontal visual field.

        Used when full frame rendering is slow or for interpretability studies.

        Biological basis: Dm3 edge-detecting neurons → TmY21 frontal input
        (Berg et al. p.5511). The gap edge creates a luminance boundary in
        the frontal field.
        """
        H, W = CFG.GAME_RENDER_H, CFG.GAME_RENDER_W

        # Build a minimal synthetic "visual scene"
        frame = np.zeros((H, W), dtype=np.float32)

        # Pipe columns (dark = obstacle)
        pipe_x_px = int((1.0 - pipe_dist) * W)    # pipe approaching from right
        gap_top_px    = int((gap_center - 0.15) * H)
        gap_bottom_px = int((gap_center + 0.15) * H)

        # High contrast on pipe edges (Dm3 edge detectors sensitive to these)
        frame[:, max(0, pipe_x_px):min(W, pipe_x_px+20)] = 0.0
        frame[:gap_top_px, max(0, pipe_x_px):min(W, pipe_x_px+20)] = -1.0
        frame[gap_bottom_px:, max(0, pipe_x_px):min(W, pipe_x_px+20)] = -1.0

        # Bird position (bright dot)
        by = int(bird_y * H)
        bx = W // 4
        frame[max(0, by-3):min(H, by+3), max(0, bx-3):min(W, bx+3)] = 1.0

        return self.encode(frame, gain=1.5)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _norm01(arr: np.ndarray) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-6:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)
