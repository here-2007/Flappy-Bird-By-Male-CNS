"""
simulation/motor.py — Decode wing pre-motor neuron activity into a
binary flap decision for the Flappy Bird game.

Motor population (Berg et al. 2026, p.5512):
  "key wing pre-motor neurons in the VNC (TN1A, vPR9, and dPR1)"

Decision rule: if the fraction of wing pre-motor neurons spiking
within the current frame window exceeds FLAP_THRESHOLD → flap.
"""

import numpy as np
import torch
import config as CFG


class MotorDecoder:
    """
    Reads spike rates from TN1A, vPR9, dPR1 populations and
    outputs a boolean flap decision.

    Parameters
    ----------
    wing_motor_idx : int array — neuron indices of TN1A + vPR9 + dPR1
    N              : total neuron count
    device         : torch.device
    threshold      : fraction of wing MNs spiking → flap (default CFG.FLAP_THRESHOLD)
    """

    def __init__(
        self,
        wing_motor_idx: np.ndarray,
        N:              int,
        device:         torch.device = CFG.DEVICE,
        threshold:      float        = CFG.FLAP_THRESHOLD,
    ) -> None:
        self.threshold = threshold
        self.n_motor   = len(wing_motor_idx)
        self._device   = device

        if self.n_motor == 0:
            print(
                "[MotorDecoder] WARNING: no wing motor neurons found. "
                "Flap decision will always be False.\n"
                "  Check neuron types TN1A, vPR9, dPR1 in the MaleCNS data."
            )
            self._motor_t = None
        else:
            self._motor_t = (
                torch.from_numpy(wing_motor_idx.astype(np.int64))
                .to(device)
            )

        # Running stats for logging
        self._recent_rates: list[float] = []

    def decode(self, spike_history_sum: torch.Tensor, n_steps: int) -> bool:
        """
        Parameters
        ----------
        spike_history_sum : Tensor [N] — cumulative spike count over last
                            n_steps LIF steps (engine.spike_history_sum)
        n_steps           : number of LIF steps in this frame window

        Returns
        -------
        flap : bool
        """
        if self._motor_t is None or n_steps == 0:
            return False

        # Mean spike rate (spikes/step) for wing motor neurons
        motor_counts = spike_history_sum[self._motor_t]
        mean_rate    = (motor_counts.float().mean() / n_steps).item()

        self._recent_rates.append(mean_rate)
        if len(self._recent_rates) > 100:
            self._recent_rates.pop(0)

        return mean_rate >= self.threshold

    def recent_mean_rate(self) -> float:
        """Smoothed wing motor rate — used for dashboard display."""
        if not self._recent_rates:
            return 0.0
        return float(np.mean(self._recent_rates[-20:]))

    def calibrate_threshold(self, observed_rates: list[float]) -> None:
        """
        Auto-calibrate threshold to 50th percentile of observed rates.
        Call this after a few random-input warm-up episodes.
        """
        if not observed_rates:
            return
        new_thresh = float(np.percentile(observed_rates, 50))
        print(f"[MotorDecoder] Calibrated threshold: {self.threshold:.4f} → {new_thresh:.4f}")
        self.threshold = new_thresh
