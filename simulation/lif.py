"""
simulation/lif.py — Vectorised Leaky Integrate-and-Fire engine on GPU/CPU.

LIF formulation (Dayan & Abbott 2001, Theoretical Neuroscience Ch.5):

    τ_m · dV/dt = -(V - E_L) + R_m · I(t)      [subthreshold]
    V → V_reset  when V ≥ V_thresh
    V = V_reset  for τ_ref ms after a spike

Synaptic current (current-based model; Brette & Gerstner 2005):
    g(t+dt) = g(t) · exp(-dt/τ_g)  +  W @ spikes(t)
    I_syn(t)= g(t)

Parameters follow Lappalainen et al. (2024) Nature 634, 1132-1140
connectome-constrained Drosophila model.
"""

import math
import torch
import numpy as np
import config as CFG


class LIFEngine:
    """
    Full-connectome LIF simulation for MaleCNS (166,700 neurons).

    All tensors live on CFG.DEVICE (CUDA on Kaggle T4 / MPS on M4 / CPU).
    The weight matrix W must be a torch.Tensor (sparse CSR or COO).

    Usage
    -----
    engine = LIFEngine(W, N)
    engine.reset()
    for frame in game:
        for _ in range(CFG.LIF_STEPS_PER_FRAME):
            spikes = engine.step(I_ext)      # I_ext shape [N]
    flap = motor.decode(engine.spike_history_sum)
    """

    def __init__(
        self,
        W: torch.Tensor,
        N: int,
        dt: float        = CFG.DT,
        tau_m: float     = CFG.TAU_M,
        tau_g: float     = CFG.TAU_G,
        V_rest: float    = CFG.V_REST,
        V_thresh: float  = CFG.V_THRESH,
        V_reset: float   = CFG.V_RESET,
        tau_ref: float   = CFG.TAU_REF,
        device: torch.device = CFG.DEVICE,
    ) -> None:
        self.W       = W
        self.N       = N
        self.dt      = dt
        self.tau_m   = tau_m
        self.tau_g   = tau_g
        self.V_thresh = V_thresh
        self.V_reset  = V_reset
        self.V_rest   = V_rest
        self.tau_ref  = tau_ref
        self.device   = device

        # Pre-compute decay factors (avoids repeated exp each step)
        self._alpha_m = math.exp(-dt / tau_m)   # membrane decay
        self._alpha_g = math.exp(-dt / tau_g)   # conductance decay

        self.reset()

    # ── State management ──────────────────────────────────────────────────

    def reset(self) -> None:
        """Initialise / re-initialise all state tensors."""
        d = self.device
        self.V   = torch.full((self.N,), self.V_rest, dtype=torch.float32, device=d)
        self.g   = torch.zeros(self.N, dtype=torch.float32, device=d)   # synaptic conductance
        self.ref = torch.zeros(self.N, dtype=torch.float32, device=d)   # refractory countdown [ms]
        # Spike history (bool) and running sum for motor decoding
        self.spikes      = torch.zeros(self.N, dtype=torch.bool, device=d)
        self.spike_history_sum = torch.zeros(self.N, dtype=torch.float32, device=d)
        self._step_count = 0

    def clear_spike_accumulator(self) -> None:
        """Reset the running spike sum (call at start of each game frame)."""
        self.spike_history_sum.zero_()
        self._step_count = 0

    # ── Core step ─────────────────────────────────────────────────────────

    def step(self, I_ext: torch.Tensor) -> torch.Tensor:
        """
        Advance the LIF network by one timestep dt.

        Parameters
        ----------
        I_ext : Tensor shape [N], float32 — external input current
                (from sensory encoder; units: normalised voltage/ms)

        Returns
        -------
        spikes : bool Tensor shape [N]
        """
        # ── 1. Synaptic input: I_syn = W @ spikes[t-1] ──────────────────
        # W is [N_post × N_pre]; spikes_prev is [N_pre]
        # Result: [N_post]  (dense)
        spikes_f = self.spikes.float()
        if self.W.is_sparse or self.W.layout == torch.sparse_csr:
            I_syn = torch.sparse.mm(self.W, spikes_f.unsqueeze(1)).squeeze(1)
        else:
            I_syn = torch.mv(self.W, spikes_f)

        # ── 2. Conductance update (exponential decay + spike input) ──────
        self.g = self.g * self._alpha_g + I_syn

        # ── 3. Membrane voltage update (Euler integration of LIF ODE) ───
        # dV = (-V + V_rest + g + I_ext) * dt/tau_m
        # Exact exponential integration:
        # V[t+dt] = V[t] * alpha_m + (V_rest + g + I_ext)*(1 - alpha_m)
        resting_input = self.V_rest + self.g + I_ext
        V_new = self.V * self._alpha_m + resting_input * (1.0 - self._alpha_m)

        # ── 4. Refractory mask: clamp V to V_reset during refractory ────
        in_refractory = self.ref > 0.0
        V_new = torch.where(in_refractory, torch.tensor(self.V_reset, device=self.device), V_new)

        # ── 5. Threshold crossing → spike ───────────────────────────────
        spiked = (V_new >= self.V_thresh) & (~in_refractory)

        # ── 6. Reset spiking neurons ─────────────────────────────────────
        V_new = torch.where(spiked, torch.tensor(self.V_reset, device=self.device), V_new)

        # ── 7. Update refractory countdown ──────────────────────────────
        self.ref = torch.clamp(self.ref - self.dt, min=0.0)
        self.ref = torch.where(spiked, torch.tensor(self.tau_ref, device=self.device), self.ref)

        # ── 8. Store state ───────────────────────────────────────────────
        self.V      = V_new
        self.spikes = spiked
        self.spike_history_sum += spiked.float()
        self._step_count += 1

        return spiked

    # ── Utility ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def spike_rates(self, pop_idx: np.ndarray) -> np.ndarray:
        """
        Mean spike rate (spikes/step, over accumulated history)
        for a population subset. Returns CPU numpy array.
        """
        if self._step_count == 0:
            return np.zeros(len(pop_idx), dtype=np.float32)
        idx_t = torch.from_numpy(pop_idx).long().to(self.device)
        rates = self.spike_history_sum[idx_t] / self._step_count
        return rates.cpu().numpy()

    @torch.no_grad()
    def all_spike_rates(self) -> np.ndarray:
        """Full N-length spike rate array (for visualisation)."""
        if self._step_count == 0:
            return np.zeros(self.N, dtype=np.float32)
        return (self.spike_history_sum / self._step_count).cpu().numpy()

    @torch.no_grad()
    def get_state_snapshot(self) -> dict:
        """Return a CPU dict snapshot for visualisation / checkpointing."""
        return {
            "V":      self.V.cpu().numpy(),
            "spikes": self.spikes.cpu().numpy(),
            "step":   self._step_count,
        }
