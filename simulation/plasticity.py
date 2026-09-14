"""
simulation/plasticity.py — PPL101 dopamine-gated reward-modulated
Hebbian plasticity.

Biological basis:
  • Berg et al. (2026) p.5519: dopamine-releasing PPL101 neurons project
    to specific mushroom body (MB) compartments.
  • Aso et al. (2014) eLife 3:e04577: PPL101 drives aversive learning
    in MB; opposite compartments encode reward.
  • DoomFly (Wormuth 2024): positive/negative DA pulses to PPL101 as
    reward/punishment signal.
  • R-STDP rule: Fremaux & Gerstner (2016) Front. Neural Circuits 10:1.

Rule:
    e_ij(t+dt) = e_ij(t) · exp(-dt/τ_e) + s_pre_i(t) · s_post_j(t)
    dW_ij = lr · DA(t) · e_ij(t)

where:
    e_ij   = eligibility trace for synapse i→j
    DA(t)  = dopamine signal from PPL101 (± reward/punishment)
    lr     = CFG.LEARNING_RATE

Plasticity is restricted to the PPL101 neighbourhood (synapses
touching PPL101 or Kenyon cells) — a biologically motivated subset.
This is tractable: Kenyon cell connectivity is ~0.5% of total edges.
"""

import numpy as np
import torch
import scipy.sparse as sp
import config as CFG


class PPL101Plasticity:
    """
    Reward-modulated Hebbian learning gated by PPL101 dopamine signal.

    Parameters
    ----------
    W              : torch sparse Tensor [N, N] — the live weight matrix
                     (this object holds a REFERENCE; updates modify it in-place
                     via the raw scipy CSR stored separately)
    W_scipy        : scipy CSR matrix — used for efficient in-place updates
    ppl101_idx     : int array — PPL101 neuron indices
    kc_idx         : int array — Kenyon cell indices
    all_edges      : tuple (rows, cols, data) from W_scipy — all edge indices
    N              : total neuron count
    device         : torch.device
    """

    def __init__(
        self,
        W_scipy:    sp.csr_matrix,
        ppl101_idx: np.ndarray,
        kc_idx:     np.ndarray,
        N:          int,
        device:     torch.device = CFG.DEVICE,
        lr:         float        = CFG.LEARNING_RATE,
        tau_e:      float        = CFG.TAU_ELIGIBILITY,
        dt:         float        = CFG.DT,
    ) -> None:
        self.N      = N
        self.device = device
        self.lr     = lr
        self.dt     = dt
        self._alpha_e = np.exp(-dt / tau_e)

        # ── Identify plasticity-eligible synapses ────────────────────────
        # We apply R-STDP only to connections into/out of Kenyon cells
        # and PPL101 neurons. This is a biologically grounded subset.
        self._plastic_mask, self._plastic_rows, self._plastic_cols = (
            self._find_plastic_synapses(W_scipy, ppl101_idx, kc_idx)
        )
        n_plastic = self._plastic_mask.sum()
        print(f"[plasticity] Plastic synapses (PPL101 + KC neighbourhood): {n_plastic:,}")

        if n_plastic == 0:
            print("[plasticity] WARNING: no plastic synapses found. Learning disabled.")
            self._enabled = False
            return
        self._enabled = True

        # ── Eligibility traces (CPU numpy — updated each step) ──────────
        self._e = np.zeros(n_plastic, dtype=np.float32)

        # ── Sparse weight data (numpy view of W_scipy.data) ─────────────
        # We accumulate dW here and push to W_scipy.data periodically.
        self._W_scipy = W_scipy
        self._dW      = np.zeros(n_plastic, dtype=np.float32)
        self._W_data_idx = self._plastic_mask.nonzero()[0]  # into W_scipy.data

        # Row/col arrays for eligibility product computation
        self._rows_cpu = self._plastic_rows   # post-synaptic neuron index
        self._cols_cpu = self._plastic_cols   # pre-synaptic neuron index

        # DA state
        self._da_signal = 0.0

        # Episode stats
        self.episode_dw_norm: list[float] = []

    # ── Per-step update ──────────────────────────────────────────────────

    def step(
        self,
        spikes: torch.Tensor,    # bool [N] current step
        da_signal: float,        # dopamine from PPL101 this step
    ) -> None:
        """
        Update eligibility traces and accumulate weight delta.
        Call once per LIF step.
        """
        if not self._enabled:
            return

        self._da_signal = da_signal

        # Get spike values for pre and post neurons (CPU, numpy)
        spikes_np = spikes.cpu().numpy().astype(np.float32)

        pre_s  = spikes_np[self._cols_cpu]   # [n_plastic]
        post_s = spikes_np[self._rows_cpu]   # [n_plastic]

        # Eligibility trace update: e ← e * decay + pre * post
        self._e *= self._alpha_e
        self._e += pre_s * post_s

        # Accumulate weight delta: dW ← lr * DA * e
        self._dW += self.lr * da_signal * self._e

    # ── Episode-end weight application ───────────────────────────────────

    def apply_updates(self, W_t: torch.Tensor) -> torch.Tensor:
        """
        Apply accumulated dW to the weight matrix and return updated W.
        Call at end of each episode.

        Also re-broadcasts the updated scipy CSR → new sparse torch tensor
        so the LIF engine uses the updated weights next episode.

        Parameters
        ----------
        W_t : current torch sparse tensor (the one used by LIFEngine)

        Returns
        -------
        W_new : updated torch sparse tensor (same sparsity pattern)
        """
        if not self._enabled:
            return W_t

        dw_norm = float(np.abs(self._dW).mean())
        self.episode_dw_norm.append(dw_norm)

        # Apply and clamp
        self._W_scipy.data[self._W_data_idx] += self._dW
        np.clip(
            self._W_scipy.data,
            CFG.W_MIN,
            CFG.W_MAX,
            out=self._W_scipy.data,
        )

        # Reset accumulators
        self._dW[:] = 0.0
        self._e[:]  = 0.0

        # Rebuild torch sparse tensor from updated scipy data
        W_new = _scipy_csr_to_torch_sparse(self._W_scipy, self.device)
        return W_new

    # ── DA signal helpers ────────────────────────────────────────────────

    @staticmethod
    def reward_alive() -> float:
        return CFG.REWARD_ALIVE

    @staticmethod
    def reward_pass_pipe() -> float:
        return CFG.REWARD_PASS_PIPE

    @staticmethod
    def punish_death() -> float:
        return CFG.PUNISH_DEATH

    # ── Private ──────────────────────────────────────────────────────────

    @staticmethod
    def _find_plastic_synapses(
        W: sp.csr_matrix,
        ppl101_idx: np.ndarray,
        kc_idx:     np.ndarray,
    ) -> tuple:
        """
        Return a boolean mask into W.data for synapses involving
        PPL101 or Kenyon cells (as pre or post).
        """
        W_coo   = W.tocoo()
        rows_np = W_coo.row.astype(np.int32)
        cols_np = W_coo.col.astype(np.int32)

        ppl101_set = set(ppl101_idx.tolist())
        kc_set     = set(kc_idx.tolist())
        relevant   = ppl101_set | kc_set

        mask = np.array([
            (r in relevant or c in relevant)
            for r, c in zip(rows_np, cols_np)
        ], dtype=bool)

        return mask, rows_np[mask], cols_np[mask]


# ─── Helper ──────────────────────────────────────────────────────────────────

def _scipy_csr_to_torch_sparse(
    W: sp.csr_matrix,
    device: torch.device,
) -> torch.Tensor:
    crow = torch.from_numpy(W.indptr.astype(np.int32))
    col  = torch.from_numpy(W.indices.astype(np.int32))
    val  = torch.from_numpy(W.data.astype(np.float32))
    try:
        t = torch.sparse_csr_tensor(crow, col, val, size=W.shape, dtype=torch.float32)
        return t.to(device)
    except (RuntimeError, NotImplementedError):
        W_coo = W.tocoo()
        i = torch.from_numpy(np.vstack([W_coo.row, W_coo.col]).astype(np.int64))
        v = torch.from_numpy(W_coo.data.astype(np.float32))
        return torch.sparse_coo_tensor(i, v, W.shape).to(device)
