"""
connectome/builder.py — Load MaleCNS cache and build a GPU-resident
sparse weight tensor with neurotransmitter sign correction.

Synaptic weight sign follows Berg et al. (2026) p.5505:
"Feedforward connections share similar, predominantly excitatory
neurotransmitter compositions, while feedback connections … are more
inhibitory."  Concretely: ACh → +, GABA/Glu → −.
"""

import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from typing import Tuple

import config as CFG
from connectome.populations import CircuitPopulations, identify_populations, get_soma_coords


def _nt_sign_vector(neuron_df: pd.DataFrame) -> np.ndarray:
    """
    Build a per-neuron sign vector (+1 excitatory / -1 inhibitory)
    based on predictedNt. Used to flip weight signs for inhibitory neurons.

    Berg et al. p.5505: neurotransmitter predictions from Eckstein et al.
    (2024) Cell 187, 2574-2594.
    """
    nt = neuron_df["predictedNt"].fillna("unknown").str.lower()
    signs = np.full(len(neuron_df), CFG.NT_SIGN["unknown"], dtype=np.float32)
    for nt_name, sign in CFG.NT_SIGN.items():
        signs[nt.str.contains(nt_name.lower(), regex=False)] = sign
    return signs


def _scipy_to_torch_sparse(
    W_scipy: sp.csr_matrix,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Convert scipy CSR matrix → torch.sparse_csr_tensor on the target device.
    Falls back to COO format if CSR is not supported (MPS).
    """
    W = W_scipy.tocsr().astype(np.float32)

    crow = torch.from_numpy(W.indptr.astype(np.int32))
    col  = torch.from_numpy(W.indices.astype(np.int32))
    val  = torch.from_numpy(W.data)

    try:
        # Preferred: CSR is fastest for SpMV on CUDA
        W_t = torch.sparse_csr_tensor(crow, col, val, size=W.shape, dtype=dtype)
        return W_t.to(device)
    except (RuntimeError, NotImplementedError):
        pass

    try:
        # Fallback: COO (works on MPS and CPU)
        W_coo = W_scipy.tocoo()
        i = torch.from_numpy(np.vstack([W_coo.row, W_coo.col]).astype(np.int64))
        v = torch.from_numpy(W_coo.data.astype(np.float32))
        return torch.sparse_coo_tensor(i, v, W.shape, dtype=dtype).to(device)
    except (RuntimeError, NotImplementedError):
        # Final fallback: Keep on CPU
        return torch.sparse_coo_tensor(i, v, W.shape, dtype=dtype).to("cpu")


def load_connectome(
    neuron_csv: str = CFG.NEURON_CSV,
    conn_npz:   str = CFG.CONN_NPZ,
    device:     torch.device = CFG.DEVICE,
) -> Tuple[torch.Tensor, pd.DataFrame, CircuitPopulations, np.ndarray]:
    """
    Load the cached MaleCNS data and return:
      W      — sparse weight matrix [N×N] on device  (float32)
      df     — neuron metadata DataFrame
      pop    — CircuitPopulations with index arrays
      coords — N×3 soma coordinate array (for visualisation)

    Weight matrix convention (after sign correction):
      W[post, pre] = signed_weight  →  I_syn = W @ spikes

    Weights are:
      - raw synapse count × WEIGHT_SCALE × nt_sign(pre)
      - clipped to [W_MIN, W_MAX]
    """
    if not os.path.exists(neuron_csv):
        raise FileNotFoundError(
            f"{neuron_csv} not found.\n"
            "Run:  python -m data.fetch_data --token YOUR_TOKEN"
        )
    if not os.path.exists(conn_npz):
        raise FileNotFoundError(
            f"{conn_npz} not found.\n"
            "Run:  python -m data.fetch_data --token YOUR_TOKEN"
        )

    # ── 1. Neuron metadata ─────────────────────────────────────────────────
    print("[build] Loading neuron metadata …")
    df = pd.read_csv(neuron_csv, low_memory=False)
    N  = len(df)
    print(f"        {N:,} neurons")

    # Per-neuron NT sign (shape: N)
    nt_sign = _nt_sign_vector(df)     # +1 or -1 per neuron

    # ── 2. Sparse weight matrix ────────────────────────────────────────────
    print("[build] Loading sparse connectivity matrix …")
    W_raw = sp.load_npz(conn_npz)     # shape [N, N], values = synapse counts
    assert W_raw.shape == (N, N), (
        f"Shape mismatch: matrix {W_raw.shape} vs {N} neurons"
    )
    print(f"        shape={W_raw.shape}, nnz={W_raw.nnz:,}")

    # ── 3. Apply NT sign & scale ───────────────────────────────────────────
    # Each column (presynaptic neuron j) gets multiplied by nt_sign[j].
    # This converts raw synapse counts to signed, scaled currents.
    print("[build] Applying neurotransmitter sign correction …")
    W_coo = W_raw.tocoo().astype(np.float32)
    W_coo.data *= nt_sign[W_coo.col]          # sign from pre-neuron
    W_coo.data *= CFG.WEIGHT_SCALE            # global scale
    np.clip(W_coo.data, CFG.W_MIN, CFG.W_MAX, out=W_coo.data)
    W_signed = W_coo.tocsr()

    # ── 4. Move to device ─────────────────────────────────────────────────
    print(f"[build] Moving sparse matrix to {device} …")
    W_t = _scipy_to_torch_sparse(W_signed, device=device)

    # ── 5. Identify populations ────────────────────────────────────────────
    print("[build] Identifying circuit populations …")
    pop = identify_populations(df)

    # ── 6. Soma coordinates ────────────────────────────────────────────────
    coords = get_soma_coords(df)

    print("[build] Connectome ready.")
    return W_t, df, pop, coords
