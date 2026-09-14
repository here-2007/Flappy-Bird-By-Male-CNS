"""
data/mock_data.py — Offline synthetic connectome generator for FlyFlappyBird.

Generates a biologically grounded mock connectome (N=600 neurons, ~4,000 synapses)
following the MaleCNS schema (Berg et al. 2026 Cell 189, 5504-5526).
Enables zero-dependency local testing and CI/CD without requiring neuPrint credentials.
"""

from typing import Optional, Union, Tuple
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

import config as CFG
from connectome.populations import CircuitPopulations, identify_populations, get_soma_coords


def generate_mock_connectome(
    n_neurons: int = 600,
    seed: int = 42,
) -> Tuple[pd.DataFrame, sp.csr_matrix, np.ndarray]:
    """
    Generate a synthetic Drosophila connectome conforming to the male-cns:v1.0 schema.

    Parameters
    ----------
    n_neurons : int
        Total number of neurons to generate (must be >= 500; defaults to 600).
    seed : int
        Random seed for reproducible topology and coordinate generation.

    Returns
    -------
    neuron_df : pd.DataFrame
        DataFrame with columns matching neuPrint MaleCNS schema:
        ['neuron_idx', 'bodyId', 'type', 'superclass', 'class', 'subclass',
         'dimorphism', 'fru_dsx', 'predictedNt', 'soma_x', 'soma_y', 'soma_z'].
    W_raw : sp.csr_matrix
        Raw synapse count connectivity matrix [N, N] with ~4,000 non-zero edges.
        Convention: W[post, pre] = synapse_count.
    coords : np.ndarray
        Soma (x, y, z) coordinates of shape [N, 3] in MaleCNS voxel space.
    """
    if n_neurons < 500:
        raise ValueError(f"n_neurons must be >= 500 to maintain biological circuits, got {n_neurons}")

    rng = np.random.default_rng(seed)

    # ── 1. Determine population partition ────────────────────────────────────
    if n_neurons == 600:
        n_photo = 120
        n_vpn   = 60
        n_dn    = 40
        n_motor = 40
        n_ppl   = 20
        n_kc    = 80
        n_inter = 240
    else:
        n_photo = int(round(n_neurons * 0.20))
        n_vpn   = int(round(n_neurons * 0.10))
        n_dn    = int(round(n_neurons * 0.0667))
        n_motor = int(round(n_neurons * 0.0667))
        n_ppl   = int(round(n_neurons * 0.0333))
        n_kc    = int(round(n_neurons * 0.1333))
        n_inter = n_neurons - (n_photo + n_vpn + n_dn + n_motor + n_ppl + n_kc)

    records = []
    coords_list = []

    # Helper to append neuron
    def add_neuron(idx, type_name, superclass, class_name, subclass, dimorphism, fru_dsx, predicted_nt, x, y, z):
        records.append({
            "neuron_idx": np.int32(idx),
            "bodyId": np.int64(100000 + idx),
            "type": str(type_name),
            "superclass": str(superclass),
            "class": str(class_name),
            "subclass": str(subclass),
            "dimorphism": str(dimorphism),
            "fru_dsx": str(fru_dsx),
            "predictedNt": str(predicted_nt),
            "soma_x": np.float32(x),
            "soma_y": np.float32(y),
            "soma_z": np.float32(z),
        })
        coords_list.append([x, y, z])

    curr_idx = 0

    # ── 2. Populate Neurons ───────────────────────────────────────────────────

    # (a) R1-R6 Photoreceptors (120 neurons)
    # Retinotopic 2D grid in (x, z) for sensory encoder mapping
    n_cols = 12
    n_rows = (n_photo + n_cols - 1) // n_cols
    photo_start = curr_idx
    for i in range(n_photo):
        c = i % n_cols
        r = i // n_cols
        x = 100.0 + c * (275.0 / max(n_cols - 1, 1))
        z = 80.0 + r * (225.0 / max(n_rows - 1, 1))
        y = 200.0 + float(rng.uniform(-5.0, 5.0))
        r_type = f"R{(i % 6) + 1}"
        add_neuron(
            idx=curr_idx,
            type_name=r_type,
            superclass="ol-sensory",
            class_name="sensory",
            subclass="photoreceptor",
            dimorphism="isomorphic",
            fru_dsx="none",
            predicted_nt="acetylcholine",
            x=x, y=y, z=z,
        )
        curr_idx += 1
    photo_end = curr_idx

    # (b) Visual Projection Neurons (60 neurons)
    # LoVP92 (male-specific), TmY21 (dimorphic), and frontal isomorphic types (LT84, LLPC4, LT86, MeVP48)
    vpn_start = curr_idx
    vpn_subtypes = [
        ("LoVP92", "male-specific", "fru+", "anterior_lobula"),
        ("TmY21",  "dimorphic",     "dsx+", "optic_lobe"),
        ("LT84",   "isomorphic",    "none", "anterior_lobula"),
        ("LLPC4",  "isomorphic",    "none", "anterior_lobula"),
        ("LT86",   "isomorphic",    "none", "anterior_lobula"),
        ("MeVP48", "isomorphic",    "none", "anterior_lobula"),
    ]
    per_vpn = n_vpn // len(vpn_subtypes)
    for s_idx, (t_name, dim, fd, subc) in enumerate(vpn_subtypes):
        count = per_vpn if s_idx < len(vpn_subtypes) - 1 else (n_vpn - per_vpn * (len(vpn_subtypes) - 1))
        for _ in range(count):
            x = float(rng.uniform(80.0, 420.0))
            y = float(rng.uniform(150.0, 350.0))
            z = float(rng.uniform(100.0, 320.0))
            add_neuron(
                idx=curr_idx,
                type_name=t_name,
                superclass="visual_projection",
                class_name="VPN",
                subclass=subc,
                dimorphism=dim,
                fru_dsx=fd,
                predicted_nt="acetylcholine",
                x=x, y=y, z=z,
            )
            curr_idx += 1
    vpn_end = curr_idx

    # (c) Descending Neurons (40 neurons)
    # DNg13 (steering), DNa02 (pursuit), aSP22 (courtship, male-specific), DNp01
    dn_start = curr_idx
    dn_subtypes = [
        ("DNg13", "isomorphic",    "none", "steering"),
        ("DNa02", "isomorphic",    "none", "pursuit"),
        ("aSP22", "male-specific", "fru+", "courtship"),
        ("DNp01", "isomorphic",    "none", "general"),
    ]
    per_dn = n_dn // len(dn_subtypes)
    for s_idx, (t_name, dim, fd, subc) in enumerate(dn_subtypes):
        count = per_dn if s_idx < len(dn_subtypes) - 1 else (n_dn - per_dn * (len(dn_subtypes) - 1))
        for _ in range(count):
            x = float(rng.uniform(150.0, 350.0))
            y = float(rng.uniform(100.0, 250.0))
            z = float(rng.uniform(50.0, 200.0))
            add_neuron(
                idx=curr_idx,
                type_name=t_name,
                superclass="descending",
                class_name="descending",
                subclass=subc,
                dimorphism=dim,
                fru_dsx=fd,
                predicted_nt="acetylcholine",
                x=x, y=y, z=z,
            )
            curr_idx += 1
    dn_end = curr_idx

    # (d) Wing Motor Neurons (40 neurons)
    # TN1A (15), vPR9 (15), dPR1 (10)
    motor_start = curr_idx
    motor_subtypes = [
        ("TN1A", int(round(n_motor * 0.375))),
        ("vPR9", int(round(n_motor * 0.375))),
    ]
    motor_subtypes.append(("dPR1", n_motor - sum(c for _, c in motor_subtypes)))
    for t_name, count in motor_subtypes:
        for _ in range(count):
            x = float(rng.uniform(180.0, 320.0))
            y = float(rng.uniform(50.0, 180.0))
            z = float(rng.uniform(50.0, 150.0))
            add_neuron(
                idx=curr_idx,
                type_name=t_name,
                superclass="vnc-motor",
                class_name="motor",
                subclass="wing_pre_motor",
                dimorphism="isomorphic",
                fru_dsx="none",
                predicted_nt="acetylcholine",
                x=x, y=y, z=z,
            )
            curr_idx += 1
    motor_end = curr_idx

    # (e) PPL101 Dopamine Neurons (20 neurons)
    ppl_start = curr_idx
    for _ in range(n_ppl):
        x = float(rng.uniform(160.0, 340.0))
        y = float(rng.uniform(200.0, 350.0))
        z = float(rng.uniform(150.0, 300.0))
        add_neuron(
            idx=curr_idx,
            type_name="PPL101",
            superclass="cb-modulatory",
            class_name="modulatory",
            subclass="dopaminergic",
            dimorphism="isomorphic",
            fru_dsx="none",
            predicted_nt="dopamine",
            x=x, y=y, z=z,
        )
        curr_idx += 1
    ppl_end = curr_idx

    # (f) Kenyon Cells (80 neurons)
    kc_start = curr_idx
    for _ in range(n_kc):
        x = float(rng.uniform(120.0, 380.0))
        y = float(rng.uniform(220.0, 380.0))
        z = float(rng.uniform(180.0, 340.0))
        add_neuron(
            idx=curr_idx,
            type_name="KCa",
            superclass="cb-intrinsic",
            class_name="Kenyon_cell",
            subclass="alpha_beta",
            dimorphism="isomorphic",
            fru_dsx="none",
            predicted_nt="acetylcholine",
            x=x, y=y, z=z,
        )
        curr_idx += 1
    kc_end = curr_idx

    # (g) Central Interneurons (240 neurons)
    # Balanced NTs: 50% ACh, 25% GABA, 25% glutamate
    inter_start = curr_idx
    n_ach = n_inter // 2
    n_gaba = n_inter // 4
    n_glu = n_inter - n_ach - n_gaba
    inter_nts = ["acetylcholine"] * n_ach + ["GABA"] * n_gaba + ["glutamate"] * n_glu

    for nt in inter_nts:
        x = float(rng.uniform(100.0, 400.0))
        y = float(rng.uniform(150.0, 380.0))
        z = float(rng.uniform(100.0, 320.0))
        add_neuron(
            idx=curr_idx,
            type_name="CB_IN",
            superclass="cb-intrinsic",
            class_name="interneuron",
            subclass="central",
            dimorphism="isomorphic",
            fru_dsx="none",
            predicted_nt=nt,
            x=x, y=y, z=z,
        )
        curr_idx += 1
    inter_end = curr_idx

    assert curr_idx == n_neurons, f"Expected {n_neurons} neurons, created {curr_idx}"

    neuron_df = pd.DataFrame(records)
    coords = np.array(coords_list, dtype=np.float32)

    # ── 3. Construct Functional Connectivity Graph ────────────────────────────
    # W[post, pre] = synapse_count
    edges: dict[Tuple[int, int], float] = {}

    photo_indices = np.arange(photo_start, photo_end)
    vpn_indices   = np.arange(vpn_start, vpn_end)
    dn_indices    = np.arange(dn_start, dn_end)
    motor_indices = np.arange(motor_start, motor_end)
    ppl_indices   = np.arange(ppl_start, ppl_end)
    kc_indices    = np.arange(kc_start, kc_end)
    inter_indices = np.arange(inter_start, inter_end)
    inter_ach_indices = np.arange(inter_start, inter_start + n_ach)

    # Pathway 1: R1-R6 -> VPN (~420 edges)
    # Each VPN samples ~7 photoreceptors
    k_photo_per_vpn = min(len(photo_indices), 7)
    for vpn in vpn_indices:
        pre_photos = rng.choice(photo_indices, size=k_photo_per_vpn, replace=False)
        for pre in pre_photos:
            edges[(vpn, pre)] = float(rng.integers(15, 31))

    # Pathway 2: VPN -> Interneurons + Kenyon Cells (~600 edges)
    # Target pool: KCs and cholinergic interneurons
    vpn_target_pool = np.concatenate([kc_indices, inter_ach_indices])
    k_targets_per_vpn = min(len(vpn_target_pool), 10)
    for vpn in vpn_indices:
        posts = rng.choice(vpn_target_pool, size=k_targets_per_vpn, replace=False)
        for post in posts:
            edges[(post, vpn)] = float(rng.integers(12, 26))

    # Pathway 3: Interneurons + KC -> Descending Neurons (~600 edges)
    # Each DN receives from ~15 interneurons / KCs
    dn_source_pool = np.concatenate([inter_ach_indices, kc_indices])
    k_sources_per_dn = min(len(dn_source_pool), 15)
    for dn in dn_indices:
        pres = rng.choice(dn_source_pool, size=k_sources_per_dn, replace=False)
        for pre in pres:
            edges[(dn, pre)] = float(rng.integers(12, 26))

    # Pathway 4: Descending Neurons -> Wing Motor (~400 edges)
    # Each Wing Motor neuron receives from ~10 DNs
    k_dns_per_motor = min(len(dn_indices), 10)
    for motor in motor_indices:
        pres = rng.choice(dn_indices, size=k_dns_per_motor, replace=False)
        for pre in pres:
            edges[(motor, pre)] = float(rng.integers(15, 36))

    # Pathway 5: Plasticity Circuit (~500 edges)
    # PPL101 -> KC (300 edges)
    k_kc_per_ppl = min(len(kc_indices), 15)
    for ppl in ppl_indices:
        kcs = rng.choice(kc_indices, size=k_kc_per_ppl, replace=False)
        for kc in kcs:
            edges[(kc, ppl)] = float(rng.integers(10, 21))

    # KC -> PPL101 (160 edges)
    k_kc_in_per_ppl = min(len(kc_indices), 8)
    for ppl in ppl_indices:
        kcs = rng.choice(kc_indices, size=k_kc_in_per_ppl, replace=False)
        for kc in kcs:
            edges[(ppl, kc)] = float(rng.integers(10, 21))

    # PPL101 -> Interneurons (40 edges)
    k_in_per_ppl = min(len(inter_ach_indices), 2)
    for ppl in ppl_indices:
        ins = rng.choice(inter_ach_indices, size=k_in_per_ppl, replace=False)
        for in_n in ins:
            edges[(in_n, ppl)] = float(rng.integers(10, 21))

    # Pathway 6: Recurrent Background (~1,500 edges)
    # Dense recurrent interactions among central interneurons
    target_total = 4020
    needed_recurrent = max(1500, target_total - len(edges))
    recurrent_added = 0
    attempts = 0
    max_attempts = needed_recurrent * 10
    while recurrent_added < needed_recurrent and attempts < max_attempts:
        attempts += 1
        u, v = rng.choice(inter_indices, size=2, replace=False)
        if (u, v) not in edges:
            edges[(u, v)] = float(rng.integers(5, 16))
            recurrent_added += 1

    # Convert edge dict to SciPy CSR matrix
    row_arr = np.array([k[0] for k in edges.keys()], dtype=np.int32)
    col_arr = np.array([k[1] for k in edges.keys()], dtype=np.int32)
    data_arr = np.array(list(edges.values()), dtype=np.float32)

    W_raw = sp.csr_matrix(
        (data_arr, (row_arr, col_arr)),
        shape=(n_neurons, n_neurons),
        dtype=np.float32,
    )

    return neuron_df, W_raw, coords


def get_or_create_mock_connectome(
    device: Optional[Union[str, torch.device]] = None,
    seed: int = 42,
) -> Tuple[torch.Tensor, sp.csr_matrix, pd.DataFrame, CircuitPopulations, np.ndarray]:
    """
    Wrap generate_mock_connectome, apply neurotransmitter signs and scaling,
    and return the simulation-ready tensors and metadata.

    Returns
    -------
    W_t : torch.Tensor
        Sparse weight tensor on target device with Dale's principle and scaling applied.
    W_scipy : sp.csr_matrix
        CPU SciPy CSR matrix of signed and scaled weights for in-place plasticity updates.
    neuron_df : pd.DataFrame
        Neuron metadata DataFrame.
    pop : CircuitPopulations
        Biological population index arrays.
    coords : np.ndarray
        Soma coordinates [N, 3].
    """
    dev = CFG.resolve_device(device)

    # 1. Generate synthetic data
    neuron_df, W_raw, coords = generate_mock_connectome(n_neurons=600, seed=seed)

    # 2. NT sign correction & Dale's principle
    # Import builder helpers lazily to avoid circular dependencies
    from connectome.builder import _nt_sign_vector, _scipy_to_torch_sparse

    nt_sign = _nt_sign_vector(neuron_df)
    W_coo = W_raw.tocoo().astype(np.float32)
    W_coo.data *= nt_sign[W_coo.col]
    W_coo.data *= CFG.WEIGHT_SCALE
    np.clip(W_coo.data, CFG.W_MIN, CFG.W_MAX, out=W_coo.data)
    W_signed = W_coo.tocsr()

    # 3. Move to target device
    W_t = _scipy_to_torch_sparse(W_signed, device=dev)

    # 4. Identify circuit populations
    pop = identify_populations(neuron_df)

    return W_t, W_signed, neuron_df, pop, coords
