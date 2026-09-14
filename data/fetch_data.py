"""
data/fetch_data.py — Download MaleCNS v1.0 neurons and synaptic connections
from neuPrint (https://neuprint.janelia.org).

Dataset: Berg et al. (2026), Cell 189, 5504-5526.
neuPrint Python client: https://github.com/connectome-neuprint/neuprint-python

Run this once before the simulation:
    python -m data.fetch_data --token YOUR_TOKEN
"""

import os
import argparse
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
from tqdm import tqdm

from neuprint import Client, NeuronCriteria as NC, fetch_neurons

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as CFG


# ─── Neuron metadata ────────────────────────────────────────────────────────

def fetch_all_neurons(client: Client) -> pd.DataFrame:
    """
    Fetch all traced neurons from MaleCNS v1.0 with:
      bodyId, type, superclass, class, subclass, soma coordinates,
      neurotransmitter prediction, dimorphism label, fru/dsx annotation.

    Returns DataFrame (~166,700 rows, Berg et al. Table 1).
    """
    print("[fetch] Querying all traced neurons …")

    # Standard properties via neuprint-python
    neuron_df, _ = fetch_neurons(NC(status="Traced"), client=client)

    # Augment with MaleCNS-specific annotations via custom Cypher
    # (dimorphism, fru_dsx, superclass, predictedNt are MaleCNS properties)
    print("[fetch] Fetching MaleCNS-specific annotations …")
    cypher = """
    MATCH (n:Neuron)
    WHERE n.status = "Traced"
    RETURN
        n.bodyId        AS bodyId,
        n.type          AS type,
        n.superclass    AS superclass,
        n.class         AS class,
        n.subclass      AS subclass,
        n.dimorphism    AS dimorphism,
        n.fru_dsx       AS fru_dsx,
        n.predictedNt   AS predictedNt,
        n.soma_x        AS soma_x,
        n.soma_y        AS soma_y,
        n.soma_z        AS soma_z
    """
    extra = client.fetch_custom(cypher)

    # Merge on bodyId
    df = neuron_df.merge(extra, on="bodyId", how="left", suffixes=("", "_extra"))

    # Consolidate duplicate columns
    for col in ["type", "superclass"]:
        col_extra = col + "_extra"
        if col_extra in df.columns:
            df[col] = df[col].fillna(df[col_extra])
            df.drop(columns=[col_extra], inplace=True)

    print(f"[fetch] {len(df):,} neurons retrieved.")
    return df


# ─── Synaptic connections ────────────────────────────────────────────────────

def fetch_connections_paged(
    client: Client,
    page_size: int = 250_000,
    min_weight: int = CFG.MIN_SYNAPSE_WEIGHT,
) -> pd.DataFrame:
    """
    Paginate through all MaleCNS connections.
    Using Cypher pagination; neuPrint has a row limit per query.

    Berg et al. STAR Methods: "Considering only connections with a strength
    of at least 5 synapses (a threshold sometimes used to remove weak
    connections)." → default min_weight = 5.

    Returns DataFrame (body_pre, body_post, weight).
    """
    print(f"[fetch] Downloading connections (min_weight={min_weight}) …")
    print("        This may take 20–60 min on first run. Cached thereafter.")

    records = []
    skip = 0
    pbar = tqdm(desc="connections", unit=" rows", dynamic_ncols=True)

    while True:
        cypher = f"""
        MATCH (a:Neuron)-[c:ConnectsTo]->(b:Neuron)
        WHERE c.weight >= {min_weight}
          AND a.status = "Traced"
          AND b.status = "Traced"
        RETURN a.bodyId AS body_pre, b.bodyId AS body_post, c.weight AS weight
        SKIP {skip} LIMIT {page_size}
        """
        chunk = client.fetch_custom(cypher)

        if chunk.empty:
            break

        records.append(chunk)
        skip += len(chunk)
        pbar.update(len(chunk))

        # neuPrint courtesy: small pause between large pages
        if len(chunk) == page_size:
            time.sleep(0.5)
        else:
            break   # last page (partial)

    pbar.close()
    df = pd.concat(records, ignore_index=True)
    print(f"[fetch] {len(df):,} connections retrieved.")
    return df


# ─── Build & cache sparse matrix ────────────────────────────────────────────

def build_and_cache(
    neuron_df: pd.DataFrame,
    conn_df: pd.DataFrame,
    neuron_csv: str = CFG.NEURON_CSV,
    conn_npz: str = CFG.CONN_NPZ,
) -> None:
    """
    Convert body IDs to contiguous indices and save:
      • neurons.csv  — metadata with index column
      • connections.npz — scipy sparse CSR matrix + bodyId arrays
    """
    # Assign contiguous indices
    body_ids = neuron_df["bodyId"].values
    id2idx = {bid: i for i, bid in enumerate(body_ids)}
    N = len(body_ids)

    # Filter connections to neurons in our set
    mask = (
        conn_df["body_pre"].isin(id2idx) &
        conn_df["body_post"].isin(id2idx)
    )
    conn_df = conn_df[mask].copy()

    # Map to indices
    rows = conn_df["body_post"].map(id2idx).values.astype(np.int32)  # post = target row
    cols = conn_df["body_pre"].map(id2idx).values.astype(np.int32)   # pre  = source col
    data = conn_df["weight"].values.astype(np.float32)

    W = sp.csr_matrix((data, (rows, cols)), shape=(N, N))

    # Save
    neuron_df["neuron_idx"] = neuron_df["bodyId"].map(id2idx)
    neuron_df.to_csv(neuron_csv, index=False)
    sp.save_npz(conn_npz, W)

    print(f"[cache] Saved {neuron_csv}")
    print(f"[cache] Saved {conn_npz}  shape={W.shape}  nnz={W.nnz:,}")


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fetch MaleCNS data from neuPrint.")
    parser.add_argument("--token", default=CFG.NEUPRINT_TOKEN,
                        help="Your neuPrint API token")
    parser.add_argument("--min-weight", type=int, default=CFG.MIN_SYNAPSE_WEIGHT)
    parser.add_argument("--skip-neurons", action="store_true",
                        help="Skip neuron fetch if neurons.csv already exists")
    parser.add_argument("--skip-connections", action="store_true",
                        help="Skip connection fetch if connections.npz exists")
    args = parser.parse_args()

    if args.token == "PASTE_YOUR_TOKEN_HERE":
        raise ValueError(
            "Set NEUPRINT_TOKEN env var or pass --token.\n"
            "Get your token at https://neuprint.janelia.org (login → Account)"
        )

    client = Client(
        CFG.NEUPRINT_SERVER,
        dataset=CFG.NEUPRINT_DATASET,
        token=args.token,
    )
    print(f"[neuPrint] Connected to {CFG.NEUPRINT_SERVER} / {CFG.NEUPRINT_DATASET}")

    # ── Neurons ──
    if args.skip_neurons and os.path.exists(CFG.NEURON_CSV):
        print(f"[skip] Loading cached {CFG.NEURON_CSV}")
        neuron_df = pd.read_csv(CFG.NEURON_CSV)
    else:
        neuron_df = fetch_all_neurons(client)
        neuron_df.to_csv(CFG.NEURON_CSV, index=False)
        print(f"[cache] Saved {CFG.NEURON_CSV}")

    # ── Connections ──
    if args.skip_connections and os.path.exists(CFG.CONN_NPZ):
        print(f"[skip] {CFG.CONN_NPZ} already exists.")
        return

    conn_df = fetch_connections_paged(client, min_weight=args.min_weight)
    build_and_cache(neuron_df, conn_df)
    print("[done] Data ready. Run: python run.py")


if __name__ == "__main__":
    main()
