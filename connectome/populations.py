"""
connectome/populations.py — Identify circuit-relevant neuron populations.

All population definitions cite Berg et al. (2026) Cell 189, 5504-5526.
Index arrays here are positions in the contiguous 0…N-1 neuron ordering
built by loader.py, not neuPrint bodyIds.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
import config as CFG


@dataclass
class CircuitPopulations:
    """
    Holds index arrays for each biologically relevant population.
    All arrays contain integer indices into the [0, N) neuron space.
    """
    # ── Sensory input ──────────────────────────────────────────────────────
    # Berg et al. p.5511: R1–R6 photoreceptors (OLSNs, brightness/motion)
    photoreceptors_R1_R6: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))

    # ── Visual projection neurons ──────────────────────────────────────────
    # Berg et al. p.5511: "of the 335 visual projection neuron (VPN) types,
    # just one is male-specific: LoVP92"
    vpns_all: np.ndarray          = field(default_factory=lambda: np.array([], dtype=np.int32))
    lovp92: np.ndarray            = field(default_factory=lambda: np.array([], dtype=np.int32))   # male-specific VPN
    tmy21: np.ndarray             = field(default_factory=lambda: np.array([], dtype=np.int32))   # dimorphic OLIN
    frontal_vpns: np.ndarray      = field(default_factory=lambda: np.array([], dtype=np.int32))   # 7 frontal-biased types

    # ── Descending neurons (brain→VNC) ────────────────────────────────────
    # Berg et al. p.5512: DNg13 = "general-purpose, coarse steering pathway"
    # DNa02, aSP22 = "required for effective visual pursuit when aroused"
    dns_all: np.ndarray           = field(default_factory=lambda: np.array([], dtype=np.int32))
    dng13: np.ndarray             = field(default_factory=lambda: np.array([], dtype=np.int32))
    dna02: np.ndarray             = field(default_factory=lambda: np.array([], dtype=np.int32))
    asp22: np.ndarray             = field(default_factory=lambda: np.array([], dtype=np.int32))

    # ── VNC wing pre-motor neurons (motor output) ──────────────────────────
    # Berg et al. p.5512: "key wing pre-motor neurons in the VNC
    # (TN1A, vPR9, and dPR1)"
    tn1a: np.ndarray              = field(default_factory=lambda: np.array([], dtype=np.int32))
    vpr9: np.ndarray              = field(default_factory=lambda: np.array([], dtype=np.int32))
    dpr1: np.ndarray              = field(default_factory=lambda: np.array([], dtype=np.int32))

    # ── PPL101 dopamine neurons (plasticity signal) ────────────────────────
    # Berg et al. Discussion p.5519; Aso et al. (2014) eLife.
    ppl101: np.ndarray            = field(default_factory=lambda: np.array([], dtype=np.int32))

    # ── Mushroom body (Kenyon cells — plasticity targets) ─────────────────
    kenyon_cells: np.ndarray      = field(default_factory=lambda: np.array([], dtype=np.int32))

    # ── Male-specific / dimorphic central brain neurons ───────────────────
    # Berg et al. p.5508: "3.4% of neurons are male-specific, 1.4% dimorphic"
    male_specific: np.ndarray     = field(default_factory=lambda: np.array([], dtype=np.int32))
    dimorphic: np.ndarray         = field(default_factory=lambda: np.array([], dtype=np.int32))

    # ── Summary ───────────────────────────────────────────────────────────
    @property
    def wing_motor(self) -> np.ndarray:
        """Union of TN1A, vPR9, dPR1 — the flap readout population."""
        return np.unique(np.concatenate([self.tn1a, self.vpr9, self.dpr1]))

    def summary(self) -> str:
        lines = ["CircuitPopulations:"]
        for attr, label in [
            ("photoreceptors_R1_R6", "Photoreceptors R1–R6"),
            ("vpns_all",             "VPNs (all)"),
            ("lovp92",               "LoVP92 (male-specific VPN)"),
            ("tmy21",                "TmY21 (dimorphic OLIN)"),
            ("dns_all",              "Descending neurons"),
            ("dng13",                "DNg13 (steering)"),
            ("dna02",                "DNa02 (visual pursuit)"),
            ("asp22",                "aSP22 (courtship)"),
            ("wing_motor",           "Wing pre-motor (TN1A+vPR9+dPR1)"),
            ("ppl101",               "PPL101 (dopamine / plasticity)"),
            ("kenyon_cells",         "Kenyon cells (MB plasticity target)"),
            ("male_specific",        "Male-specific neurons"),
            ("dimorphic",            "Dimorphic neurons"),
        ]:
            arr = getattr(self, attr)
            lines.append(f"  {label:40s}: {len(arr):6,} neurons")
        return "\n".join(lines)


# ─── Builder ─────────────────────────────────────────────────────────────────

def identify_populations(neuron_df: pd.DataFrame) -> CircuitPopulations:
    """
    Extract index arrays for each population from the neuron metadata DataFrame.

    Population criteria follow Berg et al. (2026) Table 1, Results sections,
    and the MaleCNS neuron annotation scheme described in STAR Methods.

    Parameters
    ----------
    neuron_df : pd.DataFrame
        Must have columns: neuron_idx, type, superclass, class, subclass,
        dimorphism, predictedNt, soma_x, soma_y, soma_z.
    """
    pop = CircuitPopulations()

    def idx(mask: pd.Series) -> np.ndarray:
        return neuron_df.loc[mask, "neuron_idx"].values.astype(np.int32)

    def type_match(patterns: list[str]) -> pd.Series:
        """Case-insensitive substring match on the 'type' column."""
        t = neuron_df["type"].fillna("").str.lower()
        combined = "|".join(p.lower() for p in patterns)
        return t.str.contains(combined, regex=True)

    # ── Photoreceptors ────────────────────────────────────────────────────
    # superclass = 'ol-sensory' AND type matches R1–R6 patterns
    # MaleCNS annotation: OLSNs are optic lobe sensory neurons
    is_r1r6 = (
        (neuron_df["superclass"].fillna("").str.lower().str.contains("ol-sensory|ol_sensory"))
        & (neuron_df["type"].fillna("").str.upper().str.contains(
            r"^R[1-6]$|^R[1-6][_\-]|^R1$|^R2$|^R3$|^R4$|^R5$|^R6$", regex=True
        ))
    )
    # Fallback: any ol-sensory neuron with small type code
    if is_r1r6.sum() < 100:
        is_r1r6 = neuron_df["superclass"].fillna("").str.lower().str.contains(
            "ol-sensory|ol_sensory"
        )
    pop.photoreceptors_R1_R6 = idx(is_r1r6)

    # ── Visual projection neurons ─────────────────────────────────────────
    is_vpn = neuron_df["superclass"].fillna("").str.lower().str.contains(
        "visual_projection|visual projection"
    )
    pop.vpns_all = idx(is_vpn)

    # LoVP92 — Berg et al. p.5511: only male-specific VPN type
    pop.lovp92 = idx(type_match(["LoVP92", "LOVP92"]))

    # TmY21 — Berg et al. p.5511: only dimorphic OL intrinsic type
    pop.tmy21 = idx(type_match(["TmY21", "TMY21"]))

    # Frontal-biased VPNs — Berg et al. p.5511: LoVP92 + 6 isomorphic types
    # in the anterior lobula. Identified by their lobula neuropil innervation.
    frontal_vp_types = [
        "LoVP92", "LT84", "LLPC4", "LT86", "MeVP48", "LC31b", "LT78", "AOTU045"
    ]
    pop.frontal_vpns = idx(type_match(frontal_vp_types))

    # ── Descending neurons ────────────────────────────────────────────────
    is_dn = neuron_df["superclass"].fillna("").str.lower().str.contains(
        "^descending$|cb-descending|central_descending"
    )
    pop.dns_all = idx(is_dn)

    pop.dng13 = idx(type_match(["DNg13", "Dng13"]))   # coarse steering
    pop.dna02 = idx(type_match(["DNa02", "Dna02"]))   # visual pursuit
    pop.asp22 = idx(type_match(["aSP22", "asp22"]))   # courtship

    # ── Wing pre-motor neurons ────────────────────────────────────────────
    pop.tn1a = idx(type_match(["TN1A", "TN1a"]))
    pop.vpr9 = idx(type_match(["vPR9", "VPR9"]))
    pop.dpr1 = idx(type_match(["dPR1", "DPR1"]))

    # ── PPL101 dopamine neurons ───────────────────────────────────────────
    # Berg et al. Discussion p.5519; DoomFly (Wormuth 2024)
    pop.ppl101 = idx(
        neuron_df["type"].fillna("").str.upper().str.startswith(
            CFG.PPL101_TYPE_PREFIX.upper()
        )
    )

    # ── Kenyon cells (MB) ─────────────────────────────────────────────────
    # superclass = 'cb-intrinsic', class contains 'Kenyon' or 'KCa' etc.
    is_kc = (
        neuron_df["superclass"].fillna("").str.lower().str.contains(
            CFG.MB_SUPERCLASS.lower()
        )
        & neuron_df["class"].fillna("").str.lower().str.contains(
            "kenyon|kca|kcg|kc[_\\s]", regex=True
        )
    )
    pop.kenyon_cells = idx(is_kc)

    # ── Dimorphism labels ─────────────────────────────────────────────────
    dim_col = neuron_df["dimorphism"].fillna("").str.lower()
    pop.male_specific = idx(dim_col.str.contains("male-specific|male_specific"))
    pop.dimorphic     = idx(dim_col.str.contains("^dimorphic$|sexually.dimorphic"))

    # ── Report ─────────────────────────────────────────────────────────────
    print(pop.summary())

    # Sanity checks
    _warn_if_empty(pop.photoreceptors_R1_R6, "Photoreceptors R1-R6",
                   "Check superclass='ol-sensory' in neuPrint data")
    _warn_if_empty(pop.wing_motor, "Wing pre-motor (TN1A/vPR9/dPR1)",
                   "These neuron types must exist in MaleCNS v1.0")
    _warn_if_empty(pop.ppl101, "PPL101",
                   "Check type prefix 'PPL1' in neuPrint data")

    return pop


def _warn_if_empty(arr: np.ndarray, name: str, hint: str = "") -> None:
    if len(arr) == 0:
        msg = f"[WARNING] Population '{name}' is empty."
        if hint:
            msg += f" Hint: {hint}"
        print(msg)


def get_soma_coords(neuron_df: pd.DataFrame) -> np.ndarray:
    """
    Return N×3 array of soma (x, y, z) coordinates in MaleCNS voxel space.
    Used for 2-D visualisation projection.
    """
    coords = neuron_df[["soma_x", "soma_y", "soma_z"]].fillna(0).values
    return coords.astype(np.float32)
