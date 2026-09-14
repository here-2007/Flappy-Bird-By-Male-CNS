"""
config.py — All hyperparameters and paths for FlyFlappyBird.

Neural parameters follow Lappalainen et al. (2024) connectome-constrained
Drosophila visual model (Nature 634, 1132-1140) and Berg et al. (2026)
MaleCNS annotations (Cell 189, 5504-5526).
"""

import os
from typing import Optional, Union
import torch

# ─── Paths ──────────────────────────────────────────────────────────────────
ROOT          = os.environ.get("FLY_BRAIN_ROOT", os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR     = os.environ.get("FLY_BRAIN_CACHE_DIR", os.path.join(ROOT, "data", "cache"))
VIDEO_OUT     = os.environ.get("FLY_BRAIN_VIDEO_OUT", os.path.join(ROOT, "fly_plays_flappy.mp4"))

def _resolve_data_path(filename: str) -> str:
    """
    Search for connectome dataset files across:
    1. Direct environment variable (e.g. FLY_BRAIN_NEURONS_CSV, FLY_BRAIN_CONNECTIONS_NPZ)
    2. Explicit Kaggle dataset input paths:
       - /kaggle/input/datasets/pernavjain/male-fruit-fly-cns/<filename>
       - /kaggle/input/male-fruit-fly-cns/<filename>
    3. Auto-discovery across any /kaggle/input subfolder
    4. Local cache directory (CACHE_DIR/<filename>)
    """
    env_key = f"FLY_BRAIN_{filename.replace('.', '_').upper()}"
    if env_key in os.environ and os.path.exists(os.environ[env_key]):
        return os.environ[env_key]

    # Specific Kaggle dataset mount locations
    kaggle_candidates = [
        os.path.join("/kaggle/input/datasets/pernavjain/male-fruit-fly-cns", filename),
        os.path.join("/kaggle/input/male-fruit-fly-cns", filename),
    ]
    for path in kaggle_candidates:
        if os.path.exists(path):
            return path

    # Auto-discovery if running on Kaggle
    if os.path.exists("/kaggle/input"):
        try:
            for root_dir, _, files in os.walk("/kaggle/input"):
                if filename in files:
                    return os.path.join(root_dir, filename)
        except Exception:
            pass

    # Default project cache location
    return os.path.join(CACHE_DIR, filename)

NEURON_CSV    = _resolve_data_path("neurons.csv")
CONN_NPZ      = _resolve_data_path("connections.npz")

def _resolve_weights_path() -> str:
    """Ensure weights checkpoint path is writable (Kaggle /kaggle/input is read-only)."""
    if "FLY_BRAIN_WEIGHTS_PATH" in os.environ:
        return os.environ["FLY_BRAIN_WEIGHTS_PATH"]
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        test_file = os.path.join(CACHE_DIR, ".write_test")
        with open(test_file, "w") as f:
            f.write("ok")
        os.remove(test_file)
        return os.path.join(CACHE_DIR, "trained_weights.pt")
    except Exception:
        return os.path.join(ROOT, "trained_weights.pt")

WEIGHTS_PATH  = _resolve_weights_path()

# ─── neuPrint ────────────────────────────────────────────────────────────────
NEUPRINT_SERVER  = "https://neuprint.janelia.org"
NEUPRINT_DATASET = "male-cns:v1.0"          # Berg et al. 2026 dataset
NEUPRINT_TOKEN   = os.environ.get("NEUPRINT_TOKEN", "PASTE_YOUR_TOKEN_HERE")

# ─── Connectome filtering ────────────────────────────────────────────────────
# Berg et al. p.5507: "labeled weak edges (<11 synapses for male CNS) as noisy"
MIN_SYNAPSE_WEIGHT = 5      # keep connections ≥ 5 synapses (paper threshold)
MAX_NEURONS        = 166_700  # full MaleCNS (Berg et al. Table 1)

# ─── LIF parameters ──────────────────────────────────────────────────────────
# From Lappalainen et al. 2024 connectome-constrained model;
# Dayan & Abbott (2001) Theoretical Neuroscience, Ch.5 for LIF formulation.
DT          = 1.0    # ms — simulation timestep (0.5 ms biologically ideal,
                     #       1 ms used here for speed/accuracy balance)
TAU_M       = 20.0   # ms — membrane time constant
TAU_G       = 5.0    # ms — synaptic conductance decay
V_REST      = 0.0    # mV (normalised; biological -65 mV)
V_THRESH    = 1.0    # mV (normalised; biological -50 mV)
V_RESET     = -0.2   # mV (normalised; biological -70 mV)
TAU_REF     = 2.0    # ms — absolute refractory period
WEIGHT_SCALE = 0.01  # global synaptic weight scaling factor

# Neurotransmitter sign mapping
# Berg et al. p.5505: "Feedforward connections share similar,
# predominantly excitatory neurotransmitter compositions"
# In Drosophila: ACh excitatory, GABA/Glu inhibitory.
NT_SIGN = {
    "acetylcholine": +1.0,
    "GABA":          -1.0,
    "glutamate":     -1.0,   # inhibitory in fly CNS
    "dopamine":      +0.5,   # modulatory; net excitatory here
    "serotonin":     +0.5,
    "octopamine":    +0.5,
    "histamine":     -1.0,
    "unknown":       +0.5,   # default to mild excitation
}

# ─── Sensory encoding ────────────────────────────────────────────────────────
# Visual field covered: frontal ±60° az, ±40° el — the region sampled by
# LoVP92 and frontal-biased VPNs. Berg et al. p.5511:
# "LoVP92 and 6 isomorphic types are confined to the anterior lobula,
#  which samples the frontal visual field."
VISUAL_AZ_RANGE  = (-60.0, 60.0)   # degrees azimuth
VISUAL_EL_RANGE  = (-40.0, 40.0)   # degrees elevation
GAME_RENDER_W    = 144              # game canvas width  (pixels)
GAME_RENDER_H    = 256              # game canvas height (pixels)

# ─── Motor decoding ──────────────────────────────────────────────────────────
# Berg et al. p.5512: "key wing pre-motor neurons in the VNC (TN1A, vPR9,
# and dPR1)" cited for courtship song; wing motor circuit overlap.
WING_MOTOR_TYPES  = ["TN1A", "vPR9", "dPR1"]
FLAP_THRESHOLD    = 0.15  # fraction of wing MNs spiking → flap decision

# ─── PPL101 plasticity ───────────────────────────────────────────────────────
# DoomFly (Wormuth 2024); dopamine gating follows Aso et al. (2014)
# "The neuronal architecture of the mushroom body provides a logic for
# associative learning" eLife 3:e04577.
LEARNING_RATE      = 1e-5   # synaptic weight update rate
TAU_ELIGIBILITY    = 50.0   # ms — eligibility trace decay
REWARD_ALIVE       = +0.1   # DA signal each frame alive
REWARD_PASS_PIPE   = +1.0   # DA signal for passing a pipe
PUNISH_DEATH       = -5.0   # DA signal on collision
PPL101_TYPE_PREFIX = "PPL1" # neuPrint type prefix for PPL1 dopamine neurons
MB_SUPERCLASS      = "cb-intrinsic"  # Kenyon cells & MB neurons

# PPL101 Hebbian update clamp
W_MAX =  2.0
W_MIN = -2.0

# ─── Simulation loop ─────────────────────────────────────────────────────────
LIF_STEPS_PER_FRAME = 5    # LIF steps per game frame (5 ms biological window)
MAX_EPISODES         = 200
MAX_FRAMES_PER_EP    = 5_000
VIZ_EVERY_N_FRAMES   = 5    # update dashboard every N frames

# ─── Device ──────────────────────────────────────────────────────────────────
def resolve_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """
    Resolve and return a valid torch.device without throwing errors.
    If device is specified, verifies availability and safely falls back
    (CUDA -> MPS -> CPU) if the accelerator is not available.
    """
    if device is not None:
        dev_str = str(device).lower().strip()
        if dev_str.startswith("cuda"):
            try:
                if torch.cuda.is_available():
                    return torch.device(device)
            except Exception:
                pass
        elif dev_str.startswith("mps"):
            try:
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built():
                    return torch.device(device)
            except Exception:
                pass
        elif dev_str == "cpu":
            return torch.device("cpu")
        else:
            try:
                return torch.device(device)
            except Exception:
                pass

    # Auto-detection: CUDA > MPS > CPU
    try:
        if torch.cuda.is_available():
            return torch.device("cuda")
    except Exception:
        pass

    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
    except Exception:
        pass

    return torch.device("cpu")


def get_device(verbose: bool = False) -> torch.device:
    """Select best available device. T4 on Kaggle > MPS on M4 > CPU."""
    dev = resolve_device()
    if verbose:
        if dev.type == "cuda":
            try:
                name = torch.cuda.get_device_name(0)
                print(f"[device] CUDA: {name}")
            except Exception:
                print("[device] CUDA")
        elif dev.type == "mps":
            print("[device] Apple MPS (M-series)")
        else:
            print("[device] CPU — simulation will be slow")
    return dev


DEVICE = get_device()

