"""
tests/test_m1_adversarial_edge_cases.py — Adversarial Edge-Case Stress Suite
Created by Milestone 1 Challenger 1.

Verifies:
1. Unexpected and adversarial inputs to `resolve_device()`.
2. Extreme edge cases in `_norm01` (simulation.sensory and viz.dashboard).
3. Empty photoreceptor populations in `SensoryEncoder`.
4. Dashboard resilience against degenerate soma coordinate arrays.
"""

from unittest import mock
import numpy as np
import pytest
import torch

import config as CFG
from connectome.populations import CircuitPopulations
from simulation.sensory import SensoryEncoder, _norm01 as sensory_norm01
from viz import Dashboard, DualPanelDashboard
from viz.dashboard import _norm01 as dash_norm01


# ─── 1. resolve_device() Adversarial Tests ───────────────────────────────────

def test_resolve_device_core_inputs():
    """Verify all prompt-specified inputs resolve safely without unhandled exceptions."""
    inputs = [
        ("None", None),
        ("'cuda'", "cuda"),
        ("'mps'", "mps"),
        ("'cpu'", "cpu"),
        ("'invalid_device'", "invalid_device"),
        ("torch.device('cpu')", torch.device("cpu")),
    ]
    for label, inp in inputs:
        dev = CFG.resolve_device(inp)
        assert isinstance(dev, torch.device), f"Failed for {label}: got {type(dev)}"
        assert dev.type in ("cuda", "mps", "cpu"), f"Unexpected type for {label}: {dev.type}"


def test_resolve_device_extended_adversarial_inputs():
    """Stress-test resolve_device with boundary, whitespace, malformed, and non-string inputs."""
    adversarial_inputs = [
        "",
        "   ",
        "CPU",
        "  cuda  ",
        "  mps  ",
        "cuda:0",
        "cuda:99",
        "mps:0",
        "cpu:0",
        123,
        True,
        False,
        ["cpu"],
        {"device": "cpu"},
        object(),
    ]
    for inp in adversarial_inputs:
        dev = CFG.resolve_device(inp)
        assert isinstance(dev, torch.device), f"Failed for input {inp!r}: got {type(dev)}"


def test_resolve_device_simulated_cpu_only():
    """Verify fallback to CPU when neither CUDA nor MPS is available."""
    with mock.patch("torch.cuda.is_available", return_value=False), \
         mock.patch("torch.backends.mps.is_available", return_value=False):
        for req in [None, "cuda", "mps", "cpu", "invalid_device", torch.device("cpu")]:
            dev = CFG.resolve_device(req)
            assert isinstance(dev, torch.device)
            assert dev.type == "cpu", f"Expected cpu fallback for {req!r}, got {dev.type}"


def test_resolve_device_simulated_cuda_available():
    """Verify CUDA resolution when CUDA is available."""
    with mock.patch("torch.cuda.is_available", return_value=True):
        assert CFG.resolve_device(None).type == "cuda"
        assert CFG.resolve_device("cuda").type == "cuda"
        assert CFG.resolve_device("cpu").type == "cpu"
        assert CFG.resolve_device("invalid_device").type == "cuda"


# ─── 2. _norm01 Edge-Case Tests (sensory & dashboard) ───────────────────────

@pytest.mark.parametrize("norm_func", [sensory_norm01, dash_norm01])
def test_norm01_empty_arrays(norm_func):
    """Verify 0-length arrays return empty float32 array without error."""
    for arr in [np.array([]), np.array([], dtype=np.int32), np.array([], dtype=np.float64)]:
        res = norm_func(arr)
        assert isinstance(res, np.ndarray)
        assert res.size == 0
        assert res.dtype == np.float32


@pytest.mark.parametrize("norm_func", [sensory_norm01, dash_norm01])
def test_norm01_single_element_arrays(norm_func):
    """Verify 1-element arrays return [0.0] without zero-division error."""
    for val in [0.0, 42.0, -10.0, 1e-9]:
        arr = np.array([val], dtype=np.float32)
        res = norm_func(arr)
        assert res.shape == (1,)
        assert res[0] == 0.0
        assert not np.isnan(res[0])


@pytest.mark.parametrize("norm_func", [sensory_norm01, dash_norm01])
def test_norm01_all_identical_elements(norm_func):
    """Verify arrays with identical elements return all zeros without zero division."""
    cases = [
        np.zeros(10, dtype=np.float32),
        np.full(10, 100.0, dtype=np.float32),
        np.full(5, -50.0, dtype=np.float32),
        np.full(20, 1e-8, dtype=np.float32),
    ]
    for arr in cases:
        res = norm_func(arr)
        assert res.shape == arr.shape
        assert (res == 0.0).all()
        assert not np.isnan(res).any()


@pytest.mark.parametrize("norm_func", [sensory_norm01, dash_norm01])
def test_norm01_nans_and_infs_no_unhandled_crash(norm_func):
    """Verify arrays with NaNs and Infs do not cause unhandled crashes under normal execution."""
    cases = [
        np.array([np.nan]),
        np.array([np.nan, np.nan]),
        np.array([1.0, np.nan, 3.0]),
        np.array([np.inf]),
        np.array([np.inf, np.inf]),
        np.array([1.0, np.inf, 3.0]),
        np.array([-np.inf, np.inf]),
        np.array([np.nan, np.inf, 0.0]),
    ]
    for arr in cases:
        res = norm_func(arr)
        assert isinstance(res, np.ndarray)
        assert res.shape == arr.shape


# ─── 3. SensoryEncoder Empty Photoreceptors Tests ────────────────────────────

@pytest.mark.parametrize("N", [0, 1, 10, 100])
@pytest.mark.parametrize("empty_photo", [
    np.array([], dtype=np.int64),
    np.array([], dtype=np.int32),
    [],
])
def test_sensory_encoder_empty_photoreceptors_encode(N, empty_photo):
    """Verify encode() returns zero tensor of shape [N] without NaNs when photo_idx is empty."""
    devices = [torch.device("cpu")]
    if torch.backends.mps.is_available():
        devices.append(torch.device("mps"))

    for dev in devices:
        soma_coords = np.zeros((N, 3), dtype=np.float32)
        encoder = SensoryEncoder(empty_photo, soma_coords, N=N, device=dev)

        # Standard frame
        frame = np.zeros((CFG.GAME_RENDER_H, CFG.GAME_RENDER_W), dtype=np.float32)
        I_ext = encoder.encode(frame)
        assert I_ext.shape == (N,)
        assert not torch.isnan(I_ext).any()
        assert (I_ext == 0.0).all()
        assert I_ext.device.type == dev.type

        # Frame with random values
        frame_rand = np.random.rand(CFG.GAME_RENDER_H, CFG.GAME_RENDER_W).astype(np.float32)
        I_ext_rand = encoder.encode(frame_rand, gain=5.0)
        assert I_ext_rand.shape == (N,)
        assert not torch.isnan(I_ext_rand).any()
        assert (I_ext_rand == 0.0).all()

        # Frame with NaNs
        frame_nan = np.full((CFG.GAME_RENDER_H, CFG.GAME_RENDER_W), np.nan, dtype=np.float32)
        I_ext_nan = encoder.encode(frame_nan)
        assert I_ext_nan.shape == (N,)
        assert not torch.isnan(I_ext_nan).any()
        assert (I_ext_nan == 0.0).all()


def test_sensory_encoder_empty_photoreceptors_structured():
    """Verify encode_structured() handles empty photoreceptor populations safely."""
    empty_photo = np.array([], dtype=np.int32)
    soma_coords = np.zeros((20, 3), dtype=np.float32)
    encoder = SensoryEncoder(empty_photo, soma_coords, N=20, device=torch.device("cpu"))

    I_ext = encoder.encode_structured(bird_y=0.5, gap_center=0.4, pipe_dist=0.3, bird_vy=0.1)
    assert I_ext.shape == (20,)
    assert not torch.isnan(I_ext).any()
    assert (I_ext == 0.0).all()


# ─── 4. Dashboard Coordinate Edge Cases ──────────────────────────────────────

def test_dashboard_degenerate_coordinates():
    """Verify Dashboard handles 0-size, 1-element, and identical soma coordinates without crash."""
    pop = CircuitPopulations()

    # N = 0
    d0 = Dashboard(coords=np.zeros((0, 3), dtype=np.float32), pop=pop, video_path=None)
    assert d0._px.shape == (0,)
    d0.close()

    # N = 1
    d1 = Dashboard(coords=np.array([[10.0, 20.0, 30.0]], dtype=np.float32), pop=pop, video_path=None)
    assert d1._px.shape == (1,)
    assert d1._px[0] == 0.0
    d1.close()

    # Identical coordinates
    d_ident = DualPanelDashboard(coords=np.full((30, 3), 50.0, dtype=np.float32), pop=pop, video_path=None)
    assert d_ident._px.shape == (30,)
    assert (d_ident._px == 0.0).all()
    d_ident.close()
