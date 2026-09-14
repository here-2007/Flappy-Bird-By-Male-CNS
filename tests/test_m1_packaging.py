"""
tests/test_m1_packaging.py — Verification suite for Milestone 1:
Packaging, import cleanliness, device resolution, and syntax fixes.
"""

import sys
import re
import warnings
import numpy as np
import torch
import pytest

import config as CFG
import data
import connectome
import simulation
import game
import viz


def test_pytest_installed_and_version():
    """Verify pytest is installed and meets the version requirement (>= 8.0.0)."""
    assert pytest.__version__ >= "8.0.0"


def test_device_resolution_cpu():
    """Verify resolve_device('cpu') returns a CPU torch.device."""
    dev = CFG.resolve_device("cpu")
    assert isinstance(dev, torch.device)
    assert dev.type == "cpu"


def test_device_resolution_default():
    """Verify resolve_device() returns a valid torch.device without error."""
    dev = CFG.resolve_device()
    assert isinstance(dev, torch.device)
    assert dev.type in ("cuda", "mps", "cpu")


def test_device_resolution_unavailable_cuda_graceful_fallback():
    """Verify requesting CUDA when unavailable does not crash and returns a fallback device."""
    dev = CFG.resolve_device("cuda")
    assert isinstance(dev, torch.device)
    assert dev.type in ("cuda", "mps", "cpu")


def test_device_resolution_bogus_fallback():
    """Verify requesting an invalid device string does not crash."""
    dev = CFG.resolve_device("non_existent_accelerator_xyz")
    assert isinstance(dev, torch.device)
    assert dev.type in ("cuda", "mps", "cpu")


def test_get_device_no_side_effects(capsys):
    """Verify get_device() without verbose=True emits no stdout messages."""
    CFG.get_device(verbose=False)
    captured = capsys.readouterr()
    assert captured.out == ""


def test_package_exports():
    """Verify all packages export their expected public APIs."""
    assert hasattr(data, "fetch_all_neurons")
    assert hasattr(data, "fetch_connections_paged")
    assert hasattr(data, "build_and_cache")

    assert hasattr(connectome, "CircuitPopulations")
    assert hasattr(connectome, "identify_populations")
    assert hasattr(connectome, "get_soma_coords")
    assert hasattr(connectome, "load_connectome")

    assert hasattr(simulation, "LIFEngine")
    assert hasattr(simulation, "SensoryEncoder")
    assert hasattr(simulation, "MotorDecoder")
    assert hasattr(simulation, "PPL101Plasticity")

    assert hasattr(game, "FlappyBird")
    assert hasattr(game, "GameState")

    assert hasattr(viz, "Dashboard")
    assert hasattr(viz, "DualPanelDashboard")


def test_sensory_norm01_empty_array():
    """Verify _norm01 handles empty arrays safely without ValueError or NaN."""
    from simulation.sensory import _norm01
    empty_arr = np.array([], dtype=np.float32)
    result = _norm01(empty_arr)
    assert isinstance(result, np.ndarray)
    assert result.size == 0
    assert not np.isnan(result).any()


def test_sensory_encoder_with_empty_photoreceptors():
    """Verify SensoryEncoder runs safely without crashing when photo_idx is empty."""
    from simulation.sensory import SensoryEncoder
    empty_photo = np.array([], dtype=np.int32)
    soma_coords = np.zeros((10, 3), dtype=np.float32)
    encoder = SensoryEncoder(empty_photo, soma_coords, N=10, device=torch.device("cpu"))
    frame = np.zeros((CFG.GAME_RENDER_H, CFG.GAME_RENDER_W), dtype=np.float32)
    I_ext = encoder.encode(frame)
    assert I_ext.shape == (10,)
    assert not torch.isnan(I_ext).any()
    assert (I_ext == 0.0).all()


def test_kenyon_cell_regex_syntax():
    """Verify the Kenyon cell regex pattern compiles cleanly without syntax warnings."""
    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")
        pattern = re.compile(r"kenyon|kca|kcg|kc[_\s]", flags=re.IGNORECASE)
        # Verify it matches expected Kenyon cell variants
        assert pattern.search("Kenyon_cell") is not None
        assert pattern.search("KCa_1") is not None
        assert pattern.search("KCg") is not None
        assert pattern.search("KC alpha") is not None
        assert pattern.search("KC_beta") is not None

    syntax_warnings = [
        w for w in recorded_warnings if issubclass(w.category, (SyntaxWarning, DeprecationWarning))
    ]
    assert len(syntax_warnings) == 0, f"Found unexpected warnings: {syntax_warnings}"
