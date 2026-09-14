import os
import sys
import subprocess
import pytest
import torch
import numpy as np
import scipy.sparse as sp

import config as CFG
from data.mock_data import get_or_create_mock_connectome
from game.flappy import FlappyBird
from simulation.lif import LIFEngine
from simulation.sensory import SensoryEncoder
from simulation.motor import MotorDecoder
from simulation.plasticity import PPL101Plasticity

def test_game_engine_100_frames():
    game = FlappyBird()
    game.reset(seed=42)
    for _ in range(100):
        if not game.state.alive:
            break
        game.step(False)
    assert game.state.score >= 0

def test_lif_engine_with_mock():
    W_t, W_scipy, neuron_df, pop, coords = get_or_create_mock_connectome(device="cpu", seed=42)
    N = len(neuron_df)
    engine = LIFEngine(W=W_t, N=N, device="cpu")
    I_ext = torch.randn(N)
    spikes = engine.step(I_ext)
    assert spikes.shape == (N,)
    assert spikes.dtype == torch.bool

def test_sensory_encoder_shape():
    W_t, W_scipy, neuron_df, pop, coords = get_or_create_mock_connectome(device="cpu", seed=42)
    N = len(neuron_df)
    encoder = SensoryEncoder(photo_idx=pop.photoreceptors_R1_R6, soma_coords=coords, N=N, device="cpu")
    game = FlappyBird()
    game.reset(seed=42)
    frame_gray, _ = game.render()
    I_ext = encoder.encode(frame_gray)
    assert I_ext.shape == (N,)
    assert I_ext.sum() > 0

def test_motor_decoder_returns_bool():
    W_t, W_scipy, neuron_df, pop, coords = get_or_create_mock_connectome(device="cpu", seed=42)
    N = len(neuron_df)
    decoder = MotorDecoder(wing_motor_idx=pop.wing_motor, N=N, device="cpu")
    spike_history_sum = torch.randint(0, 10, (N,))
    flap = decoder.decode(spike_history_sum, 10)
    assert isinstance(flap, bool) or isinstance(flap, np.bool_) or isinstance(flap, int)
    assert flap in [True, False, 0, 1]

def test_plasticity_updates_weights():
    W_t, W_scipy, neuron_df, pop, coords = get_or_create_mock_connectome(device="cpu", seed=42)
    N = len(neuron_df)
    plasticity = PPL101Plasticity(W_scipy=W_scipy, ppl101_idx=pop.ppl101, kc_idx=pop.kenyon_cells, N=N, device="cpu")
    if not plasticity._enabled:
        pytest.skip("No plastic synapses found.")
    
    spikes = torch.randint(0, 2, (N,), dtype=torch.bool)
    plasticity.step(spikes, da_signal=1.0)
    W_t_new = plasticity.apply_updates(W_t)
    
    assert len(plasticity.episode_dw_norm) > 0
    assert plasticity.episode_dw_norm[-1] >= 0.0

def test_end_to_end_mock():
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    res = subprocess.run([sys.executable, "run.py", "--mock", "--episodes", "1", "--no-viz"], 
                         capture_output=True, text=True, env=env, cwd=env["PYTHONPATH"])
    assert res.returncode == 0, f"Error running run.py:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    assert "Training complete" in res.stdout
