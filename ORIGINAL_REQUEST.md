# Original User Request

## Initial Request — 2026-09-13T09:00:03Z

Continue building and debugging the FlyFlappyBird project — a research-accurate neural simulation where the complete male *Drosophila* CNS connectome (166K neurons, 25M synapses from Berg et al. 2026) plays Flappy Bird via biologically grounded vision-to-motor circuits and dopamine-gated Hebbian plasticity. The target is a fully runnable, tested codebase that works on both Mac M4 (local dev) and Kaggle 2×T4 GPUs (full-scale runs).

Working directory: /Users/pernavjain/Documents/Fly Brain/fly_flappy_bird
Integrity mode: development

## Context

A prior session generated 12 skeleton Python files implementing the full pipeline: neuPrint data fetching → connectome building → LIF simulation → sensory encoding → motor decoding → plasticity → game physics → visualization. The session was terminated before any testing, debugging, or validation. **The code has never been run.** Expect import errors, undefined attributes, missing packages, and API mismatches.

### Existing files (all in working directory):
- `requirements.txt` — dependencies
- `config.py` — hyperparameters, paths, device selection
- `data/fetch_data.py` — neuPrint data fetcher (paged Cypher queries)
- `connectome/populations.py` — neuron population identification (R1–R6, LoVP92, PPL101, TN1A/vPR9/dPR1, Kenyon cells)
- `connectome/builder.py` — sparse weight tensor with NT sign correction
- `simulation/lif.py` — vectorised GPU LIF engine (Dayan & Abbott 2001)
- `simulation/sensory.py` — game frame → photoreceptor current encoder
- `simulation/motor.py` — wing motor neuron → flap decision decoder
- `simulation/plasticity.py` — PPL101 dopamine-gated reward-modulated Hebbian plasticity (R-STDP)
- `game/flappy.py` — headless Flappy Bird physics + numpy renderer
- `viz/dashboard.py` — matplotlib dual-panel real-time dashboard
- `run.py` — main training loop orchestrator

### Known issues to fix:
1. Missing `__init__.py` for `data/`, `connectome/`, `simulation/`, `game/`, `viz/` packages
2. Code never import-tested — expect missing imports, circular references, undefined names
3. `config.py` calls `get_device()` at import time (runs CUDA detection on load)
4. No tests exist
5. No mock/synthetic data mode — neuPrint token required for any run
6. Plasticity `_find_plastic_synapses` uses a Python loop over all edges (slow for 25M edges)

### Biological references embedded in code:
- Berg et al. (2026) Cell 189, 5504–5526 (MaleCNS connectome)
- Lappalainen et al. (2024) Nature 634, 1132–1140 (LIF parameters)
- Aso et al. (2014) eLife 3:e04577 (PPL101 dopamine learning)
- Dayan & Abbott (2001) Theoretical Neuroscience Ch.5 (LIF formulation)

## Requirements

### R1. Make the codebase import-clean and runnable end-to-end

Fix all import errors, missing `__init__.py` files, undefined references, type mismatches, and API inconsistencies so the full pipeline runs without errors. `config.py` should not crash if CUDA is unavailable. All modules should be importable independently.

### R2. Add a synthetic data mode for local testing

Create a mock data generator that produces a small synthetic connectome (500–1000 neurons with random sparse connectivity, appropriate column schema matching real neuPrint output). The full pipeline must be testable without network access. Activate via `python run.py --mock` or similar flag.

### R3. Validate against real neuPrint data

The user has a neuPrint API token (env var `NEUPRINT_TOKEN`). After mock tests pass, ensure the data fetcher works against `neuprint.janelia.org` dataset `male-cns:v1.0`. The fetch can be tested in isolation via `python -m data.fetch_data --token $NEUPRINT_TOKEN`. Document any API quirks discovered.

### R4. Add a test suite

Write tests (`tests/` directory) covering:
- Flappy Bird game engine: 100 frames without crash, correct reward signals
- LIF engine: accepts input, produces spikes, correct tensor shapes
- Sensory encoder: output shape [N], non-zero photoreceptor entries
- Motor decoder: returns boolean, threshold calibration works
- Plasticity: weight updates are non-zero after reward signal
- End-to-end: `run.py --mock --episodes 2 --no-viz` completes

### R5. Optimize for Mac M4 (MPS backend)

The primary dev machine is Mac M4 16GB. Ensure:
- CPU and MPS backends work (CUDA optional)
- Sparse matrix operations fall back correctly when CSR is unsupported on MPS
- Plasticity's `_find_plastic_synapses` is vectorized (no Python loop over 25M edges)
- Mock mode runs at reasonable speed (>1 fps)

### R6. Create a Kaggle notebook

Produce a `.ipynb` notebook (`notebooks/fly_flappy_bird.ipynb`) that:
- Installs dependencies via `!pip install`
- Sets the neuPrint token from Kaggle secrets or user input
- Runs the full pipeline with visualization
- Shows the dual-panel dashboard inline
- Includes markdown cells explaining the biology and architecture

### R7. Add a README.md

Write a `README.md` at the project root covering:
- Project overview and scientific motivation
- Architecture diagram (text-based or mermaid)
- Setup instructions (local Mac + Kaggle)
- Quick start (mock mode + real data)
- Citation/reference list

## Acceptance Criteria

### Pipeline integrity
- [ ] `python -c "from config import *; from run import main"` succeeds
- [ ] `python run.py --mock --no-viz --episodes 2` completes and prints a final score
- [ ] All `.py` files pass `python -m py_compile <file>` without syntax errors
- [ ] No `ImportError` when importing any module individually

### Synthetic data mode
- [ ] `--mock` flag works without any network access or cached data
- [ ] Synthetic connectome has ≥500 neurons, ≥1000 edges
- [ ] At least some neurons spike during mock runs (not all-zero rates)

### Real data path
- [ ] `python -m data.fetch_data --token $NEUPRINT_TOKEN` completes (may take time)
- [ ] Downloaded data loads correctly via `connectome/builder.py`

### Test suite
- [ ] `python -m pytest tests/ -v` passes all tests
- [ ] Tests complete in under 60 seconds on CPU
- [ ] ≥6 test functions covering the modules listed in R4

### Device compatibility
- [ ] Code runs on CPU without CUDA installed
- [ ] No `RuntimeError` from missing CUDA or MPS
- [ ] ffmpeg absence degrades gracefully (no crash)

### Kaggle notebook
- [ ] Notebook cells execute top-to-bottom without errors (in mock mode)
- [ ] Contains ≥5 markdown explanation cells
- [ ] Dashboard visualization renders inline

### README
- [ ] README.md exists at project root
- [ ] Contains setup instructions for both local and Kaggle
- [ ] Contains at least one architecture/pipeline diagram

Expecting this to run as a full project build across multiple files and modules — not a single contained change.
