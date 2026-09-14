# 🪰 FlyFlappyBird

**A biologically grounded neural simulation where the complete *Drosophila* male CNS connectome plays Flappy Bird.**

166,700 neurons. 25 million synapses. Real connectome data. Dopamine-gated learning.

## Overview

FlyFlappyBird connects the [Berg et al. (2026)](https://doi.org/10.1016/j.cell.2025.10.045) male *Drosophila* CNS connectome to a Flappy Bird game engine. The fly "sees" the game through photoreceptor neurons (R1–R6), processes visual information through its biological visual circuits, and generates wing motor commands through descending neurons — all simulated with biophysically accurate Leaky Integrate-and-Fire (LIF) dynamics. The fly learns via dopamine-gated reward-modulated Hebbian plasticity (PPL101 → mushroom body circuit).

Inspired by [DoomFly](https://github.com/awormuth/DoomFly) (Wormuth 2024).

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    FlyFlappyBird Pipeline                           │
│                                                                     │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────────────┐  │
│  │  Flappy   │───▶│   Sensory    │───▶│      LIF Engine          │  │
│  │  Bird     │    │   Encoder    │    │  (166K neurons, sparse   │  │
│  │  Engine   │    │  (R1–R6      │    │   SpMV on GPU/MPS/CPU)   │  │
│  │  (NumPy)  │    │   frontal    │    │                          │  │
│  └─────▲─────┘    │   visual     │    │  ┌──────────────────┐    │  │
│        │          │   field)     │    │  │  Connectome       │    │  │
│        │          └──────────────┘    │  │  (Berg et al.     │    │  │
│   ┌────┴─────┐                       │  │   2026 MaleCNS)   │    │  │
│   │  Motor   │◀──────────────────────│  └──────────────────┘    │  │
│   │  Decoder │   Wing pre-motor      │                          │  │
│   │  (TN1A,  │   spike rates         │  ┌──────────────────┐    │  │
│   │  vPR9,   │                       │  │  PPL101 Plasticity│    │  │
│   │  dPR1)   │                       │  │  (DA-gated STDP   │    │  │
│   └──────────┘                       │  │   → MB/KC)        │    │  │
│                                      │  └──────────────────┘    │  │
│  ┌──────────────────────────────┐    └──────────────────────────┘  │
│  │  Dashboard (matplotlib)      │                                   │
│  │  Game view + Neural activity │                                   │
│  └──────────────────────────────┘                                   │
└─────────────────────────────────────────────────────────────────────┘
```

### Signal Flow (biological)

```
Game Frame → R1–R6 Photoreceptors → Visual Projection Neurons (LoVP92)
  → Central Brain Hubs → Descending Neurons (DNg13, DNa02)
  → VNC Wing Pre-Motor (TN1A, vPR9, dPR1) → Flap Action

Reward: PPL101 dopamine burst on pipe clear (+1.0), dip on collision (−5.0)
  → Three-factor eligibility-trace Hebbian STDP on KC synapses
```

## Setup

### Local (Mac M4 / CPU)

```bash
cd fly_flappy_bird
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Kaggle (2× T4 GPU)

Use the provided notebook: `notebooks/fly_flappy_bird.ipynb`

Or install manually:
```bash
!pip install -q neuprint-python torch numpy scipy matplotlib Pillow tqdm
```

## Quick Start

### Mock mode (no neuPrint account needed)

# Run with synthetic 600-neuron connectome and record video (fly_plays_flappy.mp4)
python run.py --mock --episodes 5

# Fast headless benchmark run (no dashboard or video recording)
python run.py --mock --no-viz --episodes 5

# With structured (feature-based) sensory encoder
python run.py --mock --no-viz --episodes 10 --structured-encoder
```

### Real data mode

```bash
# 1. Set your neuPrint API token
export NEUPRINT_TOKEN=your_token_here

# 2. Download MaleCNS data (~20-60 min first run, cached thereafter)
python -m data.fetch_data

# 3. Run full simulation
python run.py --episodes 200
```

Get your neuPrint token at [neuprint.janelia.org](https://neuprint.janelia.org) → Login → Account.

## Testing

```bash
python -m pytest tests/ -v
```

42 tests covering packaging, device resolution, game engine, LIF dynamics, sensory encoding, motor decoding, plasticity, and full end-to-end pipeline.

## Project Structure

```
fly_flappy_bird/
├── config.py                 # Hyperparameters, paths, device selection
├── run.py                    # Main training loop orchestrator
├── requirements.txt          # Dependencies
├── README.md
├── data/
│   ├── fetch_data.py         # neuPrint data fetcher (Cypher queries)
│   └── mock_data.py          # Synthetic connectome generator
├── connectome/
│   ├── builder.py            # Sparse weight tensor with NT sign correction
│   └── populations.py        # Neuron population identification
├── simulation/
│   ├── lif.py                # Vectorised LIF engine (GPU/MPS/CPU)
│   ├── sensory.py            # Game frame → photoreceptor current
│   ├── motor.py              # Wing motor → flap decision
│   └── plasticity.py         # PPL101 dopamine-gated R-STDP
├── game/
│   └── flappy.py             # Headless Flappy Bird (NumPy renderer)
├── viz/
│   └── dashboard.py          # Dual-panel matplotlib dashboard
├── tests/
│   ├── test_pipeline.py      # Core pipeline tests
│   ├── test_m1_packaging.py  # Import and device tests
│   └── test_m1_adversarial_edge_cases.py
└── notebooks/
    └── fly_flappy_bird.ipynb  # Kaggle demo notebook
```

## References

1. **Berg, S., et al.** (2026). Sexual dimorphism in the complete *Drosophila* male central nervous system. *Cell*, 189, 5504–5526. [doi:10.1016/j.cell.2025.10.045](https://doi.org/10.1016/j.cell.2025.10.045)

2. **Lappalainen, J.K., et al.** (2024). Connectome-constrained networks predict neural activity across the fly visual system. *Nature*, 634, 1132–1140. [doi:10.1038/s41586-024-07939-3](https://doi.org/10.1038/s41586-024-07939-3)

3. **Aso, Y., et al.** (2014). The neuronal architecture of the mushroom body provides a logic for associative learning. *eLife*, 3:e04577. [doi:10.7554/eLife.04577](https://doi.org/10.7554/eLife.04577)

4. **Dayan, P. & Abbott, L.F.** (2001). *Theoretical Neuroscience: Computational and Mathematical Modeling of Neural Systems*. MIT Press. Ch. 5: Model Neurons I — Leaky Integrate-and-Fire.

5. **Wormuth, A.** (2024). DoomFly: MaleCNS connectome playing Doom. [GitHub](https://github.com/awormuth/DoomFly)

## License

Research/educational use. The MaleCNS connectome data is from [neuPrint](https://neuprint.janelia.org) (Janelia Research Campus, HHMI).
