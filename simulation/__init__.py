"""
simulation package — Vectorized LIF neural simulation, sensory encoder, motor decoder, and dopamine plasticity.
"""

from simulation.lif import LIFEngine
from simulation.sensory import SensoryEncoder
from simulation.motor import MotorDecoder
from simulation.plasticity import PPL101Plasticity

__all__ = [
    "LIFEngine",
    "SensoryEncoder",
    "MotorDecoder",
    "PPL101Plasticity",
]
