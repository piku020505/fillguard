"""Process model of a multi-head piston filler feeding a 100% checkweigher.

All parameters are documented assumptions for a 340 ml shampoo line; they are
inputs, not measured plant data. Replace them with real checkweigher data.
"""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class LineConfig:
    heads: int = 6                 # filling heads
    cycles_per_min: float = 10.0   # 6 heads x 10 cycles = 60 bottles/min
    shift_min: int = 480           # one 8-hour shift
    target_g: float = 347.0        # set point (340 ml x 1.02 g/ml)
    lsl_g: float = 343.5           # lower spec limit: target - 3.5 sigma (assumption)
    usl_g: float = 350.5           # upper spec limit: target + 3.5 sigma (assumption)
    sigma_g: float = 1.0           # within-head repeatability (1 sigma)
    head_bias_sd_g: float = 0.15   # small fixed offset between heads after set-up

    @property
    def cycles(self) -> int:
        return int(self.shift_min * self.cycles_per_min)


@dataclass
class Fault:
    """A known disturbance injected into the simulation (ground truth)."""
    kind: str                 # 'step', 'ramp' or 'variance'
    start_cycle: int
    heads: tuple = ()         # empty tuple = all heads
    size: float = 0.0         # g for step, g per cycle for ramp, sigma multiplier for variance
    cap: float = None         # max |shift| for a ramp, g
    label: str = ""


def simulate_shift(cfg: LineConfig, faults=(), seed: int = 0):
    """Return fill weights as an array of shape (cycles, heads)."""
    rng = np.random.default_rng(seed)
    n, h = cfg.cycles, cfg.heads
    bias = rng.normal(0, cfg.head_bias_sd_g, h)
    mean = np.full((n, h), cfg.target_g) + bias
    sd = np.full((n, h), cfg.sigma_g)
    t = np.arange(n)[:, None]
    for f in faults:
        cols = list(f.heads) if f.heads else list(range(h))
        on = t >= f.start_cycle
        if f.kind == "step":
            mean[:, cols] += np.where(on, f.size, 0.0)
        elif f.kind == "ramp":
            shift = np.where(on, f.size * (t - f.start_cycle), 0.0)
            if f.cap is not None:
                shift = np.sign(shift) * np.minimum(np.abs(shift), abs(f.cap))
            mean[:, cols] += shift
        elif f.kind == "variance":
            sd[:, cols] = np.where(on, cfg.sigma_g * f.size, cfg.sigma_g)
        else:
            raise ValueError(f.kind)
    return mean + rng.standard_normal((n, h)) * sd


# The three disturbances studied (sizes chosen to be realistic but subtle)
NOZZLE_CLOG = Fault("step", 1200, heads=(2,), size=-2.0, label="Partial nozzle clog, head 3 (-2 g step)")
COLD_BULK = Fault("ramp", 1200, size=-0.006, cap=-4.0, label="Bulk cooling raises viscosity, all heads (-0.06 g/min drift)")
VALVE_WEAR = Fault("variance", 1200, heads=(4,), size=2.0, label="Check-valve wear, head 5 (sigma x 2)")
FAULTS = [NOZZLE_CLOG, COLD_BULK, VALVE_WEAR]
