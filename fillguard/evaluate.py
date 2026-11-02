"""FillGuard monitor (Phase I baseline + Phase II monitoring) and evaluation helpers."""
from dataclasses import dataclass
import numpy as np
from .detect import subgroup_stats, ewma_alarms, cusum_alarms, var_ewma_alarms

N_SUB = 5            # fills per head per subgroup
PHASE1_CYCLES = 200  # first 20 minutes after start-up used to learn each head's mean


@dataclass
class FillGuard:
    sigma: float = 1.0
    lam: float = 0.2
    L: float = 3.0
    k: float = 0.5
    h: float = 5.0
    var_lam: float = 0.1
    var_L: float = 1.0
    method: str = "ewma"  # 'ewma' or 'cusum'

    def alarms(self, w):
        """Return (cycle_of_each_subgroup, mean_alarms, var_alarms) for the monitoring phase."""
        mu0 = w[:PHASE1_CYCLES].mean(axis=0)               # Phase I: per-head mean
        m, v = subgroup_stats(w[PHASE1_CYCLES:], N_SUB)
        sd = self.sigma / np.sqrt(N_SUB)
        if self.method == "ewma":
            ma = ewma_alarms(m, mu0, sd, self.lam, self.L)
        else:
            ma = cusum_alarms(m, mu0, sd, self.k, self.h)
        va = var_ewma_alarms(v, self.sigma ** 2, self.var_lam, self.var_L)
        cyc = PHASE1_CYCLES + (np.arange(m.shape[0]) + 1) * N_SUB
        return cyc, ma, va


def first_after(cycles_bool_rows, cyc, onset, cols=None):
    """First alarm cycle at/after onset, and the column that raised it."""
    a = cycles_bool_rows if cols is None else cycles_bool_rows[:, cols]
    idx = np.where((cyc >= onset)[:, None] & a)
    if idx[0].size == 0:
        return -1, -1
    j = np.argmin(idx[0])
    col = idx[1][j] if cols is None else cols[idx[1][j]]
    return int(cyc[idx[0][j]]), int(col)


def out_of_spec_between(w, lsl, usl, start, stop):
    seg = w[start:stop]
    return int(((seg < lsl) | (seg > usl)).sum())
