"""FillGuard: early warning for fill-weight drift on a multi-head filling line."""
from .model import LineConfig, Fault, FAULTS, simulate_shift
from .detect import subgroup_stats, ewma_alarms, cusum_alarms, var_ewma_alarms, periodic_check_alarms
from .evaluate import FillGuard
