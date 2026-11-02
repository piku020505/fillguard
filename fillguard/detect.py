"""Detection methods.

Baseline : the periodic QC check (weigh 5 random bottles every 5 minutes, X-bar 3-sigma rule).
FillGuard: per-head monitors on 100% checkweigher data, grouped into subgroups of 5 fills
           per head (one subgroup = 30 s): EWMA and CUSUM for the mean, EWMA for the variance.
Each monitor returns a boolean alarm matrix (subgroups x heads); after an alarm the
statistic resets, as it would after an operator acknowledges and corrects.
"""
import numpy as np


def subgroup_stats(w, n=5):
    """Per-head means and variances of consecutive subgroups of n fills."""
    k = w.shape[0] // n
    g = w[: k * n].reshape(k, n, w.shape[1])
    return g.mean(axis=1), g.var(axis=1, ddof=1)


def ewma_alarms(x, mu0, sd, lam=0.2, L=3.0):
    lim = L * sd * np.sqrt(lam / (2 - lam))
    z = np.array(mu0, dtype=float) * np.ones(x.shape[1])
    out = np.zeros(x.shape, bool)
    for i in range(x.shape[0]):
        z = lam * x[i] + (1 - lam) * z
        hit = np.abs(z - mu0) > lim
        out[i] = hit
        z = np.where(hit, mu0, z)
    return out


def cusum_alarms(x, mu0, sd, k=0.5, h=5.0):
    sp = np.zeros(x.shape[1]); sn = np.zeros(x.shape[1])
    out = np.zeros(x.shape, bool)
    for i in range(x.shape[0]):
        d = (x[i] - mu0) / sd
        sp = np.maximum(0, sp + d - k); sn = np.maximum(0, sn - d - k)
        hit = (sp > h) | (sn > h)
        out[i] = hit
        sp = np.where(hit, 0, sp); sn = np.where(hit, 0, sn)
    return out


def var_ewma_alarms(v, s2, lam=0.1, L=1.0):
    z = np.full(v.shape[1], s2, dtype=float)
    out = np.zeros(v.shape, bool)
    for i in range(v.shape[0]):
        z = lam * v[i] + (1 - lam) * z
        hit = z > s2 * (1 + L)
        out[i] = hit
        z = np.where(hit, s2, z)
    return out


def periodic_check_alarms(w, target, sigma, every_cycles=50, n=5, seed=0):
    """Return the list of cycles at which the periodic QC check raises an alarm."""
    rng = np.random.default_rng(seed + 10_000)
    lim = 3 * sigma / np.sqrt(n)
    hits = []
    for c in range(every_cycles, w.shape[0], every_cycles):
        heads = rng.integers(0, w.shape[1], n)
        cyc = c - rng.integers(0, 3, n)
        if abs(w[cyc, heads].mean() - target) > lim:
            hits.append(c)
    return hits
