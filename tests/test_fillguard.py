import numpy as np
from fillguard import LineConfig, Fault, simulate_shift, FillGuard, subgroup_stats
from fillguard.detect import ewma_alarms, cusum_alarms
from fillguard.evaluate import first_after

CFG = LineConfig()


def test_shape_and_in_control_mean():
    w = simulate_shift(CFG, (), seed=1)
    assert w.shape == (CFG.cycles, CFG.heads)
    assert abs(w.mean() - CFG.target_g) < 0.2
    assert abs(w.std() - CFG.sigma_g) < 0.1


def test_step_fault_only_hits_its_head():
    f = Fault("step", 100, heads=(2,), size=-2.0)
    a = simulate_shift(CFG, (), seed=3); b = simulate_shift(CFG, [f], seed=3)
    d = (b - a)[100:]
    assert np.allclose(d[:, 2], -2.0) and np.allclose(np.delete(d, 2, axis=1), 0)


def test_ramp_is_capped_and_signed():
    f = Fault("ramp", 0, size=-0.01, cap=-1.5)
    d = simulate_shift(CFG, [f], seed=4) - simulate_shift(CFG, (), seed=4)
    assert d.max() <= 0 and np.isclose(d.min(), -1.5)


def test_subgroup_stats():
    w = np.arange(20, dtype=float).reshape(10, 2)
    m, v = subgroup_stats(w, 5)
    assert np.allclose(m[:, 0], [4, 14]) and m.shape == (2, 2)


def test_monitors_flag_a_large_shift_quickly():
    x = np.vstack([np.zeros((50, 1)), np.full((50, 1), 3.0)])
    for a in (ewma_alarms(x, 0.0, 1.0), cusum_alarms(x, 0.0, 1.0)):
        assert not a[:50].any() and a[50:56].any()


def test_fillguard_finds_the_clogged_head():
    f = Fault("step", 1200, heads=(2,), size=-2.0)
    w = simulate_shift(CFG, [f], seed=11)
    cyc, ma, _ = FillGuard(L=4.4).alarms(w)
    det, col = first_after(ma, cyc, 1200)
    assert col == 2 and 0 <= det - 1200 <= 60
