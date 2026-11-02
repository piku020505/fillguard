"""Reproduce every number, table and figure in README.md.

    python run_analysis.py            # ~1-2 minutes

1. Calibrate FillGuard's thresholds so its false-alarm rate on fault-free shifts
   matches the baseline periodic QC check (a fair comparison).
2. Inject three known faults and measure, over 200 Monte Carlo shifts each:
   detection delay, out-of-spec bottles made before detection, and whether the
   faulty head was identified.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fillguard import LineConfig, FAULTS, simulate_shift, periodic_check_alarms, FillGuard, subgroup_stats
from fillguard.evaluate import first_after, out_of_spec_between, PHASE1_CYCLES, N_SUB

ROOT = Path(__file__).parent
FIG, DATA = ROOT / "figures", ROOT / "data"
FIG.mkdir(exist_ok=True); DATA.mkdir(exist_ok=True)
CFG = LineConfig()
N_CAL, N_MC = 80, 200
G, CORAL, GREY = "#1D5C38", "#C9473C", "#8A9A91"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def false_alarms(gf, seed):
    w = simulate_shift(CFG, (), seed=seed)
    _, ma, va = gf.alarms(w)
    return ma.sum() + va.sum()


def baseline_false_alarms(seed):
    w = simulate_shift(CFG, (), seed=seed)
    return len(periodic_check_alarms(w, CFG.target_g, CFG.sigma_g, seed=seed))


# ---------------------------------------------------------------- 1. calibration
base_fa = float(np.mean([baseline_false_alarms(s) for s in range(1000, 1000 + N_CAL)]))
print(f"Baseline periodic check: {base_fa:.2f} false alarms per fault-free shift")

def calibrate(make, grid):
    curve = []
    for thr in grid:
        fa = float(np.mean([false_alarms(make(thr), s) for s in range(2000, 2000 + N_CAL)]))
        curve.append((thr, fa))
        if fa <= base_fa:
            return thr, fa, curve
    return grid[-1], curve[-1][1], curve

# variance monitor first (mean monitor off by using a huge L), then the mean monitor
vL, vfa, vcurve = calibrate(lambda t: FillGuard(L=99, var_L=t), [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.4, 1.6, 2.0])
share = base_fa / 2
def cal_mean(make, grid):
    for thr in grid:
        fa = float(np.mean([false_alarms(make(thr), s) for s in range(2000, 2000 + N_CAL)]))
        if fa <= base_fa:
            return thr, fa
    return grid[-1], fa
eL, efa = cal_mean(lambda t: FillGuard(method="ewma", L=t, var_L=vL), [3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0])
ch, cfa = cal_mean(lambda t: FillGuard(method="cusum", h=t, var_L=vL), [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20])
print(f"Calibrated: variance EWMA L={vL}, mean EWMA L={eL} (FA {efa:.2f}), CUSUM h={ch} (FA {cfa:.2f})")
METHODS = {"FillGuard EWMA": FillGuard(method="ewma", L=eL, var_L=vL),
           "FillGuard CUSUM": FillGuard(method="cusum", h=ch, var_L=vL)}

# ---------------------------------------------------------------- 2. fault experiments
rows = []
for f in FAULTS:
    cols = list(f.heads) if f.heads else list(range(CFG.heads))
    res = {"Periodic QC check": [], **{m: [] for m in METHODS}}
    for s in range(N_MC):
        w = simulate_shift(CFG, [f], seed=s)
        # baseline
        hits = [c for c in periodic_check_alarms(w, CFG.target_g, CFG.sigma_g, seed=s) if c >= f.start_cycle]
        det = hits[0] if hits else -1
        stop = det if det >= 0 else CFG.cycles
        res["Periodic QC check"].append(dict(det=det, delay=(stop - f.start_cycle) / CFG.cycles_per_min,
                                             oos=out_of_spec_between(w, CFG.lsl_g, CFG.usl_g, f.start_cycle, stop), head_ok=False))
        for name, gf in METHODS.items():
            cyc, ma, va = gf.alarms(w)
            a = va if f.kind == "variance" else ma
            det, col = first_after(a, cyc, f.start_cycle)
            stop = det if det >= 0 else CFG.cycles
            res[name].append(dict(det=det, delay=(stop - f.start_cycle) / CFG.cycles_per_min,
                                  oos=out_of_spec_between(w, CFG.lsl_g, CFG.usl_g, f.start_cycle, stop),
                                  head_ok=(col in cols) if f.heads else None))
    for name, r in res.items():
        d = np.array([x["delay"] for x in r]); o = np.array([x["oos"] for x in r])
        caught = np.mean([x["det"] >= 0 for x in r])
        rows.append(dict(fault=f.label, method=name, caught_in_shift=float(caught),
                         median_delay_min=float(np.median(d)), p90_delay_min=float(np.percentile(d, 90)),
                         median_oos=float(np.median(o)), mean_oos=float(np.mean(o)),
                         head_identified=(float(np.mean([x["head_ok"] for x in r])) if (name != "Periodic QC check" and f.heads) else None),
                         delays=d.tolist(), oos=o.tolist()))

# ---------------------------------------------------------------- 3. outputs
summary = {"config": CFG.__dict__ | {"cycles": CFG.cycles}, "n_monte_carlo": N_MC,
           "false_alarms_per_shift": {"Periodic QC check": base_fa, "FillGuard EWMA": efa, "FillGuard CUSUM": cfa},
           "thresholds": {"mean_ewma_L": eL, "cusum_h": ch, "var_ewma_L": vL},
           "results": [{k: v for k, v in r.items() if k not in ("delays", "oos")} for r in rows]}
(ROOT / "results.json").write_text(json.dumps(summary, indent=2))

lines = ["| Fault | Method | Caught in shift | Median delay (min) | 90th pct delay (min) | Median out-of-spec bottles before detection | Faulty head identified |",
         "|---|---|---|---|---|---|---|"]
for r in rows:
    hid = ("all heads" if "all heads" in r["fault"] and r["method"] != "Periodic QC check" else "not possible") if r["head_identified"] is None else f"{r['head_identified']:.0%}"
    lines.append(f"| {r['fault']} | {r['method']} | {r['caught_in_shift']:.0%} | {r['median_delay_min']:.1f} | {r['p90_delay_min']:.1f} | {r['median_oos']:.0f} | {hid} |")
(ROOT / "results.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

# figure 1: example shift, nozzle clog
f = FAULTS[0]
w = simulate_shift(CFG, [f], seed=7)
m, _ = subgroup_stats(w[PHASE1_CYCLES:], N_SUB)
cyc = PHASE1_CYCLES + (np.arange(m.shape[0]) + 1) * N_SUB
gf = METHODS["FillGuard EWMA"]; c2, ma, _ = gf.alarms(w)
det, col = first_after(ma, c2, f.start_cycle)
hits = [c for c in periodic_check_alarms(w, CFG.target_g, CFG.sigma_g, seed=7) if c >= f.start_cycle]
fig, ax = plt.subplots(figsize=(10, 4.2))
for j in range(CFG.heads):
    ax.plot(cyc / 10 / 60, m[:, j], lw=0.8, color=CORAL if j == 2 else GREY, alpha=1 if j == 2 else 0.45, label="Head 3 (clogging)" if j == 2 else ("Other heads" if j == 0 else None))
ax.axhline(CFG.lsl_g, ls="--", color="k", lw=0.8); ax.text(7.6, CFG.lsl_g + 0.12, "LSL", fontsize=8)
X = ax.get_xaxis_transform()
ax.axvline(f.start_cycle / 600, color="k", lw=0.8, ls=":"); ax.text(f.start_cycle / 600 - 0.05, 0.97, "fault starts", fontsize=8, ha="right", va="top", transform=X)
if det > 0: ax.axvline(det / 600, color=G, lw=2); ax.text(det / 600 + 0.08, 0.97, f"FillGuard alarm: +{(det - f.start_cycle) / 10:.1f} min, head {col + 1}", color=G, fontsize=8.5, va="top", fontweight="bold", transform=X, bbox=dict(fc="white", ec="none", pad=1))
if hits: ax.axvline(hits[0] / 600, color=CORAL, lw=2, ls="--"); ax.text(hits[0] / 600 + 0.08, 0.89, f"Periodic QC alarm: +{(hits[0] - f.start_cycle) / 10:.0f} min", color=CORAL, fontsize=8.5, va="top", transform=X, bbox=dict(fc="white", ec="none", pad=1))
ax.set_xlabel("Hours into shift"); ax.set_ylabel("Fill weight, 30-s mean per head (g)")
ax.set_title("Example shift: a partial nozzle clog on head 3", loc="left", fontweight="bold"); ax.legend(frameon=True, loc="lower right", ncol=2, framealpha=0.9)
ax.set_ylim(CFG.lsl_g - 0.6, None)
fig.tight_layout(); fig.savefig(FIG / "example_nozzle_clog.png", dpi=160); plt.close(fig)

# figure 2: detection delay boxplots
fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
for ax, f in zip(axes, FAULTS):
    sub = [r for r in rows if r["fault"] == f.label]
    ax.boxplot([r["delays"] for r in sub], tick_labels=[r["method"].replace("FillGuard ", "FG\n").replace("Periodic QC check", "Periodic\nQC") for r in sub],
               patch_artist=True, boxprops=dict(facecolor="#E2F3E8"), medianprops=dict(color=G, lw=2), showfliers=False)
    ax.set_title(f.label.split(",")[0], fontsize=10, loc="left"); ax.set_ylabel("Minutes to detection")
fig.suptitle(f"Detection delay over {N_MC} simulated shifts per fault (lower is better)", x=0.01, ha="left", fontweight="bold")
fig.tight_layout(); fig.savefig(FIG / "detection_delay.png", dpi=160); plt.close(fig)

# figure 3: out-of-spec bottles before detection
fig, ax = plt.subplots(figsize=(10, 4))
labels = [f.label.split(",")[0] for f in FAULTS]; x = np.arange(len(labels)); wd = 0.26
for i, (meth, colr) in enumerate([("Periodic QC check", CORAL), ("FillGuard EWMA", G), ("FillGuard CUSUM", "#58B97F")]):
    vals = [next(r["mean_oos"] for r in rows if r["fault"] == f.label and r["method"] == meth) for f in FAULTS]
    b = ax.bar(x + (i - 1) * wd, vals, wd, color=colr, label=meth); ax.bar_label(b, fmt="%.0f", fontsize=8)
ax.set_xticks(x, labels); ax.set_ylabel("Out-of-spec bottles before detection (mean)"); ax.legend(frameon=False)
ax.set_title("Bad bottles made before the problem is caught", loc="left", fontweight="bold")
fig.tight_layout(); fig.savefig(FIG / "out_of_spec_before_detection.png", dpi=160); plt.close(fig)

# sample data for anyone to inspect
import csv
w = simulate_shift(CFG, [FAULTS[0]], seed=7)
with open(DATA / "example_shift_nozzle_clog.csv", "w", newline="") as fh:
    wr = csv.writer(fh); wr.writerow(["cycle", "minute", "head", "fill_g"])
    for c in range(w.shape[0]):
        for j in range(w.shape[1]):
            wr.writerow([c, round(c / CFG.cycles_per_min, 1), j + 1, round(float(w[c, j]), 2)])
print("done")
