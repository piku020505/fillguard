"""Interactive FillGuard demo:  streamlit run app.py"""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from fillguard import LineConfig, Fault, simulate_shift, periodic_check_alarms, FillGuard, subgroup_stats
from fillguard.evaluate import first_after, out_of_spec_between, PHASE1_CYCLES, N_SUB

st.set_page_config(page_title="FillGuard", layout="wide")
st.title("FillGuard · early warning for fill-weight drift")
st.caption("Simulated 6-head filling line (340 ml shampoo, 60 bottles/min). Compare the periodic QC check with per-head monitoring of 100% checkweigher data.")

with st.sidebar:
    st.header("Inject a fault")
    kind = st.selectbox("Fault", ["Partial nozzle clog (step)", "Bulk cooling (drift, all heads)", "Check-valve wear (more scatter)", "No fault"])
    head = st.slider("Head", 1, 6, 3)
    size = st.slider("Size", 0.5, 4.0, 2.0, 0.5, help="g for a step, g per 10 min for a drift, sigma multiplier for wear")
    start_h = st.slider("Fault starts at (hours)", 1.0, 6.0, 2.0, 0.5)
    method = st.radio("FillGuard statistic", ["ewma", "cusum"], horizontal=True)
    seed = st.number_input("Random seed", 0, 9999, 7)

cfg = LineConfig()
start = int(start_h * 60 * cfg.cycles_per_min)
faults = {"Partial nozzle clog (step)": [Fault("step", start, (head - 1,), -size)],
          "Bulk cooling (drift, all heads)": [Fault("ramp", start, (), -size / 100, cap=-4.0)],
          "Check-valve wear (more scatter)": [Fault("variance", start, (head - 1,), 1 + size / 2)],
          "No fault": []}[kind]
w = simulate_shift(cfg, faults, seed=int(seed))
gf = FillGuard(method=method, L=4.4, h=13, var_L=0.8)
cyc, ma, va = gf.alarms(w)
alarms = ma | va
fg_det, fg_col = first_after(alarms, cyc, start)
pq = [c for c in periodic_check_alarms(w, cfg.target_g, cfg.sigma_g, seed=int(seed)) if c >= start]
pq_det = pq[0] if pq else -1

c1, c2, c3 = st.columns(3)
fmt = lambda d: "not caught" if d < 0 else f"{(d - start) / cfg.cycles_per_min:.1f} min"
c1.metric("FillGuard: time to detect", fmt(fg_det), f"head {fg_col + 1}" if fg_col >= 0 else None)
c2.metric("Periodic QC: time to detect", fmt(pq_det))
stop_fg = fg_det if fg_det >= 0 else cfg.cycles; stop_pq = pq_det if pq_det >= 0 else cfg.cycles
c3.metric("Bad bottles before detection", f"{out_of_spec_between(w, cfg.lsl_g, cfg.usl_g, start, stop_fg)} vs {out_of_spec_between(w, cfg.lsl_g, cfg.usl_g, start, stop_pq)}", help="FillGuard vs periodic check")

m, _ = subgroup_stats(w[PHASE1_CYCLES:], N_SUB)
hrs = cyc / cfg.cycles_per_min / 60
fig = go.Figure()
for j in range(cfg.heads):
    fig.add_trace(go.Scatter(x=hrs, y=m[:, j], mode="lines", name=f"Head {j + 1}", line=dict(width=1)))
fig.add_hline(cfg.lsl_g, line_dash="dash", annotation_text="LSL")
fig.add_vline(start_h, line_dash="dot", annotation_text="fault")
if fg_det >= 0: fig.add_vline(fg_det / 600, line_color="#1D5C38", annotation_text="FillGuard")
if pq_det >= 0: fig.add_vline(pq_det / 600, line_color="#C9473C", line_dash="dash", annotation_text="Periodic QC")
fig.update_layout(height=430, xaxis_title="Hours into shift", yaxis_title="30-s mean fill per head (g)", margin=dict(t=20))
st.plotly_chart(fig, use_container_width=True)
st.caption("Thresholds were calibrated so FillGuard raises no more false alarms per fault-free shift than the periodic check (see README). All data are simulated.")
