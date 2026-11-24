import sys; sys.path.insert(0, "/home/claude/work")
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, json
from sklearn.model_selection import KFold
from sklearn.inspection import permutation_importance
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from synth import simulate, BASE, HIST
d, _ = simulate("decoupled"); FW = BASE + HIST
tr, te = next(KFold(5, shuffle=True, random_state=0).split(d))
m = make_pipeline(StandardScaler(), LinearRegression()).fit(d.iloc[tr][FW], d.iloc[tr].soh)
pi = permutation_importance(m, d.iloc[te][FW], d.iloc[te].soh, n_repeats=20, random_state=0)
R = json.load(open("out/results.json"))
assert all(abs(R["perm_importance"][k] - v) < 1e-9 for k, v in zip(FW, pi.importances_mean)), "mismatch"
lab = {"efc": "Cumulative EFC", "dod_mean": "Mean DOD", "dod_p90": "DOD 90th pct.", "fc_frac": "Fast-charge fraction",
       "temp_mean": "Mean temperature", "age": "Pack age", "ir_growth": "IR growth", "ce_trend": "CE trend",
       "ir_slope": "IR slope (6 mo)", "efc_rate": "EFC rate (6 mo)"}
o = np.argsort(pi.importances_mean)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(4.8, 3.0))
ax.barh([lab[FW[i]] for i in o], pi.importances_mean[o], xerr=pi.importances_std[o], color="#2a6f97")
ax.set_xlabel("Permutation importance (drop in R$^2$)")
fig.savefig("figs/fig8_importance.png", dpi=300, bbox_inches="tight"); print("ok, matches results.json")
