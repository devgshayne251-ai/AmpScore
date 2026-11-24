"""All experiments for the revised AmpScore manuscript. Writes out/results.json and figs/*.png."""
import json, sys, warnings, numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, roc_auc_score
from xgboost import XGBRegressor
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from synth import simulate, BASE, HIST, USAGE, ELEC

warnings.filterwarnings("ignore")
REAL = sys.argv[1] if len(sys.argv) > 1 else "out/real_cycles.csv"
RES = {}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight"})
C = ["#2a6f97", "#e07a5f", "#3d405b", "#81b29a", "#f2cc8f", "#9c6644", "#6d597a", "#b5838d"]

def metrics(y, p):
    return dict(r2=float(r2_score(y, p)), mae=float(100 * mean_absolute_error(y, p)),
                rmse=float(100 * np.sqrt(mean_squared_error(y, p))), p90=float(100 * np.quantile(np.abs(y - p), 0.9)))

# monotone constraints: SoH non-increasing in EFC, DOD, fast-charge, temp, age, IR growth; non-decreasing in CE
MONO = {"efc": -1, "dod_mean": -1, "dod_p90": -1, "fc_frac": -1, "temp_mean": -1, "age": -1, "ir_growth": -1,
        "ce_trend": 1, "ir_slope": -1, "efc_rate": -1, "cycle": -1, "temp": 0, "days": -1}

def proposed(feats, loss="squared_error", q=None):
    kw = dict(max_iter=300, max_depth=3, learning_rate=0.05, random_state=42,
              monotonic_cst=[MONO.get(f, 0) for f in feats])
    if loss == "quantile": kw.update(loss="quantile", quantile=q)
    return HistGradientBoostingRegressor(**kw)

from sklearn.linear_model import QuantileRegressor
def linq(feats, loss="squared_error", q=None):
    if loss == "quantile":
        return make_pipeline(StandardScaler(), QuantileRegressor(quantile=q, alpha=0.0, solver="highs"))
    return make_pipeline(StandardScaler(), LinearRegression())
FAMILIES = {"linear": linq, "mono-HGB": proposed}

def baselines(feats):
    return {"Linear regression": make_pipeline(StandardScaler(), LinearRegression()),
            "Random forest": RandomForestRegressor(300, max_depth=6, random_state=42, n_jobs=-1),
            "SVR (RBF)": make_pipeline(StandardScaler(), SVR(C=10, epsilon=0.005)),
            "XGBoost": XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42, verbosity=0),
            "Gradient boosting (orig.)": GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42),
            "Monotone HGB": proposed(feats)}

def repcv(df, feats, target, models, reps=5, k=5):
    """5x5 repeated CV; returns per-fold metrics per model."""
    out = {m: [] for m in models}
    X, y = df[feats].values, df[target].values
    for r in range(reps):
        for tr, te in KFold(k, shuffle=True, random_state=r).split(X):
            for m, est in models.items():
                est.fit(X[tr], y[tr]); out[m].append(metrics(y[te], est.predict(X[te])))
    return out

def corrected_ttest(a, b, n_tr, n_te):
    """Nadeau-Bengio corrected resampled t-test for repeated k-fold."""
    d = np.asarray(a) - np.asarray(b); J = len(d)
    var = np.var(d, ddof=1) * (1 / J + n_te / n_tr)
    t = d.mean() / np.sqrt(var) if var > 0 else 0.0
    return float(2 * stats.t.sf(abs(t), J - 1))

def summarize(out):
    return {m: {k: [float(np.mean([f[k] for f in v])), float(np.std([f[k] for f in v]))] for k in v[0]} for m, v in out.items()}

# ---------------------------------------------------------------- 1. coupled vs decoupled, baselines
cohorts = {k: simulate(k) for k in ["coupled", "decoupled"]}
RES["cohort"] = {k: {"soh_mean": float(d.soh.mean()), "soh_sd": float(d.soh.std()), "soh_h_mean": float(d.soh_h.mean()),
                     "corr_ir_soh": float(d[["ir_growth", "soh"]].corr().iloc[0, 1]),
                     "corr_ce_soh": float(d[["ce_trend", "soh"]].corr().iloc[0, 1])} for k, (d, h) in cohorts.items()}
RES["baselines"] = {}
for k, (d, h) in cohorts.items():
    feats = BASE if k == "coupled" else BASE + HIST
    out = repcv(d, feats, "soh", baselines(feats))
    RES["baselines"][k] = summarize(out)
    ref = "Linear regression"
    RES["baselines"][k + "_pvals_mae"] = {m: corrected_ttest([f["mae"] for f in out[m]], [f["mae"] for f in out[ref]], 480, 120)
                                          for m in out if m != ref}
print(json.dumps(RES["baselines"], indent=1)[:3000])

# ---------------------------------------------------------------- 2. ablation (both cohorts)
def cv_one(d, feats, target="soh", est=None):
    out = repcv(d, feats, target, {"m": est or linq(feats)}, reps=3)
    return {k: float(np.mean([f[k] for f in out["m"]])) for k in out["m"][0]}

RES["ablation"] = {}
for k, (d, h) in cohorts.items():
    full = BASE if k == "coupled" else BASE + HIST
    sets = {"All features": full,
            "Without IR growth": [f for f in full if f not in ("ir_growth", "ir_slope")],
            "Without CE trend": [f for f in full if f != "ce_trend"],
            "Usage-pattern only": USAGE,
            "Electrical-health only (IR, CE)": ELEC}
    RES["ablation"][k] = {n: cv_one(d, f) for n, f in sets.items()}
    dn = d.copy(); rng = np.random.default_rng(7)
    dn["ir_growth"] += rng.normal(0, 0.05, len(dn)); dn["ce_trend"] += rng.normal(0, 0.0005, len(dn))
    RES["ablation"][k]["All + extra IR/CE sensor noise"] = cv_one(dn, full)
print(json.dumps(RES["ablation"], indent=1))

# ---------------------------------------------------------------- 3. forward-looking, uncertainty-aware score (decoupled)
d, h = cohorts["decoupled"]
FW = BASE + HIST
H = 12
TAUS = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
ZT = stats.norm.ppf(TAUS)

def qforecast(fam, dtr, dte, feats, target, alpha=0.1, seed=0):
    """Multi-quantile forecast + conformalized (CQR) widening on a held-out calibration split."""
    rng = np.random.default_rng(seed); idx = rng.permutation(len(dtr)); ncal = len(dtr) // 4
    cal, fit = dtr.iloc[idx[:ncal]], dtr.iloc[idx[ncal:]]
    ms = [FAMILIES[fam](feats, "quantile", t).fit(fit[feats], fit[target]) for t in TAUS]
    qc = np.sort(np.c_[[m.predict(cal[feats]) for m in ms]].T, axis=1)
    qt = np.sort(np.c_[[m.predict(dte[feats]) for m in ms]].T, axis=1)
    s_ = np.maximum(qc[:, 0] - cal[target].values, cal[target].values - qc[:, -1])
    qhat = np.quantile(s_, min(1, np.ceil((ncal + 1) * (1 - alpha)) / ncal))
    return qt + qhat * (ZT / ZT[-1])[None, :]           # widen outer quantiles most, median untouched

def qfun(Q, u):
    """Evaluate piecewise-linear quantile function with Gaussian tails at probabilities u."""
    out = np.empty((len(Q), len(u)))
    zu = stats.norm.ppf(u)
    for j, uu in enumerate(u):
        if uu < TAUS[0]:
            sd = (Q[:, 1] - Q[:, 0]) / (ZT[1] - ZT[0]); out[:, j] = Q[:, 0] + sd * (zu[j] - ZT[0])
        elif uu > TAUS[-1]:
            sd = (Q[:, -1] - Q[:, -2]) / (ZT[-1] - ZT[-2]); out[:, j] = Q[:, -1] + sd * (zu[j] - ZT[-1])
        else:
            k = np.searchsorted(TAUS, uu) - 1; k = min(max(k, 0), len(TAUS) - 2)
            w = (uu - TAUS[k]) / (TAUS[k + 1] - TAUS[k]); out[:, j] = (1 - w) * Q[:, k] + w * Q[:, k + 1]
    return out
UG = np.linspace(0.001, 0.999, 999)
def prob_below(Q, thr):
    qq = qfun(Q, UG); return (qq < thr).mean(1)

S_GUAR = 0.80
fwres, oofs = {}, {}
for fam in FAMILIES:
    oof = d[["pack", "soh", "soh_h"]].copy()
    Qall = np.zeros((len(d), len(TAUS))); soh_hat = np.zeros(len(d)); naive = np.zeros(len(d))
    for f, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=0).split(d)):
        dtr, dte = d.iloc[tr], d.iloc[te]
        Qall[te] = qforecast(fam, dtr, dte, FW, "soh_h", seed=f)
        soh_hat[te] = FAMILIES[fam](FW).fit(dtr[FW], dtr["soh"]).predict(dte[FW])
        naive[te] = soh_hat[te] - float((dtr["soh"] - dtr["soh_h"]).mean())
    y = d["soh_h"].values
    oof["soh_hat"], oof["naive_h"], oof["med"], oof["lo"], oof["hi"] = soh_hat, naive, Qall[:, 3], Qall[:, 0], Qall[:, -1]
    oof["p_bad"] = prob_below(Qall, S_GUAR)
    ev = (y < S_GUAR).astype(int)
    fwres[fam] = {"persistence": metrics(y, soh_hat), "current_soh_minus_popfade": metrics(y, naive),
                  "forward_median": metrics(y, Qall[:, 3]),
                  "coverage_90": float(np.mean((y >= oof.lo) & (y <= oof.hi))), "mean_width_pp": float(100 * (oof.hi - oof.lo).mean()),
                  "auc_current_soh": float(roc_auc_score(ev, -soh_hat)), "auc_p_bad": float(roc_auc_score(ev, oof.p_bad)),
                  "brier_p_bad": float(np.mean((oof.p_bad - ev) ** 2)), "event_rate": float(ev.mean())}
    oofs[fam] = (oof, Qall)
RES["forward"] = {"H_months": H, "S_guar": S_GUAR, "families": fwres}
FAM = min(fwres, key=lambda k: fwres[k]["forward_median"]["mae"])   # chosen by validation MAE
RES["forward"]["selected_family"] = FAM
oof, Qall = oofs[FAM]
# same current SoH, different futures
oof["bin"] = pd.cut(oof["soh"], np.arange(0.80, 1.06, 0.02))
g = oof.groupby("bin", observed=True)["soh_h"].agg(["count", "std", lambda s: s.max() - s.min()])
g = g[g["count"] >= 10]
RES["forward"]["same_soh_spread"] = {"mean_sd_pp": float(100 * g["std"].mean()), "max_range_pp": float(100 * g.iloc[:, 2].max()),
                                     "bins": int(len(g))}
sub = oof[(oof.soh > 0.90) & (oof.soh < 0.92)]
RES["forward"]["example_bin_90_92"] = {"n": int(len(sub)), "min": float(sub.soh_h.min()), "max": float(sub.soh_h.max()),
                                       "frac_below80": float((sub.soh_h < 0.8).mean())}
print(json.dumps(RES["forward"], indent=1))

# ---------------------------------------------------------------- 4. residual value V(SoH), guarantee pricing, scorecard
P_PACK, S_FLOOR, S_EOL, GAMMA, SALV = 60000.0, 0.60, 0.70, 1.0, 0.10
def V(s):
    s = np.asarray(s, float)
    first = P_PACK * (SALV + (1 - SALV) * np.clip((s - S_EOL) / (1 - S_EOL), 0, 1) ** GAMMA)
    second = P_PACK * SALV * np.clip((s - S_FLOOR) / (S_EOL - S_FLOOR), 0, 1)
    return np.where(s >= S_EOL, first, second)
G = float(V(S_GUAR))
def payout(s): return np.maximum(0.0, G - V(s))
def exp_payout(Q): return payout(qfun(Q, UG)).mean(1)
oof["pay_real"] = payout(oof["soh_h"].values)
oof["pay_fw"] = exp_payout(Qall)
sd_pop = float(np.std(oof["soh_h"] - oof["naive_h"]))
Qcur = oof["naive_h"].values[:, None] + sd_pop * ZT[None, :]
oof["pay_cur"] = exp_payout(Qcur)
oof["p_bad_cur"] = prob_below(Qcur, S_GUAR)
def dec(col):
    k = pd.qcut(oof[col].rank(method="first"), 10, labels=False); return oof.groupby(k)[[col, "pay_real"]].mean()
cal_tab, cal_c = dec("pay_fw"), dec("pay_cur")
RES["guarantee"] = {"P_pack": P_PACK, "G": G, "S_guar": S_GUAR, "mean_real_payout": float(oof.pay_real.mean()),
                    "mean_priced_fw": float(oof.pay_fw.mean()), "mean_priced_cur": float(oof.pay_cur.mean()),
                    "mae_payout_fw": float(np.abs(oof.pay_fw - oof.pay_real).mean()),
                    "mae_payout_cur": float(np.abs(oof.pay_cur - oof.pay_real).mean()),
                    "decile_slope_fw": float(np.polyfit(cal_tab.pay_fw, cal_tab.pay_real, 1)[0]),
                    "decile_slope_cur": float(np.polyfit(cal_c.pay_cur, cal_c.pay_real, 1)[0]),
                    "top_decile_priced_fw": float(cal_tab.pay_fw.iloc[-1]), "top_decile_real_fw": float(cal_tab.pay_real.iloc[-1]),
                    "top_decile_priced_cur": float(cal_c.pay_cur.iloc[-1]), "top_decile_real_cur": float(cal_c.pay_real.iloc[-1]),
                    "V_at": {str(x): float(V(x)) for x in [1.0, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6]}}
# scorecard scaling (Siddiqi): score = offset + factor*ln(odds good:bad); 700 at 50:1, PDO = 50
PDO, S0, ODDS0 = 50.0, 700.0, 50.0
FACT = PDO / np.log(2); OFFS = S0 - FACT * np.log(ODDS0)
def score(p):
    p = np.clip(p, 1e-4, 1 - 1e-4); return np.clip(OFFS + FACT * np.log((1 - p) / p), 300, 900)
oof["score"] = score(oof.p_bad); oof["score_cur"] = score(oof.p_bad_cur)
oof["score_lin"] = 300 + 600 * np.clip((oof.soh_hat - 0.30) / 0.70, 0, 1)   # original v1 mapping
bands = [300, 500, 600, 700, 800, 901]; names = ["E (300-499)", "D (500-599)", "C (600-699)", "B (700-799)", "A (800-900)"]
def bandtab(col):
    b = pd.cut(oof[col], bands, right=False, labels=names)
    t = oof.groupby(b, observed=True).agg(n=("pack", "size"), soh_now=("soh", "mean"), soh_h=("soh_h", "mean"),
                                          pred_bad=("p_bad", "mean"), real_bad=("soh_h", lambda s: float((s < S_GUAR).mean())),
                                          pay=("pay_real", "mean"))
    return {str(i): {k: float(v) for k, v in r.items()} for i, r in t.iterrows()}
RES["bands"] = bandtab("score")
ev = (oof.soh_h < S_GUAR).astype(int)
RES["score_discrimination"] = {"auc_score": float(roc_auc_score(ev, -oof.score)), "auc_linear_current": float(roc_auc_score(ev, -oof.score_lin)),
                               "auc_score_cur": float(roc_auc_score(ev, -oof.score_cur)),
                               "spearman_score_payout": float(stats.spearmanr(-oof.score, oof.pay_real)[0]),
                               "spearman_lin_payout": float(stats.spearmanr(-oof.score_lin, oof.pay_real)[0]),
                               "PDO": PDO, "S0": S0, "ODDS0": ODDS0}
# reliability of p_bad
rb = pd.cut(oof.p_bad, [0, .02, .05, .1, .2, .4, .7, 1.0001], include_lowest=True)
RES["reliability"] = {str(i): [int(r["n"]), float(r["p"]), float(r["o"])] for i, r in
                      oof.assign(o=ev).groupby(rb, observed=True).agg(n=("o", "size"), p=("p_bad", "mean"), o=("o", "mean")).iterrows()}
# current-SoH estimation error by original linear band (threshold-consistency fix)
oof["band_cur"] = pd.cut(oof.score_lin, [300, 600, 700, 800, 901], right=False)
eb = oof.assign(ae=100 * np.abs(oof.soh_hat - oof.soh)).groupby("band_cur", observed=True)["ae"].agg(["count", "mean"])
RES["band_error_current"] = {str(i): [int(r["count"]), float(r["mean"])] for i, r in eb.iterrows()}
print(json.dumps({k: RES[k] for k in ["guarantee", "bands", "score_discrimination", "reliability", "band_error_current"]}, indent=1))
oof.to_csv("out/synthetic_oof.csv", index=False)

# ---------------------------------------------------------------- 5. real cells
r = pd.read_csv(REAL)
# CALCE cells cycled far past EOL; keep the financing-relevant range up to first crossing of 0.60
keep = []
for cell, g in r.groupby("cell", sort=False):
    g = g.sort_values("cycle"); cut = np.argmax(g.soh.values < 0.60) if (g.soh.values < 0.60).any() else len(g)
    keep.append(g.iloc[:cut])
r = pd.concat(keep, ignore_index=True)
r["ce_trend"] = r.groupby("cell")["ce"].transform(lambda s: s.rolling(10, min_periods=1).mean())
r["ir_slope"] = r.groupby("cell")["ir_growth"].transform(lambda s: s.diff(10).fillna(s.diff().fillna(0) * 10) / 10)
r["cycle_frac"] = r["cycle"]  # raw cycle index (EFC proxy; 100 % DOD protocol)
RF = ["cycle", "ir_growth", "ce_trend", "ir_slope", "temp_mean"]
RES["real_cells"] = r.groupby("cell").agg(n=("cycle", "size"), soh_end=("soh", "last"), ir_end=("ir_growth", "last"),
                                          vmin=("v_min", "median"), cap0=("capacity", "first")).round(4).to_dict("index")

def loco(r, feats, cells, est_fn=proposed, anchor=0, rng_seed=0):
    preds = []
    rng = np.random.default_rng(rng_seed)
    for c in cells:
        tr, te = r[(r.cell != c) & r.cell.isin(cells)], r[r.cell == c].copy()
        m = est_fn(feats).fit(tr[feats], tr["soh"]); te["pred"] = m.predict(te[feats])
        te["pred_raw"] = te["pred"]
        if anchor:
            # per-cell calibration anchor: `anchor` reference capacity tests at onboarding (random point in first
            # 40 % of the record); offset = mean(true - pred) at those points, applied to all *later* cycles only.
            n = len(te); start = int(rng.integers(0, max(1, int(0.4 * n))))
            pts = te.iloc[start:start + 10 * anchor:10] if anchor > 1 else te.iloc[[start]]
            te["pred"] = te["pred"] + float((pts.soh - pts.pred).mean())
            te = te.iloc[start + 1:]
        preds.append(te)
    return pd.concat(preds)

def cell_table(p, col="pred"):
    if col != "pred": p = p.assign(pred=p[col])
    return {c: metrics(g.soh.values, g.pred.values) for c, g in p.groupby("cell")} | {"Pooled": metrics(p.soh.values, p.pred.values)}

NASA, CALCE = ["B0005", "B0006", "B0007", "B0018"], ["CS2_35", "CS2_36", "CS2_37", "CS2_38"]
RES["real"] = {}
p_old = loco(r, ["cycle", "days", "ir_growth", "ce", "temp_mean"], NASA,
             lambda f: GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42))
RES["real"]["nasa_loco_original_setup"] = cell_table(p_old)
RES["real"]["family_compare"] = {}
for fam, fn in [("linear", linq), ("mono-HGB", proposed),
                ("random forest", lambda f: RandomForestRegressor(300, max_depth=6, random_state=42, n_jobs=-1))]:
    RES["real"]["family_compare"][fam] = {nm: cell_table(loco(r, RF, cl, est_fn=fn))["Pooled"]
                                          for nm, cl in [("nasa", NASA), ("calce", CALCE), ("pooled8", NASA + CALCE)]}
RFAM_NAME = min(RES["real"]["family_compare"], key=lambda k: RES["real"]["family_compare"][k]["pooled8"]["mae"])
RFAM = {"linear": linq, "mono-HGB": proposed,
        "random forest": lambda f: RandomForestRegressor(300, max_depth=6, random_state=42, n_jobs=-1)}[RFAM_NAME]
RES["real"]["selected_family"] = RFAM_NAME
p_n = loco(r, RF, NASA, est_fn=RFAM); RES["real"]["nasa_loco"] = cell_table(p_n)
p_c = loco(r, RF, CALCE, est_fn=RFAM); RES["real"]["calce_loco"] = cell_table(p_c)
p_all = loco(r, RF, NASA + CALCE, est_fn=RFAM); RES["real"]["pooled8_loco"] = cell_table(p_all)
# anchor calibration, averaged over 20 random onboarding points
for a in [1, 3]:
    for name, cells in [("nasa", NASA), ("calce", CALCE), ("pooled8", NASA + CALCE)]:
        runs = [loco(r, RF, cells, est_fn=RFAM, anchor=a, rng_seed=s) for s in range(20)]
        for tag, col in [(f"{name}_sameSubset_noanchor_a{a}", "pred_raw"), (f"{name}_loco_anchor{a}", "pred")]:
            ms = [cell_table(p, col) for p in runs]
            RES["real"][tag] = {c: {k: float(np.mean([m[c][k] for m in ms])) for k in ms[0][c]} for c in ms[0]}
# true sim-to-real transfer: train on decoupled synthetic cohort, test on real cells with the shared features
dsyn = d.copy(); SH = ["efc", "ir_growth", "ce_trend"]
r2 = r.assign(efc=r["cycle"])
m = FAMILIES[FAM](SH).fit(dsyn[SH], dsyn["soh"])
for name, cells in [("nasa", NASA), ("calce", CALCE)]:
    te = r2[r2.cell.isin(cells)].assign(pred=lambda x: m.predict(x[SH]))
    RES["real"][f"sim2real_{name}"] = cell_table(te)
    ms = []
    for s in range(20):
        rng = np.random.default_rng(s); parts = []
        for c, g in te.groupby("cell"):
            n = len(g); st = int(rng.integers(0, max(1, int(0.4 * n))))
            g = g.copy(); g["pred"] += float(g.soh.iloc[st] - g.pred.iloc[st]); parts.append(g.iloc[st + 1:])
        ms.append(cell_table(pd.concat(parts)))
    RES["real"][f"sim2real_{name}_anchor1"] = {c: {k: float(np.mean([mm[c][k] for mm in ms])) for k in ms[0][c]} for c in ms[0]}

# cluster bootstrap over cells for pooled metrics
def cluster_boot(p, B=5000, seed=42):
    rng = np.random.default_rng(seed); cells = p.cell.unique(); g = {c: p[p.cell == c] for c in cells}; out = []
    for _ in range(B):
        s = pd.concat([g[c] for c in rng.choice(cells, len(cells))]); out.append(metrics(s.soh.values, s.pred.values))
    return {k: [float(np.quantile([o[k] for o in out], 0.025)), float(np.quantile([o[k] for o in out], 0.975))] for k in out[0]}
RES["real"]["boot_pooled8"] = cluster_boot(p_all)
RES["real"]["boot_pooled8_anchor3"] = cluster_boot(loco(r, RF, NASA + CALCE, est_fn=RFAM, anchor=3, rng_seed=0))
RES["real"]["boot_nasa_original"] = cluster_boot(p_old)

# forward horizon on real cells: predict SoH h cycles ahead, with cell-level jackknife+ intervals
def forward_real(r, cells, h, feats=RF, alpha=0.1, fn=None):
    fn = fn or RFAM
    rr = []
    for c, g in r[r.cell.isin(cells)].groupby("cell"):
        g = g.sort_values("cycle").copy(); g["soh_h"] = g["soh"].shift(-h); rr.append(g.dropna(subset=["soh_h"]))
    rr = pd.concat(rr); out = []
    for c in cells:
        tr, te = rr[rr.cell != c], rr[rr.cell == c].copy()
        # cell-level jackknife residuals among training cells
        res = []
        for c2 in tr.cell.unique():
            t2, v2 = tr[tr.cell != c2], tr[tr.cell == c2]
            res.append(np.abs(v2.soh_h - fn(feats).fit(t2[feats], t2.soh_h).predict(v2[feats])))
        qhat = np.quantile(np.concatenate(res), 1 - alpha)
        te["pred"] = fn(feats).fit(tr[feats], tr.soh_h).predict(te[feats])
        te["persist"] = fn(feats).fit(tr[feats], tr.soh).predict(te[feats])
        te["lo"], te["hi"] = te.pred - qhat, te.pred + qhat; out.append(te)
    o = pd.concat(out)
    return {"h": h, "forward": metrics(o.soh_h.values, o.pred.values), "persistence": metrics(o.soh_h.values, o.persist.values),
            "coverage_90": float(((o.soh_h >= o.lo) & (o.soh_h <= o.hi)).mean()), "width_pp": float(100 * (o.hi - o.lo).mean()),
            "per_cell_coverage": o.groupby("cell").apply(lambda g: float(((g.soh_h >= g.lo) & (g.soh_h <= g.hi)).mean())).to_dict()}
RES["real"]["forward_nasa_h30"] = forward_real(r, NASA, 30)
RES["real"]["forward_calce_h100"] = forward_real(r, CALCE, 100)
print(json.dumps(RES["real"], indent=1, default=str)[:6000])

# ---------------------------------------------------------------- figures
def save(fig, name): fig.savefig(f"figs/{name}.png"); plt.close(fig)

# Fig 3: coupled vs decoupled parity (proposed, 5-fold OOF)
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True)
for a, (k, (dd, _)) in zip(ax, cohorts.items()):
    feats = BASE if k == "coupled" else BASE + HIST; p = np.zeros(len(dd))
    for tr, te in KFold(5, shuffle=True, random_state=0).split(dd):
        p[te] = linq(feats).fit(dd.iloc[tr][feats], dd.iloc[tr].soh).predict(dd.iloc[te][feats])
    mm = metrics(dd.soh.values, p)
    a.scatter(100 * dd.soh, 100 * p, s=7, alpha=.5, color=C[0] if k == "coupled" else C[1], edgecolor="none")
    a.plot([30, 105], [30, 105], "k--", lw=.8); a.set_xlim(55, 105); a.set_ylim(55, 105)
    a.set_title(f"({'a' if k == 'coupled' else 'b'}) {k.capitalize()} generator", loc="left", fontsize=9)
    a.text(57, 100, f"R$^2$ = {mm['r2']:.3f}\nMAE = {mm['mae']:.2f} pp", fontsize=8, va="top")
    a.set_xlabel("Simulated SoH (%)")
ax[0].set_ylabel("Estimated SoH (%)"); save(fig, "fig3_parity")

# Fig 4: IR growth vs SoH, coupled vs decoupled vs real cells (diagnosis)
fig, ax = plt.subplots(1, 3, figsize=(9.5, 3.0))
for a, k, col in zip(ax[:2], ["coupled", "decoupled"], [C[0], C[1]]):
    dd = cohorts[k][0]; a.scatter(dd.ir_growth, 100 * dd.soh, s=5, alpha=.4, color=col, edgecolor="none")
    a.set_title(f"({'a' if k == 'coupled' else 'b'}) {k.capitalize()} synthetic", loc="left", fontsize=9)
    a.set_xlim(-0.1, 1.2); a.set_xlabel("IR growth (relative)"); a.set_ylabel("SoH (%)")
for i, (c, g) in enumerate(r.groupby("cell")):
    ax[2].plot(g.ir_growth, 100 * g.soh, lw=1.1, color=C[i], label=c)
ax[2].set_title("(c) Real cells", loc="left", fontsize=9); ax[2].set_xlabel("IR growth (relative to first reading)")
ax[2].legend(fontsize=6.5, ncol=1, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5)); save(fig, "fig4_ir_mapping")

# Fig 5: real LOCO pooled8 without / with anchor
pa = loco(r, RF, NASA + CALCE, est_fn=RFAM, anchor=3, rng_seed=0)
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True)
p_all_s = pa.assign(pred=pa.pred_raw)
for a, p, t in zip(ax, [p_all_s, pa], ["(a) Population model (same cycles)", "(b) With three onboarding reference tests"]):
    for i, (c, g) in enumerate(p.groupby("cell")):
        a.scatter(100 * g.soh, 100 * g.pred, s=4, color=C[i], label=c, alpha=.7, edgecolor="none")
    mm = metrics(p.soh.values, p.pred.values)
    a.plot([55, 105], [55, 105], "k--", lw=.8); a.set_xlim(55, 105); a.set_ylim(55, 105)
    a.set_title(t, loc="left", fontsize=9); a.set_xlabel("Measured SoH (%)")
    a.text(57, 102, f"Pooled R$^2$ = {mm['r2']:.2f}, MAE = {mm['mae']:.1f} pp", fontsize=8, va="top")
ax[0].set_ylabel("Estimated SoH (%)"); ax[1].legend(fontsize=6.5, ncol=2, frameon=False, loc="lower right"); save(fig, "fig5_real_loco")

# Fig 6: same current SoH, different futures + forward intervals
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.2))
sub = oof[(oof.soh > 0.90) & (oof.soh < 0.92)]
ax[0].scatter(100 * sub.soh, 100 * sub.soh_h, s=10, color=C[2], alpha=.7)
ax[0].set_xlabel("Current SoH (%)"); ax[0].set_ylabel(f"SoH after {H} months (%)")
ax[0].set_title("(a) Packs at 90–92 % SoH today", loc="left", fontsize=9)
o2 = oof.sort_values("med").reset_index(drop=True).iloc[::4]
ax[1].fill_between(np.arange(len(o2)), 100 * o2.lo, 100 * o2.hi, color=C[0], alpha=.25, label="90 % CQR interval")
ax[1].plot(np.arange(len(o2)), 100 * o2.med, color=C[0], lw=1, label="Median forecast")
ax[1].scatter(np.arange(len(o2)), 100 * o2.soh_h, s=4, color=C[1], label="Realised", zorder=3)
ax[1].set_ylim(30, 105); ax[1].set_xlabel("Test packs, every 4th (sorted by forecast)"); ax[1].set_ylabel(f"SoH after {H} months (%)")
ax[1].set_title("(b) Forward forecast with conformal interval", loc="left", fontsize=9); ax[1].legend(fontsize=7, frameon=False)
save(fig, "fig6_forward")

# Fig 7: residual-value function and guarantee calibration
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.1))
s = np.linspace(0.5, 1.0, 200)
ax[0].plot(100 * s, V(s) / 1000, color=C[0]); ax[0].axhline(G / 1000, ls="--", color=C[1], lw=.9)
ax[0].text(51, G / 1000 + 1, f"Guaranteed floor G = ₹{G/1000:.0f}k", fontsize=7.5, color=C[1])
ax[0].set_xlabel("SoH at horizon (%)"); ax[0].set_ylabel("Residual value V (₹ thousand)")
ax[0].set_title("(a) Residual-value function V(SoH)", loc="left", fontsize=9)
ax[1].plot(cal_tab.pay_fw / 1000, cal_tab.pay_real / 1000, "o-", color=C[0], label="Forward, uncertainty-aware")
ax[1].plot(cal_c.pay_cur / 1000, cal_c.pay_real / 1000, "s-", color=C[1], label="Current SoH only")
mx = max(cal_tab.max().max(), cal_c.max().max()) / 1000
ax[1].plot([0, mx], [0, mx], "k--", lw=.8); ax[1].set_xlabel("Priced expected payout (₹ thousand)")
ax[1].set_ylabel("Realised mean payout (₹ thousand)"); ax[1].legend(fontsize=7, frameon=False)
ax[1].set_title("(b) Guarantee pricing calibration (deciles)", loc="left", fontsize=9); save(fig, "fig7_value_guarantee")

# Fig 8: feature importance via permutation on decoupled
from sklearn.inspection import permutation_importance
tr, te = next(KFold(5, shuffle=True, random_state=0).split(d))
mdl = FAMILIES[FAM](FW).fit(d.iloc[tr][FW], d.iloc[tr].soh)
pi = permutation_importance(mdl, d.iloc[te][FW], d.iloc[te].soh, n_repeats=20, random_state=0)
order = np.argsort(pi.importances_mean)
fig, ax = plt.subplots(figsize=(4.6, 3.0))
ax.barh(np.array(FW)[order], pi.importances_mean[order], xerr=pi.importances_std[order], color=C[0])
ax.set_xlabel("Permutation importance (drop in R$^2$)"); save(fig, "fig8_importance")
RES["perm_importance"] = dict(zip(FW, map(float, pi.importances_mean)))

json.dump(RES, open("out/results.json", "w"), indent=1, default=str)
print("done")
