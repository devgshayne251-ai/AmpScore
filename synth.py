"""Synthetic e-rickshaw pack cohorts.

coupled   : replicates the original manuscript's generator - IR growth and CE are noisy
            functions of the *same* SoH value the model must recover (circular by design).
decoupled : mechanism-based generator - capacity fade is the sum of two latent modes
            (loss of lithium inventory, LLI; loss of active material, LAM, with a knee),
            plus an unobservable per-pack initial-capacity factor. IR growth is driven by
            the latent modes through a pack-specific coefficient plus an independent
            interconnect-drift term and sensor noise; CE reflects the *rate* of LLI, not the
            SoH level. No observable feature is a deterministic function of SoH.
Both return one row per pack at a random inspection month, with the current SoH and the
SoH H months later (forward target), plus a monthly history used for trajectory features.
"""
import numpy as np, pandas as pd

R_GAS, EA = 8.314, 30_000.0  # J/mol, activation energy for calendar ageing


def _usage(rng, n):
    return dict(cyc=np.clip(rng.normal(55, 12, n), 20, 100), dod=np.clip(rng.normal(0.72, 0.12, n), 0.3, 0.98),
                fc=rng.beta(2, 5, n), temp=np.clip(rng.normal(34, 5, n), 20, 48))


def simulate(kind="decoupled", n=600, months=48, horizon=12, seed=42, insp_max=36):
    rng = np.random.default_rng(seed)
    u = _usage(rng, n)
    q = np.clip(rng.normal(1.0, 0.03, n), 0.9, 1.08)             # initial capacity vs rated (unobserved)
    a_cal = rng.lognormal(np.log(0.010), 0.25, n)                 # calendar LLI coefficient
    a_cyc = rng.lognormal(np.log(3.0e-5), 0.30, n)                # cyclic LLI per stressed EFC
    knee = rng.lognormal(np.log(1500), 0.25, n)                   # stressed-EFC knee onset
    b_lam = rng.lognormal(np.log(1.0e-6), 0.35, n)                # LAM after knee
    c_ir = rng.lognormal(np.log(2.0), 0.35, n)                    # IR sensitivity to latent fade (pack specific)
    drift = np.abs(rng.normal(0.0, 0.004, n))                     # interconnect/contact IR drift per month
    rows, hist = [], []
    for i in range(n):
        t = np.arange(1, months + 1)
        cyc_m = np.clip(u['cyc'][i] * (1 + 0.15 * rng.standard_normal(months)), 5, None)
        dod_m = np.clip(u['dod'][i] + 0.06 * rng.standard_normal(months), 0.2, 1.0)
        fc_m = np.clip(u['fc'][i] + 0.05 * rng.standard_normal(months), 0, 1)
        T_m = u['temp'][i] + 4 * np.sin(2 * np.pi * (t + rng.integers(12)) / 12) + rng.standard_normal(months)
        arr = np.exp(EA / R_GAS * (1 / 298.15 - 1 / (T_m + 273.15)))
        efc = np.cumsum(cyc_m * dod_m)
        stress = np.cumsum(cyc_m * dod_m ** 1.5 * (1 + 1.5 * fc_m) * arr ** 0.5)
        lli = a_cal[i] * np.sqrt(np.cumsum(arr * 30.4 / 365)) * 3 + a_cyc[i] * stress
        lam = b_lam[i] * np.clip(stress - knee[i], 0, None) ** 1.6
        if kind == "coupled":
            soh = q[i] * (1 - lli - lam)
            ir = 2.2 * (1 - soh) + 0.02 * rng.standard_normal(months)
            ce = 1 - 0.004 * (1 - soh) + 0.0002 * rng.standard_normal(months)
        else:
            soh = q[i] * (1 - lli - lam)
            ir = c_ir[i] * (lli + 1.8 * lam) + drift[i] * t + 0.03 * rng.standard_normal(months)
            dlli = np.diff(np.r_[0, lli]) / np.maximum(cyc_m, 1)
            ce = 1 - 4.0 * dlli - 0.002 * fc_m + 0.0005 * rng.standard_normal(months)
        soh = np.clip(soh, 0.3, 1.1)
        hist.append(pd.DataFrame(dict(pack=i, month=t, soh=soh, efc=efc, ir=ir, ce=ce, temp=T_m, dod=dod_m, fc=fc_m)))
        ti = int(np.clip(np.round(rng.triangular(3, 10, insp_max)), 3, insp_max))
        k = ti - 1
        w = slice(max(0, k - 5), k + 1)
        rows.append(dict(pack=i, age=ti, efc=efc[k], dod_mean=dod_m[:ti].mean(),
                         dod_p90=np.quantile(dod_m[:ti], 0.9), fc_frac=fc_m[:ti].mean(), temp_mean=T_m[:ti].mean(),
                         ir_growth=ir[k], ce_trend=ce[w].mean(),
                         ir_slope=np.polyfit(t[w], ir[w], 1)[0], efc_rate=cyc_m[w].mean() * dod_m[w].mean(),
                         soh=soh[k], soh_h=soh[min(k + horizon, months - 1)], q=q[i]))
    return pd.DataFrame(rows), pd.concat(hist, ignore_index=True)


BASE = ["efc", "dod_mean", "dod_p90", "fc_frac", "temp_mean", "age", "ir_growth", "ce_trend"]
HIST = ["ir_slope", "efc_rate"]
USAGE = ["efc", "dod_mean", "dod_p90", "fc_frac", "temp_mean", "age"]
ELEC = ["ir_growth", "ce_trend"]

if __name__ == "__main__":
    for k in ["coupled", "decoupled"]:
        d, h = simulate(k)
        print(k, d[["soh", "soh_h", "ir_growth", "ce_trend", "efc"]].describe().round(3).T[["mean", "std", "min", "max"]])
        print(d[BASE + HIST + ["soh"]].corr()["soh"].round(3).to_dict())
