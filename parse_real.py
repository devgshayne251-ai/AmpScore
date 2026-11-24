"""Parse NASA (B0005/6/7/18) and CALCE (CS2_35-38) into per-cycle tables."""
import sys, os, glob, re, numpy as np, pandas as pd, scipy.io as sio
NASA_DIR, CALCE_DIR, OUT = sys.argv[1], sys.argv[2], sys.argv[3]

def nasa_cell(name):
    m = sio.loadmat(os.path.join(NASA_DIR, f"{name}.mat"), simplify_cells=True)[name]['cycle']
    rows, last_R, last_charge_ah, t0 = [], None, None, None
    R0 = None
    for c in m:
        typ = c['type']; d = c['data']
        t = pd.Timestamp(*[int(x) for x in c['time'][:5]]) + pd.Timedelta(seconds=float(c['time'][5]))
        if t0 is None: t0 = t
        if typ == 'impedance':
            re_, rct = float(np.real(d['Re'])), float(np.real(d['Rct']))
            if np.isfinite(re_) and np.isfinite(rct) and 0 < re_ + rct < 1:
                last_R = re_ + rct
                if R0 is None: R0 = last_R
        elif typ == 'charge':
            tt, I = np.asarray(d['Time'], float), np.asarray(d['Current_measured'], float)
            last_charge_ah = np.trapezoid(np.clip(I, 0, None), tt) / 3600.0
        elif typ == 'discharge':
            cap = float(d['Capacity'])
            T = np.asarray(d['Temperature_measured'], float)
            V = np.asarray(d['Voltage_measured'], float)
            rows.append(dict(cell=name, capacity=cap, R=last_R, charge_ah=last_charge_ah,
                             temp_mean=T.mean(), temp_max=T.max(), v_min=V.min(),
                             days=(t - t0).total_seconds() / 86400))
    df = pd.DataFrame(rows)
    df['R'] = df['R'].ffill().bfill()
    df['cycle'] = np.arange(1, len(df) + 1)
    df['ir_growth'] = df['R'] / df['R'].iloc[0] - 1.0
    df['ce'] = (df['capacity'] / df['charge_ah']).clip(0.8, 1.05)
    df['dataset'] = 'NASA'; df['chem'] = 'LCO-18650 (2.0 Ah)'
    return df

def calce_date(fn):
    m = re.search(r'_(\d+)_(\d+)_(\d+)\.xlsx$', fn); mo, dd, yy = map(int, m.groups())
    return pd.Timestamp(2000 + yy, mo, dd)

def calce_cell(name):
    files = sorted(glob.glob(os.path.join(CALCE_DIR, name, '*.xlsx')), key=calce_date)
    rows = []
    for f in files:
        x = pd.ExcelFile(f); sh = [s for s in x.sheet_names if s.startswith('Channel')][0]
        d = pd.read_excel(x, sh)
        for ci, g in d.groupby('Cycle_Index'):
            ch = g['Charge_Capacity(Ah)'].max()-g['Charge_Capacity(Ah)'].min(); dis = g['Discharge_Capacity(Ah)'].max()-g['Discharge_Capacity(Ah)'].min()
            ir = g['Internal_Resistance(Ohm)']; ir = ir[ir > 0]
            dur = g['Test_Time(s)'].iloc[-1] - g['Test_Time(s)'].iloc[0]
            idis = -g.loc[g['Current(A)'] < -0.05, 'Current(A)']
            rows.append(dict(cell=name, capacity=dis, charge_ah=ch,
                             R=ir.median() if len(ir) else np.nan,
                             i_dis=idis.mean() if len(idis) else np.nan,
                             v_min=g['Voltage(V)'].min(),
                             days=(calce_date(f) - calce_date(files[0])).days + g['Test_Time(s)'].iloc[0] / 86400,
                             file=os.path.basename(f)))
    df = pd.DataFrame(rows)
    # keep regular 1C cycling cycles with sensible capacity (drops characterisation / partial cycles)
    df = df[(df['i_dis'] > 0.9) & (df['i_dis'] < 1.3) & (df['capacity'] > 0.1) & (df['capacity'] < 1.2)
            & (df['charge_ah'] > 0.1)].reset_index(drop=True)
    # remove isolated outliers (>5% from rolling median)
    med = df['capacity'].rolling(9, center=True, min_periods=1).median()
    df = df[(df['capacity'] - med).abs() / med < 0.05].reset_index(drop=True)
    df['R'] = df['R'].interpolate().bfill().ffill()
    df['cycle'] = np.arange(1, len(df) + 1)
    df['ir_growth'] = df['R'] / df['R'].iloc[:5].median() - 1.0
    df['ce'] = (df['capacity'] / df['charge_ah']).clip(0.8, 1.05)
    df['temp_mean'] = 25.0; df['temp_max'] = 25.0  # CALCE CS2 test chamber, no per-cycle temperature logged
    df['dataset'] = 'CALCE'; df['chem'] = 'LCO-prismatic (1.1 Ah)'
    return df

frames = [nasa_cell(c) for c in ['B0005', 'B0006', 'B0007', 'B0018']]
frames += [calce_cell(c) for c in ['CS2_35', 'CS2_36', 'CS2_37', 'CS2_38']]
all_ = pd.concat(frames, ignore_index=True)
all_['soh'] = all_.groupby('cell')['capacity'].transform(lambda s: s / s.iloc[:3].median())
all_.to_csv(OUT, index=False)
print(all_.groupby('cell').agg(n=('cycle', 'size'), cap0=('capacity', 'first'), soh_min=('soh', 'min'),
      ir_end=('ir_growth', 'last'), ce=('ce', 'median'), vmin=('v_min', 'median')))
