# AmpScore – reproducibility package

Reproduces every table and figure in the manuscript. Python 3.11+, scikit-learn 1.x, xgboost, numpy, pandas, scipy, matplotlib, openpyxl.

1. Get the raw data (public mirrors of the NASA PCoE and CALCE CS2 archives):
   - `git clone https://github.com/psanabriaUC/BatteryDatasetImplementation` (NASA B0005/6/7/18, `battery_data/`)
   - `git clone https://github.com/XiuzeZhou/RUL` (CALCE CS2_35–38, `datasets/CALCE/`)
2. `python parse_real.py <NASA battery_data dir> <CALCE dir> real_cycles.csv` – per-cycle feature table (already included).
3. `mkdir -p out figs && cp real_cycles.csv out/ && python experiments.py out/real_cycles.csv` – all experiments (~4 min); writes `out/results.json` and `figs/*.png`.
4. `python fig_importance.py` (labelled Fig. 3) and `python diagram.py` (Fig. 1, graphical abstract).

All random seeds are fixed (42 for models, 0–19 for folds/onboarding draws). `synth.py` contains both the coupled (circular) and the decoupled generators.
