import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({"font.family": "DejaVu Sans"})
def box(ax, x, y, w, h, title, body, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec="#3d405b", lw=1))
    ax.text(x + w / 2, y + h - 0.13, title, ha="center", va="top", fontsize=9, weight="bold", color="#1d1d1d")
    ax.text(x + w / 2, y + h - 0.45, body, ha="center", va="top", fontsize=7.4, color="#2b2b2b", linespacing=1.35)
def arr(ax, x1, y1, x2, y2):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", color="#3d405b", lw=1.1))
def draw(fname, ga=False):
    fig, ax = plt.subplots(figsize=(11, 4.2 if not ga else 3.6)); ax.set_xlim(0, 11); ax.set_ylim(0, 4.2); ax.axis("off")
    box(ax, 0.1, 1.3, 1.85, 1.9, "1. Telemetry", "BMS or retrofit\ndongle: current,\nvoltage, temperature,\ncycle counts", "#e8f1f7")
    box(ax, 2.25, 1.3, 1.95, 1.9, "2. Health indicators", "Usage: EFC, DOD,\nfast-charge, temp., age\nElectrical: IR growth,\nCE trend\nTrajectory: IR slope,\nEFC rate", "#e8f1f7")
    box(ax, 4.5, 2.35, 2.1, 1.55, "3a. Current SoH", "Point estimator\n(selected by validation)\n+ optional onboarding\nreference-test anchor", "#fbe9e2")
    box(ax, 4.5, 0.35, 2.1, 1.75, "3b. Forward SoH at H", "Multi-quantile forecast\nof SoH(t+H), widened by\nconformal calibration\n(90 % coverage target)", "#fbe9e2")
    box(ax, 6.95, 2.35, 1.85, 1.55, "4. Risk", "P[SoH(t+H) < s_g]\nE[guarantee payout]\nvia residual value\nV(SoH)", "#eef5ee")
    box(ax, 6.95, 0.35, 1.85, 1.75, "5. AmpScore", "Scorecard scaling of\nlog-odds (PDO = 50):\n300–900, bands A–E", "#eef5ee")
    box(ax, 9.1, 1.3, 1.8, 1.9, "6. Lender use", "Certificate +\nrisk-priced residual-\nvalue guarantee;\none input to\nunderwriting", "#f6f0e1")
    arr(ax, 1.95, 2.25, 2.25, 2.25); arr(ax, 4.2, 2.6, 4.5, 3.0); arr(ax, 4.2, 1.9, 4.5, 1.3)
    arr(ax, 6.6, 1.3, 6.95, 3.0); arr(ax, 6.6, 3.1, 6.95, 3.1); arr(ax, 7.875, 2.35, 7.875, 2.1)
    arr(ax, 8.8, 1.3, 9.1, 2.0); arr(ax, 8.8, 3.1, 9.1, 2.6)
    if ga:
        ax.text(5.5, 0.05, "Validated on a decoupled synthetic cohort (600 packs) and 8 real cells (NASA, CALCE); sim-to-real gap reported",
                ha="center", fontsize=8, style="italic", color="#3d405b")
    fig.savefig(fname, dpi=300, bbox_inches="tight"); plt.close(fig)
draw("figs/fig1_framework.png"); draw("figs/graphical_abstract.png", ga=True)
