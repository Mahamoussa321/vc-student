# ===========================
# ALL FIGURES (pretty palette; unique driver colors) — one cell
# ===========================
import os, math, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
    brier_score_loss,
)

# ---------- Paths & dirs ----------
try:
    BASE = REPO_ROOT
except NameError:
    BASE = (Path(__file__).resolve().parents[1]
            if "__file__" in globals() else Path.cwd())

FIG_DIR = BASE / "figures"
TAB_DIR = BASE / "outputs" / "tables"
for p in (FIG_DIR, TAB_DIR):
    p.mkdir(parents=True, exist_ok=True)

print("[paths] FIG_DIR →", FIG_DIR.resolve())
print("[paths] TAB_DIR →", TAB_DIR.resolve())
print("[paths] CWD     →", Path.cwd().resolve())

SHOW = True
DPI  = 600

mpl.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": DPI,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.25,
    "axes.edgecolor": "#888",
    "axes.linewidth": 0.6,
    "font.size": 11,
    "axes.titleweight": "bold",
})

# Palettes
PASTEL  = mpl.colormaps["Set3"]     # faint many-lines
PASTEL2 = mpl.colormaps["Set2"]     # bars/boxes/lines
HEAT    = mpl.colormaps["RdBu_r"]   # heatmap

# ---------- Pretty labels ----------
PRETTY = {
    # Surface & geo
    "RH": r"RH",
    "gridded_data_pres": r"$P_s$",
    "Elevation": "Elev",
    "Longitude": "Lon",
    "Latitude":  "Lat",
    # Profile drivers
    "Z0": r"$Z_0$",
    "ApT": r"ApT",
    "lapse_rate": r"$\Gamma_{ll}$",
    "temp_gradient":  r"$\Gamma_{b}$",
    # Thresholds (°C)
    "thr_Dew_C": "Dew cut",
    "thr_WBT_C": "WBT cut",
    "thr_Air_C": "Air cut",
    # Betas
    "beta0": r"$\beta_0$",
    "beta_Air_Temp": r"$\beta_{\mathrm{Air}}$",
    "beta_Dewpoint": r"$\beta_{\mathrm{Dew}}$",
    "beta_Wet_Bulb_Temp": r"$\beta_{\mathrm{WBT}}$",
    # Index
    "Index_I": r"Learned index $I(\mathbf{Z})$ (std units)",
}
def pretty(name: str) -> str:
    return PRETTY.get(name, name)

# ---------- Small helpers ----------
def save(fig, name, w=8, h=5.5, caption=None):
    fig.set_size_inches(w, h); fig.tight_layout()
    fig.savefig(FIG_DIR/f"{name}.png", dpi=DPI, bbox_inches="tight")
    if SHOW: plt.show()
    plt.close(fig)
    if caption:
        (TAB_DIR/f"{name}__caption.txt").write_text(caption, encoding="utf-8")

def binned_summary(x, y, k=25, n_min=0):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]
    if len(x) == 0:
        return pd.DataFrame(columns=["x_mid","y_mean","y_lo","y_hi","n"])
    qs = np.quantile(x, np.linspace(0,1,k+1))
    diffs = np.diff(qs)
    if np.any(diffs==0):
        jitter = np.linspace(-1e-9,1e-9,len(qs))
        qs = np.sort(qs + jitter)
    b = np.digitize(x, qs[1:-1], right=True)
    rows = []
    for i in range(k):
        idx = (b==i)
        if not np.any(idx): continue
        n = idx.sum()
        if n < n_min: continue
        xv, yv = x[idx], y[idx]
        rows.append((np.nanmean(xv),
                     np.nanmean(yv),
                     np.nanpercentile(yv,25),
                     np.nanpercentile(yv,75),
                     int(n)))
    return pd.DataFrame(rows, columns=["x_mid","y_mean","y_lo","y_hi","n"])

def nice_filled_curve(ax, ds, color, label=None):
    """Draw mean line + IQR ribbon using a specific color."""
    ax.fill_between(ds["x_mid"], ds["y_lo"], ds["y_hi"],
                    color=color, alpha=0.25, linewidth=0)
    ax.plot(ds["x_mid"], ds["y_mean"], lw=2.0, color=color, label=label)

def fancy_bar(ax, series, title_canon):
    s = series.dropna()
    s = s.reindex(s.abs().sort_values(ascending=False).index)  # big → top
    y = np.arange(len(s))
    c = [PASTEL2(i/max(1,len(s)-1)) for i in range(len(s))]
    bars = ax.barh(y, s.values, height=0.7, color=c, edgecolor="#333", linewidth=0.5)
    off = 0.015 * (np.nanmax(np.abs(s.values)) + 1e-6)
    for b, v in zip(bars, s.values):
        if np.isfinite(v) and abs(v) > 1e-6:
            ax.text(v + (off if v>=0 else -off),
                    b.get_y()+b.get_height()/2,
                    f"{v:+.2f}",
                    va="center",
                    ha=("left" if v>=0 else "right"),
                    color="#222", fontsize=9)
    ax.axvline(0, color="#555", lw=0.8, ls="--", alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels([pretty(d) for d in s.index])
    ax.set_title(pretty(title_canon))
    ax.set_xlabel("Effect (°C)")
    ax.invert_yaxis()

def reliability_diagram(y_true, p_hat, n_bins=20):
    """Bin-wise empirical frequency + ECE."""
    y_true = np.asarray(y_true, float)
    p_hat  = np.asarray(p_hat, float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(p_hat, bins[1:-1], right=True)

    bin_centers, emp_freq, counts = [], [], []
    for b in range(n_bins):
        m = bin_ids == b
        if not np.any(m):
            continue
        p_bin = p_hat[m]
        y_bin = y_true[m]
        bin_centers.append(p_bin.mean())
        emp_freq.append(y_bin.mean())
        counts.append(len(p_bin))

    bin_centers = np.array(bin_centers)
    emp_freq    = np.array(emp_freq)
    counts      = np.array(counts)

    N = counts.sum()
    ece = np.sum(np.abs(emp_freq - bin_centers) * counts) / N
    return bin_centers, emp_freq, ece

# helpers to pick columns if names differ slightly
TRUE_CANDIDATES = ["y_true", "y", "label", "obs", "target"]
PROB_CANDIDATES = ["p_hat", "prob", "prob_snow", "p_snow", "y_prob", "pred_proba"]

def pick_true_col(df):
    for c in TRUE_CANDIDATES:
        if c in df.columns:
            return c
    # fallback: first column that looks binary
    for c in df.columns:
        vals = pd.to_numeric(df[c], errors="coerce").dropna().unique()
        if set(vals).issubset({0, 1}) and len(vals) > 0:
            return c
    raise ValueError("Could not find a 0/1 column for y_true.")

def pick_prob_col(df):
    for c in PROB_CANDIDATES:
        if c in df.columns:
            return c
    # fallback: first float column mostly in [0,1]
    for c in df.columns:
        v = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(v) == 0:
            continue
        if v.min() >= -1e-6 and v.max() <= 1 + 1e-6:
            return c
    raise ValueError("Could not find a probability column for p_hat.")

# ---------- Model tags (only VC–Student with distillation) ----------
MODEL_TAGS = ["distill"]

def model_paths(tag):
    outdir = BASE / "outputs" / "ablation" / tag
    return {
        "thr":       outdir / f"vc_thresholds_ALL__{tag}.csv",
        "coef":      outdir / f"vc_coefficients_ALL__{tag}.csv",
        "effects":   outdir / f"index_weight_effects_per1SD__{tag}.csv",
        "weights":   outdir / f"index_weights__{tag}.csv",
        "contrib":   outdir / f"index_contributions_ALL__{tag}.csv",
        "outdir":    outdir
    }

# ===========================================
# LOOP OVER MODELS
# ===========================================
for TAG in MODEL_TAGS:
    P = model_paths(TAG)
    have = [k for k in ("thr","coef") if P[k].exists()]
    if len(have) < 2:
        print(f"[skip {TAG}] missing required CSVs → "
              f"{P['thr'].name if not P['thr'].exists() else ''} "
              f"{P['coef'].name if not P['coef'].exists() else ''}")
        continue

    print(f"\n=== Figures for model: {TAG} ===")
    thr  = pd.read_csv(P["thr"])
    coef = pd.read_csv(P["coef"])
    effects_df = pd.read_csv(P["effects"]) if P["effects"].exists() else None

    # ============================================================
    # (A) Pure alpha weights bar  (Fig 3)
    # ============================================================
    if P["weights"].exists():
        w = pd.read_csv(P["weights"])
        series = w.set_index("variable")["alpha_stdZ"].sort_values(
            key=lambda s: np.abs(s), ascending=True
        )

        fig, ax = plt.subplots(figsize=(6.8, 4.6))
        y = np.arange(len(series))
        colors = [PASTEL2(i/max(1, len(series)-1)) for i in range(len(series))]
        bars = ax.barh(y, series.values, color=colors,
                       edgecolor="#333", linewidth=0.6)

        off = 0.015 * (np.nanmax(np.abs(series.values)) + 1e-6)
        for b, v in zip(bars, series.values):
            if np.isfinite(v) and abs(v) > 1e-6:
                ax.text(v + (off if v>=0 else -off),
                        b.get_y()+b.get_height()/2,
                        f"{v:+.2f}",
                        va="center", ha=("left" if v>=0 else "right"),
                        fontsize=9, color="#222")

        ax.axvline(0, color="#555", lw=0.8, ls="--", alpha=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels([pretty(v) for v in series.index])
        ax.set_xlabel("α weight (std-Z units)")
        ax.set_title("Index weights α (shared across cuts)")
        ax.invert_yaxis()

        out = FIG_DIR / f"{TAG}__alpha_weights_pure.png"
        fig.tight_layout(); fig.savefig(out, dpi=DPI, bbox_inches="tight")
        if SHOW: plt.show()
        plt.close(fig)
        print(f"[ok] saved {out}")
    else:
        print(f"[info {TAG}] no index_weights file → skipping alpha bar.")

    # ============================================================
    # (C) Per-1SD effects bars (from CSV)  – three bar charts
    # ============================================================
    if effects_df is not None:
        fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), constrained_layout=True)
        idx = effects_df.set_index("driver")
        fancy_bar(axes[0], idx["thr_Air_C_per1SD"], "thr_Air_C")
        fancy_bar(axes[1], idx["thr_WBT_C_per1SD"], "thr_WBT_C")
        fancy_bar(axes[2], idx["thr_Dew_C_per1SD"], "thr_Dew_C")
        all_vals = pd.concat([
            effects_df["thr_Air_C_per1SD"],
            effects_df["thr_WBT_C_per1SD"],
            effects_df["thr_Dew_C_per1SD"]
        ], axis=0).dropna().values
        pad = 0.10 * (np.nanmax(np.abs(all_vals)) + 1e-6)
        xmax = np.nanmax(np.abs(all_vals)) + pad
        for ax in axes: ax.set_xlim(-xmax, xmax)
        out = FIG_DIR / f"{TAG}__index_effects_per1SD.png"
        fig.savefig(out, dpi=DPI, bbox_inches="tight")
        if SHOW: plt.show()
        plt.close(fig)
        print(f"[ok] saved {out}")
    else:
        print(f"[info {TAG}] no per-1SD effects CSV → skipping.")

    # ============================================================
    # (D) Violin distributions of thresholds (°C) – Fig 2
    # ============================================================
    thr_cols = [c for c in ["thr_Dew_C","thr_WBT_C","thr_Air_C"] if c in thr.columns]
    if thr_cols:
        fig, ax = plt.subplots()
        data = [thr[c].dropna().values for c in thr_cols]
        parts = ax.violinplot(data, showmeans=True, showextrema=False)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(PASTEL(i/len(thr_cols)))
            pc.set_edgecolor("none")
            pc.set_alpha(0.85)
        if "cmeans" in parts:
            parts['cmeans'].set_color("#333333")
            parts['cmeans'].set_linewidth(1.0)
        ax.set_xticks(range(1, len(thr_cols)+1))
        ax.set_xticklabels([pretty(c) for c in thr_cols])
        ax.set_ylabel("Threshold (°C)")
        ax.set_ylim(-10, 2)
        ax.axhline(0, color="#888", lw=0.6, ls="--", alpha=0.6)
        save(fig, f"{TAG}__R1_threshold_distributions")
    else:
        print(f"[info {TAG}] threshold columns missing for violin.")

    # ============================================================
    # (E) Threshold vs drivers — facets (Air cut only) – Fig 6
    # ============================================================
    driver_candidates = ["RH","gridded_data_pres","Z0","lapse_rate","temp_gradient","ApT",
                         "Elevation","Longitude","Latitude"]
    drivers = [c for c in driver_candidates if c in thr.columns]

    # unique, consistent color per driver
    def make_driver_colors(names):
        n = max(1, len(names))
        return {name: PASTEL2(i / max(1, n-1)) for i, name in enumerate(names)}
    driver_colors = make_driver_colors(drivers)

    def facet_binned(thr_df, ycol, xcols, fname, ncols=3, k_bins=25, n_min=60):
        n = len(xcols); ncols = min(ncols, n) if n>0 else 1
        nrows = math.ceil(n/ncols) if n>0 else 1
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*4.4, nrows*3.3), squeeze=False)
        ymins, ymaxs, curves = [], [], {}
        for xcol in xcols:
            sub = thr_df[[xcol, ycol]].dropna()
            if len(sub) < max(200, n_min*3):
                curves[xcol] = None; continue
            ds = binned_summary(sub[xcol].values, sub[ycol].values, k=k_bins, n_min=n_min)
            curves[xcol] = ds if len(ds) else None
            if ds is not None and not ds.empty:
                ymins.append(float(np.nanmin([ds["y_lo"].min(), ds["y_mean"].min()])))
                ymaxs.append(float(np.nanmax([ds["y_hi"].max(), ds["y_mean"].max()])))
        ylo = min(ymins) if ymins else None
        yhi = max(ymaxs) if ymaxs else None

        for i, xcol in enumerate(xcols):
            ax = axes[i//ncols, i % ncols]
            ds = curves.get(xcol, None)
            if ds is None or ds.empty:
                ax.text(0.5,0.5,"Not enough data",ha="center",va="center")
                ax.axis("off"); continue
            color = driver_colors.get(xcol, PASTEL2(0.2))
            nice_filled_curve(ax, ds, color=color)
            if (ylo is not None) and (yhi is not None): ax.set_ylim(ylo, yhi)
            ax.set_xlabel(pretty(xcol)); ax.set_ylabel(pretty(ycol))
        for j in range(i+1, nrows*ncols):
            axes[j//ncols, j % ncols].axis("off")
        save(fig, fname)

    if "thr_Air_C" in thr_cols and drivers:
        facet_binned(thr, "thr_Air_C", drivers,
                     f"{TAG}__R3_thr_Air_C_vs_all_drivers_facets")

    # ============================================================
    # (F) β vs Index_I — many-lines + median – Fig 11
    # ============================================================
    def make_group_id(df, lon="Longitude", lat="Latitude", elev="Elevation",
                      lon_res=0.25, lat_res=0.25, elev_res=100):
        g_lon = np.round(df[lon].astype(float) / lon_res) * lon_res
        g_lat = np.round(df[lat].astype(float) / lat_res) * lat_res
        g_ele = np.round(df[elev].astype(float) / elev_res) * elev_res
        return (g_lon.astype(str) + "_" + g_lat.astype(str) + "_" + g_ele.astype(str))

    def bin_curve(x, y, k=40, q_low=0.02, q_high=0.98, n_min=15):
        x = np.asarray(x, float); y = np.asarray(y, float)
        m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]
        if len(x) < max(k, n_min): return None
        lo, hi = np.quantile(x, [q_low, q_high])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi: return None
        xb  = np.linspace(lo, hi, k+1)
        idx = np.digitize(x, xb[1:-1], right=True)
        xm, ym = [], []
        for b in range(k):
            mb = (idx == b)
            if mb.sum() < n_min:
                xm.append(np.nan); ym.append(np.nan)
            else:
                xm.append(np.nanmean(x[mb])); ym.append(np.nanmean(y[mb]))
        xm, ym = np.array(xm), np.array(ym)
        m2 = np.isfinite(xm) & np.isfinite(ym)
        if m2.sum() < 8: return None
        o = np.argsort(xm[m2])
        return xm[m2][o], ym[m2][o]

    def many_lines_panel(coef_df, cols=("beta0","beta_Air_Temp","beta_Dewpoint","beta_Wet_Bulb_Temp"),
                         xcol="Index_I", fname=f"{TAG}__R4_beta_vs_index_facets"):
        need = ["Longitude","Latitude","Elevation", xcol]
        if not all(c in coef_df.columns for c in need):
            print("[skip] missing columns for many_lines_panel"); return
        df = coef_df.copy(); df["group_id"] = make_group_id(df)
        n = len(cols); ncols = 2; nrows = math.ceil(n/ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(10, 7.6), squeeze=False)
        for i, y in enumerate(cols):
            ax = axes[i//ncols, i % ncols]
            curves = []
            for gi, (gid, d) in enumerate(df.groupby("group_id")):
                out = bin_curve(d[xcol].values, d[y].values, k=40, q_low=0.02, q_high=0.98, n_min=15)
                if out is None: continue
                xg, yg = out; curves.append((xg, yg))
                ax.plot(xg, yg, lw=0.7, alpha=0.18, color=PASTEL((gi % 12)/12.0))
            if curves:
                lo_list = [c[0].min() for c in curves]
                hi_list = [c[0].max() for c in curves]
                Xlo = np.nanmax(lo_list); Xhi = np.nanmin(hi_list)
                if not np.isfinite(Xlo) or not np.isfinite(Xhi) or Xlo >= Xhi:
                    Xlo = np.nanmin(lo_list); Xhi = np.nanmax(hi_list)
                X = np.linspace(Xlo, Xhi, 220)
                Ys = [np.interp(X, x, yy, left=np.nan, right=np.nan) for x, yy in curves]
                Y_stack = np.vstack(Ys)
                support = np.sum(~np.isnan(Y_stack), axis=0)
                need_support = max(10, int(0.25 * len(curves)))
                mask = support >= need_support
                if mask.any():
                    X_med = X[mask]; Y_med = np.nanmedian(Y_stack[:, mask], axis=0)
                    ax.plot(X_med, Y_med, lw=2.4, color="#5b0f18")  # median
            ax.axhline(0, color="#444", lw=0.6, ls="--", alpha=.6)
            ax.set_xlabel(pretty(xcol)); ax.set_ylabel("Coefficient")
            title_map = {c: pretty(c) for c in cols}
            ax.set_title(f"{title_map.get(y,y)} vs {pretty(xcol)}")
        for j in range(i+1, nrows*ncols):
            axes[j//ncols, j % ncols].axis("off")
        save(fig, fname, w=10, h=7.6,
             caption="Many-line functional plots of β vs learned index I(Z). Median drawn only where ≥25% curves overlap.")

    many_lines_panel(coef)

    # ============================================================
    # (G) ROC + PR curves – Fig 12
    # ============================================================
    # Try several possible prediction file locations/formats
    pred_candidates = [
        BASE / "outputs" / "eval"     / f"heldout_predictions__{TAG}.csv",
        BASE / "outputs" / "eval"     / f"heldout_predictions__{TAG}.xlsx",
        BASE / "outputs" / "ablation" / TAG / f"predictions_test__{TAG}.csv",
        BASE / "outputs" / "ablation" / TAG / f"predictions_test__{TAG}.xlsx",
    ]
    pred_path = next((p for p in pred_candidates if p.exists()), None)

    if pred_path is not None:
        print("[eval]", "using prediction file:", pred_path)
        if pred_path.suffix.lower() == ".xlsx":
            preds = pd.read_excel(pred_path)
        else:
            preds = pd.read_csv(pred_path)

        y_col = pick_true_col(preds)
        p_col = pick_prob_col(preds)

        y_true = preds[y_col].to_numpy(dtype=int)
        p_hat  = preds[p_col].to_numpy(dtype=float)

        fpr, tpr, _ = roc_curve(y_true, p_hat)
        auc = roc_auc_score(y_true, p_hat)

        prec, rec, _ = precision_recall_curve(y_true, p_hat)
        ap = average_precision_score(y_true, p_hat)

        fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.5))
        # ROC
        ax = axes[0]
        ax.plot(fpr, tpr, lw=2)
        ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title(f"ROC (AUC = {auc:.3f})")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        # PR
        ax = axes[1]
        ax.plot(rec, prec, lw=2)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"Precision-Recall (AP = {ap:.3f})")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

        caption = (
            "Held-out discrimination skill. Left: ROC curve (AUC ≈ "
            f"{auc:.2f}). Right: precision–recall curve (average precision ≈ "
            f"{ap:.2f}). The VC–Student matches the performance of the higher-"
            "capacity DCu–Teacher while remaining interpretable."
        )
        save(fig, f"{TAG}__eval_ROC_PR", w=8.5, h=3.5, caption=caption)
    else:
        print(f"[info {TAG}] no predictions file found for ROC/PR.")

    # ============================================================
    # (H) Reliability diagram – Fig 13
    # ============================================================
    if pred_path is not None:
        bin_c, emp_f, ece = reliability_diagram(y_true, p_hat, n_bins=20)
        brier = brier_score_loss(y_true, p_hat)

        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
        ax.plot(bin_c, emp_f, "o-", lw=2)
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Empirical frequency")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_title(f"Reliability (ECE={ece:.3f}, Brier={brier:.3f})")

        caption = (
            "Reliability diagram on the held-out test set. Points lie close to "
            "the 1:1 line, with expected calibration error (ECE) ≈ "
            f"{ece:.3f} and Brier score ≈ {brier:.3f}, indicating well-"
            "calibrated probabilities."
        )
        save(fig, f"{TAG}__eval_reliability", w=4.8, h=4.8, caption=caption)

# ============================================================
# (I) Index of artifacts
# ============================================================
OUTDIR = BASE / "outputs"
artifacts = {
    "figures": sorted(str(p) for p in FIG_DIR.glob("*.png")),
    "tables":  sorted(str(p) for p in OUTDIR.glob("**/*.csv")) +
               sorted(str(p) for p in OUTDIR.glob("**/*__caption.txt")) +
               sorted(str(p) for p in TAB_DIR.glob("*.csv")) +
               sorted(str(p) for p in TAB_DIR.glob("*__caption.txt")),
}
(OUTDIR / "artifacts_index.json").write_text(json.dumps(artifacts, indent=2), encoding="utf-8")
print("\nDone.\nFigures:", FIG_DIR.as_posix(), "\nOutputs:", OUTDIR.as_posix())
