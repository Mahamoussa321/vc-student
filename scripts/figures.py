
# ===========================
# ALL FIGURES (with pretty labels) — one cell
# ===========================
import os, math, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
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
PASTEL2 = mpl.colormaps["Set2"]     # bars/boxes
HEAT    = mpl.colormaps["RdBu_r"]   # heatmap
MEDIAN_COLOR = "#5b0f18"
IQR_SHADE    = "#8ab6d6"
LINE_BLUE    = "#2a6f97"

# ---------- Pretty labels (keep canonical names in CSVs!) ----------
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
    "temp_gradient": r"$\Gamma_b$",
    "lapse_rate": r"$\Gamma_{ll}$",
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

def nice_filled_curve(ax, ds, label=None):
    ax.fill_between(ds["x_mid"], ds["y_lo"], ds["y_hi"], alpha=0.25,
                    color=IQR_SHADE, linewidth=0)
    ax.plot(ds["x_mid"], ds["y_mean"], lw=2.0, color=LINE_BLUE, label=label)

# ---------- Load CSVs ----------
thr_path     = BASE / "outputs" / "vc_thresholds_ALL.csv"
coef_path    = BASE / "outputs" / "vc_coefficients_ALL.csv"
effects_path = BASE / "outputs" / "index_weight_effects_per1SD.csv"


thr  = pd.read_csv(thr_path)
coef = pd.read_csv(coef_path)
effects_df = pd.read_csv(effects_path) if effects_path.exists() else None

# ============================================================
# (0) Index-I effects bar charts (per +1 SD in Z)
# ============================================================
if effects_df is not None:
    # Stable driver order & color map
    drivers = list(effects_df["driver"].unique())
    color_map = {d: PASTEL2(i/max(1,len(drivers)-1)) for i,d in enumerate(drivers)}

    def fancy_bar(ax, series, title_canon):
        # 'series' is a Series indexed by canonical driver names
        order = series.reindex(drivers)  # stable order
        y = np.arange(len(order))
        c = [color_map.get(d, "#888") for d in order.index]
        bars = ax.barh(y, order.values, height=0.7, color=c, edgecolor="#333", linewidth=0.5)
        # value labels
        for b, v in zip(bars, order.values):
            if np.isfinite(v) and abs(v) > 1e-6:
                ax.text(v + (0.02 if v >= 0 else -0.02),
                        b.get_y()+b.get_height()/2,
                        f"{v:+.2f}",
                        va="center",
                        ha=("left" if v>=0 else "right"),
                        color="#222", fontsize=9)
        ax.axvline(0, color="#555", lw=0.8, ls="--", alpha=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels([pretty(d) for d in order.index])
        ax.set_title(pretty(title_canon))
        ax.set_xlabel("Effect (°C)")
        ax.invert_yaxis()

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), constrained_layout=True)
    idx = effects_df.set_index("driver")
    fancy_bar(axes[0], idx["thr_Air_C_per1SD"], "thr_Air_C")
    fancy_bar(axes[1], idx["thr_WBT_C_per1SD"], "thr_WBT_C")
    fancy_bar(axes[2], idx["thr_Dew_C_per1SD"], "thr_Dew_C")

    # equal x-lims
    all_vals = pd.concat([
        effects_df["thr_Air_C_per1SD"],
        effects_df["thr_WBT_C_per1SD"],
        effects_df["thr_Dew_C_per1SD"]
    ], axis=0).dropna().values
    pad = 0.10 * (np.nanmax(np.abs(all_vals)) + 1e-6)
    xmax = np.nanmax(np.abs(all_vals)) + pad
    for ax in axes: ax.set_xlim(-xmax, xmax)

    out = FIG_DIR / "index_effects_per1SD.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    if SHOW: plt.show()
    plt.close(fig)
    print(f"[ok] saved {out}")
else:
    print("[info] index_weight_effects_per1SD.csv not found → skipping Index-I effects bars.")

# ============================================================
# (1) Violin of thresholds (°C)
# ============================================================
thr_cols = [c for c in ["thr_Dew_C","thr_WBT_C","thr_Air_C"] if c in thr.columns]

fig, ax = plt.subplots()

# collect data for each threshold
data = [thr[c].dropna().values for c in thr_cols]

# make violins
parts = ax.violinplot(
    data,
    showmeans=True,
    showextrema=False
)

# color + style
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(PASTEL(i/len(thr_cols)))
    pc.set_edgecolor("none")
    pc.set_alpha(0.85)

if "cmeans" in parts:
    parts['cmeans'].set_color("#333333")
    parts['cmeans'].set_linewidth(1.0)

# x-axis labeling
ax.set_xticks(range(1, len(thr_cols)+1))
ax.set_xticklabels([pretty(c) for c in thr_cols])

# y-axis labeling
ax.set_ylabel("Threshold (°C)")

# *** NEW: tighten the y-limits so we zoom in on the useful range ***
ax.set_ylim(-10, 2)   # <- adjust if you want a bit more/less padding

# optional but nice: light horizontal zero line
ax.axhline(0, color="#888", lw=0.6, ls="--", alpha=0.6)

save(fig, "R1_threshold_distributions")


# ============================================================
# (2) Threshold vs drivers — facet grids (binned mean + IQR)
# ============================================================
driver_candidates = ["RH","gridded_data_pres","Z0","lapse_rate","temp_gradient","ApT",
                     "Elevation","Longitude","Latitude"]
drivers = [c for c in driver_candidates if c in thr.columns]

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
        nice_filled_curve(ax, ds)
        if (ylo is not None) and (yhi is not None): ax.set_ylim(ylo, yhi)
        ax.set_xlabel(pretty(xcol)); ax.set_ylabel(pretty(ycol))
    for j in range(i+1, nrows*ncols):
        axes[j//ncols, j % ncols].axis("off")
    save(fig, fname)

for t in thr_cols:
    if drivers:
        facet_binned(thr, t, drivers, f"R3_{t}_vs_all_drivers_facets")

# ============================================================
# (3) β vs Index_I — 2×2 many-lines + median
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
                     xcol="Index_I", fname="R4_beta_vs_index_facets"):
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
                ax.plot(X_med, Y_med, lw=2.4, color=MEDIAN_COLOR)
        ax.axhline(0, color="#444", lw=0.6, ls="--", alpha=.6)
        ax.set_xlabel(pretty(xcol)); ax.set_ylabel("Coefficient")
        title_map = {c: pretty(c) for c in cols}
        ax.set_title(f"{title_map.get(y,y)} vs {pretty(xcol)}")
    for j in range(i+1, nrows*ncols):
        axes[j//ncols, j % ncols].axis("off")
    save(fig, fname, w=10, h=7.6,
         caption="Many-line functional plots of β vs learned index I(Z). Median drawn only where ≥25% (≥10) curves overlap.")

many_lines_panel(coef)

# ============================================================
# (4) Correlation heatmap for profile drivers
# ============================================================
hm_vars = [c for c in ["Z0","ApT","temp_gradient","lapse_rate"] if c in thr.columns]
if len(hm_vars) >= 2:
    C = thr[hm_vars].corr().values
    fig, ax = plt.subplots(figsize=(6,5.2))
    im = ax.imshow(C, cmap=HEAT, vmin=-1, vmax=1)
    ax.set_xticks(range(len(hm_vars))); ax.set_xticklabels([pretty(v) for v in hm_vars], rotation=45, ha="right")
    ax.set_yticks(range(len(hm_vars))); ax.set_yticklabels([pretty(v) for v in hm_vars])
    for i in range(len(hm_vars)):
        for j in range(len(hm_vars)):
            ax.text(j, i, f"{C[i,j]:.2f}", ha="center", va="center", color="#111", fontsize=9)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cbar.set_label("corr")
    save(fig, "R_corr_profile_drivers")

# ============================================================
# (5) Index of artifacts
# ============================================================
artifacts = {
    "figures": sorted(str(p) for p in FIG_DIR.glob("*.png")),
    "tables":  sorted(str(p) for p in TAB_DIR.glob("*.csv")) + sorted(str(p) for p in TAB_DIR.glob("*__caption.txt")),
}
(BASE / "outputs" / "artifacts_index.json").write_text(json.dumps(artifacts, indent=2), encoding="utf-8")
print("Done.\nFigures:", FIG_DIR.as_posix(), "\nCaptions:", TAB_DIR.as_posix())
