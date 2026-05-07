import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def wilson_interval(k, n, z=1.96):
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    p = np.divide(k, n, out=np.zeros_like(k), where=n > 0)
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    lo = np.clip(center - half, 0, 1)
    hi = np.clip(center + half, 0, 1)
    return lo, hi


def probability_vs_index_figure(index_vals, p_hat, y_true, fname, *, save, pretty,
                                k_bins=20, min_bin_n=400, qtrim=(0.01, 0.99)):
    """
    Cleaner held-out figure for probability vs learned index.

    Shows:
      - mean fitted rain probability by equal-count index bins
      - empirical rain frequency with 95% Wilson intervals
      - subtle support bars at the bottom (same x scale, no second y-axis)

    This figure is more journal-friendly than the earlier version because it
    avoids a distracting secondary axis and very wide IQR ribbons.
    """
    idx = np.asarray(index_vals, float)
    p = np.asarray(p_hat, float)
    y = np.asarray(y_true, float)
    m = np.isfinite(idx) & np.isfinite(p) & np.isfinite(y)
    idx, p, y = idx[m], p[m], y[m]

    if len(idx) < max(2000, k_bins * min_bin_n):
        print("[info] not enough aligned rows for improved probability-vs-index figure.")
        return

    # Trim extreme tails so the x-axis focuses on where the data actually are.
    xlo, xhi = np.quantile(idx, qtrim)
    keep = (idx >= xlo) & (idx <= xhi)
    idx, p, y = idx[keep], p[keep], y[keep]

    # Equal-count bins are usually better here than equal-width bins.
    q = np.linspace(0, 1, k_bins + 1)
    edges = np.quantile(idx, q)
    # guard against duplicate quantiles
    edges = np.unique(edges)
    if len(edges) - 1 < 8:
        print("[info] too few unique bins for improved probability-vs-index figure.")
        return

    bin_id = np.digitize(idx, edges[1:-1], right=True)

    rows = []
    for b in range(len(edges) - 1):
        mb = bin_id == b
        n = int(mb.sum())
        if n < min_bin_n:
            continue
        xb = idx[mb]
        pb = p[mb]
        yb = y[mb]
        xmid = float(np.mean(xb))
        pmean = float(np.mean(pb))
        ymean = float(np.mean(yb))
        lo, hi = wilson_interval(int(np.sum(yb)), n)
        rows.append((xmid, pmean, ymean, float(lo), float(hi), n))

    ds = pd.DataFrame(rows, columns=["x_mid", "p_mean", "y_mean", "y_lo", "y_hi", "n"])
    if ds.empty:
        print("[info] empty bin summary for improved probability-vs-index figure.")
        return

    fig, ax = plt.subplots(figsize=(6.8, 4.8))

    fit_color = "#E67E5F"
    emp_color = "#C8A36A"
    support_color = "#BEBEBE"

    # subtle support bars anchored at the bottom; no secondary axis
    counts_scaled = 0.12 * ds["n"] / ds["n"].max()
    ax.bar(ds["x_mid"], counts_scaled, width=np.diff(edges).mean() * 0.8,
           bottom=0, color=support_color, alpha=0.25, linewidth=0, zorder=1)

    # fitted probability line
    ax.plot(ds["x_mid"], ds["p_mean"], lw=2.4, marker="o", ms=4,
            color=fit_color, label="Mean fitted rain probability", zorder=3)

    # empirical frequency with Wilson interval ribbon
    ax.fill_between(ds["x_mid"], ds["y_lo"], ds["y_hi"],
                    color=emp_color, alpha=0.18, linewidth=0, zorder=2)
    ax.plot(ds["x_mid"], ds["y_mean"], lw=2.1, marker="s", ms=3.8,
            color=emp_color, label="Empirical rain frequency", zorder=4)

    ax.set_xlabel(pretty("Index_I"))
    ax.set_ylabel("Rain probability")
    ax.set_title("Rain probability versus learned index")
    ax.set_ylim(0, 1)
    ax.set_xlim(ds["x_mid"].min() - 0.05, ds["x_mid"].max() + 0.05)
    ax.legend(frameon=False, loc="upper left")

    caption = (
        "Held-out relationship between the learned contextual index and rain probability. "
        "Points are grouped into equal-count bins of the learned index. The first line shows "
        "the mean fitted rain probability in each bin, and the second line shows the empirical "
        "rain frequency with 95% Wilson intervals. The close agreement indicates that the "
        "learned one-dimensional index organizes predicted probabilities in a coherent way, "
        "in addition to its role in the threshold and coefficient functions."
    )
    save(fig, fname, w=6.8, h=4.8, caption=caption)
