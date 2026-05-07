import json
from pathlib import Path
import numpy as np
import pandas as pd

# -------------------------------------------------
# Fix sign-indeterminacy in stability summaries
# -------------------------------------------------

THIS = Path(__file__).resolve()
REPO_ROOT = THIS.parent.parent

RUN_DIR = REPO_ROOT / "outputs" / "stability" / "distill"
RUNS_CSV = RUN_DIR / "stability_runs.csv"
FULL_JSON = RUN_DIR / "full_data_summary.json"

OUT_RUNS = RUN_DIR / "stability_runs_aligned.csv"
OUT_SUMMARY = RUN_DIR / "stability_summary_aligned.csv"

if not RUNS_CSV.exists():
    raise FileNotFoundError(f"Missing file: {RUNS_CSV}")
if not FULL_JSON.exists():
    raise FileNotFoundError(f"Missing file: {FULL_JSON}")

runs = pd.read_csv(RUNS_CSV)
full = json.loads(FULL_JSON.read_text())

alpha_cols = [
    "alpha_RH",
    "alpha_gridded_data_pres",
    "alpha_Z0",
    "alpha_lapse_rate",
    "alpha_temp_gradient",
    "alpha_ApT",
    "alpha_Elevation",
    "alpha_Longitude",
    "alpha_Latitude",
]

signed_cols = ["corr_index_air"] + alpha_cols

full_alpha = np.array([full[c] for c in alpha_cols], dtype=float)

runs_aligned = runs.copy()
flip_flags = []
dot_products = []

for i, row in runs_aligned.iterrows():
    rep_alpha = row[alpha_cols].to_numpy(dtype=float)
    dot = float(np.dot(rep_alpha, full_alpha))
    flip = dot < 0

    dot_products.append(dot)
    flip_flags.append(flip)

    if flip:
        runs_aligned.loc[i, signed_cols] = -runs_aligned.loc[i, signed_cols]

runs_aligned["sign_flipped"] = np.array(flip_flags, dtype=int)
runs_aligned["alignment_dot"] = dot_products


def summarize_runs(df: pd.DataFrame, full_row: dict | None = None) -> pd.DataFrame:
    exclude = {"label", "seed", "sign_flipped"}
    summary_rows = []

    for col in df.columns:
        if col in exclude:
            continue

        if pd.api.types.is_bool_dtype(df[col]):
            continue

        vals = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy()
        if len(vals) == 0:
            continue

        row = {
            "quantity": col,
            "n_reps": int(len(vals)),
            "median": float(np.median(vals)),
            "q025": float(np.quantile(vals, 0.025)),
            "q975": float(np.quantile(vals, 0.975)),
            "mean": float(np.mean(vals)),
            "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "full_data_estimate": np.nan,
        }

        if full_row is not None and col in full_row:
            try:
                row["full_data_estimate"] = float(full_row[col])
            except Exception:
                pass

        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


summary_aligned = summarize_runs(runs_aligned, full_row=full)

runs_aligned.to_csv(OUT_RUNS, index=False)
summary_aligned.to_csv(OUT_SUMMARY, index=False)

print(f"Saved aligned runs:    {OUT_RUNS}")
print(f"Saved aligned summary: {OUT_SUMMARY}")
print(f"Replicates flipped: {int(np.sum(flip_flags))} / {len(flip_flags)}")

print("\nKey corrected rows:\n")
keep = [
    "auc_test",
    "acc_test",
    "corr_index_air",
    "corr_student_teacher",
    "mae_student_teacher",
] + alpha_cols

print(summary_aligned[summary_aligned["quantity"].isin(keep)].to_string(index=False))