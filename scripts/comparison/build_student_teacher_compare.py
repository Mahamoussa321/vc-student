from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "outputs" / "comparison"
OUTDIR.mkdir(parents=True, exist_ok=True)

student_path = ROOT / "outputs" / "ablation" / "distill" / "vc_thresholds_ALL__distill.csv"
teacher_path = ROOT / "outputs" / "comparison" / "teacher_thresholds_ALL.csv"

if not student_path.exists():
    raise FileNotFoundError(f"Missing student file: {student_path}")
if not teacher_path.exists():
    raise FileNotFoundError(f"Missing teacher file: {teacher_path}")

student = pd.read_csv(student_path)
teacher = pd.read_csv(teacher_path)

need = ["Longitude", "Latitude", "thr_Air_C", "thr_WBT_C", "thr_Dew_C"]
for c in need:
    if c not in student.columns:
        raise KeyError(f"Student file missing column: {c}")
    if c not in teacher.columns:
        raise KeyError(f"Teacher file missing column: {c}")

student_station = (
    student
    .groupby(["Longitude", "Latitude"], as_index=False)
    .agg({
        "thr_Air_C": "mean",
        "thr_WBT_C": "mean",
        "thr_Dew_C": "mean",
        **({"Elevation": "mean"} if "Elevation" in student.columns else {})
    })
    .rename(columns={
        "thr_Air_C": "student_thr_Air_C",
        "thr_WBT_C": "student_thr_WBT_C",
        "thr_Dew_C": "student_thr_Dew_C",
        "Elevation": "student_Elevation"
    })
)

teacher_station = (
    teacher
    .groupby(["Longitude", "Latitude"], as_index=False)
    .agg({
        "thr_Air_C": "mean",
        "thr_WBT_C": "mean",
        "thr_Dew_C": "mean",
        **({"Elevation": "mean"} if "Elevation" in teacher.columns else {})
    })
    .rename(columns={
        "thr_Air_C": "teacher_thr_Air_C",
        "thr_WBT_C": "teacher_thr_WBT_C",
        "thr_Dew_C": "teacher_thr_Dew_C",
        "Elevation": "teacher_Elevation"
    })
)

compare = student_station.merge(
    teacher_station,
    on=["Longitude", "Latitude"],
    how="inner",
    validate="one_to_one"
)

compare["diff_Air_C"] = compare["student_thr_Air_C"] - compare["teacher_thr_Air_C"]
compare["diff_WBT_C"] = compare["student_thr_WBT_C"] - compare["teacher_thr_WBT_C"]
compare["diff_Dew_C"] = compare["student_thr_Dew_C"] - compare["teacher_thr_Dew_C"]

compare.to_csv(OUTDIR / "student_teacher_station_compare.csv", index=False)

def summarize_pair(df, s_col, t_col, label):
    diff = df[s_col] - df[t_col]
    return {
        "variable": label,
        "n_station": int(df[[s_col, t_col]].dropna().shape[0]),
        "corr": float(df[[s_col, t_col]].corr().iloc[0, 1]),
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "mean_diff": float(np.mean(diff)),
        "median_diff": float(np.median(diff)),
    }

summary = pd.DataFrame([
    summarize_pair(compare, "student_thr_Air_C", "teacher_thr_Air_C", "Air"),
    summarize_pair(compare, "student_thr_WBT_C", "teacher_thr_WBT_C", "WBT"),
    summarize_pair(compare, "student_thr_Dew_C", "teacher_thr_Dew_C", "Dew"),
])

summary.to_csv(OUTDIR / "student_teacher_compare_summary.csv", index=False)

with open(OUTDIR / "student_teacher_compare_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary.to_dict(orient="records"), f, indent=2)

print("[ok] saved:", OUTDIR / "student_teacher_station_compare.csv")
print("[ok] saved:", OUTDIR / "student_teacher_compare_summary.csv")
print(summary.to_string(index=False))
