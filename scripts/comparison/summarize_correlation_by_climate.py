from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(r"C:/Users/maham/Desktop/vc-student")
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

need = ["Cls", "Longitude", "Latitude", "thr_Air_C", "thr_WBT_C", "thr_Dew_C"]
for c in need:
    if c not in student.columns:
        raise KeyError(f"Student file missing column: {c}")
    if c not in teacher.columns:
        raise KeyError(f"Teacher file missing column: {c}")


def station_mode(x: pd.Series):
    x = x.dropna().astype(str)
    if len(x) == 0:
        return np.nan
    m = x.mode()
    return m.iloc[0] if len(m) else np.nan


student_station = (
    student
    .groupby(["Longitude", "Latitude"], as_index=False)
    .agg({
        "Cls": station_mode,
        "thr_Air_C": "mean",
        "thr_WBT_C": "mean",
        "thr_Dew_C": "mean",
    })
    .rename(columns={
        "Cls": "student_Cls",
        "thr_Air_C": "student_thr_Air_C",
        "thr_WBT_C": "student_thr_WBT_C",
        "thr_Dew_C": "student_thr_Dew_C",
    })
)

teacher_station = (
    teacher
    .groupby(["Longitude", "Latitude"], as_index=False)
    .agg({
        "Cls": station_mode,
        "thr_Air_C": "mean",
        "thr_WBT_C": "mean",
        "thr_Dew_C": "mean",
    })
    .rename(columns={
        "Cls": "teacher_Cls",
        "thr_Air_C": "teacher_thr_Air_C",
        "thr_WBT_C": "teacher_thr_WBT_C",
        "thr_Dew_C": "teacher_thr_Dew_C",
    })
)

compare = student_station.merge(
    teacher_station,
    on=["Longitude", "Latitude"],
    how="inner",
    validate="one_to_one"
)

compare["climate_group"] = compare["student_Cls"]
compare["climate_match"] = compare["student_Cls"].astype(str) == compare["teacher_Cls"].astype(str)

compare["diff_Air_C"] = compare["student_thr_Air_C"] - compare["teacher_thr_Air_C"]
compare["diff_WBT_C"] = compare["student_thr_WBT_C"] - compare["teacher_thr_WBT_C"]
compare["diff_Dew_C"] = compare["student_thr_Dew_C"] - compare["teacher_thr_Dew_C"]

compare.to_csv(OUTDIR / "student_teacher_station_compare_by_climate.csv", index=False)


def summarize_one(df: pd.DataFrame, s_col: str, t_col: str, label: str):
    df = df[[s_col, t_col]].dropna()
    if len(df) < 3:
        return {
            "variable": label,
            "n_station": int(len(df)),
            "corr": np.nan,
            "mae": np.nan,
            "rmse": np.nan,
            "mean_diff": np.nan,
            "median_diff": np.nan,
        }
    diff = df[s_col] - df[t_col]
    return {
        "variable": label,
        "n_station": int(len(df)),
        "corr": float(df[[s_col, t_col]].corr().iloc[0, 1]),
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "mean_diff": float(np.mean(diff)),
        "median_diff": float(np.median(diff)),
    }


rows = []
for grp, sub in compare.groupby("climate_group"):
    rows.append({
        "climate_group": grp,
        **summarize_one(sub, "student_thr_Air_C", "teacher_thr_Air_C", "Air")
    })
    rows.append({
        "climate_group": grp,
        **summarize_one(sub, "student_thr_WBT_C", "teacher_thr_WBT_C", "WBT")
    })
    rows.append({
        "climate_group": grp,
        **summarize_one(sub, "student_thr_Dew_C", "teacher_thr_Dew_C", "Dew")
    })

summary = pd.DataFrame(rows).sort_values(["variable", "climate_group"])
summary.to_csv(OUTDIR / "student_teacher_correlation_by_climate.csv", index=False)

agreement = (
    compare.groupby("climate_group", as_index=False)
    .agg(
        n_station=("climate_match", "size"),
        climate_label_agreement=("climate_match", "mean")
    )
    .sort_values("n_station", ascending=False)
)
agreement.to_csv(OUTDIR / "student_teacher_climate_label_agreement.csv", index=False)

with open(OUTDIR / "student_teacher_correlation_by_climate.json", "w", encoding="utf-8") as f:
    json.dump(summary.to_dict(orient="records"), f, indent=2)

print("\n[ok] saved:", OUTDIR / "student_teacher_station_compare_by_climate.csv")
print("[ok] saved:", OUTDIR / "student_teacher_correlation_by_climate.csv")
print("[ok] saved:", OUTDIR / "student_teacher_climate_label_agreement.csv")

print("\nTop climate groups for AIR by correlation:")
air_top = (
    summary[summary["variable"] == "Air"]
    .sort_values(["corr", "n_station"], ascending=[False, False])
    .head(15)
)
print(air_top.to_string(index=False))
