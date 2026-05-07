import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score


def convert_to_binary_rain(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        vals = set(series.dropna().astype(int).unique())
        if vals.issubset({0, 1}):
            return series.astype(int)
        raise ValueError(f"Numeric outcome is not coded as 0/1. Found: {sorted(vals)}")

    s = series.astype(str).str.strip().str.lower()
    rain_levels = {"rain", "r", "1", "yes"}
    snow_levels = {"snow", "s", "0", "no"}

    out = pd.Series(np.nan, index=series.index)
    out[s.isin(rain_levels)] = 1
    out[s.isin(snow_levels)] = 0

    bad = s[out.isna()].unique().tolist()
    if bad:
        raise ValueError(f"Could not map some outcome values to rain/snow: {bad}")
    return out.astype(int)


def compute_rule_metrics(y_true: np.ndarray, score: np.ndarray, threshold: float, model_name: str):
    y_pred = (score > threshold).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    n = tp + tn + fp + fn
    accuracy = (tp + tn) / n
    rain_recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    snow_recall = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    f1 = (
        2 * precision * rain_recall / (precision + rain_recall)
        if pd.notna(precision) and pd.notna(rain_recall) and (precision + rain_recall) > 0
        else np.nan
    )
    auc = roc_auc_score(y_true, score)

    return {
        "Model": model_name,
        "Threshold_C": threshold,
        "N": n,
        "Accuracy": accuracy,
        "Rain_Recall": rain_recall,
        "Snow_Recall": snow_recall,
        "Precision": precision,
        "F1": f1,
        "AUC": auc,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
    }


def fmt_pct(x):
    return f"{100 * x:.2f}%"


def fmt_num(x):
    return f"{x:.4f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--y", required=True)
    parser.add_argument("--tair", required=True)
    parser.add_argument("--twbt", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = input_path.parent

    df = pd.read_csv(input_path)

    df["y_true_bin"] = convert_to_binary_rain(df[args.y])
    df["T_air_raw"] = pd.to_numeric(df[args.tair], errors="coerce")
    df["T_wbt_raw"] = pd.to_numeric(df[args.twbt], errors="coerce")

    dat = df.dropna(subset=["y_true_bin", "T_air_raw", "T_wbt_raw"]).copy()

    res_air_0 = compute_rule_metrics(
        dat["y_true_bin"].to_numpy(),
        dat["T_air_raw"].to_numpy(),
        0.0,
        "Fixed air-temperature rule (0C)",
    )

    res_twbt_1 = compute_rule_metrics(
        dat["y_true_bin"].to_numpy(),
        dat["T_wbt_raw"].to_numpy(),
        1.0,
        "Fixed wet-bulb rule (1.0C)",
    )

    res_twbt_1_1 = compute_rule_metrics(
        dat["y_true_bin"].to_numpy(),
        dat["T_wbt_raw"].to_numpy(),
        1.1,
        "Fixed wet-bulb rule (1.1C) [optional]",
    )

    results_main = pd.DataFrame([res_air_0, res_twbt_1])
    results_all = pd.DataFrame([res_air_0, res_twbt_1, res_twbt_1_1])

    results_main.to_csv(out_dir / "simple_rule_baselines_main.csv", index=False)
    results_all.to_csv(out_dir / "simple_rule_baselines_with_optional_1p1C.csv", index=False)

    print("=============== MAIN RESULTS ===============")
    paper_main = results_main.copy()
    for col in ["Accuracy", "Rain_Recall", "Snow_Recall", "Precision"]:
        paper_main[col] = paper_main[col].map(fmt_pct)
    for col in ["F1", "AUC"]:
        paper_main[col] = paper_main[col].map(fmt_num)
    print(
        paper_main[
            ["Model", "N", "Accuracy", "Rain_Recall", "Snow_Recall", "Precision", "F1", "AUC"]
        ].to_string(index=False)
    )

    print("\n=============== CONFUSION-COUNT DETAILS ===============")
    print(results_all[["Model", "TP", "TN", "FP", "FN"]].to_string(index=False))


if __name__ == "__main__":
    main()
