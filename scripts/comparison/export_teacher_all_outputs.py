import os
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from pathlib import Path
import json
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score, brier_score_loss
from tensorflow.keras import layers, models

SEED = 2025
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "point_data_with_atmospheric_predictors.csv"
TEACHER_WEIGHTS = ROOT / "data" / "best.weights.h5"
OUTDIR = ROOT / "outputs" / "comparison"
OUTDIR.mkdir(parents=True, exist_ok=True)

y_var = "Rain_Phase"
region_col = "Cls"
x_input_vars = ["Air_Temp", "Wet_Bulb_Temp", "Dewpoint"]
z_input_vars_all = ["RH","gridded_data_pres","Longitude","Latitude","Elevation","lapse_rate","Z0","temp_gradient","ApT"]
Z_KEEP = ["RH","gridded_data_pres","Z0","lapse_rate","temp_gradient","ApT"]

TEMPORAL_SPLIT = True
TEST_YEARS = 2
VAL_FRACTION = 0.20
DATE_CANDIDATES = ["Date", "Timestamp", "date", "timestamp", "datetime"]

def build_teacher(x_dim, z_dim, r_dim, gate_type="relu", gate_k=5.0, dropout_rate=0.30):
    x_input = layers.Input(shape=(x_dim,), name="x_input")
    z_input = layers.Input(shape=(z_dim,), name="z_input")
    r_input = layers.Input(shape=(r_dim,), name="region_input")

    r_embed = layers.Dense(64, activation="relu", name="r_embed_1")(r_input)
    r_embed = layers.BatchNormalization(name="r_embed_bn")(r_embed)
    r_embed = layers.Dense(32, activation="relu", name="r_embed_2")(r_embed)

    zr = layers.Concatenate(name="zr_concat")([z_input, r_embed])
    h  = layers.Dense(128, activation="relu", name="cut_h1")(zr)
    h  = layers.BatchNormalization(name="cut_bn1")(h)
    h  = layers.Dropout(dropout_rate, seed=SEED, name="cut_do1")(h)
    h  = layers.Dense(64, activation="relu", name="cut_h2")(h)
    h  = layers.BatchNormalization(name="cut_bn2")(h)
    h  = layers.Dropout(dropout_rate, seed=SEED, name="cut_do2")(h)

    dew_base = layers.Dense(1, activation="linear", name="dew_base")(h)
    wb_off   = layers.Dense(1, activation="linear", name="wb_off")(h)
    air_off  = layers.Dense(1, activation="linear", name="air_off")(h)

    wb_thr  = layers.Lambda(lambda t: t[0] + tf.nn.softplus(t[1]), name="wb_thr")([dew_base, wb_off])
    air_thr = layers.Lambda(lambda t: t[0] + tf.nn.softplus(t[1]), name="air_thr")([wb_thr, air_off])

    cutting_points = layers.Concatenate(name="cutting_points")([air_thr, wb_thr, dew_base])

    def gate_pair(x, c):
        if gate_type == "relu":
            return tf.nn.relu(x - c)
        elif gate_type == "softplus":
            return tf.math.softplus(gate_k * (x - c)) / gate_k
        else:
            return tf.sigmoid(gate_k * (x - c)) * (x - c)

    thresholded_x = layers.Lambda(lambda t: gate_pair(t[0], t[1]), name="thresholded_x")([x_input, cutting_points])

    all_feat = layers.Concatenate(name="clf_concat")([thresholded_x, z_input, r_embed])
    h  = layers.Dense(128, activation="relu", name="clf_h1")(all_feat)
    h  = layers.BatchNormalization(name="clf_bn1")(h)
    h  = layers.Dropout(dropout_rate, seed=SEED, name="clf_do1")(h)
    h  = layers.Dense(64, activation="relu", name="clf_h2")(h)
    h  = layers.BatchNormalization(name="clf_bn2")(h)
    h  = layers.Dropout(dropout_rate, seed=SEED, name="clf_do2")(h)
    h  = layers.Dense(32, activation="relu", name="clf_h3")(h)
    y_out = layers.Dense(1, activation="sigmoid", name="output")(h)

    return models.Model([x_input, z_input, r_input], [y_out, cutting_points], name="DeepCutRegionEmbed")

print("[info] ROOT:", ROOT)
print("[info] DATA_PATH exists:", DATA_PATH.exists())
print("[info] TEACHER_WEIGHTS exists:", TEACHER_WEIGHTS.exists())

raw = pd.read_csv(DATA_PATH, low_memory=False)
raw[region_col] = raw[region_col].astype(str)

z_input_vars = [c for c in z_input_vars_all if c in raw.columns]
raw[x_input_vars] = raw[x_input_vars].apply(pd.to_numeric, errors="coerce")
raw[z_input_vars] = raw[z_input_vars].apply(pd.to_numeric, errors="coerce")
raw = raw.dropna(subset=[y_var, region_col]).reset_index(drop=True)

date_col = next((c for c in DATE_CANDIDATES if c in raw.columns), None)
if not TEMPORAL_SPLIT or date_col is None:
    raise RuntimeError("This script expects a date column and the same temporal split as vc_student.py.")

df = raw.copy()
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=[date_col, y_var, region_col]).sort_values(date_col).reset_index(drop=True)

cutoff_test = df[date_col].max() - pd.DateOffset(years=TEST_YEARS)
trainval_df = df[df[date_col] <= cutoff_test].copy()
test_df     = df[df[date_col]  > cutoff_test].copy()

n_val = max(1, int(len(trainval_df) * VAL_FRACTION))
val_df   = trainval_df.tail(n_val).copy()
train_df = trainval_df.iloc[:-n_val].copy()

print("[info] Train/Val/Test shapes:", train_df.shape, val_df.shape, test_df.shape)
# --- export raw held-out test data for simple physical-rule baselines ---
simple_rule_test = test_df[[y_var, x_input_vars[0], x_input_vars[1]]].copy()
simple_rule_test.columns = ["y_true", "T_air", "T_wbt"]

if region_col in test_df.columns:
    simple_rule_test["region"] = test_df[region_col].values
if date_col is not None and date_col in test_df.columns:
    simple_rule_test["date"] = test_df[date_col].values

out_simple = OUTDIR / "simple_rule_test_input.csv"
simple_rule_test.to_csv(out_simple, index=False)
print(f"[ok] wrote simple-rule benchmark input to: {out_simple}")
X_meds = train_df[x_input_vars].median(numeric_only=True)
Z_meds = train_df[z_input_vars].median(numeric_only=True)

keep = []
for c in z_input_vars:
    if pd.isna(Z_meds.get(c, np.nan)):
        continue
    if train_df[c].std(skipna=True) == 0:
        continue
    keep.append(c)
z_input_vars = keep
print("[info] Z variables used:", z_input_vars)

for d in (train_df, val_df, test_df):
    d[x_input_vars] = d[x_input_vars].fillna(X_meds)
    d[z_input_vars] = d[z_input_vars].fillna(Z_meds[z_input_vars])

X_scaler = StandardScaler().fit(train_df[x_input_vars].to_numpy())
Z_scaler = StandardScaler().fit(train_df[z_input_vars].to_numpy())
try:
    R_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
except TypeError:
    R_encoder = OneHotEncoder(handle_unknown="ignore", sparse=True)
R_encoder.fit(train_df[[region_col]])

def to_arrays(df_):
    X = X_scaler.transform(df_[x_input_vars].to_numpy()).astype("float32")
    Z = Z_scaler.transform(df_[z_input_vars].to_numpy()).astype("float32")
    R = R_encoder.transform(df_[[region_col]]).toarray().astype("float32")
    y = df_[y_var].to_numpy().astype("float32")
    return X, Z, R, y

def to_arrays_no_y(df_):
    X = X_scaler.transform(df_[x_input_vars].to_numpy()).astype("float32")
    Z = Z_scaler.transform(df_[z_input_vars].to_numpy()).astype("float32")
    R = R_encoder.transform(df_[[region_col]]).toarray().astype("float32")
    return X, Z, R

X_test, Z_test, R_test, y_test = to_arrays(test_df)

teacher = build_teacher(
    x_dim=len(x_input_vars),
    z_dim=len(z_input_vars),
    r_dim=R_test.shape[1],
    gate_type="relu",
    gate_k=5.0,
)
teacher.load_weights(TEACHER_WEIGHTS)
print("[ok] loaded teacher weights")

# TEST predictions
y_prob_test, cuts_test = teacher.predict([X_test, Z_test, R_test], batch_size=4096, verbose=1)
y_prob_test = y_prob_test.ravel()

metrics_out = {
    "n_test": int(len(y_test)),
    "AUC": float(roc_auc_score(y_test, y_prob_test)),
    "ACC_0.5": float(accuracy_score(y_test, (y_prob_test >= 0.5).astype(int))),
    "AP": float(average_precision_score(y_test, y_prob_test)),
    "Brier": float(brier_score_loss(y_test, y_prob_test)),
}
print("[teacher TEST metrics]", metrics_out)

pd.DataFrame({
    "y_true": y_test.astype(int),
    "y_prob": y_prob_test
}).to_csv(OUTDIR / "teacher_predictions_TEST.csv", index=False)

with open(OUTDIR / "teacher_metrics_TEST.json", "w", encoding="utf-8") as f:
    json.dump(metrics_out, f, indent=2)

# ALL thresholds
full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
X_all, Z_all, R_all = to_arrays_no_y(full_df)
_, cuts_all = teacher.predict([X_all, Z_all, R_all], batch_size=4096, verbose=1)

thr_df = pd.DataFrame(cuts_all, columns=["thr_Air_std", "thr_WBT_std", "thr_Dew_std"])

means = X_scaler.mean_
scales = X_scaler.scale_
i_air = x_input_vars.index("Air_Temp")
i_wbt = x_input_vars.index("Wet_Bulb_Temp")
i_dew = x_input_vars.index("Dewpoint")

thr_df["thr_Air_C"] = thr_df["thr_Air_std"] * scales[i_air] + means[i_air]
thr_df["thr_WBT_C"] = thr_df["thr_WBT_std"] * scales[i_wbt] + means[i_wbt]
thr_df["thr_Dew_C"] = thr_df["thr_Dew_std"] * scales[i_dew] + means[i_dew]

thr_df["thr_WBT_C"] = np.maximum(thr_df["thr_WBT_C"], thr_df["thr_Dew_C"])
thr_df["thr_Air_C"] = np.maximum(thr_df["thr_Air_C"], thr_df["thr_WBT_C"])

meta_cols = [c for c in [region_col, "Longitude", "Latitude", "Elevation"] if c in full_df.columns]
z_present = [c for c in Z_KEEP if c in full_df.columns]

teacher_all_df = pd.concat(
    [full_df.reset_index(drop=True)[meta_cols + z_present], thr_df.reset_index(drop=True)],
    axis=1
)
teacher_all_df.to_csv(OUTDIR / "teacher_thresholds_ALL.csv", index=False)

station_teacher = (
    teacher_all_df
    .groupby(["Longitude", "Latitude"], as_index=False)
    .agg({
        "thr_Air_C": "mean",
        "thr_WBT_C": "mean",
        "thr_Dew_C": "mean"
    })
)
station_teacher.to_csv(OUTDIR / "teacher_thresholds_station_mean.csv", index=False)

print("[ok] saved teacher_predictions_TEST.csv")
print("[ok] saved teacher_metrics_TEST.json")
print("[ok] saved teacher_thresholds_ALL.csv")
print("[ok] saved teacher_thresholds_station_mean.csv")
print("Done.")
