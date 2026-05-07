# ========================== VC-Student Stability Analysis ==========================
# Station-blocked subsampling stability analysis for the AOAS VC-Student model.
#
# How to use:
# 1) Put this file in your vc-student project root (same level as data/ and outputs/),
#    or run it from that project root.
# 2) Adjust STABILITY_REPS / STATION_SAMPLE_FRAC below if needed.
# 3) Run: python vc_student_stability.py
#
# Main outputs:
#   outputs/stability/<distill|nodistill>/
#       stability_runs.csv
#       stability_summary.csv
#       full_data_summary.json   (optional, if RUN_FULL_DATA = True)
#       rep_00/, rep_01/, ...    (per-replicate small artifacts)
#
# Notes:
# - This script performs station-blocked subsampling, not formal inferential UQ.
# - It preserves the same chronological split logic as the main training script.
# - It auto-detects a station identifier if possible; otherwise it falls back to a
#   rounded lon/lat/elevation key.

import os
SEED = 2025
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import gc
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow.keras import layers, losses, metrics, models, optimizers, regularizers

# -------------------- Determinism --------------------
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# -------------------- Project paths --------------------
THIS = Path(__file__).resolve()
CANDIDATE_ROOTS = [THIS.parent, THIS.parent.parent, Path.cwd()]
REPO_ROOT = None
for cand in CANDIDATE_ROOTS:
    if (cand / "data" / "point_data_with_atmospheric_predictors.csv").exists():
        REPO_ROOT = cand
        break
if REPO_ROOT is None:
    REPO_ROOT = Path.cwd()

DATA_PATH = REPO_ROOT / "data" / "point_data_with_atmospheric_predictors.csv"
TEACHER_WEIGHTS = REPO_ROOT / "data" / "best.weights.h5"
OUTDIR = REPO_ROOT / "outputs" / "stability"

# -------------------- User config --------------------
y_var = "Rain_Phase"
region_col = "Cls"

# Station detection: if none of these exist, the script falls back to a rounded coordinate key.
STATION_CANDIDATES = [
    "Station_ID", "station_id", "station", "Station", "STATION",
    "station_name", "Site_ID", "site_id", "USAF", "WBAN", "ID", "id"
]

x_input_vars = ["Air_Temp", "Wet_Bulb_Temp", "Dewpoint"]
assert x_input_vars == ["Air_Temp", "Wet_Bulb_Temp", "Dewpoint"], \
    f"X order must be [Air, WBT, Dew]; got {x_input_vars}"

z_input_vars_all = [
    "RH", "gridded_data_pres", "Longitude", "Latitude", "Elevation",
    "lapse_rate", "Z0", "temp_gradient", "ApT"
]

DATE_CANDIDATES = ["Date", "Timestamp", "date", "timestamp", "datetime"]
TEMPORAL_SPLIT = True
TEST_YEARS = 2
VAL_FRACTION = 0.20

# Student training
EPOCHS = 40
PATIENCE = 8
BATCH_SIZE = 256
LR = 1e-3
GATE_TYPE = "softplus"     # {"relu", "softplus", "sigmoid"}
GATE_K = 5.0
FORCE_BETA_NONNEG = True
USE_INTERNAL_INDEX = True
ALPHA_L1 = 1e-4
BETA_L2 = 1e-5

# Regularizers
USE_REGION_REG = True
REGION_LAMBDA = 2e-2
USE_SPATIAL_REG = False
SPATIAL_K = 5
SPATIAL_LAMBDA = 5e-4

# Teacher distillation
USE_DISTILLATION = True
TEACHER_GATE_TYPE = "relu"
if USE_DISTILLATION and not TEACHER_WEIGHTS.exists():
    print("[warn] USE_DISTILLATION=True, but teacher weights not found → switching to no-distill.")
    USE_DISTILLATION = False
RUN_TAG = "distill" if USE_DISTILLATION else "nodistill"

# Stability analysis settings
STABILITY_REPS = 10
STATION_SAMPLE_FRAC = 0.70
RUN_FULL_DATA = False        # set True to refit the full-data model; False reuses full_data_summary.json if present
MIN_STATIONS_PER_SAMPLE = 50

# Optional quick mode for debugging
QUICK_MODE = False
if QUICK_MODE:
    EPOCHS = 8
    PATIENCE = 3
    STABILITY_REPS = 3
    RUN_FULL_DATA = False

RUN_DIR = OUTDIR / RUN_TAG
RUN_DIR.mkdir(parents=True, exist_ok=True)
print("[paths] REPO_ROOT →", REPO_ROOT)
print("[paths] DATA_PATH  →", DATA_PATH)
print("[paths] RUN_DIR    →", RUN_DIR)


# ========================== Helper functions ==========================
def set_all_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def detect_station_col(df: pd.DataFrame) -> str:
    for c in STATION_CANDIDATES:
        if c in df.columns:
            print(f"[station] Using station column: {c}")
            return c
    if all(c in df.columns for c in ["Longitude", "Latitude", "Elevation"]):
        df["_station_key"] = (
            df["Longitude"].round(4).astype(str) + "_" +
            df["Latitude"].round(4).astype(str) + "_" +
            df["Elevation"].round(1).astype(str)
        )
        print("[station] No explicit station ID found. Using rounded Longitude/Latitude/Elevation key.")
        return "_station_key"
    raise KeyError(
        "Could not detect a station identifier and no Longitude/Latitude/Elevation fallback is available."
    )


def load_raw_data() -> tuple[pd.DataFrame, str]:
    raw = pd.read_csv(DATA_PATH, low_memory=False)
    raw[region_col] = raw[region_col].astype(str)

    missing = [c for c in [y_var, region_col] + x_input_vars if c not in raw.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    z_input_vars = [c for c in z_input_vars_all if c in raw.columns]
    raw[x_input_vars] = raw[x_input_vars].apply(pd.to_numeric, errors="coerce")
    raw[z_input_vars] = raw[z_input_vars].apply(pd.to_numeric, errors="coerce")
    raw = raw.dropna(subset=[y_var, region_col]).reset_index(drop=True)

    station_col = detect_station_col(raw)
    return raw, station_col


def temporal_split(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str | None]:
    date_col = next((c for c in DATE_CANDIDATES if c in raw.columns), None)
    if TEMPORAL_SPLIT and date_col is not None:
        df = raw.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col, y_var, region_col]).sort_values(date_col).reset_index(drop=True)

        cutoff_test = df[date_col].max() - pd.DateOffset(years=TEST_YEARS)
        trainval_df = df[df[date_col] <= cutoff_test].copy()
        test_df = df[df[date_col] > cutoff_test].copy()

        n_val = max(1, int(len(trainval_df) * VAL_FRACTION))
        val_df = trainval_df.tail(n_val).copy()
        train_df = trainval_df.iloc[:-n_val].copy()

        return train_df, val_df, test_df, date_col

    from sklearn.model_selection import train_test_split
    train_df, tmp_df = train_test_split(raw, test_size=0.30, random_state=SEED, stratify=raw[y_var])
    val_df, test_df = train_test_split(tmp_df, test_size=0.50, random_state=SEED, stratify=tmp_df[y_var])
    return train_df, val_df, test_df, None


def fit_preprocessors(train_df, val_df, test_df, z_input_vars, all_region_levels):
    X_meds = train_df[x_input_vars].median(numeric_only=True)
    Z_meds = train_df[z_input_vars].median(numeric_only=True)

    keep = []
    for c in z_input_vars:
        if pd.isna(Z_meds.get(c, np.nan)):
            print(f"[warn] drop Z '{c}' (all-NaN on train)")
            continue
        if train_df[c].std(skipna=True) == 0:
            print(f"[warn] drop Z '{c}' (constant on train)")
            continue
        keep.append(c)
    z_input_vars = keep

    def _impute(df):
        df[x_input_vars] = df[x_input_vars].fillna(X_meds)
        df[z_input_vars] = df[z_input_vars].fillna(Z_meds[z_input_vars])

    for d in (train_df, val_df, test_df):
        _impute(d)

    X_scaler = StandardScaler().fit(train_df[x_input_vars].to_numpy())
    Z_scaler = StandardScaler().fit(train_df[z_input_vars].to_numpy())

    cats = [np.array(all_region_levels, dtype=object)]
    try:
        R_encoder = OneHotEncoder(
            categories=cats,
            handle_unknown="ignore",
            sparse_output=True,
        )
    except TypeError:
        R_encoder = OneHotEncoder(
            categories=cats,
            handle_unknown="ignore",
            sparse=True,
        )

    # IMPORTANT:
    # Fit the encoder on the full, fixed list of region classes from the complete dataset,
    # not on the current replicate's training subset. This keeps r_dim constant across
    # replicates so teacher/student architectures always match the saved teacher weights.
    R_encoder.fit(pd.DataFrame({region_col: all_region_levels}))
    return train_df, val_df, test_df, z_input_vars, X_scaler, Z_scaler, R_encoder


def to_arrays(df, X_scaler, Z_scaler, R_encoder, z_input_vars):
    X = X_scaler.transform(df[x_input_vars].to_numpy()).astype("float32")
    Z = Z_scaler.transform(df[z_input_vars].to_numpy()).astype("float32")
    R = R_encoder.transform(df[[region_col]]).toarray().astype("float32")
    y = df[y_var].to_numpy().astype("float32")
    return X, Z, R, y


# ========================== Physics/VC layers ==========================
class Shape1D(tf.keras.layers.Layer):
    def __init__(self, n_knots=16, direction="free", l1=1e-4, name=None):
        super().__init__(name=name)
        assert direction in ("free", "increasing", "decreasing")
        self.n_knots, self.direction, self.l1 = int(n_knots), direction, float(l1)

    def build(self, input_shape):
        k = np.linspace(0.0, 1.0, self.n_knots).astype("float32")
        self.knots = self.add_weight(
            shape=(self.n_knots,),
            initializer=tf.keras.initializers.Constant(k),
            trainable=False,
            name="knots",
            dtype=tf.float32,
        )
        self.w = self.add_weight(
            shape=(self.n_knots,),
            initializer="zeros",
            regularizer=regularizers.L1(self.l1) if self.l1 > 0 else None,
            name="w",
            dtype=tf.float32,
        )
        self.b = self.add_weight(shape=(), initializer="zeros", name="b", dtype=tf.float32)

    def call(self, z01):
        z = tf.clip_by_value(z01, 0.0, 1.0)
        basis = tf.nn.relu(z - tf.reshape(self.knots, (1, self.n_knots)))
        if self.direction == "free":
            slopes = self.w
        else:
            base = tf.nn.softplus(self.w)
            slopes = base if self.direction == "increasing" else -base
        return self.b + tf.reduce_sum(basis * slopes, axis=-1, keepdims=True)


class LinearIndex(tf.keras.layers.Layer):
    def __init__(self, l1=0.0, name=None):
        super().__init__(name=name)
        self.l1 = float(l1)

    def build(self, input_shape):
        d = int(input_shape[-1])
        self.alpha = self.add_weight(
            shape=(d, 1),
            initializer="zeros",
            regularizer=regularizers.L1(self.l1) if self.l1 > 0 else None,
            name="alpha",
        )
        self.bias = self.add_weight(shape=(1,), initializer="zeros", name="bias")

    def call(self, z_std):
        return tf.squeeze(z_std @ self.alpha + self.bias, axis=-1)


# ========================== Teacher & student builders ==========================
def build_teacher(x_dim, z_dim, r_dim, gate_type="relu", gate_k=5.0, dropout_rate=0.30):
    x_input = layers.Input(shape=(x_dim,), name="x_input")
    z_input = layers.Input(shape=(z_dim,), name="z_input")
    r_input = layers.Input(shape=(r_dim,), name="region_input")

    r_embed = layers.Dense(64, activation="relu", name="r_embed_1")(r_input)
    r_embed = layers.BatchNormalization(name="r_embed_bn")(r_embed)
    r_embed = layers.Dense(32, activation="relu", name="r_embed_2")(r_embed)

    zr = layers.Concatenate(name="zr_concat")([z_input, r_embed])
    h = layers.Dense(128, activation="relu", name="cut_h1")(zr)
    h = layers.BatchNormalization(name="cut_bn1")(h)
    h = layers.Dropout(dropout_rate, seed=SEED, name="cut_do1")(h)
    h = layers.Dense(64, activation="relu", name="cut_h2")(h)
    h = layers.BatchNormalization(name="cut_bn2")(h)
    h = layers.Dropout(dropout_rate, seed=SEED, name="cut_do2")(h)

    dew_base = layers.Dense(1, activation="linear", name="dew_base")(h)
    wb_off = layers.Dense(1, activation="linear", name="wb_off")(h)
    air_off = layers.Dense(1, activation="linear", name="air_off")(h)

    wb_thr = layers.Lambda(lambda t: t[0] + tf.nn.softplus(t[1]), name="wb_thr")([dew_base, wb_off])
    air_thr = layers.Lambda(lambda t: t[0] + tf.nn.softplus(t[1]), name="air_thr")([wb_thr, air_off])
    cutting_points = layers.Concatenate(name="cutting_points")([air_thr, wb_thr, dew_base])

    def gate_pair(x, c):
        if gate_type == "relu":
            return tf.nn.relu(x - c)
        if gate_type == "softplus":
            return tf.math.softplus(gate_k * (x - c)) / gate_k
        return tf.sigmoid(gate_k * (x - c)) * (x - c)

    thresholded_x = layers.Lambda(lambda t: gate_pair(t[0], t[1]), name="thresholded_x")([x_input, cutting_points])

    all_feat = layers.Concatenate(name="clf_concat")([thresholded_x, z_input, r_embed])
    h = layers.Dense(128, activation="relu", name="clf_h1")(all_feat)
    h = layers.BatchNormalization(name="clf_bn1")(h)
    h = layers.Dropout(dropout_rate, seed=SEED, name="clf_do1")(h)
    h = layers.Dense(64, activation="relu", name="clf_h2")(h)
    h = layers.BatchNormalization(name="clf_bn2")(h)
    h = layers.Dropout(dropout_rate, seed=SEED, name="clf_do2")(h)
    h = layers.Dense(32, activation="relu", name="clf_h3")(h)
    y_out = layers.Dense(1, activation="sigmoid", name="output")(h)

    return models.Model([x_input, z_input, r_input], [y_out, cutting_points], name="DeepCutRegionEmbed")


def build_vc_student_v2(
    x_dim, z_dim, r_dim, z_names,
    gate_type="relu", gate_k=5.0,
    force_beta_nonneg=True, use_internal_index=True,
    alpha_l1=1e-4, beta_l2=1e-5,
):
    x_in = layers.Input(shape=(x_dim,), name="x_input")
    z_in = layers.Input(shape=(z_dim,), name="z_input")
    r_in = layers.Input(shape=(r_dim,), name="region_input")

    r = layers.Dense(64, activation="relu", name="r_embed_1")(r_in)
    r = layers.BatchNormalization(name="r_embed_bn")(r)
    r = layers.Dense(32, activation="relu", name="r_embed_2")(r)

    I_exp = None
    if use_internal_index:
        I = LinearIndex(l1=alpha_l1, name="Index_I_raw")(z_in)
        I_exp = layers.Lambda(lambda t: tf.expand_dims(t, -1), name="Index_I")(I)

    parts = []
    for j, nm in enumerate(z_names):
        z_col = layers.Lambda(lambda t, j=j: t[:, j:j+1], name=f"pick_{nm}")(z_in)
        z_std = layers.Lambda(lambda t: tf.clip_by_value(t, -3.0, 3.0), name=f"clip_{nm}")(z_col)
        z01 = layers.Lambda(lambda t: (t + 3.0) / 6.0, name=f"to01_{nm}")(z_std)
        direction = "decreasing" if nm == "Elevation" else ("increasing" if nm == "Z0" else "free")
        parts.append(Shape1D(16, direction=direction, l1=1e-4, name=f"shape_{nm}")(z01))

    sumZ = layers.Add(name="sumZ")(parts) if len(parts) > 1 else parts[0]
    if use_internal_index:
        sumZ = layers.Add(name="sumZ_plus_I")([sumZ, layers.Dense(1, use_bias=False, name="I_to_thr")(I_exp)])

    rb = layers.Dense(1, activation="linear", name="region_bias")(r)
    dew_b = layers.Add(name="dew_base")([sumZ, rb])
    wb_off = layers.Dense(1, activation="linear", name="wb_off")(r)
    air_off = layers.Dense(1, activation="linear", name="air_off")(r)
    wbt_thr = layers.Lambda(lambda t: t[0] + tf.nn.softplus(t[1]), name="wbt_thr")([dew_b, wb_off])
    air_thr = layers.Lambda(lambda t: t[0] + tf.nn.softplus(t[1]), name="air_thr")([wbt_thr, air_off])
    cuts = layers.Concatenate(name="cutting_points")([air_thr, wbt_thr, dew_b])

    if gate_type == "relu":
        gx = layers.Lambda(lambda t: tf.nn.relu(t[0] - t[1]), name="thresholded_x")([x_in, cuts])
    elif gate_type == "softplus":
        gx = layers.Lambda(
            lambda t, k=gate_k: tf.math.softplus(k * (t[0] - t[1])) / k,
            arguments={"k": gate_k},
            name="thresholded_x",
        )([x_in, cuts])
    else:
        gx = layers.Lambda(
            lambda t, k=gate_k: tf.sigmoid(k * (t[0] - t[1])) * (t[0] - t[1]),
            arguments={"k": gate_k},
            name="thresholded_x",
        )([x_in, cuts])

    if not use_internal_index:
        raise ValueError("Single-index VC requires USE_INTERNAL_INDEX=True")
    zr = I_exp
    beta0 = layers.Dense(1, activation="linear", name="beta0", kernel_regularizer=regularizers.l2(beta_l2))(zr)
    beta = layers.Dense(
        x_dim,
        activation=("softplus" if force_beta_nonneg else "linear"),
        name="beta_vec",
        kernel_regularizer=regularizers.l2(beta_l2),
    )(zr)

    dot = layers.Lambda(lambda t: tf.reduce_sum(t[0] * t[1], axis=-1, keepdims=True), name="dot_beta_gx")([beta, gx])
    logit = layers.Add(name="logit")([beta0, dot])
    y_out = layers.Activation("sigmoid", name="y_out")(logit)

    outs = [y_out, cuts]
    if use_internal_index:
        outs.append(I_exp)
    outs.extend([beta0, beta])
    return models.Model([x_in, z_in, r_in], outs, name="VCStudent_v2")


@tf.function
def region_pairwise_regularization(c_points, region_matrix, lambda_reg=0.02):
    dtype = c_points.dtype
    reg_loss = tf.zeros((), dtype=dtype)
    region_ids = tf.argmax(region_matrix, axis=1)
    unique_regions = tf.unique(region_ids)[0]

    for r in unique_regions:
        mask = tf.equal(region_ids, r)
        c_r = tf.boolean_mask(c_points, mask)
        nr = tf.shape(c_r)[0]
        xdim = tf.cast(tf.shape(c_r)[1], dtype)

        def _region_loss():
            c0 = c_r - tf.reduce_mean(c_r, axis=0, keepdims=True)
            sum_sq = tf.reduce_sum(tf.reduce_sum(tf.square(c0), axis=1))
            n_pairs = tf.cast(nr * (nr - 1) // 2, dtype)
            denom = tf.maximum(n_pairs, tf.constant(1.0, dtype=dtype)) * xdim
            return (tf.constant(2.0, dtype=dtype) * sum_sq) / denom

        reg_loss += tf.cond(nr > 1, _region_loss, lambda: tf.zeros((), dtype=dtype))

    return tf.cast(lambda_reg, dtype) * reg_loss


@tf.function
def spatial_smoothness_regularization(c_points, coords_std, k=5, lambda_space=5e-4):
    dtype = c_points.dtype
    B = tf.shape(coords_std)[0]
    xdim = tf.cast(tf.shape(c_points)[1], dtype)

    c2 = tf.reduce_sum(tf.square(coords_std), axis=1, keepdims=True)
    d2 = c2 - tf.cast(2.0, dtype) * tf.matmul(coords_std, coords_std, transpose_b=True) + tf.transpose(c2)
    big = tf.reduce_max(d2) + tf.constant(1.0, dtype=dtype)
    d2 = tf.linalg.set_diag(d2, tf.fill([B], big))

    _, idxs = tf.nn.top_k(-d2, k=k)
    c_nb = tf.gather(c_points, idxs)
    c_i = tf.expand_dims(c_points, axis=1)
    diffs = c_i - c_nb
    sq = tf.reduce_sum(tf.square(diffs), axis=2)
    loss = tf.reduce_mean(tf.reduce_mean(sq, axis=1))

    return tf.cast(lambda_space, dtype) * (loss / tf.maximum(xdim, tf.constant(1.0, dtype=dtype)))


class VCStudentModel(tf.keras.Model):
    def __init__(self, core, use_region=True, region_lambda=0.02,
                 use_spatial=True, spatial_lambda=5e-4, k_neighbors=5, coord_idxs=None):
        super().__init__(name="VCStudentModel")
        self.core = core
        self.use_region = use_region
        self.region_lambda = float(region_lambda)
        self.use_spatial = use_spatial
        self.spatial_lambda = float(spatial_lambda)
        self.k_neighbors = int(k_neighbors)
        self.coord_idxs = coord_idxs if coord_idxs is not None else []

    def call(self, inputs, training=False):
        return self.core(inputs, training=training)

    def train_step(self, data):
        (inputs, y_true) = data
        Xb, Zb, Rb = inputs
        with tf.GradientTape() as tape:
            outs = self.core([Xb, Zb, Rb], training=True)
            y_pred = {"y_out": outs[0], "cutting_points": outs[1], "beta0": outs[-2], "beta_vec": outs[-1]}
            if isinstance(y_true, dict) and "Index_I" in y_true:
                y_pred["Index_I"] = outs[2]
            loss = self.compute_loss(x=inputs, y=y_true, y_pred=y_pred, sample_weight=None)
            c_points = outs[1]
            if self.use_region:
                loss += region_pairwise_regularization(c_points, Rb, lambda_reg=self.region_lambda)
            if self.use_spatial and (len(self.coord_idxs) == 3):
                coords_std = tf.gather(Zb, self.coord_idxs, axis=1)
                loss += spatial_smoothness_regularization(
                    c_points, coords_std, k=self.k_neighbors, lambda_space=self.spatial_lambda
                )
        grads = tape.gradient(loss, self.core.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.core.trainable_variables))
        self.compute_metrics(x=inputs, y=y_true, y_pred=y_pred)
        return {"loss": loss, **{m.name: m.result() for m in self.metrics}}

    def test_step(self, data):
        (inputs, y_true) = data
        outs = self.core(inputs, training=False)
        y_pred = {"y_out": outs[0], "cutting_points": outs[1], "beta0": outs[-2], "beta_vec": outs[-1]}
        if isinstance(y_true, dict) and "Index_I" in y_true:
            y_pred["Index_I"] = outs[2]
        self.compute_loss(x=inputs, y=y_true, y_pred=y_pred, sample_weight=None)
        self.compute_metrics(x=inputs, y=y_true, y_pred=y_pred)
        return {m.name: m.result() for m in self.metrics}


# ========================== Per-run training function ==========================
def run_single_fit(
    raw_input: pd.DataFrame,
    station_col: str,
    run_dir: Path,
    seed: int,
    label: str,
    all_region_levels,
) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    set_all_seeds(seed)
    tf.keras.backend.clear_session()

    z_input_vars = [c for c in z_input_vars_all if c in raw_input.columns]
    train_df, val_df, test_df, date_col = temporal_split(raw_input)
    print(f"[{label}] split sizes → train={train_df.shape}, val={val_df.shape}, test={test_df.shape}, date_col={date_col}")

    train_df, val_df, test_df, z_input_vars, X_scaler, Z_scaler, R_encoder = fit_preprocessors(
        train_df.copy(), val_df.copy(), test_df.copy(), z_input_vars, all_region_levels
    )

    X_train, Z_train, R_train, y_train = to_arrays(train_df, X_scaler, Z_scaler, R_encoder, z_input_vars)
    X_val, Z_val, R_val, y_val = to_arrays(val_df, X_scaler, Z_scaler, R_encoder, z_input_vars)
    X_test, Z_test, R_test, y_test = to_arrays(test_df, X_scaler, Z_scaler, R_encoder, z_input_vars)

    coord_names = ["Longitude", "Latitude", "Elevation"]
    coord_idxs = [z_input_vars.index(c) for c in coord_names if c in z_input_vars]
    has_coords = len(coord_idxs) == 3
    use_spatial = USE_SPATIAL_REG and has_coords

    x_dim = len(x_input_vars)
    z_dim = len(z_input_vars)
    r_dim = R_train.shape[1]
    print(
        f"[{label}] region dims → global={len(all_region_levels)}, "
        f"train_unique={train_df[region_col].nunique()}, encoded_r_dim={r_dim}"
    )

    teacher = None
    if USE_DISTILLATION:
        teacher = build_teacher(x_dim, z_dim, r_dim, gate_type=TEACHER_GATE_TYPE, gate_k=GATE_K)
        teacher.load_weights(TEACHER_WEIGHTS)

    base_student = build_vc_student_v2(
        x_dim, z_dim, r_dim, z_names=z_input_vars,
        gate_type=GATE_TYPE, gate_k=GATE_K,
        force_beta_nonneg=FORCE_BETA_NONNEG,
        use_internal_index=USE_INTERNAL_INDEX,
        alpha_l1=ALPHA_L1, beta_l2=BETA_L2,
    )

    vc_student = VCStudentModel(
        base_student,
        use_region=USE_REGION_REG,
        region_lambda=REGION_LAMBDA,
        use_spatial=use_spatial,
        spatial_lambda=SPATIAL_LAMBDA,
        k_neighbors=SPATIAL_K,
        coord_idxs=coord_idxs,
    )

    if teacher is not None:
        _, tr_cuts = teacher.predict([X_train, Z_train, R_train], batch_size=4096, verbose=0)
        _, va_cuts = teacher.predict([X_val, Z_val, R_val], batch_size=4096, verbose=0)
        train_targets = {"y_out": y_train, "cutting_points": tr_cuts}
        val_targets = {"y_out": y_val, "cutting_points": va_cuts}
        lw_cuts = 1.0
    else:
        train_targets = {"y_out": y_train, "cutting_points": np.zeros((len(y_train), 3), np.float32)}
        val_targets = {"y_out": y_val, "cutting_points": np.zeros((len(y_val), 3), np.float32)}
        lw_cuts = 0.0

    if USE_INTERNAL_INDEX:
        train_targets["Index_I"] = np.zeros((len(y_train), 1), np.float32)
        val_targets["Index_I"] = np.zeros((len(y_val), 1), np.float32)

    train_targets["beta0"] = np.zeros((len(y_train), 1), np.float32)
    val_targets["beta0"] = np.zeros((len(y_val), 1), np.float32)
    train_targets["beta_vec"] = np.zeros((len(y_train), x_dim), np.float32)
    val_targets["beta_vec"] = np.zeros((len(y_val), x_dim), np.float32)

    losses_dict = {
        "y_out": losses.BinaryCrossentropy(),
        "cutting_points": losses.MeanSquaredError(),
        "beta0": losses.MeanSquaredError(),
        "beta_vec": losses.MeanSquaredError(),
    }
    loss_wts = {"y_out": 1.0, "cutting_points": lw_cuts, "beta0": 0.0, "beta_vec": 0.0}
    metrics_dict = {"y_out": [metrics.BinaryAccuracy(name="acc"), metrics.AUC(name="auc")]}
    if USE_INTERNAL_INDEX:
        losses_dict["Index_I"] = losses.MeanSquaredError()
        loss_wts["Index_I"] = 0.0

    vc_student.compile(
        optimizer=optimizers.Adam(learning_rate=LR),
        loss=losses_dict,
        loss_weights=loss_wts,
        metrics=metrics_dict,
    )

    _ = vc_student([X_train[:4], Z_train[:4], R_train[:4]], training=False)

    weights_path = run_dir / f"vc_student__{label}.weights.h5"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_y_out_loss", mode="min", patience=PATIENCE, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            weights_path.as_posix(), monitor="val_y_out_loss", mode="min",
            save_best_only=True, save_weights_only=True
        ),
    ]

    hist = vc_student.fit(
        [X_train, Z_train, R_train], train_targets,
        validation_data=([X_val, Z_val, R_val], val_targets),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        callbacks=callbacks,
    )

    history_path = run_dir / f"history__{label}.csv"
    hist_df = pd.DataFrame(hist.history)
    hist_df["epoch"] = np.arange(len(hist_df))
    hist_df.to_csv(history_path, index=False)

    outs_test = vc_student.core.predict([X_test, Z_test, R_test], batch_size=4096, verbose=0)
    yprob_te = outs_test[0].ravel()
    auc_test = roc_auc_score(y_test, yprob_te)
    acc_test = accuracy_score(y_test, (yprob_te > 0.5).astype(int))

    pred_test_path = run_dir / f"predictions_test__{label}.csv"
    pd.DataFrame({"y_true": y_test.astype(int), "y_prob": yprob_te}).to_csv(pred_test_path, index=False)

    # Full subset outputs for stability summaries
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    X_all, Z_all, R_all, _ = to_arrays(full_df, X_scaler, Z_scaler, R_encoder, z_input_vars)
    outs_all = vc_student.core.predict([X_all, Z_all, R_all], batch_size=4096, verbose=0)
    cuts_all = outs_all[1]
    Index_I_all = outs_all[2].reshape(-1) if USE_INTERNAL_INDEX else np.full(len(full_df), np.nan)

    # Alpha weights
    idx_layer = vc_student.core.get_layer("Index_I_raw")
    alpha_std = idx_layer.get_weights()[0].reshape(-1)
    bias_std = float(idx_layer.get_weights()[1].reshape(()))
    alpha_map = dict(zip(z_input_vars, alpha_std))

    # Convert thresholds to physical units; model output order is [Air, WBT, Dew]
    means, scales = X_scaler.mean_, X_scaler.scale_
    i_air = x_input_vars.index("Air_Temp")
    i_wbt = x_input_vars.index("Wet_Bulb_Temp")
    i_dew = x_input_vars.index("Dewpoint")

    thr_air_c = cuts_all[:, 0] * scales[i_air] + means[i_air]
    thr_wbt_c = cuts_all[:, 1] * scales[i_wbt] + means[i_wbt]
    thr_dew_c = cuts_all[:, 2] * scales[i_dew] + means[i_dew]

    station_summary = pd.DataFrame({
        station_col: full_df[station_col].values,
        "Index_I": Index_I_all,
        "thr_Air_C": thr_air_c,
        "thr_WBT_C": thr_wbt_c,
        "thr_Dew_C": thr_dew_c,
    })
    if "Longitude" in full_df.columns:
        station_summary["Longitude"] = full_df["Longitude"].values
    if "Latitude" in full_df.columns:
        station_summary["Latitude"] = full_df["Latitude"].values
    if "Elevation" in full_df.columns:
        station_summary["Elevation"] = full_df["Elevation"].values

    station_mean = (
        station_summary
        .groupby(station_col, as_index=False)
        .agg(
            Index_I_mean=("Index_I", "mean"),
            thr_Air_C_mean=("thr_Air_C", "mean"),
            thr_WBT_C_mean=("thr_WBT_C", "mean"),
            thr_Dew_C_mean=("thr_Dew_C", "mean"),
        )
    )
    corr_index_air = station_mean["Index_I_mean"].corr(station_mean["thr_Air_C_mean"])

    station_mean_path = run_dir / f"station_means__{label}.csv"
    station_mean.to_csv(station_mean_path, index=False)

    corr_student_teacher = np.nan
    mae_student_teacher = np.nan
    if teacher is not None:
        _, teacher_all_cuts = teacher.predict([X_all, Z_all, R_all], batch_size=4096, verbose=0)
        teacher_air_c = teacher_all_cuts[:, 0] * scales[i_air] + means[i_air]
        teacher_station = pd.DataFrame({
            station_col: full_df[station_col].values,
            "teacher_air_c": teacher_air_c,
        })
        teacher_station_mean = teacher_station.groupby(station_col, as_index=False)["teacher_air_c"].mean()
        st_comp = station_mean[[station_col, "thr_Air_C_mean"]].merge(teacher_station_mean, on=station_col, how="inner")
        if len(st_comp) > 1:
            corr_student_teacher = st_comp["thr_Air_C_mean"].corr(st_comp["teacher_air_c"])
            mae_student_teacher = np.mean(np.abs(st_comp["thr_Air_C_mean"] - st_comp["teacher_air_c"]))
        st_comp.to_csv(run_dir / f"student_teacher_station_compare__{label}.csv", index=False)

    # Save alpha weights for this run
    alpha_df = pd.DataFrame({"variable": z_input_vars, "alpha_stdZ": alpha_std})
    alpha_df.to_csv(run_dir / f"index_weights__{label}.csv", index=False)

    result = {
        "label": label,
        "seed": seed,
        "n_rows_total": int(len(full_df)),
        "n_rows_train": int(len(train_df)),
        "n_rows_val": int(len(val_df)),
        "n_rows_test": int(len(test_df)),
        "n_stations": int(full_df[station_col].nunique()),
        "auc_test": float(auc_test),
        "acc_test": float(acc_test),
        "corr_index_air": float(corr_index_air) if pd.notnull(corr_index_air) else np.nan,
        "corr_student_teacher": float(corr_student_teacher) if pd.notnull(corr_student_teacher) else np.nan,
        "mae_student_teacher": float(mae_student_teacher) if pd.notnull(mae_student_teacher) else np.nan,
        "alpha_bias_stdZ": float(bias_std),
        "alpha_RH": float(alpha_map.get("RH", np.nan)),
        "alpha_gridded_data_pres": float(alpha_map.get("gridded_data_pres", np.nan)),
        "alpha_Z0": float(alpha_map.get("Z0", np.nan)),
        "alpha_lapse_rate": float(alpha_map.get("lapse_rate", np.nan)),
        "alpha_temp_gradient": float(alpha_map.get("temp_gradient", np.nan)),
        "alpha_ApT": float(alpha_map.get("ApT", np.nan)),
        "alpha_Elevation": float(alpha_map.get("Elevation", np.nan)),
        "alpha_Longitude": float(alpha_map.get("Longitude", np.nan)),
        "alpha_Latitude": float(alpha_map.get("Latitude", np.nan)),
    }

    print(f"[{label}] TEST AUC={auc_test:.4f} | Acc={acc_test:.4f} | Corr(Index,AirThr)={corr_index_air:.4f}")
    if teacher is not None and pd.notnull(corr_student_teacher):
        print(f"[{label}] Student-Teacher Corr={corr_student_teacher:.4f} | MAE={mae_student_teacher:.4f}")

    # Aggressive cleanup between runs
    del vc_student, base_student
    if teacher is not None:
        del teacher
    tf.keras.backend.clear_session()
    gc.collect()

    return result


def summarize_runs(df: pd.DataFrame, full_row: dict | None = None) -> pd.DataFrame:
    exclude = {"label", "seed"}
    summary_rows = []
    for col in df.columns:
        if col in exclude:
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
                row["full_data_estimate"] = np.nan
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)


# ========================== Main driver ==========================
def main():
    raw, station_col = load_raw_data()

    all_region_levels = sorted(raw[region_col].dropna().astype(str).unique())
    print(f"[main] total region classes = {len(all_region_levels)}")

    all_stations = np.array(sorted(raw[station_col].dropna().astype(str).unique()))
    print(f"[main] unique stations = {len(all_stations):,}")
    if len(all_stations) < MIN_STATIONS_PER_SAMPLE:
        raise ValueError(
            f"Too few stations detected ({len(all_stations)}). Please check station_col detection."
        )

    full_row = None
    full_summary_path = RUN_DIR / "full_data_summary.json"
    if RUN_FULL_DATA:
        print("\n==================== FULL-DATA FIT ====================")
        full_dir = RUN_DIR / "full_data"
        full_row = run_single_fit(
            raw.copy(),
            station_col=station_col,
            run_dir=full_dir,
            seed=SEED,
            label="full_data",
            all_region_levels=all_region_levels,
        )
        full_summary_path.write_text(json.dumps(full_row, indent=2))
    elif full_summary_path.exists():
        full_row = json.loads(full_summary_path.read_text())
        print(f"[main] Reusing existing full-data summary → {full_summary_path}")

    rng = np.random.default_rng(SEED)
    rep_rows = []

    for b in range(STABILITY_REPS):
        print(f"\n==================== REPLICATE {b+1}/{STABILITY_REPS} ====================")
        n_take = max(MIN_STATIONS_PER_SAMPLE, int(STATION_SAMPLE_FRAC * len(all_stations)))
        chosen = rng.choice(all_stations, size=n_take, replace=False)
        raw_b = raw[raw[station_col].astype(str).isin(chosen)].copy()
        rep_dir = RUN_DIR / f"rep_{b:02d}"
        rep_seed = SEED + 100 + b
        out = run_single_fit(
            raw_b,
            station_col=station_col,
            run_dir=rep_dir,
            seed=rep_seed,
            label=f"rep_{b:02d}",
            all_region_levels=all_region_levels,
        )
        out["replicate"] = b
        out["sample_frac_stations"] = STATION_SAMPLE_FRAC
        rep_rows.append(out)

    runs_df = pd.DataFrame(rep_rows)
    runs_path = RUN_DIR / "stability_runs.csv"
    runs_df.to_csv(runs_path, index=False)
    print(f"[ok] Saved replicate-level results → {runs_path}")

    summary_df = summarize_runs(runs_df, full_row=full_row)
    summary_path = RUN_DIR / "stability_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"[ok] Saved stability summary → {summary_path}")

    print("\nTop rows of stability summary:")
    with pd.option_context("display.max_rows", 20, "display.max_columns", None):
        print(summary_df.head(20))


if __name__ == "__main__":
    main()
