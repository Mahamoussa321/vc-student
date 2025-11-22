# ========================== VC-Student  ==========================
# Reproducibility must come BEFORE importing TensorFlow
import os
SEED = 2025  
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  
import json
import gc, time, random
import math
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score, accuracy_score
from tensorflow.keras import layers, models, regularizers, optimizers, losses, metrics

# Seed all RNGs
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)
# (Optional) single-threaded BLAS for stricter determinism
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# 1---------------------------------- User config ----------------------------------
# Robust data path (project default, override via env var RAIN_SNOW_DATA)
REPO_ROOT   = Path(__file__).resolve().parents[1]
DATA_PATH  = REPO_ROOT / "data" / "point_data_with_atmospheric_predictors.csv"
y_var            = "Rain_Phase"                      
region_col       = "Cls"


x_input_vars = ["Air_Temp","Wet_Bulb_Temp","Dewpoint"]
assert x_input_vars == ["Air_Temp","Wet_Bulb_Temp","Dewpoint"], \
    f"X order must be [Air, WBT, Dew]; got {x_input_vars}"


z_input_vars_all = ["RH","gridded_data_pres","Longitude","Latitude","Elevation","lapse_rate","Z0","temp_gradient","ApT"]
Z_KEEP = ["RH","gridded_data_pres","Z0","lapse_rate","temp_gradient","ApT"]

# ---------- Temporal split settings ----------
TEMPORAL_SPLIT   = True
TEST_YEARS       = 2
VAL_FRACTION     = 0.20
DATE_CANDIDATES  = ["Date", "Timestamp", "date", "timestamp", "datetime"]

# Student training
EPOCHS     = 40
PATIENCE   = 8   
BATCH_SIZE = 256
LR         = 1e-3
GATE_TYPE  = "softplus"      # {"relu","softplus","sigmoid"}
GATE_K     = 5.0
FORCE_BETA_NONNEG = True
USE_INTERNAL_INDEX  = True
ALPHA_L1            = 1e-4
BETA_L2             = 1e-5

# Regularizers
USE_REGION_REG      = True
REGION_LAMBDA       = 2e-2
USE_SPATIAL_REG     = False
SPATIAL_K           = 5
SPATIAL_LAMBDA      = 5e-4

# Teacher distillation (optional)
TEACHER_WEIGHTS = REPO_ROOT / "data" / "best.weights.h5"
# --- Distillation toggle + preflight (no model building here) ---
USE_DISTILLATION = True  # flip to False for ablation
RUN_TAG = "distill" if USE_DISTILLATION else "nodistill"

if USE_DISTILLATION and not Path(TEACHER_WEIGHTS).exists():
    print("[warn] USE_DISTILLATION=True, but teacher weights not found → switching to no-distill.")
    USE_DISTILLATION = False
    RUN_TAG = "nodistill"

TEACHER_GATE_TYPE   = "relu"



# Outputs
OUTDIR = (REPO_ROOT / "outputs").resolve()
RUN_DIR = (OUTDIR / "ablation" / RUN_TAG)
RUN_DIR.mkdir(parents=True, exist_ok=True)
print("[paths] OUTDIR →", OUTDIR)
print("[paths] RUN_DIR →", RUN_DIR)

# Tagged artifacts (no clobbering between runs)
WEIGHTS_STUDENT = RUN_DIR / f"vc_student_final__{RUN_TAG}.weights.h5"
PRED_TEST_CSV   = RUN_DIR / f"predictions_test__{RUN_TAG}.csv"
HIST_CSV        = RUN_DIR / f"history__{RUN_TAG}.csv"
COEF_CSV_ALL    = RUN_DIR / f"vc_coefficients_ALL__{RUN_TAG}.csv"
THR_CSV_ALL     = RUN_DIR / f"vc_thresholds_ALL__{RUN_TAG}.csv"
INDEX_W_CSV     = RUN_DIR / f"index_weights__{RUN_TAG}.csv"
INDEX_BIAS_JSON = RUN_DIR / f"index_bias__{RUN_TAG}.json"
INDEX_CONTRIB_CSV = RUN_DIR / f"index_contributions_ALL__{RUN_TAG}.csv"
EFFECTS_OUT     = RUN_DIR / f"index_weight_effects_per1SD__{RUN_TAG}.csv"


# 2---------------------------------- Load & basic prep ----------------------------------
raw = pd.read_csv(DATA_PATH, low_memory=False)
raw[region_col] = raw[region_col].astype(str)

missing = [c for c in [y_var, region_col] + x_input_vars if c not in raw.columns]
if missing: raise KeyError(f"Missing required columns: {missing}")

z_input_vars = [c for c in z_input_vars_all if c in raw.columns]
raw[x_input_vars] = raw[x_input_vars].apply(pd.to_numeric, errors="coerce")
raw[z_input_vars] = raw[z_input_vars].apply(pd.to_numeric, errors="coerce")
raw = raw.dropna(subset=[y_var, region_col]).reset_index(drop=True)

#3 --------- Temporal split (fallback to random if no date col) -----------------------------------------------------
date_col = next((c for c in DATE_CANDIDATES if c in raw.columns), None)
if TEMPORAL_SPLIT and date_col is not None:
    df = raw.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col, y_var, region_col]).sort_values(date_col).reset_index(drop=True)

    cutoff_test = df[date_col].max() - pd.DateOffset(years=TEST_YEARS)
    trainval_df = df[df[date_col] <= cutoff_test].copy()
    test_df     = df[df[date_col]  > cutoff_test].copy()

    n_val = max(1, int(len(trainval_df) * VAL_FRACTION))
    val_df   = trainval_df.tail(n_val).copy()
    train_df = trainval_df.iloc[:-n_val].copy()

    print(f"[temporal] Using '{date_col}'. Train/Val/Test shapes:",
          train_df.shape, val_df.shape, test_df.shape)
else:
    from sklearn.model_selection import train_test_split
    train_df, tmp_df = train_test_split(raw, test_size=0.30, random_state=SEED, stratify=raw[y_var])
    val_df,   test_df = train_test_split(tmp_df, test_size=0.50, random_state=SEED, stratify=tmp_df[y_var])
    print("[random] Stratified split. Train/Val/Test shapes:",
          train_df.shape, val_df.shape, test_df.shape)

# Impute using TRAIN medians; drop Z that are all-NaN or constant on TRAIN
X_meds = train_df[x_input_vars].median(numeric_only=True)
Z_meds = train_df[z_input_vars].median(numeric_only=True)

_keep = []
for c in z_input_vars:
    if pd.isna(Z_meds.get(c, np.nan)):
        print(f"[warn] drop Z '{c}' (all-NaN on train)"); continue
    if train_df[c].std(skipna=True)==0:
        print(f"[warn] drop Z '{c}' (constant on train)"); continue
    _keep.append(c)
z_input_vars = _keep
print(f"[ok] Using Z features: {z_input_vars}")

def _impute(df):
    df[x_input_vars] = df[x_input_vars].fillna(X_meds)
    df[z_input_vars] = df[z_input_vars].fillna(Z_meds[z_input_vars])
for _d in (train_df, val_df, test_df): _impute(_d)

# scalers/encoder (fit on TRAIN only)
X_scaler = StandardScaler().fit(train_df[x_input_vars].to_numpy())
Z_scaler = StandardScaler().fit(train_df[z_input_vars].to_numpy())
try: R_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
except TypeError: R_encoder = OneHotEncoder(handle_unknown="ignore", sparse=True)
R_encoder.fit(train_df[[region_col]])

means, scales = X_scaler.mean_, X_scaler.scale_
name_map = dict(zip(x_input_vars, zip(means, scales)))
print("[sanity] X order & stats:", name_map)
print("[sanity] cuts are [Air, WBT, Dew] matching x_input_vars:", x_input_vars)

def to_arrays(df):
    X = X_scaler.transform(df[x_input_vars].to_numpy()).astype("float32")
    Z = Z_scaler.transform(df[z_input_vars].to_numpy()).astype("float32")
    R = R_encoder.transform(df[[region_col]]).toarray().astype("float32")
    y = df[y_var].to_numpy().astype("float32")
    return X, Z, R, y

X_train, Z_train, R_train, y_train = to_arrays(train_df)
X_val,   Z_val,   R_val,   y_val   = to_arrays(val_df)
X_test,  Z_test,  R_test,  y_test  = to_arrays(test_df)

# indices for coords in standardized Z (for spatial reg)
coord_names = ["Longitude","Latitude","Elevation"]
coord_idxs = [z_input_vars.index(c) for c in coord_names if c in z_input_vars]
HAS_COORDS = (len(coord_idxs) == 3)
if USE_SPATIAL_REG and not HAS_COORDS:
    print("[warn] Spatial regularizer disabled (Longitude/Latitude/Elevation not all in Z).")
    USE_SPATIAL_REG = False

#4 ---------------------------------- Physics-correct layers --------------------------------------------------------
class Shape1D(tf.keras.layers.Layer):
    def __init__(self, n_knots=16, direction="free", l1=1e-4, name=None):
        super().__init__(name=name); assert direction in ("free","increasing","decreasing")
        self.n_knots, self.direction, self.l1 = int(n_knots), direction, float(l1)
    def build(self, input_shape):
        k = np.linspace(0.0, 1.0, self.n_knots).astype("float32")
        self.knots = self.add_weight(shape=(self.n_knots,), initializer=tf.keras.initializers.Constant(k),
                                     trainable=False, name="knots", dtype=tf.float32)
        self.w = self.add_weight(shape=(self.n_knots,), initializer="zeros",
                                 regularizer=regularizers.L1(self.l1) if self.l1>0 else None,
                                 name="w", dtype=tf.float32)
        self.b = self.add_weight(shape=(), initializer="zeros", name="b", dtype=tf.float32)
    def call(self, z01):
        z = tf.clip_by_value(z01, 0.0, 1.0)
        basis = tf.nn.relu(z - tf.reshape(self.knots, (1, self.n_knots)))
        if self.direction=="free":
            slopes = self.w
        else:
            base = tf.nn.softplus(self.w)
            slopes = base if self.direction=="increasing" else -base
        return self.b + tf.reduce_sum(basis * slopes, axis=-1, keepdims=True)

class LinearIndex(tf.keras.layers.Layer):
    def __init__(self, l1=0.0, name=None): super().__init__(name=name); self.l1=float(l1)
    def build(self, input_shape):
        d = int(input_shape[-1])
        self.alpha = self.add_weight(shape=(d,1), initializer="zeros",
                                     regularizer=regularizers.L1(self.l1) if self.l1>0 else None, name="alpha")
        self.bias  = self.add_weight(shape=(1,), initializer="zeros", name="bias")
    def call(self, z_std): return tf.squeeze(z_std @ self.alpha + self.bias, axis=-1)

#5 ---------------------------------- Teacher (optional, for distillation) ---------------------------------------
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
    air_thr = layers.Lambda(lambda t: t[0] + tf.nn.softplus(t[1]), name="air_thr")([wb_thr,   air_off])

    cutting_points = layers.Concatenate(name="cutting_points")([air_thr, wb_thr, dew_base])  

    def gate_pair(x, c):
        if gate_type == "relu":
            return tf.nn.relu(x - c)
        elif gate_type == "softplus":
            return tf.math.softplus(gate_k * (x - c)) / gate_k
        else:
            return tf.sigmoid(gate_k * (x - c)) * (x - c)

    thresholded_x = layers.Lambda(lambda t: gate_pair(t[0], t[1]),
                                  name="thresholded_x")([x_input, cutting_points])

    all_feat = layers.Concatenate(name="clf_concat")([thresholded_x, z_input, r_embed])
    h  = layers.Dense(128, activation="relu", name="clf_h1")(all_feat)
    h  = layers.BatchNormalization(name="clf_bn1")(h)
    h  = layers.Dropout(dropout_rate, seed=SEED, name="clf_do1")(h)
    h  = layers.Dense(64, activation="relu", name="clf_h2")(h)
    h  = layers.BatchNormalization(name="clf_bn2")(h)
    h  = layers.Dropout(dropout_rate, seed=SEED, name="clf_do2")(h)
    h  = layers.Dense(32, activation="relu", name="clf_h3")(h)
    y_out = layers.Dense(1, activation="sigmoid", name="output")(h)

    return models.Model([x_input, z_input, r_input], [y_out, cutting_points],
                        name="DeepCutRegionEmbed")

teacher = None
x_dim, z_dim = len(x_input_vars), len(z_input_vars)

#6 ---------------------------------- Build VC-Student (varying coefficients, physics rules) ----------------------------------
# Part 6: define how predictions are made
def build_vc_student_v2(x_dim, z_dim, r_dim, z_names,
                        gate_type="relu", gate_k=5.0,
                        force_beta_nonneg=True, use_internal_index=True,
                        alpha_l1=1e-4, beta_l2=1e-5):
    x_in = layers.Input(shape=(x_dim,), name="x_input")
    z_in = layers.Input(shape=(z_dim,), name="z_input")
    r_in = layers.Input(shape=(r_dim,), name="region_input")

    # region embedding
    r = layers.Dense(64, activation="relu", name="r_embed_1")(r_in)
    r = layers.BatchNormalization(name="r_embed_bn")(r)
    r = layers.Dense(32, activation="relu", name="r_embed_2")(r)

    # ---------- Single index I ----------
    I_exp = None
    if use_internal_index:
        I = LinearIndex(l1=alpha_l1, name="Index_I_raw")(z_in)   
        I_exp = layers.Lambda(lambda t: tf.expand_dims(t, -1), name="Index_I")(I)  


    # ---------- Cutting points (thresholds) ----------
    parts = []
    for j, nm in enumerate(z_names):
        z_col = layers.Lambda(lambda t, j=j: t[:, j:j+1], name=f"pick_{nm}")(z_in)
        z_std = layers.Lambda(lambda t: tf.clip_by_value(t, -3.0, 3.0), name=f"clip_{nm}")(z_col)
        z01   = layers.Lambda(lambda t: (t + 3.0)/6.0, name=f"to01_{nm}")(z_std)
        direction = "decreasing" if nm == "Elevation" else ("increasing" if nm == "Z0" else "free")
        parts.append(Shape1D(16, direction=direction, l1=1e-4, name=f"shape_{nm}")(z01))

    sumZ = layers.Add(name="sumZ")(parts) if len(parts) > 1 else parts[0]
    if use_internal_index:
        sumZ = layers.Add(name="sumZ_plus_I")(
            [sumZ, layers.Dense(1, use_bias=False, name="I_to_thr")(I_exp)]
        )

    rb      = layers.Dense(1, activation="linear", name="region_bias")(r)
    dew_b   = layers.Add(name="dew_base")([sumZ, rb])
    wb_off  = layers.Dense(1, activation="linear", name="wb_off")(r)
    air_off = layers.Dense(1, activation="linear", name="air_off")(r)
    wbt_thr = layers.Lambda(lambda t: t[0] + tf.nn.softplus(t[1]), name="wbt_thr")([dew_b,  wb_off])
    air_thr = layers.Lambda(lambda t: t[0] + tf.nn.softplus(t[1]), name="air_thr")([wbt_thr, air_off])
    cuts    = layers.Concatenate(name="cutting_points")([air_thr, wbt_thr, dew_b]) 

    # Gate X by cuts
    if gate_type == "relu":
        gx = layers.Lambda(lambda t: tf.nn.relu(t[0] - t[1]), name="thresholded_x")([x_in, cuts])
    elif gate_type == "softplus":
        gx = layers.Lambda(lambda t, k=gate_k: tf.math.softplus(k*(t[0]-t[1]))/k,
                           arguments={"k": gate_k}, name="thresholded_x")([x_in, cuts])
    else:
        gx = layers.Lambda(lambda t, k=gate_k: tf.sigmoid(k*(t[0]-t[1]))*(t[0]-t[1]),
                           arguments={"k": gate_k}, name="thresholded_x")([x_in, cuts])

    # ---------- β depends on I ----------
    if not use_internal_index:
        raise ValueError("Single-index VC requires USE_INTERNAL_INDEX=True")
    zr = I_exp  
    beta0 = layers.Dense(1, activation="linear", name="beta0",
                         kernel_regularizer=regularizers.l2(BETA_L2))(zr)
    beta  = layers.Dense(x_dim,
                         activation=("softplus" if force_beta_nonneg else "linear"),
                         name="beta_vec",
                         kernel_regularizer=regularizers.l2(BETA_L2))(zr)

    # Logit and output
    dot   = layers.Lambda(lambda t: tf.reduce_sum(t[0]*t[1], axis=-1, keepdims=True),
                          name="dot_beta_gx")([beta, gx])
    logit = layers.Add(name="logit")([beta0, dot])
    y_out = layers.Activation("sigmoid", name="y_out")(logit)

    outs = [y_out, cuts]
    if use_internal_index: outs.append(I_exp)
    outs.extend([beta0, beta])
    return models.Model([x_in, z_in, r_in], outs, name="VCStudent_v2")


#7 ---------------------------------- Regularizers ----------------------------------
@tf.function
def region_pairwise_regularization(c_points, region_matrix, lambda_reg=0.02):
    """Pairwise dispersion of cutting points within each region (scale-invariant)."""
    dtype = c_points.dtype
    reg_loss = tf.zeros((), dtype=dtype)  # dtype/device-safe init
    region_ids = tf.argmax(region_matrix, axis=1)
    unique_regions = tf.unique(region_ids)[0]

    for r in unique_regions:
        mask = tf.equal(region_ids, r)
        c_r  = tf.boolean_mask(c_points, mask)     # [Nr, 3]
        nr   = tf.shape(c_r)[0]
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
    """K-NN smoothness of cutting points over standardized coordinates."""
    dtype = c_points.dtype
    B = tf.shape(coords_std)[0]
    xdim = tf.cast(tf.shape(c_points)[1], dtype)

    # pairwise squared distances in standardized coord space
    c2 = tf.reduce_sum(tf.square(coords_std), axis=1, keepdims=True)              # [B,1]
    d2 = c2 - tf.cast(2.0, dtype) * tf.matmul(coords_std, coords_std, transpose_b=True) + tf.transpose(c2)
    big = tf.reduce_max(d2) + tf.constant(1.0, dtype=dtype)
    d2 = tf.linalg.set_diag(d2, tf.fill([B], big))

    # k nearest neighbors (by small distance => large -d2)
    _, idxs = tf.nn.top_k(-d2, k=k)                                               # [B,k]
    c_nb = tf.gather(c_points, idxs)                                              # [B,k,3]
    c_i  = tf.expand_dims(c_points, axis=1)                                       # [B,1,3]
    diffs = c_i - c_nb                                                            # [B,k,3]
    sq = tf.reduce_sum(tf.square(diffs), axis=2)                                  # [B,k]
    loss = tf.reduce_mean(tf.reduce_mean(sq, axis=1))                             # scalar

    return tf.cast(lambda_space, dtype) * (loss / tf.maximum(xdim, tf.constant(1.0, dtype=dtype)))


#8 ---------------------------------- Keras wrapper ----------------------------------
    # Part 8: define how training is performed (adds custom losses)
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
                loss += spatial_smoothness_regularization(c_points, coords_std,
                                                          k=self.k_neighbors, lambda_space=self.spatial_lambda)
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

# 9 ---------------------------------- Build models ----------------------------------
r_dim = R_train.shape[1]
x_dim, z_dim = len(x_input_vars), len(z_input_vars)

base_student = build_vc_student_v2(
    x_dim, z_dim, r_dim, z_names=z_input_vars,
    gate_type=GATE_TYPE, gate_k=GATE_K,
    force_beta_nonneg=FORCE_BETA_NONNEG,
    use_internal_index=USE_INTERNAL_INDEX,
    alpha_l1=ALPHA_L1, beta_l2=BETA_L2
)
print("Student outputs:", base_student.output_names)

# --- Teacher (conditional on toggle; no path logic here—the preflight already set RUN_TAG) ---
teacher = None
if USE_DISTILLATION:
    teacher = build_teacher(x_dim, z_dim, r_dim, gate_type=TEACHER_GATE_TYPE, gate_k=GATE_K)
    teacher.load_weights(TEACHER_WEIGHTS)
    print(f"[ok] Loaded teacher weights: {TEACHER_WEIGHTS}")
else:
    print("[info] Training student without distillation (USE_DISTILLATION=False).")

vc_student = VCStudentModel(
    base_student,
    use_region=USE_REGION_REG, region_lambda=REGION_LAMBDA,
    use_spatial=USE_SPATIAL_REG, spatial_lambda=SPATIAL_LAMBDA,
    k_neighbors=SPATIAL_K, coord_idxs=coord_idxs
)
print("Wrapper ready. RegionReg:", USE_REGION_REG, " SpatialReg:", USE_SPATIAL_REG, " coords ok:", HAS_COORDS)



#10 ---------------------------------- Distillation targets ----------------------------------
if teacher is not None:
    _, tr_cuts = teacher.predict([X_train, Z_train, R_train], batch_size=4096, verbose=0)
    _, va_cuts = teacher.predict([X_val,   Z_val,   R_val],   batch_size=4096, verbose=0)
    train_targets = {"y_out": y_train, "cutting_points": tr_cuts}
    val_targets   = {"y_out": y_val,   "cutting_points": va_cuts}
    lw_cuts = 1.0
else:
    zeros_tr = np.zeros((len(y_train),3), np.float32)
    zeros_va = np.zeros((len(y_val), 3), np.float32)
    train_targets = {"y_out": y_train, "cutting_points": zeros_tr}
    val_targets   = {"y_out": y_val,   "cutting_points": zeros_va}
    lw_cuts = 0.0

if USE_INTERNAL_INDEX:
    train_targets["Index_I"] = np.zeros((len(y_train),1), np.float32)
    val_targets["Index_I"]   = np.zeros((len(y_val),  1), np.float32)

train_targets["beta0"]    = np.zeros((len(y_train),1),      np.float32)
val_targets["beta0"]      = np.zeros((len(y_val),  1),      np.float32)
train_targets["beta_vec"] = np.zeros((len(y_train),x_dim),  np.float32)
val_targets["beta_vec"]   = np.zeros((len(y_val),  x_dim),  np.float32)

#11 ---------------------------------- Compile, warm-up build, & Train ----------------------------------
losses_dict  = {"y_out": losses.BinaryCrossentropy(),
                "cutting_points": losses.MeanSquaredError(),
                "beta0": losses.MeanSquaredError(),
                "beta_vec": losses.MeanSquaredError()}
loss_wts     = {"y_out": 1.0, "cutting_points": lw_cuts, "beta0": 0.0, "beta_vec": 0.0}
metrics_dict = {"y_out": [metrics.BinaryAccuracy(name="acc"), metrics.AUC(name="auc")]}

if USE_INTERNAL_INDEX:
    losses_dict["Index_I"] = losses.MeanSquaredError()
    loss_wts["Index_I"] = 0.0

vc_student.compile(optimizer=optimizers.Adam(learning_rate=LR),
                   loss=losses_dict, loss_weights=loss_wts, metrics=metrics_dict)

_ = vc_student([X_train[:4], Z_train[:4], R_train[:4]], training=False)

cb = [
    tf.keras.callbacks.EarlyStopping(monitor="val_y_out_loss", mode="min",
                                     patience=PATIENCE, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(WEIGHTS_STUDENT.as_posix(), monitor="val_y_out_loss",
                                       mode="min", save_best_only=True, save_weights_only=True)
]

hist = vc_student.fit([X_train, Z_train, R_train], train_targets,
                      validation_data=([X_val, Z_val, R_val], val_targets),
                      epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1, callbacks=cb)


# -------- Evaluate --------
outs_test = vc_student.core.predict([X_test, Z_test, R_test], batch_size=4096, verbose=0)
yprob_te  = outs_test[0].ravel()
auc = roc_auc_score(y_test, yprob_te)
acc = accuracy_score(y_test, (yprob_te > 0.5).astype(int))
print(f"[TEST] AUC={auc:.4f} | Acc={acc:.4f}")

# Save test-set predictions
pd.DataFrame({"y_true": y_test.astype(int), "y_prob": yprob_te}).to_csv(PRED_TEST_CSV, index=False)
print(f"[ok] Saved test predictions → {PRED_TEST_CSV}")

# Distillation diagnostics (if applicable)
if teacher is not None:
    _, teacher_val_cuts = teacher.predict([X_val, Z_val, R_val], batch_size=4096, verbose=0)
    student_val_cuts    = vc_student.core.predict([X_val, Z_val, R_val], batch_size=4096, verbose=0)[1]
    mse_cuts = np.mean((teacher_val_cuts - student_val_cuts)**2)
    print(f"[distill] Teacher–Student cut MSE (val): {mse_cuts:.6f}")

# Save training history (once, to the run-tagged path)
hist_df = pd.DataFrame(hist.history)
hist_df["epoch"] = np.arange(len(hist_df))
hist_df.to_csv(HIST_CSV, index=False)
print(f"[ok] Saved training history → {HIST_CSV}")


#13 -----------------------------------Export interpretable outputs (all rows) ---------------------------------
# ---------------------------------- Export β,β0​,I(z) ----------------------------------
full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
def arrays_from_df(df):
    X = X_scaler.transform(df[x_input_vars].to_numpy()).astype("float32")
    Z = Z_scaler.transform(df[z_input_vars].to_numpy()).astype("float32")
    R = R_encoder.transform(df[[region_col]]).toarray().astype("float32")
    return X, Z, R
X_all, Z_all, R_all = arrays_from_df(full_df)

outs_all = vc_student.core.predict([X_all, Z_all, R_all], batch_size=4096, verbose=0)
cuts_all    = outs_all[1]
if USE_INTERNAL_INDEX:
    Index_I_all = outs_all[2].reshape(-1)
    b0_all      = outs_all[3].reshape(-1)
    b_all       = outs_all[4]
else:
    b0_all      = outs_all[2].reshape(-1)
    b_all       = outs_all[3]

coef_df = pd.DataFrame(b_all, columns=[f"beta_{nm}" for nm in x_input_vars], index=full_df.index)
coef_df["beta0"] = b0_all
if USE_INTERNAL_INDEX:
    coef_df["Index_I"] = Index_I_all
keep_cols = [c for c in [region_col, "Longitude","Latitude","Elevation"] if c in full_df.columns]
coef_df = pd.concat([full_df[keep_cols].reset_index(drop=True), coef_df.reset_index(drop=True)], axis=1)
coef_df.to_csv(COEF_CSV_ALL, index=False)
print(f"[ok] Saved β0/β per row → {COEF_CSV_ALL}")


# ---------------------------------- Export Index_I weights & contributions ----------------------------------
if USE_INTERNAL_INDEX:
    
    core = vc_student.core  
    idx_layer = core.get_layer("Index_I_raw") 
    alpha_std = idx_layer.get_weights()[0].reshape(-1)  
    bias_std  = float(idx_layer.get_weights()[1].reshape(()))
    
    z_names  = z_input_vars
    z_means  = Z_scaler.mean_
    z_scales = Z_scaler.scale_
    assert len(z_names) == len(alpha_std) == len(z_scales), "Mismatch in Z dimension/order."

    alpha_orig = alpha_std / z_scales
    bias_orig  = bias_std - np.sum(alpha_std * (z_means / z_scales))

    weights_df = pd.DataFrame({
        "variable":  z_names,
        "alpha_stdZ": alpha_std,
        "alpha_orig": alpha_orig,
        "Z_mean":     z_means,
        "Z_std":      z_scales
    }).sort_values("alpha_stdZ", key=lambda s: np.abs(s), ascending=False)
    weights_df.to_csv(INDEX_W_CSV, index=False)
    print(f"[ok] Saved index weights → {INDEX_W_CSV}")


    INDEX_BIAS_JSON.write_text(
    json.dumps({"bias_stdZ": float(bias_std), "bias_orig": float(bias_orig)}, indent=2)
    )


    Z_std_all = Z_all  
    contrib_mat = Z_std_all * alpha_std.reshape(1, -1)  
    contrib_sum = contrib_mat.sum(axis=1) + bias_std     
    diff = np.nanmax(np.abs(contrib_sum - Index_I_all))
    print(f"[check] max|Index_I_from_parts - Index_I_model| = {diff:.6e}")

    meta_cols = [c for c in [region_col, "Longitude","Latitude","Elevation"] if c in full_df.columns]
    contrib_cols = {f"contrib_{nm}": contrib_mat[:, j] for j, nm in enumerate(z_names)}
    contrib_df = pd.DataFrame({
        **{c: full_df[c].values for c in meta_cols},
        "Index_I_model": Index_I_all,
        "bias_stdZ": bias_std,
        **contrib_cols
    })
    contrib_df["Index_I_from_parts"] = contrib_df.filter(like="contrib_", axis=1).sum(axis=1).astype(float) + contrib_df["bias_stdZ"].astype(float)

    contrib_df.to_csv(INDEX_CONTRIB_CSV, index=False)
    print("[ok] Saved per-row Index_I contributions →", INDEX_CONTRIB_CSV)
   

    # 4) Quick summary
    top_k = min(10, len(weights_df))
    print("\nTop drivers of Index_I (standardized space):")
    print(weights_df.head(top_k)[["variable","alpha_stdZ","alpha_orig"]])
else:
    print("[skip] USE_INTERNAL_INDEX is False → no Index_I weights to export.")

# right after outs_all
thr_df = pd.DataFrame(cuts_all, columns=["thr_Air_std","thr_WBT_std","thr_Dew_std"])

i_air = x_input_vars.index("Air_Temp")
i_wbt = x_input_vars.index("Wet_Bulb_Temp")
i_dew = x_input_vars.index("Dewpoint")
means, scales = X_scaler.mean_, X_scaler.scale_

thr_df["thr_Air_C"] = thr_df["thr_Air_std"]*scales[i_air] + means[i_air]
thr_df["thr_WBT_C"] = thr_df["thr_WBT_std"]*scales[i_wbt] + means[i_wbt]
thr_df["thr_Dew_C"] = thr_df["thr_Dew_std"]*scales[i_dew] + means[i_dew]

# enforce physics: Dew ≤ WBT ≤ Air
thr_df["thr_WBT_C"] = np.maximum(thr_df["thr_WBT_C"], thr_df["thr_Dew_C"])
thr_df["thr_Air_C"] = np.maximum(thr_df["thr_Air_C"], thr_df["thr_WBT_C"])


meta_cols = [c for c in [region_col, "Longitude","Latitude","Elevation"] if c in full_df.columns]
z_present = [c for c in (Z_KEEP if "Z_KEEP" in globals() else []) if c in full_df.columns]

thr_df = pd.concat(
    [full_df.reset_index(drop=True)[meta_cols + z_present], thr_df.reset_index(drop=True)],
    axis=1
)
thr_df.to_csv(THR_CSV_ALL, index=False)
print(f"[ok] Saved thresholds per row → {THR_CSV_ALL}  (added Z: {z_present})")
thr_std = cuts_all  # [Air, WBT, Dew]
ord_mask = (thr_std[:, 2] <= thr_std[:, 1]) & (thr_std[:, 1] <= thr_std[:, 0])  # Dew ≤ WBT ≤ Air

print(f"[ordering] Dew ≤ WBT ≤ Air holds for {ord_mask.mean()*100:.4f}% "
      f"({ord_mask.sum():,}/{len(ord_mask):,})")

# --------------------Effects of each Z (per +1 SD) on thresholds -----------------------------------------

# For speed you can subsample; set N_SAMPLES = len(Z_all) to use all rows
N_SAMPLES = min(100_000, len(Z_all))
rng = np.random.default_rng(SEED)
idx = rng.choice(len(Z_all), size=N_SAMPLES, replace=False) if N_SAMPLES < len(Z_all) else np.arange(len(Z_all))

Xb = X_all[idx]
Zb = Z_all[idx].copy()   # standardized Z
Rb = R_all[idx]

# Warm up
_ = vc_student.core.predict([Xb, Zb, Rb], batch_size=4096, verbose=0)

# Order of cuts from the model is [Air_std, WBT_std, Dew_std]
i_air = x_input_vars.index("Air_Temp")
i_wbt = x_input_vars.index("Wet_Bulb_Temp")
i_dew = x_input_vars.index("Dewpoint")
thr_scales = np.array([X_scaler.scale_[i_air], X_scaler.scale_[i_wbt], X_scaler.scale_[i_dew]], dtype=float)

rows = []
for j, driver in enumerate(z_input_vars):
    # +/- 1 SD in Z_j (Z is already standardized)
    Z_plus  = Zb.copy(); Z_plus[:, j]  += 1.0
    Z_minus = Zb.copy(); Z_minus[:, j] -= 1.0

    cuts_plus  = vc_student.core.predict([Xb, Z_plus,  Rb], batch_size=4096, verbose=0)[1]  # [N,3]
    cuts_minus = vc_student.core.predict([Xb, Z_minus, Rb], batch_size=4096, verbose=0)[1]  # [N,3]

    # symmetric finite difference in std-X units → convert to °C
    effect_std = (cuts_plus - cuts_minus) / 2.0                   # [N,3] = [Air_std, WBT_std, Dew_std]
    effect_C   = effect_std * thr_scales.reshape(1, 3)            # [N,3] in °C

    mu = np.nanmean(effect_C, axis=0)  # average effect over sample
    rows.append({
        "driver": driver,
        "thr_Air_C_per1SD": float(mu[0]),
        "thr_WBT_C_per1SD": float(mu[1]),
        "thr_Dew_C_per1SD": float(mu[2]),
    })

effects_df = pd.DataFrame(rows).sort_values("driver")
effects_df.to_csv(EFFECTS_OUT, index=False)
print("[effects] saved →", EFFECTS_OUT)


gc.collect()
# ==========================================================================================================================

