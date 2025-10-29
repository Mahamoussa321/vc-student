# ===========================
# Spatial maps for paper
# - Dewpoint threshold (°C)
# - Contextual index I(z)
# ===========================

# Load libraries
suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(readr)
})

# ---------------------------
# Paths
# ---------------------------
# Assume this script lives in project root, same level as "outputs" and "figures"
BASE_DIR <- getwd()

thr_path   <- file.path(BASE_DIR, "outputs", "vc_thresholds_ALL.csv")
idx_path   <- file.path(BASE_DIR, "outputs", "index_contributions_ALL.csv")
fig_dir    <- file.path(BASE_DIR, "figures")

if (!dir.exists(fig_dir)) dir.create(fig_dir, recursive = TRUE)

message("[paths]")
message("  thresholds CSV : ", thr_path)
message("  index CSV      : ", idx_path)
message("  figures outdir : ", fig_dir)

# ---------------------------
# Helpers
# ---------------------------

# downsample to avoid overplotting thousands of points in dense regions
downsample_points <- function(df, n_max = 4000, seed = 2025) {
  set.seed(seed)
  if (nrow(df) > n_max) {
    df[sample.int(nrow(df), n_max), , drop = FALSE]
  } else {
    df
  }
}

# a clean, journal-ish theme
theme_map <- function() {
  theme_minimal(base_size = 11) +
    theme(
      panel.grid = element_line(color = "grey80", linewidth = 0.3, linetype = "dashed"),
      panel.grid.minor = element_blank(),
      panel.background = element_rect(fill = "white", color = NA),
      axis.title = element_text(color = "black"),
      axis.text  = element_text(color = "black"),
      plot.title = element_text(face = "bold", hjust = 0, size = 12),
      legend.title = element_text(face = "bold"),
      legend.position = "right",
      legend.key.height = unit(0.6, "cm"),
      legend.key.width  = unit(0.4, "cm")
    )
}

# ---------------------------
# 1. Dewpoint threshold map
# ---------------------------

if (file.exists(thr_path)) {
  
  thr_df <- read_csv(thr_path, show_col_types = FALSE)
  
  # required columns we expect from your export
  needed_thr_cols <- c("Longitude", "Latitude", "thr_Dew_C")
  missing_thr <- setdiff(needed_thr_cols, names(thr_df))
  if (length(missing_thr) > 0) {
    stop("Missing required columns for threshold map: ", paste(missing_thr, collapse = ", "))
  }
  
  # Drop NA and obviously bad coords (if any)
  thr_map <- thr_df %>%
    filter(
      is.finite(Longitude),
      is.finite(Latitude),
      is.finite(thr_Dew_C)
    )
  
  # Optional: restrict to plausible domain if you only work N. Hemisphere land obs
  # e.g. Latitude > 10, Latitude < 80, etc.
  # thr_map <- thr_map %>% filter(Latitude > 10, Latitude < 80)
  
  # Thin points for clarity
  thr_plot <- downsample_points(thr_map, n_max = 4000)
  
  # Build plot
  p_thr <- ggplot(thr_plot, aes(x = Longitude, y = Latitude, color = thr_Dew_C)) +
    geom_point(size = 1.2, alpha = 0.8, stroke = 0) +
    scale_color_viridis_c(
      name = "Dewpoint\nthreshold (°C)",
      option = "C"  # perceptually uniform
    ) +
    labs(
      x = "Longitude",
      y = "Latitude",
      title = "Learned dewpoint transition temperature (°C)"
    ) +
    coord_equal(ratio = 1.2) +
    theme_map()
  
  out_thr <- file.path(fig_dir, "map_dewpoint_threshold.png")
  ggsave(out_thr, p_thr, width = 8, height = 4, dpi = 600)
  message("[ok] saved dewpoint-threshold map → ", out_thr)
  
} else {
  warning("Threshold file not found: ", thr_path)
}


# ---------------------------
# 2. Contextual index I(z) map
# ---------------------------

if (file.exists(idx_path)) {
  
  idx_df <- read_csv(idx_path, show_col_types = FALSE)
  
  # required columns we expect from your export
  # from index_contributions_ALL.csv:
  #   "Longitude","Latitude","Index_I_model"
  needed_idx_cols <- c("Longitude", "Latitude", "Index_I_model")
  missing_idx <- setdiff(needed_idx_cols, names(idx_df))
  if (length(missing_idx) > 0) {
    stop("Missing required columns for I(z) map: ", paste(missing_idx, collapse = ", "))
  }
  
  idx_map <- idx_df %>%
    filter(
      is.finite(Longitude),
      is.finite(Latitude),
      is.finite(Index_I_model)
    )
  
  # Thin points
  idx_plot <- downsample_points(idx_map, n_max = 4000)
  
  p_idx <- ggplot(idx_plot, aes(x = Longitude, y = Latitude, color = Index_I_model)) +
    geom_point(size = 1.2, alpha = 0.8, stroke = 0) +
    scale_color_viridis_c(
      name = "Context index\nI(z) (std units)",
      option = "B"
    ) +
    labs(
      x = "Longitude",
      y = "Latitude",
      title = "Spatial pattern of learned contextual index I(z)"
    ) +
    coord_equal(ratio = 1.2) +
    theme_map()
  
  out_idx <- file.path(fig_dir, "map_index_Iz.png")
  ggsave(out_idx, p_idx, width = 8, height = 4, dpi = 600)
  message("[ok] saved I(z) map → ", out_idx)
  
} else {
  warning("Index file not found: ", idx_path)
}


# ---------------------------
# 3. Artifact index (mirroring your Python style)
# ---------------------------

# We will write a small manifest JSON-like text file so you can track generated figures.
artifact_list <- list(
  figures = list.files(fig_dir, pattern = "\\.png$", full.names = TRUE),
  sources = list(thresholds_csv = thr_path,
                 index_csv      = idx_path)
)

art_idx_path <- file.path(BASE_DIR, "outputs", "map_artifacts_index.txt")
if (!dir.exists(file.path(BASE_DIR, "outputs"))) {
  dir.create(file.path(BASE_DIR, "outputs"), recursive = TRUE)
}
writeLines(
  paste(capture.output(str(artifact_list)), collapse = "\n"),
  con = art_idx_path
)
message("[ok] wrote artifact index → ", art_idx_path)

message("Done.")
