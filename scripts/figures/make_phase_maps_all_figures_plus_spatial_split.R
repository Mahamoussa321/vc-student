# ===========================
# make_phase_maps_climate_boundaries.R
# Uses KÃ¶ppenâ€“Geiger climate-zone boundaries from ASCII point grid
# and creates:
#   1) ONE spatial split coverage map from saved split assignments
#   2) context-index maps for the standard runs
#   3) air-threshold maps for the standard runs
#   4) threshold-vs-index scatter plots for the standard runs
# ===========================

pacman::p_load(
  tidyverse, scales, viridis, RColorBrewer,
  rnaturalearth, rnaturalearthdata,
  sf, units, dplyr, grid
)

args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)

if (length(file_arg) > 0) {
  SCRIPT_PATH <- normalizePath(sub("^--file=", "", file_arg[1]), winslash = "/", mustWork = TRUE)
  SCRIPT_DIR  <- dirname(SCRIPT_PATH)
} else {
  SCRIPT_DIR <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
}

ROOT_DIR <- normalizePath(file.path(SCRIPT_DIR, "..", ".."), winslash = "/", mustWork = TRUE)
setwd(ROOT_DIR)

FIG_DIR  <- file.path(ROOT_DIR, "figures")
OUT_DIR  <- file.path(ROOT_DIR, "outputs")

# These are the runs for the regular manuscript maps/scatters
MODEL_RUN_TAGS <- c("distill", "nodistill")

# This is the run whose saved split_assignments file is used for the spatial split map
SPLIT_RUN_TAG <- "distill_spatial_blocked_latlon"

# KÃ¶ppenâ€“Geiger ASCII table with columns: Lat, Lon, Cls
KG_ASCII <- file.path(ROOT_DIR, "data", "Koeppen-Geiger-ASCII.txt")

dir.create(FIG_DIR, showWarnings = FALSE, recursive = TRUE)

COMMON_W   <- 7.2
COMMON_H   <- 3.8
COMMON_DPI <- 600

# ------- Coastline for orientation --------------------------------------------
coast <- rnaturalearth::ne_coastline(scale = "medium", returnclass = "sf")

if (!file.exists(KG_ASCII)) {
  stop("KÃ¶ppenâ€“Geiger ASCII file not found. Update KG_ASCII to the correct file path.")
}

# ------- Read climate table ----------------------------------------------------
kg_df <- read.table(KG_ASCII, header = TRUE, sep = "", fill = TRUE)

# Keep only the Northern Hemisphere map extent used in your figures
kg_df <- kg_df %>%
  dplyr::filter(Lat >= 0, Lat <= 90, Lon >= -180, Lon <= 180)

# The ASCII file stores 0.5-degree grid-cell centers, so each cell extends by 0.25
kg_cells <- kg_df %>%
  mutate(
    xmin = Lon - 0.25,
    xmax = Lon + 0.25,
    ymin = Lat - 0.25,
    ymax = Lat + 0.25
  )

# Build polygons for each cell
polys <- lapply(seq_len(nrow(kg_cells)), function(i) {
  with(kg_cells[i, ], {
    sf::st_polygon(list(matrix(
      c(
        xmin, ymin,
        xmax, ymin,
        xmax, ymax,
        xmin, ymax,
        xmin, ymin
      ),
      ncol = 2, byrow = TRUE
    )))
  })
})

# Convert to sf and dissolve by climate class
kg_poly <- sf::st_sf(
  Cls = kg_cells$Cls,
  geometry = sf::st_sfc(polys, crs = 4326)
) %>%
  dplyr::group_by(Cls) %>%
  dplyr::summarise(do_union = TRUE, .groups = "drop") %>%
  sf::st_make_valid()

# Crop coastline to same extent
coast <- sf::st_crop(
  coast,
  sf::st_bbox(c(xmin = -180, xmax = 180, ymin = 0, ymax = 90), crs = sf::st_crs(4326))
)

# ------- Theme helpers ---------------------------------------------------------
theme_map_pretty <- function() {
  theme_minimal(base_size = 12) +
    theme(
      axis.text         = element_blank(),
      axis.title        = element_blank(),
      axis.ticks        = element_blank(),
      panel.grid        = element_blank(),
      panel.border      = element_rect(color = "black", fill = NA, linewidth = 0.8),
      panel.background  = element_rect(fill = "gray95", color = NA),
      plot.background   = element_rect(fill = "white", color = "black", linewidth = 1),
      legend.key.height = grid::unit(0.55, "cm"),
      legend.key.width  = grid::unit(0.45, "cm"),
      legend.title      = element_text(size = 13, face = "bold"),
      legend.text       = element_text(size = 12)
    )
}

theme_common <- theme(
  legend.position  = "bottom",
  legend.direction = "horizontal",
  plot.margin      = grid::unit(rep(6, 4), "pt")
)

guide_bar_h <- function() {
  guides(color = guide_colorbar(
    direction      = "horizontal",
    title.position = "left",
    label.position = "bottom",
    barwidth       = grid::unit(8, "cm"),
    barheight      = grid::unit(0.5, "cm"),
    title.vjust    = 0.5,
    title.hjust    = 0.5
  ))
}

station_mean_sf <- function(df, value_col, lon = "Longitude", lat = "Latitude") {
  stopifnot(all(c(lon, lat, value_col) %in% names(df)))
  df %>%
    filter(
      is.finite(.data[[lon]]),
      is.finite(.data[[lat]]),
      is.finite(.data[[value_col]])
    ) %>%
    group_by(.data[[lon]], .data[[lat]]) %>%
    summarise(val = mean(.data[[value_col]], na.rm = TRUE), .groups = "drop") %>%
    rename(Longitude = all_of(lon), Latitude = all_of(lat)) %>%
    sf::st_as_sf(coords = c("Longitude", "Latitude"), crs = 4326)
}

save_plot <- function(p, path_png, w = COMMON_W, h = COMMON_H, dpi = COMMON_DPI, bg = "white") {
  ggsave(filename = path_png, plot = p, width = w, height = h, dpi = dpi, bg = bg)
  message("[ok] saved: ", path_png)
}

# Climate boundaries + coastline
map_layers <- list(
  geom_sf(data = kg_poly, fill = NA, color = "gray70", linewidth = 0.18, alpha = 0.9),
  geom_sf(data = coast, color = "black", linewidth = 0.22)
)

# ------- Spatial split coverage map from saved split assignments ---------------
make_split_coverage_map <- function(run_tag) {
  split_csv <- file.path(
    OUT_DIR, "ablation", run_tag,
    paste0("split_assignments__", run_tag, ".csv")
  )

  if (!file.exists(split_csv)) {
    message("[skip] split map: split assignments file not found -> ", split_csv)
    return(invisible(NULL))
  }

  split_df <- readr::read_csv(split_csv, show_col_types = FALSE)

  needed <- c("split_set", "Longitude", "Latitude")
  miss <- needed[!needed %in% names(split_df)]
  if (length(miss) > 0) {
    stop("split_assignments file is missing: ", paste(miss, collapse = ", "))
  }

  split_df <- split_df %>%
    mutate(
      Longitude = as.numeric(Longitude),
      Latitude  = as.numeric(Latitude)
    ) %>%
    filter(
      is.finite(Longitude),
      is.finite(Latitude),
      !is.na(split_set)
    ) %>%
    distinct(Longitude, Latitude, split_set) %>%
    mutate(
      split_group = case_when(
        split_set %in% c("train", "val") ~ "Train/validation",
        split_set == "test" ~ "Test",
        split_set == "buffer_excluded" ~ "Buffer excluded",
        TRUE ~ split_set
      )
    )

  split_sf <- sf::st_as_sf(split_df, coords = c("Longitude", "Latitude"), crs = 4326)

  p_split <- ggplot() +
    map_layers +
    geom_sf(
      data = split_sf,
      aes(color = split_group),
      size = 0.45, alpha = 0.75
    ) +
    scale_color_manual(
      values = c(
        "Train/validation" = "#f95738",# #1f77b4 #ff7f0e #2ca02c
        "Test"             = "#0f4c5c",
        "Buffer excluded"  = "#009E73"
      ),
      breaks = c("Buffer excluded", "Test", "Train/validation"),
      name = NULL
    ) +
    coord_sf(xlim = c(-180, 180), ylim = c(0, 90), expand = FALSE) +
    theme_map_pretty() +
    theme_common +
    theme(
      legend.position = "bottom",
      legend.text = element_text(size = 12)
    )

  save_plot(
    p_split,
    file.path(FIG_DIR, paste0("split_coverage_map__", run_tag, ".png")),
    w = COMMON_W,
    h = COMMON_H
  )
}

# ------- Create ONE spatial split map -----------------------------------------
make_split_coverage_map(SPLIT_RUN_TAG)

# ========================= Main loop for regular figures =======================
for (RUN_TAG in MODEL_RUN_TAGS) {

  run_dir <- file.path(OUT_DIR, "ablation", RUN_TAG)
  thr_csv <- file.path(run_dir, paste0("vc_thresholds_ALL__", RUN_TAG, ".csv"))
  idx_csv <- file.path(run_dir, paste0("index_contributions_ALL__", RUN_TAG, ".csv"))

  if (!file.exists(thr_csv) || !file.exists(idx_csv)) {
    message("[skip:", RUN_TAG, "] missing: ",
            if (!file.exists(thr_csv)) paste0("\n  - ", thr_csv) else "",
            if (!file.exists(idx_csv)) paste0("\n  - ", idx_csv) else "")
    next
  }

  thr_df <- readr::read_csv(thr_csv, show_col_types = FALSE)
  idx_df <- readr::read_csv(idx_csv, show_col_types = FALSE)

  # ---- 1) Map of Context Index I(z) ------------------------------------------
  iz_sf <- station_mean_sf(idx_df, value_col = "Index_I_model")

  iz_range  <- range(iz_sf$val, na.rm = TRUE)
  iz_limits <- c(
    max(-2.5, floor(iz_range[1] * 2) / 2),
    min(-0.5, ceiling(iz_range[2] * 2) / 2)
  )
  if (iz_limits[1] >= iz_limits[2] || any(!is.finite(iz_limits))) iz_limits <- iz_range
  iz_breaks <- pretty(iz_limits, n = 5)

  p_iz_map <- ggplot() +
    map_layers +
    geom_sf(data = iz_sf, aes(color = val), size = 0.7) +
    scale_color_viridis_c(
      option = "viridis",
      limits = iz_limits,
      breaks = iz_breaks,
      oob    = scales::squish,
      name   = "Context index  I(z)  (std units)"
    ) +
    coord_sf(xlim = c(-180, 180), ylim = c(0, 90), expand = FALSE) +
    theme_map_pretty() + theme_common + guide_bar_h()

  save_plot(
    p_iz_map,
    file.path(FIG_DIR, paste0("map_index_Iz__", RUN_TAG, ".png"))
  )

  # ---- 2) Map of Air Temperature Threshold -----------------------------------
  air_col <- "thr_Air_C"
  stopifnot(air_col %in% names(thr_df))
  air_sf <- station_mean_sf(thr_df, value_col = air_col)

  air_limits <- c(-2.5, 0.5)
  air_breaks <- seq(-2.5, 0, by = 0.5)

  p_air_map <- ggplot() +
    map_layers +
    geom_sf(data = air_sf, aes(color = val), size = 0.7) +
    scale_color_viridis_c(
      option = "viridis",
      limits = air_limits,
      breaks = air_breaks,
      oob    = scales::squish,
      name   = "Mean air temp threshold (Â°C)"
    ) +
    coord_sf(xlim = c(-180, 180), ylim = c(0, 90), expand = FALSE) +
    theme_map_pretty() + theme_common + guide_bar_h()

  save_plot(
    p_air_map,
    file.path(FIG_DIR, paste0("map_air_threshold__", RUN_TAG, ".png"))
  )

  # ---- 3) Join station means (Air + I(z)) ------------------------------------
  thr_station_air <- thr_df %>%
    filter(is.finite(Longitude), is.finite(Latitude), is.finite(.data[[air_col]])) %>%
    group_by(Longitude, Latitude) %>%
    summarise(thr_Air_C_mean = mean(.data[[air_col]], na.rm = TRUE), .groups = "drop")

  idx_station_iz <- idx_df %>%
    filter(is.finite(Longitude), is.finite(Latitude), is.finite(Index_I_model)) %>%
    group_by(Longitude, Latitude) %>%
    summarise(
      Index_I_mean = mean(Index_I_model, na.rm = TRUE),
      Latitude     = mean(Latitude, na.rm = TRUE),
      .groups = "drop"
    )

  merged_air <- inner_join(thr_station_air, idx_station_iz, by = c("Longitude", "Latitude"))

  # ---- 4) Correlation ---------------------------------------------------------
  cor_air_vs_Iz <- cor(merged_air$thr_Air_C_mean, merged_air$Index_I_mean, use = "complete.obs")
  message("[", RUN_TAG, "] Pearson r (Air threshold vs I(z)) = ", round(cor_air_vs_Iz, 3))

  # ---- 5) Scatter plot (Air threshold vs I(z)) --------------------------------
  pearson_label_air <- paste0("r = ", sprintf("%.2f", cor_air_vs_Iz))

  p_scatter_air <- ggplot(
    merged_air,
    aes(x = Index_I_mean, y = thr_Air_C_mean, color = Latitude)
  ) +
    geom_point(alpha = 0.55, size = 1.15, stroke = 0) +
    geom_smooth(
      aes(x = Index_I_mean, y = thr_Air_C_mean),
      method = "loess", formula = y ~ x, se = TRUE,
      color  = "black", fill  = "gray75",
      linewidth = 0.8, alpha = 0.22, inherit.aes = FALSE
    ) +
    scale_color_viridis_c(option = "turbo", name = "Latitude (Â°N)") +
    labs(
      x = "Context index  I(z)  (std units)",
      y = "Air temperature threshold (Â°C)"
    ) +
    annotate(
      "text",
      x = Inf, y = -Inf, label = pearson_label_air,
      hjust = 1.1, vjust = -0.8, fontface = "bold", size = 4
    ) +
    theme_minimal(base_size = 12) +
    theme(
      panel.border     = element_rect(color = "black", fill = NA, linewidth = 0.8),
      panel.grid.minor = element_blank()
    ) +
    theme_common +
    scale_x_continuous(expand = expansion(mult = 0.03)) +
    coord_cartesian(
      ylim = if (RUN_TAG == "nodistill") {
        quantile(merged_air$thr_Air_C_mean, probs = c(0.01, 0.99), na.rm = TRUE)
      } else {
        range(merged_air$thr_Air_C_mean, na.rm = TRUE)
      }
    )

  save_plot(
    p_scatter_air,
    file.path(FIG_DIR, paste0("threshold_vs_index_scatter_air__", RUN_TAG, ".png")),
    w = COMMON_W, h = COMMON_H
  )
}

