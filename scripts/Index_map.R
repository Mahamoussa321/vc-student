# ===========================
# make_phase_maps.R  (run-tag aware; pretty palettes; one-line legends)
# ===========================

# Packages (same stack you already use; no 'here')
pacman::p_load(
  tidyverse, scales, viridis, RColorBrewer,
  rnaturalearth, rnaturalearthdata,
  sf, units, dplyr, grid
)

# ------- Minimal paths (assume working dir = project root) --------------------
FIG_DIR  <- "figures"
OUT_DIR  <- "outputs"
RUN_TAGS <- c("distill", "nodistill")   # change/trim as needed

dir.create(FIG_DIR, showWarnings = FALSE, recursive = TRUE)
message("[paths] FIG_DIR=", normalizePath(FIG_DIR),
        "  |  OUT_DIR=", normalizePath(OUT_DIR))

# ------- Common export size (identical for all panels) ------------------------
COMMON_W <- 7.2   # inches
COMMON_H <- 3.8   # inches
COMMON_DPI <- 600

# ------- World basemap --------------------------------------------------------
world <- rnaturalearth::ne_countries(scale = "medium", returnclass = "sf")

# ------- Small helpers --------------------------------------------------------
theme_map_pretty <- function() {
  theme_minimal(base_size = 12) +
    theme(
      axis.text        = element_blank(),
      axis.title       = element_blank(),
      axis.ticks       = element_blank(),
      panel.grid       = element_blank(),
      panel.border     = element_rect(color = "black", fill = NA, linewidth = 0.8),
      panel.background = element_rect(fill = "gray95", color = NA),
      plot.background  = element_rect(fill = "white", color = "black", linewidth = 1),
      legend.key.height = grid::unit(0.55, "cm"),
      legend.key.width  = grid::unit(0.45, "cm"),
      legend.title     = element_text(size = 13, face = "bold"),
      legend.text      = element_text(size = 12)
    )
}

# identical legend position + outer margins for all figures
theme_common <- theme(
  legend.position = "bottom",
  legend.direction = "horizontal",
  plot.margin = grid::unit(rep(6, 4), "pt")  # (t, r, b, l)
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
    filter(is.finite(.data[[lon]]),
           is.finite(.data[[lat]]),
           is.finite(.data[[value_col]])) %>%
    group_by(.data[[lon]], .data[[lat]]) %>%
    summarise(val = mean(.data[[value_col]], na.rm = TRUE), .groups = "drop") %>%
    rename(Longitude = all_of(lon), Latitude = all_of(lat)) %>%
    sf::st_as_sf(coords = c("Longitude", "Latitude"), crs = 4326)
}

save_plot <- function(p, path_png, w = COMMON_W, h = COMMON_H, dpi = COMMON_DPI, bg = "white") {
  ggsave(filename = path_png, plot = p, width = w, height = h, dpi = dpi, bg = bg)
  message("[ok] saved: ", path_png)
}

# ========================= Main loop ==========================================
for (RUN_TAG in RUN_TAGS) {
  
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
  
  # ---------- 0) Map of Context Index I(z) ----------
  iz_sf <- station_mean_sf(idx_df, value_col = "Index_I_model")
  
  iz_range  <- range(iz_sf$val, na.rm = TRUE)
  iz_limits <- c(max(-2.5, floor(iz_range[1]*2)/2), min(-0.5, ceiling(iz_range[2]*2)/2))
  if (iz_limits[1] >= iz_limits[2] || any(!is.finite(iz_limits))) iz_limits <- iz_range
  iz_breaks <- pretty(iz_limits, n = 5)
  
  p_iz_map <- ggplot(world) +
    geom_sf(fill = "gray85", color = "gray55", linewidth = 0.2) +
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
  
  save_plot(p_iz_map, file.path(FIG_DIR, paste0("map_index_Iz__", RUN_TAG, ".png")))
  
  # ---------- A) Map of Air Temperature Threshold ----------
  air_col <- "thr_Air_C"
  stopifnot(air_col %in% names(thr_df))
  air_sf <- station_mean_sf(thr_df, value_col = air_col)
  
  air_limits <- c(-2.5, 0.5)            # fixed legend limits
  air_breaks <- seq(-2.5, 0, by = 0.5)  # ticks at 0.5°C steps
  
  p_air_map <- ggplot(world) +
    geom_sf(fill = "gray85", color = "gray55", linewidth = 0.2) +
    geom_sf(data = air_sf, aes(color = val), size = 0.7) +
    scale_color_viridis_c(
      option = "viridis",
      limits = air_limits,
      breaks = air_breaks,
      oob    = scales::squish,
      name   = "Mean air temp threshold (°C)"
    ) +
    coord_sf(xlim = c(-180, 180), ylim = c(0, 90), expand = FALSE) +
    theme_map_pretty() + theme_common + guide_bar_h()
  
  save_plot(p_air_map, file.path(FIG_DIR, paste0("map_air_threshold__", RUN_TAG, ".png")))
  
  # ---------- B) Join station means (Air + I(z)) ----------
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
  
  # ---------- C) Correlation ----------
  cor_air_vs_Iz <- cor(merged_air$thr_Air_C_mean, merged_air$Index_I_mean, use = "complete.obs")
  message("[", RUN_TAG, "] Pearson r (Air threshold vs I(z)) = ", round(cor_air_vs_Iz, 3))
  
  # ---------- D) Scatter plot (Air threshold vs I(z)) ----------
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
    scale_color_viridis_c(option = "turbo", name = "Latitude (°N)") +
    labs(x = "Context index  I(z)  (std units)", y = "Air temperature threshold (°C)") +
    annotate("text", x = Inf, y = -Inf, label = pearson_label_air,
             hjust = 1.1, vjust = -0.8, fontface = "bold", size = 4) +
    theme_minimal(base_size = 12) +
    theme(
      panel.border     = element_rect(color = "black", fill = NA, linewidth = 0.8),
      panel.grid.minor = element_blank()
    ) +
    theme_common +                                      # same margins + legend
    scale_x_continuous(expand = c(0, 0)) +              # remove side padding
    scale_y_continuous(expand = c(0, 0))
  
  save_plot(
    p_scatter_air,
    file.path(FIG_DIR, paste0("threshold_vs_index_scatter_air__", RUN_TAG, ".png")),
    w = COMMON_W, h = COMMON_H
  )
}
