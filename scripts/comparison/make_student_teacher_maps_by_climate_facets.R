suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(ggplot2)
  library(viridis)
  library(sf)
  library(rnaturalearth)
  library(rnaturalearthdata)
  library(scales)
  library(grid)
})

# =========================
# User options
# =========================
ROOT_DIR <- "C:/Users/maham/Desktop/vc-student"
USE_KG_BOUNDARIES <- TRUE
SHOW_TITLES <- FALSE

CMP_DIR <- file.path(ROOT_DIR, "outputs", "comparison")
FIG_DIR <- file.path(CMP_DIR, "figures")
dir.create(FIG_DIR, showWarnings = FALSE, recursive = TRUE)

cmp_csv <- file.path(CMP_DIR, "student_teacher_station_compare.csv")
if (!file.exists(cmp_csv)) stop("Missing comparison file: ", cmp_csv)
cmp <- readr::read_csv(cmp_csv, show_col_types = FALSE) %>%
  filter(
    is.finite(Longitude), is.finite(Latitude),
    is.finite(student_thr_Air_C), is.finite(teacher_thr_Air_C), is.finite(diff_Air_C)
  )

clim_csv <- file.path(CMP_DIR, "student_teacher_station_compare_by_climate.csv")
if (!file.exists(clim_csv)) stop("Missing climate comparison file: ", clim_csv)
cmp_clim <- readr::read_csv(clim_csv, show_col_types = FALSE) %>%
  filter(
    !is.na(climate_group), climate_group != "",
    is.finite(student_thr_Air_C), is.finite(teacher_thr_Air_C)
  ) %>%
  group_by(climate_group) %>%
  mutate(n_station = n()) %>%
  ungroup()

main_groups <- c("Cfa", "Cwa", "BSk", "Csa", "Dfb", "Cfb")
cmp_clim_main <- cmp_clim %>%
  filter(climate_group %in% main_groups) %>%
  mutate(climate_group = factor(climate_group, levels = main_groups))

facet_labels_main <- cmp_clim_main %>%
  group_by(climate_group) %>%
  summarise(
    n_station = first(n_station),
    r = cor(teacher_thr_Air_C, student_thr_Air_C, use = "complete.obs"),
    .groups = "drop"
  ) %>%
  arrange(match(climate_group, main_groups)) %>%
  mutate(lab = paste0(climate_group, " (n=", n_station, ", r=", sprintf("%.2f", r), ")")) %>%
  { setNames(.$lab, .$climate_group) }

# =========================
# Base map
# =========================
coast <- rnaturalearth::ne_coastline(scale = "medium", returnclass = "sf")
map_bbox <- sf::st_bbox(c(xmin = -180, xmax = 180, ymin = 0, ymax = 90), crs = sf::st_crs(4326))
coast <- suppressWarnings(sf::st_crop(coast, map_bbox))

if (USE_KG_BOUNDARIES) {
  KG_ASCII <- file.path(ROOT_DIR, "data", "Koeppen-Geiger-ASCII.txt")
  if (!file.exists(KG_ASCII)) stop("Missing Köppen-Geiger ASCII file: ", KG_ASCII)

  kg_df <- read.table(KG_ASCII, header = TRUE, sep = "", fill = TRUE) %>%
    dplyr::filter(Lat >= 0, Lat <= 90, Lon >= -180, Lon <= 180)

  kg_cells <- kg_df %>% mutate(
    xmin = Lon - 0.25, xmax = Lon + 0.25,
    ymin = Lat - 0.25, ymax = Lat + 0.25
  )

  polys <- lapply(seq_len(nrow(kg_cells)), function(i) {
    with(kg_cells[i, ], {
      sf::st_polygon(list(matrix(
        c(xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax, xmin, ymin),
        ncol = 2, byrow = TRUE
      )))
    })
  })

  kg_poly <- sf::st_sf(Cls = kg_cells$Cls, geometry = sf::st_sfc(polys, crs = 4326)) %>%
    dplyr::group_by(Cls) %>%
    dplyr::summarise(do_union = TRUE, .groups = "drop") %>%
    sf::st_make_valid()

  kg_boundary <- sf::st_boundary(kg_poly)
  kg_boundary <- sf::st_cast(kg_boundary, "LINESTRING", warn = FALSE)
  kg_boundary <- sf::st_sf(geometry = kg_boundary)
  bbs <- lapply(sf::st_geometry(kg_boundary), sf::st_bbox)
  kg_boundary <- kg_boundary %>% mutate(
    x_span = sapply(bbs, function(bb) as.numeric(bb["xmax"] - bb["xmin"])),
    y_span = sapply(bbs, function(bb) as.numeric(bb["ymax"] - bb["ymin"])),
    y_mid  = sapply(bbs, function(bb) as.numeric((bb["ymax"] + bb["ymin"]) / 2))
  )
  kg_boundary_plot <- kg_boundary %>%
    filter(!(x_span > 120 & y_span < 1.0 & y_mid > 55)) %>%
    select(geometry)

  map_layers <- list(
    geom_sf(data = kg_boundary_plot, color = "gray55", linewidth = 0.28, alpha = 1),
    geom_sf(data = coast, color = "black", linewidth = 0.18)
  )
} else {
  map_layers <- list(geom_sf(data = coast, color = "black", linewidth = 0.18))
}

theme_map_pretty <- function() {
  theme_minimal(base_size = 12) +
    theme(
      axis.text = element_blank(), axis.title = element_blank(), axis.ticks = element_blank(),
      panel.grid = element_blank(),
      panel.border = element_rect(color = "black", fill = NA, linewidth = 0.8),
      panel.background = element_rect(fill = "gray95", color = NA),
      plot.background = element_rect(fill = "white", color = "black", linewidth = 1),
      legend.position = "bottom", legend.direction = "horizontal",
      legend.title = element_text(size = 12, face = "bold"),
      legend.text = element_text(size = 13),
      plot.title = element_text(face = "bold", size = 13),
      plot.margin = grid::unit(rep(6, 4), "pt")
    )
}

save_plot <- function(p, filename, w = 7.2, h = 3.9, dpi = 600) {
  ggsave(filename = file.path(FIG_DIR, filename), plot = p, width = w, height = h, dpi = dpi, bg = "white")
  message("[ok] saved: ", file.path(FIG_DIR, filename))
}

# scales
air_vals <- c(cmp$student_thr_Air_C, cmp$teacher_thr_Air_C)
air_lim <- range(air_vals, na.rm = TRUE)
air_lim <- c(floor(air_lim[1] * 2) / 2, ceiling(air_lim[2] * 2) / 2)
if (any(!is.finite(air_lim)) || air_lim[1] >= air_lim[2]) air_lim <- c(-2.5, 0.5)
air_breaks <- pretty(air_lim, n = 6)

diff_q <- quantile(cmp$diff_Air_C, probs = c(0.025, 0.975), na.rm = TRUE)
diff_lim <- max(abs(diff_q))
diff_lim <- ceiling(diff_lim * 2) / 2
if (!is.finite(diff_lim) || diff_lim <= 0) diff_lim <- 1
diff_breaks <- pretty(c(-diff_lim, diff_lim), n = 7)

# maps
p_student_air <- ggplot() +
  map_layers +
  geom_point(data = cmp, aes(x = Longitude, y = Latitude, color = student_thr_Air_C), size = 0.65, alpha = 0.85) +
  scale_color_viridis_c(option = "viridis", limits = air_lim, breaks = air_breaks, oob = scales::squish, name = "Air threshold (°C)") +
  coord_sf(xlim = c(-180, 180), ylim = c(0, 90), expand = FALSE) +
  labs(title = if (SHOW_TITLES) "Student air-temperature threshold" else NULL) +
  theme_map_pretty()
save_plot(p_student_air, "map_air_threshold_student_distill.png")

p_teacher_air <- ggplot() +
  map_layers +
  geom_point(data = cmp, aes(x = Longitude, y = Latitude, color = teacher_thr_Air_C), size = 0.65, alpha = 0.85) +
  scale_color_viridis_c(option = "viridis", limits = air_lim, breaks = air_breaks, oob = scales::squish, name = "Air threshold (°C)") +
  coord_sf(xlim = c(-180, 180), ylim = c(0, 90), expand = FALSE) +
  labs(title = if (SHOW_TITLES) "Teacher air-temperature threshold" else NULL) +
  theme_map_pretty()
save_plot(p_teacher_air, "map_air_threshold_teacher.png")

p_diff_air <- ggplot() +
  map_layers +
  geom_point(data = cmp, aes(x = Longitude, y = Latitude, color = diff_Air_C), size = 0.70, alpha = 0.90) +
  scale_color_gradient2(low = "#3B6FB6", mid = "white", high = "#B63B3B", midpoint = 0,
                        limits = c(-diff_lim, diff_lim), breaks = diff_breaks, oob = scales::squish,
                        name = "Student - teacher (°C)") +
  coord_sf(xlim = c(-180, 180), ylim = c(0, 90), expand = FALSE) +
  labs(title = if (SHOW_TITLES) "Air-threshold difference map" else NULL) +
  theme_map_pretty()
save_plot(p_diff_air, "map_air_threshold_difference_student_minus_teacher.png")

# faceted scatter for main paper
p_scatter_air_facet <- ggplot(cmp_clim_main, aes(x = teacher_thr_Air_C, y = student_thr_Air_C)) +
  geom_point(alpha = 0.28, size = 0.55, color = "#B63B3B") +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", linewidth = 0.55, color ="#1F4E79") +
  geom_smooth(method = "lm", se = FALSE, linewidth = 0.55, color = "black") +
  facet_wrap(~ climate_group, ncol = 3, scales = "fixed", labeller = as_labeller(facet_labels_main)) +
  labs(x = "Teacher air threshold (°C)", y = "Student air threshold (°C)") +
  theme_minimal(base_size = 13) +
  theme(
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.6),
    panel.grid.minor = element_blank(),
    strip.text = element_text(face = "bold", size = 14),
    axis.title = element_text(size = 13),
    plot.background = element_rect(fill = "white", color = "black", linewidth = 1)
  )

save_plot(p_scatter_air_facet, "scatter_air_student_vs_teacher_by_climate_main.png", w = 8.8, h = 6.4)

message("Done.")
