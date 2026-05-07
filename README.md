# AI-Guided Single-Index Varying-Coefficient Logistic Model for Rain-Snow Partitioning

This repository contains code for reproducing the main analyses and figures for the manuscript:

**AI-Guided Single-Index Varying-Coefficient Logistic Modeling: A Rain-Snow Partitioning Case Study**

The project fits an interpretable VC-Student model for rain-snow partitioning, compares it with a higher-capacity DCut-Teacher model and simple physical threshold rules, evaluates chronological and blocked spatial generalization, and generates manuscript-ready figures.

## Repository structure

```text
vc-student/
|-- data/
|   |-- README.md
|   |-- Koeppen-Geiger-ASCII.txt
|-- figures/
|   |-- README.md
|-- outputs/
|   |-- README.md
|-- scripts/
|   |-- vc_student.py
|   |-- vc_student_spatial_split_blocked.py
|   |-- vc_student_stability.py
|   |-- fix_stability_sign.py
|   |-- comparison/
|   |   |-- compare_simple_rules.py
|   |   |-- export_teacher_all_outputs.py
|   |   |-- build_student_teacher_compare.py
|   |   |-- summarize_correlation_by_climate.py
|   |   |-- make_student_teacher_maps_by_climate_facets.R
|   |-- figures/
|   |   |-- make_all_figures.py
|   |   |-- make_phase_maps_all_figures_plus_spatial_split.R
|   |   |-- make_student_teacher_maps_by_climate_facets.R
|-- requirements.txt
|-- LICENSE
|-- .gitignore
|-- README.md
```

## Data availability

Large raw and generated files are not tracked in GitHub.

Expected local input files:

```text
data/point_data_with_atmospheric_predictors.csv
data/Koeppen-Geiger-ASCII.txt
```

The Koppen-Geiger file is included because it is small and is used to draw climate-zone boundaries. The main modeling data file is large and should be placed locally in `data/` before reproducing the full analysis.

## Software requirements

Python packages are listed in:

```text
requirements.txt
```

Install them with:

```bash
pip install -r requirements.txt
```

The R figure scripts require these R packages:

```text
readr
dplyr
ggplot2
viridis
sf
rnaturalearth
rnaturalearthdata
scales
grid
tidyverse
RColorBrewer
units
pacman
```

## Reproducing the analysis

All commands should be run from the project root.

On Windows PowerShell:

```powershell
Set-Location "path\to\vc-student"
```

### 1. Fit the main VC-Student model

```powershell
python ".\scripts\vc_student.py"
```

### 2. Fit the blocked spatial split model

```powershell
python ".\scripts\vc_student_spatial_split_blocked.py"
```

### 3. Run station-blocked stability analysis

```powershell
python ".\scripts\vc_student_stability.py"
python ".\scripts\fix_stability_sign.py"
```

### 4. Compare against simple physical threshold rules

```powershell
python ".\scripts\comparison\compare_simple_rules.py"
```

### 5. Export teacher outputs and build student-teacher comparisons

```powershell
python ".\scripts\comparison\export_teacher_all_outputs.py"
python ".\scripts\comparison\build_student_teacher_compare.py"
python ".\scripts\comparison\summarize_correlation_by_climate.py"
```

### 6. Generate main manuscript figures

```powershell
python ".\scripts\figures\make_all_figures.py"
```

### 7. Generate climate-boundary maps and spatial split maps

```powershell
& "C:\Program Files\R\R-4.4.3\bin\x64\Rscript.exe" ".\scripts\figures\make_phase_maps_all_figures_plus_spatial_split.R"
```

### 8. Generate student-teacher comparison maps

```powershell
& "C:\Program Files\R\R-4.4.3\bin\x64\Rscript.exe" ".\scripts\figures\make_student_teacher_maps_by_climate_facets.R"
```

## Main generated outputs

The scripts generate files under:

```text
outputs/
figures/
outputs/comparison/figures/
outputs/comparison/tables/
outputs/stability/
```

Generated outputs are ignored by Git because many files are large and can be recreated from the scripts.

## Key reported results

Important manuscript results include:

- Fixed 0 C air-temperature rule: AUC approximately 0.9659.
- Fixed 1.0 C wet-bulb rule: AUC approximately 0.9700.
- VC-Student held-out AUC approximately 0.98.
- Blocked spatial holdout AUC approximately 0.9698.
- Distilled model threshold-index association: Pearson r approximately 0.699.
- Non-distilled model threshold-index association: Pearson r approximately 0.274.
- Student-teacher air-threshold agreement: Pearson r approximately 0.61.

## Notes for reproducibility

The scripts use relative paths. Scripts inside `scripts/figures/` detect the project root by moving two directories up from the script location. This avoids hard-coded local paths inside the code.

Large files should remain local and should not be committed to GitHub.

## License

See `LICENSE`.

