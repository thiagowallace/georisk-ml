# GeoRisk ML

<p align="center">
  <strong>End-to-End Machine Learning pipeline for landslide susceptibility analysis using geospatial data, remote sensing and spatial validation.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue" />
  <img src="https://img.shields.io/badge/scikit--learn-Machine%20Learning-orange" />
  <img src="https://img.shields.io/badge/GeoPandas-Geospatial-green" />
  <img src="https://img.shields.io/badge/Rasterio-Raster%20Processing-green" />
  <img src="https://img.shields.io/badge/pytest-111%20tests-brightgreen" />
  <img src="https://img.shields.io/badge/Status-Stage%204%20Completed-blueviolet" />
</p>

---

## Overview

**GeoRisk ML** is an applied Data Science project that combines **Machine Learning, geospatial data processing, remote sensing and spatial validation** to support landslide susceptibility analysis.

The project was designed as an **End-to-End Machine Learning workflow**, covering the complete path from raw geospatial data to model evaluation and, in the next stage, susceptibility mapping.

The current pilot area is **Santa Tereza, Rio Grande do Sul, Brazil**, one of the regions affected by the extreme rainfall and landslide events that occurred in the state in 2024.

The project is currently at **Stage 4 — Spatial Model Validation**.

---

## Project Goals

The main goal is to build a reproducible pipeline capable of:

- processing landslide inventory data;
- integrating multiple environmental datasets;
- performing geospatial feature engineering;
- creating a supervised Machine Learning dataset;
- performing exploratory data analysis;
- training classification models;
- evaluating model generalization;
- controlling spatial dependence during validation;
- generating landslide susceptibility maps;
- extending the workflow to other affected areas in Rio Grande do Sul.

The project also explores an important challenge in geospatial Machine Learning:

> **How much does model performance change when spatial dependence between training and validation data is explicitly controlled?**

---

# Project Pipeline

```text
Landslide Inventory
        ↓
Geospatial ETL
        ↓
Pilot Study Area
        ↓
Environmental Feature Engineering
        ↓
Supervised ML Dataset
        ↓
Exploratory Data Analysis
        ↓
Baseline Machine Learning Models
        ↓
Spatial Cross-Validation
        ↓
Spatial Sensitivity Analysis
        ↓
Landslide Susceptibility Mapping
        ↓
Testing in additional areas of Rio Grande do Sul
```

---

# Study Area

The current pilot area is:

**Santa Tereza — Rio Grande do Sul, Brazil**

The project uses a **30 m spatial grid** referenced to:

```text
SIRGAS 2000 / UTM Zone 22S
EPSG:31982
```

Santa Tereza was selected as the initial area for the development and validation of the workflow.

Future stages are expected to apply the same pipeline to additional regions affected by the 2024 extreme rainfall events in Rio Grande do Sul.

---

# Data Sources

The project integrates different types of geospatial and remote sensing data.

## Landslide Inventory

The starting point is the landslide inventory related to the **May 2024 extreme rainfall event in Rio Grande do Sul**.

Initial statewide dataset:

```text
16,862 landslide initiation points
```

After ETL and spatial quality control:

```text
16,861 valid initiation points
```

For the Santa Tereza pilot area:

```text
275 original landslide occurrences
270 unique positive raster cells
```

Multiple landslide points falling inside the same 30 m raster cell are represented by a single Machine Learning sample while preserving traceability.

---

## Environmental Variables

The current model uses environmental predictors derived from:

### Copernicus DEM GLO-30

Terrain variables:

- Elevation
- Slope
- Aspect
- Plan curvature
- Profile curvature

### Sentinel-2

Pre-event vegetation information:

- NDVI

The pre-event NDVI composite was generated from Sentinel-2 imagery acquired before the May 2024 disaster.

### MapBiomas

Land use and land cover:

- MapBiomas 2023 land-cover classes

---

# Geospatial Feature Engineering

All environmental layers were aligned to a common grid:

```text
CRS: EPSG:31982
Spatial resolution: 30 × 30 m
Raster dimensions: 408 × 570
```

Environmental variables were extracted only where the joint valid-data mask was available.

The final predictors used by the models are:

```text
elevation
slope
aspect_sin
aspect_cos
plan_curvature
profile_curvature
ndvi_pre_event
land_cover
```

Raw aspect is transformed into:

```text
sin(aspect)
cos(aspect)
```

to properly represent its circular nature.

Spatial coordinates and identifiers are preserved for auditing and spatial validation but are **not used as predictive features**.

---

# Machine Learning Dataset

The supervised dataset was constructed using one observation per valid 30 m raster cell.

## Positive samples

```text
275 original landslide occurrences
↓
270 unique positive cells
```

## Background samples

A background / pseudo-absence sampling strategy was implemented.

Background cells are not interpreted as confirmed absence of landslides.

A minimum exclusion distance from mapped landslide occurrences was applied during sampling.

```text
Positive cells:    270
Background cells:  270
Total samples:     540
```

The current dataset is intentionally balanced for model development.

Because of this sampling strategy, model outputs should not automatically be interpreted as calibrated municipal landslide probabilities.

---

# Exploratory Data Analysis

The EDA stage included:

- class distribution analysis;
- descriptive statistics;
- feature distribution comparison;
- boxplots;
- outlier analysis;
- Pearson correlation;
- Spearman correlation;
- land-cover distribution;
- circular analysis of aspect;
- class-specific environmental comparison.

No feature pair presented correlation above:

```text
|r| ≥ 0.80
```

Among the continuous predictors, **slope presented the strongest univariate separation between landslide and background samples**.

Other relevant variables included:

- elevation;
- terrain orientation;
- NDVI;
- land cover.

Outliers were identified for analysis but were **not automatically removed**.

---

# Baseline Models

Two baseline classification algorithms were implemented:

### Logistic Regression

Preprocessing:

```text
StandardScaler
+
OneHotEncoder for land_cover
```

### Random Forest

Preprocessing:

```text
Continuous variables passed directly
+
OneHotEncoder for land_cover
```

No hyperparameter optimization was performed during the baseline stage.

---

# Preliminary Random Holdout Evaluation

The first evaluation used a stratified random split:

```text
Training: 75%
Testing:  25%
Random state: 42
```

Results:

| Model | Accuracy | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.807 | 0.881 | 0.819 | **0.887** |
| Random Forest | 0.837 | 0.851 | 0.838 | **0.896** |

These results were treated only as **preliminary baselines**.

Random train/test splitting may produce optimistic performance estimates for spatial datasets because nearby observations can share similar environmental characteristics.

Therefore, the next stage focused on spatial validation.

---

# Spatial Cross-Validation

To provide a more realistic evaluation of spatial generalization, the project implements spatial cross-validation using:

```text
StratifiedGroupKFold
```

The study area was divided into square spatial blocks.

Configuration:

```text
Block size: 1,500 × 1,500 m
Cells per block: 50 × 50
Occupied blocks: 42
Number of folds: 5
Random state: 42
```

All samples from the same spatial block remain in the same fold.

This prevents spatial blocks from being simultaneously used for training and validation.

Preprocessing is fitted **only on the training samples inside each fold**, avoiding data leakage.

---

# Spatial Validation Results

Mean performance across the five spatial folds:

| Model | Accuracy | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | **0.776** | **0.798** | **0.768** | **0.860** |
| Random Forest | 0.755 | 0.745 | 0.741 | 0.848 |

The spatial evaluation produced lower performance than the random holdout.

This behavior is expected because spatial validation represents a more demanding generalization scenario.

For example:

```text
Logistic Regression ROC-AUC

Random holdout:     0.887
Spatial validation: 0.860
```

```text
Random Forest ROC-AUC

Random holdout:     0.896
Spatial validation: 0.848
```

The difference demonstrates why spatial validation is important when evaluating geospatial Machine Learning models.

---

# Spatial Sensitivity Analysis

An additional sensitivity analysis was implemented using a:

```text
300 m training-validation buffer
```

The purpose was not to optimize model performance.

Instead, the objective was to evaluate whether the main conclusions remained stable when training observations close to validation areas were removed.

The original validation folds were preserved.

Only training observations located less than 300 m from validation samples were removed.

---

## Results with the 300 m Buffer

### Logistic Regression

```text
ROC-AUC

Spatial CV:          0.8600
300 m buffer:        0.8572
Difference:         -0.0028
```

```text
F1

Spatial CV:          0.7676
300 m buffer:        0.7618
Difference:         -0.0058
```

### Random Forest

```text
ROC-AUC

Spatial CV:          0.8476
300 m buffer:        0.8459
Difference:         -0.0017
```

```text
F1

Spatial CV:          0.7414
300 m buffer:        0.7267
Difference:         -0.0147
```

The average ROC-AUC changed only slightly under the stricter spatial separation.

This indicates that the main qualitative conclusions of the spatial validation remained relatively stable.

However, some individual folds showed greater variation, particularly in recall.

---

# Model Interpretation

The project intentionally avoids treating one model as universally superior.

In the current spatial validation:

**Logistic Regression presented higher mean:**

- accuracy;
- balanced accuracy;
- recall;
- F1;
- ROC-AUC.

**Random Forest presented slightly higher mean:**

- PR-AUC;
- average precision.

The comparison therefore depends on the evaluation metric and the intended application.

---

# Data Science Stack

The project combines traditional Data Science tools with geospatial processing.

## Machine Learning and Data Analysis

```text
Python
Pandas
NumPy
SciPy
scikit-learn
Matplotlib
```

Techniques include:

- Exploratory Data Analysis;
- supervised classification;
- feature engineering;
- categorical encoding;
- feature scaling;
- Logistic Regression;
- Random Forest;
- ROC-AUC;
- Precision-Recall analysis;
- confusion matrices;
- grouped cross-validation;
- spatial validation;
- model reproducibility.

---

## Geospatial Processing

```text
GeoPandas
Rasterio
Shapely
QGIS
ArcGIS Pro
```

Applications include:

- vector processing;
- raster processing;
- coordinate reference systems;
- spatial masks;
- raster alignment;
- environmental variable extraction;
- satellite imagery;
- DEM processing;
- spatial sampling.

---

## Remote Sensing

Data sources and techniques include:

```text
Sentinel-2
Copernicus DEM GLO-30
MapBiomas
NDVI
Terrain analysis
```

---

## Development and Reproducibility

```text
Git
GitHub
pytest
Python virtual environments
AWS S3
```

The project follows a modular structure separating:

- data extraction;
- transformations;
- feature engineering;
- models;
- tests;
- documentation;
- generated outputs.

---

# Automated Testing

Testing has been an important part of the project since the data preparation stage.

The test suite currently contains:

```text
111 passing tests
```

Tests cover areas such as:

- dataset integrity;
- spatial grid consistency;
- pseudo-absence sampling;
- reproducibility;
- feature generation;
- preprocessing;
- model training;
- prediction probability validation;
- train/test leakage;
- spatial block separation;
- spatial buffering;
- model metrics;
- output validation.

The objective is to keep each stage reproducible and auditable.

---

# Repository Structure

```text
georisk-ml/
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── docs/
│   ├── dataset_stage3.md
│   ├── ml_eda_stage3.md
│   ├── baseline_models_stage3.md
│   ├── spatial_validation_stage4.md
│   └── spatial_validation_buffer300.md
│
├── outputs/
│   ├── eda/
│   ├── figures/
│   └── models/
│       ├── baseline/
│       ├── spatial_validation/
│       └── spatial_validation_buffer300/
│
├── scripts/
│
├── src/
│   ├── extract/
│   ├── transform/
│   ├── features/
│   └── models/
│
├── tests/
│
├── requirements.txt
└── README.md
```

Some datasets and generated artifacts are intentionally excluded from Git because of file size, reproducibility or storage considerations.

---

# Current Project Status

## Stage 1 — Geospatial ETL

- [x] Landslide inventory ingestion
- [x] Spatial quality control
- [x] Duplicate handling
- [x] CRS standardization
- [x] Pilot study area extraction

---

## Stage 2 — Environmental Feature Engineering

- [x] Copernicus DEM
- [x] Elevation
- [x] Slope
- [x] Aspect
- [x] Plan curvature
- [x] Profile curvature
- [x] MapBiomas land cover
- [x] Sentinel-2 pre-event NDVI
- [x] Common raster grid
- [x] Joint valid-data mask

---

## Stage 3 — Machine Learning Dataset and Baselines

- [x] Positive-cell aggregation
- [x] Pseudo-absence sampling
- [x] Machine Learning dataset
- [x] Exploratory Data Analysis
- [x] Logistic Regression
- [x] Random Forest
- [x] Preliminary random holdout evaluation

---

## Stage 4 — Spatial Validation

- [x] Spatial blocks
- [x] Stratified grouped cross-validation
- [x] Model evaluation by spatial fold
- [x] Comparison with random holdout
- [x] 300 m spatial-buffer sensitivity analysis
- [x] Automated validation tests

---

## Stage 5 — Susceptibility Mapping

Current next stage:

- [ ] Train final model
- [ ] Generate predictions for valid raster cells
- [ ] Produce susceptibility score raster
- [ ] Evaluate spatial patterns
- [ ] Produce final cartographic visualization
- [ ] Document the complete End-to-End pipeline
- [ ] Test workflow in additional areas of Rio Grande do Sul

---

# Important Methodological Note

The Machine Learning dataset uses a balanced case/background sampling strategy.

Therefore:

> model outputs are not automatically equivalent to the real probability of a landslide occurring at a given location.

The first final raster product will therefore be interpreted as a:

**landslide susceptibility score**

rather than a calibrated territorial probability.

Additional calibration would be required before interpreting model scores as real-world occurrence probabilities.

---

# Next Step

The next development stage will transform the validated Machine Learning workflow into a spatial product.

The models will be applied to valid environmental raster cells across the study area to generate a continuous:

**landslide susceptibility surface**

for Santa Tereza.

After validating the complete workflow, the project is expected to be tested in additional regions affected by the **2024 extreme rainfall events in Rio Grande do Sul**.

---

# Why This Project?

GeoRisk ML was designed not only as a geospatial analysis project, but also as an applied **Data Science and Machine Learning portfolio project**.

It combines:

```text
Data Engineering
        +
Data Analysis
        +
Feature Engineering
        +
Machine Learning
        +
Model Validation
        +
Software Testing
        +
Geospatial Analytics
        +
Remote Sensing
```

The main focus is building a reproducible analytical workflow where Machine Learning models are evaluated considering the spatial structure of the data.

---

# Author

**Thiago Wallace**

Cartographic and Surveying Engineer | Data Scientist | Geospatial Data Science

- LinkedIn: [linkedin.com/in/thiagowallace](https://www.linkedin.com/in/thiagowallace)
- GitHub: [github.com/thiagowallace](https://github.com/thiagowallace)

---

## Project Status

> **Stage 4 completed — Spatial Validation** ✅  
> **Next: Stage 5 — Landslide Susceptibility Mapping** 🚀
