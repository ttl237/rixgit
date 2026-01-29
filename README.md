# Reproducible R + Python SVM Pipeline with Nix, rix & rixpress

## Overview

This project is a **fully reproducible, polyglot machine learning pipeline** built with:

* **R** (analysis, visualization, evaluation)
* **Python** (data processing, ML training, plotting)
* **Nix** (reproducible system + language environments)
* **rix / rixpress** (pipeline orchestration)
* **Git** (versioned, auditable history)

The pipeline trains an **SVM (RBF kernel)** classifier on a heart disease dataset and produces:

* Processed datasets  
* A trained SVM model  
* Predictions  
* Evaluation data  
* Accuracy metric  
* Confusion matrix  
* Multiple visualizations (PNG plots)

All steps are defined declaratively and executed reproducibly — meaning **any collaborator can reproduce the exact same results, bit-for-bit**, on any machine.

---

## Project Philosophy

This project demonstrates a production-grade scientific workflow based on:

* **Reproducible environments** → Nix  
* **Reproducible logic** → functional programming  
* **Reproducible history** → Git  
* **Reproducible execution** → rix + rixpress  
* **Reproducible validation** → unit tests (pytest + testthat)  

No notebooks. No hidden state. No manual execution order. No version ambiguity.

Everything is:

✔ declarative  
✔ deterministic  
✔ testable  
✔ automatable  
✔ reproducible  

---

## Repository Structure

```
rixgit/
│
├── data/                    # Input dataset
│   └── HeartDiseaseTrain-Test.csv
│
├── functions.py             # Python ML + plotting functions
├── functions.R              # R visualization + encoders
│
├── gen-env.R                # Environment specification (rix → Nix)
├── default.nix              # Auto-generated reproducible environment
│
├── gen-pipeline.R           # Pipeline definition (rixpress)
├── pipeline.nix             # Auto-generated pipeline build spec
│
├── tests/                   # Unit tests
│   ├── test_functions.py    # pytest (Python)
│   └── testthat/            # testthat (R)
│       └── test-functions.R
│
├── pytest.ini                # pytest configuration
├── README.md                 # Project documentation
└── _rixpress/                # rixpress internal state
```
### Step 2 — Build the environment

```bash
nix-build
nix-shell

### Step 3 — Generate environment (once)

```r
source("gen-env.R")

## Technology Stack

### R side

* rix  
* rixpress  
* ggplot2  
* yardstick  
* dplyr  
* testthat  

### Python side

* numpy  
* pandas  
* scikit-learn  
* matplotlib  
* seaborn  
* pytest  

### System

* Nix  
* Git  

---

## How It Works

There are two core scripts:

### `gen-env.R`

Defines the **entire execution environment** using `{rix}`.

This generates `default.nix`, which specifies:

* OS-level dependencies  
* R version  
* Python version  
* All R packages  
* All Python packages  
* System libraries  

This guarantees **bit-for-bit reproducibility**.

---

### `gen-pipeline.R`

Defines the **entire computational pipeline** using `{rixpress}`.

Each step is explicitly declared as:

* `rxp_py()` → Python step  
* `rxp_r()` → R step  

Each step:

* Has explicit inputs  
* Has explicit outputs  
* Produces named artifacts  
* Is language-agnostic in dependency chaining  

Artifacts flow automatically across languages.

---

## Pipeline Outputs

After running the pipeline, the main artifacts include:

### Data artifacts

* `raw_df`  
* `encoded_df`  
* `processed_data`  
* `evaluation_df`  

### Model artifacts

* `svm_rbf_model`  

### Prediction artifacts

* `y_pred`  

### Evaluation artifacts

* `accuracy`  
* `confusion_matrix`  

### Plot artifacts

* `target_dist_plot_png`  
* `correlation_heatmap_png`  
* `confusion_matrix_plot_png`  

---

## Running the Pipeline (Reproducibly)

### Step 1 — Clone the repository

```bash
git clone git@github.com:ttl237/rixgit.git
cd rixgit
