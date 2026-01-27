"""
functions.py
=============

Python execution layer for the reproducible Nix + rixpress pipeline.

This module provides:
- Data encoding
- Feature preprocessing
- Model training
- Prediction
- Evaluation data generation
- Plot generation
- Artifact encoders for rixpress/Nix

Design goals:
- Pure, deterministic functions where possible
- Explicit inputs/outputs (pipeline-friendly)
- No hidden state
- Robust behavior on bad inputs and edge cases
"""

import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# Encoders / I/O (for rixpress)
# ============================================================

def write_to_csv(df, path):
    """
    rixpress encoder: write a DataFrame to EXACTLY `path`.

    Purpose:
        Convert a Python object into a CSV file artifact that can be
        consumed by other pipeline stages (e.g. R).

    Inputs:
        df : pandas.DataFrame or array-like
            Data to be written.
        path : str
            Exact output path (no automatic ".csv" extension added).

    Behavior:
        - Converts input to DataFrame if needed
        - Creates parent directories if required (skips makedirs("") case)
        - Writes CSV deterministically
        - Returns the output path

    Returns:
        str : path to generated CSV file
    """
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    df.to_csv(path, index=False)
    return path


def copy_file(src_path, out_path):
    """
    rixpress encoder: copy a produced file into EXACTLY `out_path`.

    Purpose:
        Convert a locally generated file (PNG, CSV, etc.) into a
        Nix derivation artifact.

    Inputs:
        src_path : str
            Path to existing source file.
        out_path : str
            Target path inside Nix derivation output.

    Returns:
        str : path to copied file inside Nix store
    """
    dirpath = os.path.dirname(out_path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    shutil.copyfile(src_path, out_path)
    return out_path


# ============================================================
# Plot makers (produce local files)
# ============================================================

def make_target_dist_png(encoded_df, target_col="target", out_path="target_dist.png"):
    """
    Generate target distribution plot and save as PNG.

    Inputs:
        encoded_df : pandas.DataFrame
        target_col : str
        out_path : str

    Returns:
        str : path to generated PNG file
    """
    dirpath = os.path.dirname(out_path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    plt.figure(figsize=(8, 6))

    # Future-proof seaborn: palette without hue is deprecated.
    # Use hue=x and legend=False to keep same visual behavior.
    sns.countplot(
        x=encoded_df[target_col],
        hue=encoded_df[target_col],
        palette="Set2",
        legend=False
    )

    plt.title("Heart Disease Distribution")
    plt.xlabel("Heart Disease (0 = No, 1 = Yes)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path


def make_correlation_heatmap_png(encoded_df, out_path="correlation_heatmap.png"):
    """
    Generate feature correlation heatmap and save as PNG.

    Edge case handling:
        If the dataframe has no numeric columns (or correlation matrix is empty),
        produce a valid PNG with an informative message instead of crashing.

    Inputs:
        encoded_df : pandas.DataFrame
        out_path : str

    Returns:
        str : path to generated PNG file
    """
    dirpath = os.path.dirname(out_path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    plt.figure(figsize=(12, 8))
    numeric_df = encoded_df.select_dtypes(include=["number"])

    if numeric_df.empty:
        plt.axis("off")
        plt.title("Feature Correlation Heatmap")
        plt.text(
            0.5, 0.5,
            "No numeric columns available for correlation.",
            ha="center", va="center"
        )
        plt.tight_layout()
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        return out_path

    corr = numeric_df.corr()
    if corr.shape[0] == 0:
        plt.axis("off")
        plt.title("Feature Correlation Heatmap")
        plt.text(
            0.5, 0.5,
            "Correlation matrix is empty.",
            ha="center", va="center"
        )
        plt.tight_layout()
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        return out_path

    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path


# ============================================================
# ML pipeline steps
# ============================================================

def encode_categoricals(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical (object dtype) columns into numeric codes.

    Inputs:
        raw_df : pandas.DataFrame

    Returns:
        pandas.DataFrame : encoded dataset (object columns replaced with integer codes)
    """
    out = raw_df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].astype("category").cat.codes
    return out


def make_processed_data(encoded_df: pd.DataFrame, target_col: str = "target"):
    """
    Scale X and create train/test split.

    Inputs:
        encoded_df : pandas.DataFrame
        target_col : str

    Returns:
        dict with keys:
            - X_train
            - X_test
            - y_train
            - y_test

    Raises:
        KeyError: if target_col not found
        ValueError: if target has < 2 classes (stratify requires ≥2 classes)
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    if target_col not in encoded_df.columns:
        raise KeyError(f"Target column '{target_col}' not found in encoded_df.")

    X = encoded_df.drop(columns=[target_col])
    y = encoded_df[target_col]

    # Explicit validation: stratified split and classification require ≥2 classes.
    unique = pd.Series(y).dropna().unique()
    if len(unique) < 2:
        raise ValueError(
            f"Target column '{target_col}' must contain at least 2 classes; got {len(unique)}."
        )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )

    return {"X_train": X_tr, "X_test": X_te, "y_train": y_tr, "y_test": y_te}


def train_svm_rbf(processed_data, C: float = 1.0):
    """
    Train an RBF-kernel Support Vector Machine classifier.

    Inputs:
        processed_data : dict (output of make_processed_data)
        C : float

    Returns:
        sklearn.svm.SVC : trained model
    """
    from sklearn.svm import SVC

    model = SVC(kernel="rbf", C=C, probability=True, random_state=42)
    model.fit(processed_data["X_train"], processed_data["y_train"])
    return model


def predict_labels(model, processed_data):
    """
    Predict labels on the test set.

    Inputs:
        model : trained sklearn model
        processed_data : dict

    Returns:
        numpy.ndarray : predicted labels
    """
    return model.predict(processed_data["X_test"])


def compute_accuracy(processed_data, y_pred):
    """
    Compute prediction accuracy.

    Inputs:
        processed_data : dict containing y_test
        y_pred : array-like

    Returns:
        float : accuracy score
    """
    from sklearn.metrics import accuracy_score
    return float(accuracy_score(processed_data["y_test"], y_pred))


def make_evaluation_df(processed_data, y_pred):
    """
    Build an evaluation dataframe used by R.

    Inputs:
        processed_data : dict containing y_test
        y_pred : array-like predictions

    Returns:
        pandas.DataFrame with columns:
            - truth
            - estimate
    """
    return pd.DataFrame({"truth": processed_data["y_test"], "estimate": y_pred})

