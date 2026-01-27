import os
import pandas as pd
import numpy as np
import pytest

# Headless backend for CI/Nix builds
import matplotlib
matplotlib.use("Agg")

from functions import (
    write_to_csv,
    copy_file,
    make_target_dist_png,
    make_correlation_heatmap_png,
    encode_categoricals,
    make_processed_data,
    train_svm_rbf,
    predict_labels,
    compute_accuracy,
    make_evaluation_df,
)

# ----------------------------
# Helpers
# ----------------------------

def make_raw_df(n=40, seed=42):
    """Raw-ish dataset with one categorical col and a balanced binary target."""
    rng = np.random.default_rng(seed)
    # Ensure balanced binary target for stratify
    y = np.array([0, 1] * (n // 2))
    df = pd.DataFrame({
        "age": rng.integers(30, 80, size=n),
        "chol": rng.integers(150, 300, size=n),
        "sex": rng.choice(["M", "F"], size=n),
        "target": y
    })
    return df

def make_encoded_df(n=40):
    return encode_categoricals(make_raw_df(n=n))


def assert_is_png(path):
    """Lightweight PNG format check (signature bytes)."""
    with open(path, "rb") as f:
        sig = f.read(8)
    assert sig == b"\x89PNG\r\n\x1a\n"


# ============================================================
# 1) Encoders / I/O tests
# ============================================================

def test_write_to_csv_happy_path(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    out = tmp_path / "out.csv"

    returned = write_to_csv(df, str(out))

    assert returned == str(out)
    assert out.exists()
    loaded = pd.read_csv(out)
    pd.testing.assert_frame_equal(loaded, df)

def test_write_to_csv_accepts_non_dataframe(tmp_path):
    data = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    out = tmp_path / "out.csv"

    returned = write_to_csv(data, str(out))

    assert returned == str(out)
    assert out.exists()
    loaded = pd.read_csv(out)
    assert list(loaded.columns) == ["a", "b"]
    assert loaded.shape == (2, 2)

def test_write_to_csv_no_dirpath_does_not_crash(tmp_path, monkeypatch):
    """
    Edge case: path has no directory component. Should not call makedirs("").
    """
    monkeypatch.chdir(tmp_path)
    df = pd.DataFrame({"x": [1]})

    returned = write_to_csv(df, "file.csv")

    assert returned == "file.csv"
    assert (tmp_path / "file.csv").exists()

def test_copy_file_happy_path(tmp_path):
    src = tmp_path / "src.bin"
    dst = tmp_path / "dst.bin"
    src.write_bytes(b"hello")

    returned = copy_file(str(src), str(dst))

    assert returned == str(dst)
    assert dst.exists()
    assert dst.read_bytes() == b"hello"

def test_copy_file_bad_input_missing_source(tmp_path):
    dst = tmp_path / "dst.bin"
    with pytest.raises(FileNotFoundError):
        copy_file("does_not_exist.bin", str(dst))


# ============================================================
# 2) Plot generator tests
# ============================================================

def test_make_target_dist_png_happy_path(tmp_path):
    df = make_encoded_df(n=40)
    out = tmp_path / "target.png"

    returned = make_target_dist_png(df, target_col="target", out_path=str(out))

    assert returned == str(out)
    assert out.exists()
    assert_is_png(out)

def test_make_target_dist_png_bad_target_col(tmp_path):
    df = make_encoded_df(n=40)
    out = tmp_path / "target.png"

    with pytest.raises(KeyError):
        make_target_dist_png(df, target_col="not_a_col", out_path=str(out))

def test_make_correlation_heatmap_png_happy_path(tmp_path):
    df = make_encoded_df(n=40)
    out = tmp_path / "corr.png"

    returned = make_correlation_heatmap_png(df, out_path=str(out))

    assert returned == str(out)
    assert out.exists()
    assert_is_png(out)

def test_make_correlation_heatmap_png_edge_no_numeric_cols(tmp_path):
    """
    Edge case: if df has no numeric columns, corr() will be empty.
    Our function should still not crash (it will produce an empty heatmap).
    """
    df = pd.DataFrame({"a": ["x", "y"], "b": ["u", "v"]})
    out = tmp_path / "corr.png"

    returned = make_correlation_heatmap_png(df, out_path=str(out))

    assert returned == str(out)
    assert out.exists()
    assert_is_png(out)


# ============================================================
# 3) Data / ML pipeline tests
# ============================================================

def test_encode_categoricals_happy_path():
    df = pd.DataFrame({"cat": ["a", "b", "a"], "num": [1, 2, 3]})
    out = encode_categoricals(df)

    assert out.shape == df.shape
    assert out["cat"].dtype != object  # should be integer codes
    assert out["num"].dtype == df["num"].dtype

def test_make_processed_data_happy_path_shapes():
    df = make_encoded_df(n=40)
    proc = make_processed_data(df, target_col="target")

    assert set(proc.keys()) == {"X_train", "X_test", "y_train", "y_test"}
    assert proc["X_train"].shape[0] == proc["y_train"].shape[0]
    assert proc["X_test"].shape[0] == proc["y_test"].shape[0]
    # 70/30 on 40 => 28 train, 12 test
    assert proc["X_train"].shape[0] == 28
    assert proc["X_test"].shape[0] == 12

def test_make_processed_data_bad_input_missing_target():
    df = make_encoded_df(n=40).drop(columns=["target"])
    with pytest.raises(KeyError):
        make_processed_data(df, target_col="target")

def test_make_processed_data_edge_single_class_fails_stratify():
    """
    Edge case: stratify requires at least 2 classes in y.
    sklearn should raise a ValueError.
    """
    df = make_encoded_df(n=40)
    df["target"] = 0  # single class
    with pytest.raises(ValueError):
        make_processed_data(df, target_col="target")

def test_train_predict_happy_path_and_accuracy_range():
    df = make_encoded_df(n=60)
    proc = make_processed_data(df, target_col="target")

    model = train_svm_rbf(proc, C=1.0)
    y_pred = predict_labels(model, proc)

    assert len(y_pred) == len(proc["y_test"])
    acc = compute_accuracy(proc, y_pred)
    assert 0.0 <= acc <= 1.0

def test_make_evaluation_df_schema_and_length():
    df = make_encoded_df(n=60)
    proc = make_processed_data(df, target_col="target")
    model = train_svm_rbf(proc, C=1.0)
    y_pred = predict_labels(model, proc)

    eval_df = make_evaluation_df(proc, y_pred)

    assert isinstance(eval_df, pd.DataFrame)
    assert list(eval_df.columns) == ["truth", "estimate"]
    assert len(eval_df) == len(proc["y_test"])

def test_compute_accuracy_bad_lengths():
    """
    Bad input: y_pred must match y_test length.
    sklearn should raise a ValueError.
    """
    df = make_encoded_df(n=60)
    proc = make_processed_data(df, target_col="target")

    y_pred = np.zeros(len(proc["y_test"]) + 1, dtype=int)
    with pytest.raises(ValueError):
        compute_accuracy(proc, y_pred)
