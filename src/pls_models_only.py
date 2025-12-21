# src/pls_models_only.py
import os
import glob
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from config import META_COLORS

MAX_COMPONENTS = 15


def _mse(y_true, y_pred) -> float:
    return float(mean_squared_error(y_true, y_pred))


def _choose_best_components_cv(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
    max_components: int = MAX_COMPONENTS,
) -> Tuple[int, Dict[str, float]]:
    """
    Chọn n_components bằng K-fold CV trên TRAIN.
    Tiêu chí: MSE nhỏ nhất.
    """
    n_samples, n_features = X.shape
    max_comp = min(max_components, n_features, max(1, n_samples - 1))

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    best_n = 1
    best_mse = float("inf")
    best_metrics = {"mse": None, "mae": None, "r2": None}

    for n_comp in range(1, max_comp + 1):
        mses, maes, r2s = [], [], []

        for tr_idx, val_idx in kf.split(X):
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]

            pipe = Pipeline([
                ("scale", StandardScaler()),
                ("pls", PLSRegression(n_components=n_comp)),
            ])
            pipe.fit(X_tr, y_tr)
            pred_val = pipe.predict(X_val).ravel()

            mses.append(_mse(y_val, pred_val))
            maes.append(float(mean_absolute_error(y_val, pred_val)))
            r2s.append(float(r2_score(y_val, pred_val)))

        mse_mean = float(np.mean(mses))
        if mse_mean < best_mse:
            best_mse = mse_mean
            best_n = n_comp
            best_metrics = {
                "mse": mse_mean,
                "mae": float(np.mean(maes)),
                "r2": float(np.mean(r2s)),
            }

    return int(best_n), best_metrics


def pls_per_phone_table(
    dir_path: str,
    meta_path: str = None,
    test_size: float = 0.2,
    n_splits: int = 5,
    random_state: int = 42,
) -> List[Dict[str, Any]]:
    """
    Đọc từng file features_*{phone_norm}*.csv trong dir_path.
    Split train/test.
    CV 5-fold trên train để chọn n_components.
    Fit final và đánh giá test.
    Trả về bảng row theo format bạn cần.
    """
    meta_path = meta_path or META_COLORS
    df_meta = pd.read_csv(meta_path)
    df_meta["norm"] = df_meta["Phones"].str.lower()
    norms = df_meta["norm"].unique()
    phone_map = {n: df_meta.loc[df_meta["norm"] == n, "Phones"].iloc[0] for n in norms}

    rows: List[Dict[str, Any]] = []

    for norm in norms:
        display = phone_map[norm]
        pattern = os.path.join(dir_path, f"features_*{norm}*.csv")
        files = glob.glob(pattern)

        if not files:
            print(f"[PLS] Skip {display}: no file matched {pattern}")
            continue

        df = pd.read_csv(files[0])
        if df.empty or "ppm" not in df.columns:
            print(f"[PLS] Skip {display}: invalid CSV (empty or no ppm)")
            continue

        X = df.drop(columns=["id_img", "ppm"], errors="ignore").values.astype(float)
        y = df["ppm"].values.astype(float)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        best_n, val_metrics = _choose_best_components_cv(
            X_tr, y_tr, n_splits=n_splits, random_state=random_state
        )

        final_pipe = Pipeline([
            ("scale", StandardScaler()),
            ("pls", PLSRegression(n_components=best_n)),
        ])
        final_pipe.fit(X_tr, y_tr)

        pred_te = final_pipe.predict(X_te).ravel()
        test_mse = _mse(y_te, pred_te)
        test_mae = float(mean_absolute_error(y_te, pred_te))
        test_r2 = float(r2_score(y_te, pred_te))

        print(
            f"[PLS][{display}] best_n={best_n} | "
            f"Val(MSE)={val_metrics['mse']:.4f} | "
            f"Test: MSE={test_mse:.4f}, MAE={test_mae:.4f}, R2={test_r2:.4f}"
        )

        rows.append({
            "Model": "PLS",
            "Phone": display,
            "Validate in": "test set",
            "MSE": round(test_mse, 4),
            "MAE": round(test_mae, 4),
            "R2": round(test_r2, 4),
        })

    # sort theo Phone cho đẹp
    rows = sorted(rows, key=lambda r: r["Phone"])
    return rows
