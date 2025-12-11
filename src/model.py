import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Ensure the parent directory is in the Python path for module import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.classifiers import get_classifiers
from models.regressors import get_regressors
from config import DATA_DIR, META_COLORS

def train_models(
    meta_path: str = None,
    dir_path: str = None,
    out_path: str = None,
    n_estimators: int = 100,
    n_splits: int = 5,
) -> None:
    if dir_path is None:
        dir_path = os.path.join(DATA_DIR, 'csv')
    if meta_path is None:
        meta_path = META_COLORS
    if out_path is None:
        out_path = os.path.join(DATA_DIR, 'models')
    os.makedirs(out_path, exist_ok=True)

    df = pd.read_csv(meta_path)
    lst_phone = df['Phones'].unique().tolist()

    classifiers = get_classifiers(n_estimators=n_estimators)
    results = []

    for phone in lst_phone:
        clf_path = os.path.join(dir_path, f'clf_{phone}.csv')
        if not os.path.exists(clf_path):
            print(f"Missing file: {clf_path}, skipping")
            continue

        df_phone = pd.read_csv(clf_path)
        if df_phone['type'].nunique() < 2:
            print(f"Not enough classes for {phone}, skipping")
            continue

        X = df_phone.drop(columns=['id_img', 'type'])
        y = df_phone['type']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        for name, model in classifiers.items():
            print(f"\nCross-validating {name} for {phone}...")

            acc_scores = []

            for train_idx, test_idx in skf.split(X_scaled, y_encoded):
                X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
                y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                acc_scores.append(acc)

            avg_acc = np.mean(acc_scores)
            std_acc = np.std(acc_scores)

            # Save final model (trained on full data)
            model.fit(X_scaled, y_encoded)
            joblib.dump(model, os.path.join(out_path, f'{name}_model_{phone}.pkl'))
            joblib.dump(scaler, os.path.join(out_path, f'{name}_scaler_{phone}.pkl'))
            joblib.dump(label_encoder, os.path.join(out_path, f'{name}_label_encoder_{phone}.pkl'))

            print(f"Average Accuracy (CV): {avg_acc:.4f} ± {std_acc:.4f}")

            y_full_pred = model.predict(X_scaled)
            y_full_pred_str = label_encoder.inverse_transform(y_full_pred)
            y_true_str = label_encoder.inverse_transform(y_encoded)

            report = classification_report(y_true_str, y_full_pred_str)
            f1_macro = f1_score(y_true_str, y_full_pred_str, average='macro')
            print(report)

            results.append({
                'phone': phone,
                'model': name,
                'accuracy': avg_acc,
                'std': std_acc,
                'f1_macro': f1_macro
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(out_path, 'classification_summary.csv'), index=False)

    # Export LaTeX table
    latex_table = results_df.pivot(index='model', columns='phone', values='accuracy').round(4).to_latex()
    with open(os.path.join(out_path, 'classification_summary_table.tex'), 'w') as f:
        f.write(latex_table)

    print("\nAll models cross-validated. Summary saved to classification_summary.csv and LaTeX table exported.")

def train_regressors(
    meta_path: str = None,
    dir_path: str = None,
    out_path: str = None,
    n_estimators: int = 100,
    n_splits: int = 5,
    test_size: float = 0.2,
) -> None:
    """
    Train regression models for PPM prediction per phone type.
    """
    if dir_path is None:
        dir_path = os.path.join(DATA_DIR, 'csv')
    if meta_path is None:
        meta_path = META_COLORS
    if out_path is None:
        out_path = os.path.join(DATA_DIR, 'models')
    os.makedirs(out_path, exist_ok=True)

    df = pd.read_csv(meta_path)
    lst_phone = df['Phones'].unique().tolist()

    regressors = get_regressors(n_estimators=n_estimators)
    results = []

    for phone in lst_phone:
        rgs_path = os.path.join(dir_path, f'rgs_{phone}.csv')
        if not os.path.exists(rgs_path):
            print(f"Missing file: {rgs_path}, skipping")
            continue

        df_phone = pd.read_csv(rgs_path)
        if 'ppm' not in df_phone.columns or df_phone.empty:
            print(f"Invalid data for {phone}, skipping")
            continue

        X = df_phone.drop(columns=['id_img', 'ppm'], errors='ignore')
        y = df_phone['ppm']

        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Cross-validation setup
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        X_full_scaled = scaler.fit_transform(X)

        for name, model in regressors.items():
            print(f"\nCross-validating {name} for {phone}...")

            mse_scores = []
            mae_scores = []
            r2_scores = []

            for train_idx, val_idx in kf.split(X_full_scaled):
                X_tr, X_val = X_full_scaled[train_idx], X_full_scaled[val_idx]
                y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_val)
                
                mse_scores.append(mean_squared_error(y_val, y_pred))
                mae_scores.append(mean_absolute_error(y_val, y_pred))
                r2_scores.append(r2_score(y_val, y_pred))

            avg_mse = np.mean(mse_scores)
            avg_mae = np.mean(mae_scores)
            avg_r2 = np.mean(r2_scores)
            std_mse = np.std(mse_scores)

            print(f"Average MSE (CV): {avg_mse:.4f} ± {std_mse:.4f}")
            print(f"Average MAE (CV): {avg_mae:.4f}")
            print(f"Average R² (CV): {avg_r2:.4f}")

            # Train final model on full training set
            model.fit(X_train_scaled, y_train)
            y_test_pred = model.predict(X_test_scaled)
            
            test_mse = mean_squared_error(y_test, y_test_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            test_r2 = r2_score(y_test, y_test_pred)

            print(f"Test MSE: {test_mse:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}")

            # Save model and scaler
            joblib.dump(model, os.path.join(out_path, f'{name}_regressor_{phone}.pkl'))
            joblib.dump(scaler, os.path.join(out_path, f'{name}_regressor_scaler_{phone}.pkl'))

            results.append({
                'phone': phone,
                'model': name,
                'cv_mse': avg_mse,
                'cv_mae': avg_mae,
                'cv_r2': avg_r2,
                'test_mse': test_mse,
                'test_mae': test_mae,
                'test_r2': test_r2
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(out_path, 'regression_summary.csv'), index=False)

    # Export LaTeX table
    latex_table = results_df.pivot(index='model', columns='phone', values='test_r2').round(4).to_latex()
    with open(os.path.join(out_path, 'regression_summary_table.tex'), 'w') as f:
        f.write(latex_table)

    print("\nAll regressors cross-validated. Summary saved to regression_summary.csv and LaTeX table exported.")

if __name__ == '__main__':
    train_models(meta_path=META_COLORS)
    train_regressors(meta_path=META_COLORS)
