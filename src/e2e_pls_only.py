# src/e2e_pls_only.py
import os
import pandas as pd

from config import DATA_DIR, META_COLORS
from loading import create_meta_data
from processing import process_data
from normalize import getFeature

from pls_models_only import pls_per_phone_table


def ensure_dirs():
    os.makedirs(os.path.join(DATA_DIR, "results"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "models"), exist_ok=True)


def run_preprocess():
    # 1) metadata
    create_meta_data(
        data_dir=os.path.join(DATA_DIR, "full", "HP5_data"),
        out_dir=META_COLORS
    )
    print("✅ Metadata creation Success!")

    # 2) processing images -> data/square image
    process_data(
        data_dir=os.path.join(DATA_DIR, "full", "HP5_data"),
        meta_path=META_COLORS
    )
    print("✅ Processing Data Success!")


def run_feature_variants():
    # ROI
    csv_roi = os.path.join(DATA_DIR, "csv_roi")
    os.makedirs(csv_roi, exist_ok=True)
    getFeature(
        df_path=META_COLORS,
        dir_path=os.path.join(DATA_DIR, "square image"),
        out_path=csv_roi,
        use_square_background=False
    )
    print(f"✅ ROI features saved to: {csv_roi}")

    # Square
    csv_square = os.path.join(DATA_DIR, "csv_square")
    os.makedirs(csv_square, exist_ok=True)
    getFeature(
        df_path=META_COLORS,
        dir_path=os.path.join(DATA_DIR, "square image"),
        out_path=csv_square,
        use_square_background=True
    )
    print(f"✅ Square features saved to: {csv_square}")

    return csv_roi, csv_square


def run_pls_and_export(csv_dir: str, tag: str):
    rows = pls_per_phone_table(
        dir_path=csv_dir,
        test_size=0.2,     # để giống setup bạn đang báo cáo
        n_splits=5,        # đúng “method 1” của repo
        random_state=42
    )
    out_csv = os.path.join(DATA_DIR, "results", f"PLS_table_{tag}.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"✅ Saved table: {out_csv}")


def main():
    ensure_dirs()

    # chạy full preprocess (nếu bạn muốn “1 lệnh từ đầu”)
    run_preprocess()

    # tạo 2 bộ feature
    csv_roi, csv_square = run_feature_variants()

    # chạy PLS cho ROI & Square và xuất bảng
    run_pls_and_export(csv_roi, "ROI")
    run_pls_and_export(csv_square, "SQUARE")

    print("\n🎉 DONE. Check:")
    print(" - data/results/PLS_table_ROI.csv")
    print(" - data/results/PLS_table_SQUARE.csv")


if __name__ == "__main__":
    main()
