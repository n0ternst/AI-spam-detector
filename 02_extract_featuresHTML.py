"""
02_extract_features.py
Извлечение 46 признаков DOM-форензики и формирование выборок Train/Val/Test.
Вход: data/raw_dataset.jsonl
Выход: data/splits/train.csv, val.csv, test.csv, html_dataset_splits.zip
"""

import os
import sys
import json
import zipfile
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Поддержка импорта из корня или core/
try:
    from core.html_detector import HTMLDetector
except ImportError:
    from html_detector import HTMLDetector

SEED = 42
DATA_DIR = "data"
RAW_INPUT_PATH = os.path.join(DATA_DIR, "raw_dataset.jsonl")
SPLITS_DIR = os.path.join(DATA_DIR, "splits")
os.makedirs(SPLITS_DIR, exist_ok=True)


def main():
    if not os.path.exists(RAW_INPUT_PATH):
        print(f"Файл {RAW_INPUT_PATH} не найден. Сначала запустите 01_download_and_build_raw.py")
        sys.exit(1)

    detector = HTMLDetector(model_path="none")
    feature_names = detector.FEATURE_NAMES

    print(f"Извлечение {len(feature_names)} признаков из {RAW_INPUT_PATH}...")
    X_rows = []
    y_rows = []
    ids = []

    with open(RAW_INPUT_PATH, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            row = json.loads(line)
            feats, _, _, _ = detector.extract_features(row["raw_html"])
            X_rows.append([feats[k] for k in feature_names])
            y_rows.append(int(row["label"]))
            ids.append(row["id"])

            if (idx + 1) % 500 == 0:
                print(f"  Обработано: {idx + 1} писем...")

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.int32)
    ids = np.array(ids)

    # Стратифицированное разбиение: 70% Train, 15% Validation, 15% Test
    idx_all = np.arange(len(y))
    idx_train, idx_temp = train_test_split(idx_all, test_size=0.30, random_state=SEED, stratify=y)
    idx_val, idx_test = train_test_split(idx_temp, test_size=0.50, random_state=SEED, stratify=y[idx_temp])

    splits = {
        "train": (idx_train, X[idx_train], y[idx_train], ids[idx_train]),
        "val":   (idx_val, X[idx_val], y[idx_val], ids[idx_val]),
        "test":  (idx_test, X[idx_test], y[idx_test], ids[idx_test]),
    }

    for name, (_, x_s, y_s, id_s) in splits.items():
        df_split = pd.DataFrame(x_s, columns=feature_names)
        df_split.insert(0, "id", id_s)
        df_split.insert(1, "label", y_s)
        out_csv = os.path.join(SPLITS_DIR, f"{name}.csv")
        df_split.to_csv(out_csv, index=False, encoding="utf-8")
        print(f"  {name.upper():<6}: {len(df_split)} записей -> {out_csv}")

    # Упаковываем в архив под ТЗ
    zip_path = os.path.join(DATA_DIR, "html_dataset_splits.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in ("train", "val", "test"):
            zf.write(os.path.join(SPLITS_DIR, f"{name}.csv"), arcname=f"{name}.csv")

    print(f"\nАрхив выборок сохранен: {zip_path}")


if __name__ == "__main__":
    main()
