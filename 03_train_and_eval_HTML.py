"""
03_train_and_eval.py
Обучение бустинга, изотоническая калибровка, подбор порога и расчет метрик.
Вход: data/splits/train.csv, val.csv, test.csv
Выход: weights/html_forensic_classifier.joblib, metrics/html_model_metrics.json
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score, f1_score,
    fbeta_score, precision_score, recall_score, brier_score_loss, confusion_matrix
)

SEED = 42
DATA_DIR = "data"
SPLITS_DIR = os.path.join(DATA_DIR, "splits")
WEIGHTS_DIR = "weights"
METRICS_DIR = "metrics"

os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)


def calc_ece(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for low, high in zip(bins[:-1], bins[1:]):
        mask = (probs >= low) & (probs < high)
        if np.any(mask):
            acc = np.mean(y_true[mask])
            conf = np.mean(probs[mask])
            ece += np.abs(acc - conf) * (np.sum(mask) / len(y_true))
    return float(ece)


def main():
    train_file = os.path.join(SPLITS_DIR, "train.csv")
    val_file = os.path.join(SPLITS_DIR, "val.csv")
    test_file = os.path.join(SPLITS_DIR, "test.csv")

    for f_path in (train_file, val_file, test_file):
        if not os.path.exists(f_path):
            print(f"Файл {f_path} не найден. Сначала выполните 02_extract_features.py")
            sys.exit(1)

    print("Загрузка табличных выборок...")
    df_train = pd.read_csv(train_file)
    df_val = pd.read_csv(val_file)
    df_test = pd.read_csv(test_file)

    feature_cols = [c for c in df_train.columns if c not in ("id", "label")]

    X_train, y_train = df_train[feature_cols].values, df_train["label"].values
    X_val, y_val = df_val[feature_cols].values, df_val["label"].values
    X_test, y_test = df_test[feature_cols].values, df_test["label"].values

    print(f"Размерности: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

    # 1. Базовый классификатор HistGradientBoosting
    print("Обучение HistGradientBoostingClassifier...")
    base_hgb = HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.045,
        max_leaf_nodes=45,
        min_samples_leaf=18,
        l2_regularization=1.2,
        random_state=SEED,
    )

    # 2. Калибровка вероятностей (5-fold CV Isotonic)
    print("Изотоническая калибровка вероятностей (5-fold CV)...")
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    calibrated_model = CalibratedClassifierCV(estimator=base_hgb, method="isotonic", cv=cv_strategy)
    calibrated_model.fit(X_train, y_train)

    # 3. Подбор оптимального порога строго на Validation (максимум F0.5 при FPR <= 6%)
    val_probs = calibrated_model.predict_proba(X_val)[:, 1]
    best_th = 0.52
    best_score = -1.0

    for th in np.linspace(0.40, 0.70, 150):
        preds_v = (val_probs >= th).astype(int)
        tn_v, fp_v, fn_v, tp_v = confusion_matrix(y_val, preds_v).ravel()
        fpr_v = fp_v / max(1, fp_v + tn_v)
        f05_v = fbeta_score(y_val, preds_v, beta=0.5, zero_division=0)
        f1_v = f1_score(y_val, preds_v, zero_division=0)
        score_v = 0.55 * f1_v + 0.45 * f05_v

        if fpr_v <= 0.060 and score_v > best_score:
            best_score = score_v
            best_th = float(th)

    print(f"Оптимальный порог (найден на Val): {best_th:.4f}")

    # 4. Сохранение модели и весов
    model_path = os.path.join(WEIGHTS_DIR, "html_forensic_classifier.joblib")
    bundle = {
        "model": calibrated_model,
        "feature_names": feature_cols,
        "threshold": best_th,
    }
    joblib.dump(bundle, model_path)
    print(f"Веса модели сохранены в: {model_path}")

    # 5. Финальная оценка на закрытой тестовой выборке
    test_probs = calibrated_model.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= best_th).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, test_preds).ravel()
    fpr = float(fp / max(1, fp + tn))
    fnr = float(fn / max(1, fn + tp))

    metrics = {
        "model": "HistGradientBoosting + 5-fold Isotonic Calibration",
        "threshold": round(best_th, 4),
        "test_samples": len(y_test),
        "roc_auc": round(float(roc_auc_score(y_test, test_probs)), 4),
        "pr_auc": round(float(average_precision_score(y_test, test_probs)), 4),
        "accuracy": round(float(accuracy_score(y_test, test_preds)), 4),
        "f1_score": round(float(f1_score(y_test, test_preds)), 4),
        "f05_score": round(float(fbeta_score(y_test, test_preds, beta=0.5)), 4),
        "precision": round(float(precision_score(y_test, test_preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, test_preds)), 4),
        "specificity": round(1.0 - fpr, 4),
        "fpr": round(fpr, 4),
        "fnr": round(fnr, 4),
        "brier_loss": round(float(brier_score_loss(y_test, test_probs)), 4),
        "ece": round(calc_ece(y_test, test_probs), 4),
    }

    metrics_out = os.path.join(METRICS_DIR, "html_model_metrics.json")
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 55)
    print("ИТОГОВЫЕ МЕТРИКИ НА ТЕСТОВОЙ ВЫБОРКЕ (TEST SET):")
    print("=" * 55)
    for k, v in metrics.items():
        print(f"  {k:<16}: {v}")
    print(f"\nОтчет сохранен: {metrics_out}")


if __name__ == "__main__":
    main()
