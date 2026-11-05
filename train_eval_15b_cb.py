import json
import os
from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve

BASE_DIR = r"C:\Users\Bogdan\Documents\AI-spam-detector"

TRAIN_PATH = os.path.join(BASE_DIR, "features_15b_train.jsonl")
VAL_PATH = os.path.join(BASE_DIR, "features_15b_val.jsonl")
TEST_PATH = os.path.join(BASE_DIR, "features_15b_test.jsonl")
MIXED_PATH = os.path.join(BASE_DIR, "features_15b_mixed_test.jsonl")

LLM_COLS = [
    "lr_global",
    "lrr_global",
    "fd_global",
    "lr_win_max",
    "lr_win_mean",
    "lr_win_std",
    "lr_win_q90",
    "lrr_win_max",
    "lrr_win_mean",
    "lrr_win_std",
    "fd_win_max",
    "fd_win_mean",
    "fd_win_std",
    "merged_span_lr",
    "merged_span_ratio",
]

STYLO_COLS = [
    "avg_sent_len",
    "sent_len_var",
    "short_sent_ratio",
    "root_ttr",
    "word_len_var",
    "comma_rate",
    "dash_rate",
    "excl_quest_rate",
    "colon_semi_rate",
]

ALL_FEATURES = LLM_COLS + STYLO_COLS


def compute_char_overlap(
    pred_span: list[int], gt_intervals: list[list[int]]
) -> tuple[float, float]:
  """
    Считает:
    1) Span Precision (Overlap): какая доля нашего выделенного спана попала в реальный AI-интервал.
    2) Span Recall: какую долю реального AI-текста мы накрыли своим спаном.
    """
  p_start, p_end = pred_span
  pred_len = max(p_end - p_start, 0)
  if pred_len == 0 or not gt_intervals:
    return 0.0, 0.0

  inter_len = 0
  gt_total_len = 0
  for g_start, g_end in gt_intervals:
    gt_total_len += max(g_end - g_start, 0)
    s = max(p_start, g_start)
    e = min(p_end, g_end)
    if e > s:
      inter_len += e - s

  precision = inter_len / pred_len
  recall = inter_len / max(gt_total_len, 1)
  return precision, recall


def main():
  df_train = pd.read_json(TRAIN_PATH, lines=True)
  df_val = pd.read_json(VAL_PATH, lines=True)
  df_test = pd.read_json(TEST_PATH, lines=True)
  df_mixed = (
      pd.read_json(MIXED_PATH, lines=True)
      if os.path.exists(MIXED_PATH)
      else None
  )

  print("=" * 90)
  print(
      f"РАЗМЕРЫ ВЫБОРОК: Train={len(df_train)} | Val={len(df_val)} |"
      f" Test={len(df_test)}"
  )
  print("=" * 90)

  thresholds_cfg = {}
  models = {}

  cb_params = dict(
      iterations=1200,
      learning_rate=0.03,
      depth=5,
      l2_leaf_reg=8,
      eval_metric="AUC",
      random_seed=42,
      verbose=0,
  )

  for lang in ["ru", "eng"]:
    print(f"\n==================== ЯЗЫК: {lang.upper()} ====================")
    tr = df_train[df_train["lang"] == lang].copy()
    va = df_val[df_val["lang"] == lang].copy()
    te = df_test[df_test["lang"] == lang].copy()

    y_tr = (tr["label"] == "ai").astype(int).values
    y_va = (va["label"] == "ai").astype(int).values
    y_te = (te["label"] == "ai").astype(int).values

    # 1. Ablation Study на 3 наборах признаков
    for exp_name, cols in [
        ("1. Только Qwen-1.5B (15 фичей)", LLM_COLS),
        ("2. Только Стилометрия (9 фичей)", STYLO_COLS),
        ("3. Полный ансамбль 1.5B + Stylo (24 фичи)", ALL_FEATURES),
    ]:
      X_tr = tr[cols].fillna(0).values
      X_va = va[cols].fillna(0).values
      X_te = te[cols].fillna(0).values

      model = CatBoostClassifier(**cb_params)
      model.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=100)

      val_probs = model.predict_proba(X_va)[:, 1]
      test_probs = model.predict_proba(X_te)[:, 1]

      val_auc = roc_auc_score(y_va, val_probs)
      test_auc = roc_auc_score(y_te, test_probs)

      # Калибровка порогов СТРОГО на Validation
      fpr_v, tpr_v, thresh_v = roc_curve(y_va, val_probs)
      t_strict = float(thresh_v[np.where(fpr_v <= 0.03)[0][-1]])
      t_soft = float(thresh_v[np.where(fpr_v <= 0.10)[0][-1]])

      print(
          f"\n--- {exp_name} | Val AUC: {val_auc:.4f} | Test AUC:"
          f" {test_auc:.4f} ---"
      )
      for mode, th in [
          ("Strict (Val FPR<=3%)", t_strict),
          ("Soft   (Val FPR<=10%)", t_soft),
      ]:
        preds = (test_probs >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_te, preds).ravel()
        rec = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        prec = tp / max(tp + fp, 1)
        f1 = f1_score(y_te, preds)
        print(
            f"  {mode} (T={th:.3f}) -> Recall: {rec:6.2%} | FPR: {fpr:5.2%} |"
            f" Precision: {prec:6.2%} | F1: {f1:.3f}"
        )

      # Сохраняем финальную модель полного ансамбля
      if exp_name.startswith("3."):
        model_path = os.path.join(BASE_DIR, f"cb_{lang}.cbm")
        model.save_model(model_path)
        models[lang] = model
        thresholds_cfg[lang] = {"t_strict": t_strict, "t_soft": t_soft}

        importances = model.get_feature_importance()
        top_idx = np.argsort(importances)[::-1][:10]
        print(f"\n  Топ-10 признаков ({lang.upper()}):")
        for idx in top_idx:
          print(f"    {ALL_FEATURES[idx]:<18}: {importances[idx]:5.2f}%")
        llm_total_imp = sum(
            importances[ALL_FEATURES.index(c)] for c in LLM_COLS
        )
        print(f"  >> Суммарный вклад Qwen-1.5B в модель: {llm_total_imp:.2f}%")

  # Сохраняем пороги
  thresh_path = os.path.join(BASE_DIR, "cb_thresholds.json")
  with open(thresh_path, "w", encoding="utf-8") as f:
    json.dump(thresholds_cfg, f, indent=4)
  print(f"\n✓ Модели и пороги сохранены в {BASE_DIR}")

  # 2. Оценка на реальных смешанных текстах из LLMTrace_detection
  if df_mixed is not None and len(df_mixed) > 0:
    print("\n" + "=" * 90)
    print(
        "ТЕСТ НА РЕАЛЬНЫХ СМЕШАННЫХ ТЕКСТАХ (LLMTrace_detection,"
        " label='mixed')"
    )
    print("=" * 90)
    for lang in ["ru", "eng"]:
      sub_m = df_mixed[df_mixed["lang"] == lang].copy()
      if len(sub_m) == 0:
        continue

      X_m = sub_m[ALL_FEATURES].fillna(0).values
      probs_m = models[lang].predict_proba(X_m)[:, 1]

      t_strict = thresholds_cfg[lang]["t_strict"]
      t_soft = thresholds_cfg[lang]["t_soft"]

      strict_caught = int(np.sum(probs_m >= t_strict))
      soft_caught = int(np.sum(probs_m >= t_soft))

      any_hit = 0
      strong_hit = 0
      recalls = []

      for _, row in sub_m.iterrows():
        p_span = row.get("pred_span_chars", [0, 0])
        gt_ints = row.get("ai_char_intervals", [])
        prec, rec = compute_char_overlap(p_span, gt_ints)
        if prec > 0.0:
          any_hit += 1
        if prec >= 0.30 or rec >= 0.50:
          strong_hit += 1
        recalls.append(rec)

      n = len(sub_m)
      print(f"\n[{lang.upper()}] Всего смешанных текстов: {n}")
      print(
          f"  Глобальный отлов Strict (Block)  : {strict_caught}/{n}"
          f" ({strict_caught/n:.1%})"
      )
      print(
          f"  Глобальный отлов Soft   (Warning): {soft_caught}/{n}"
          f" ({soft_caught/n:.1%})"
      )
      print(
          f"  Касание AI-интервала (Overlap>0) : {any_hit}/{n} ({any_hit/n:.1%})"
      )
      print(
          "  Точная локализация спана         :"
          f" {strong_hit}/{n} ({strong_hit/n:.1%}) | Средний Recall спана:"
          f" {np.mean(recalls):.1%}"
      )


if __name__ == "__main__":
  main()