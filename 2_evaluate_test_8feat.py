import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix

TEST_PATH = "C:\\Users\\Bogdan\\Documents\\AI-spam-detector\\text_test_features.jsonl"
df_test = pd.read_json(TEST_PATH, lines=True)

# 1. CPU-экстрактор устойчивых признаков
def extract_robust_features(text: str) -> tuple[float, float]:
    if not isinstance(text, str) or len(text.strip()) < 10:
        return 0.0, 0.0
    words = re.findall(r'\b[^\W\d_]+\b', text.lower())
    n_words = len(words)
    if n_words == 0:
        return 0.0, 0.0
    root_ttr = float(len(set(words)) / np.sqrt(n_words))
    word_lengths = [len(w) for w in words]
    word_len_var = float(np.var(word_lengths)) if len(word_lengths) > 1 else 0.0
    return root_ttr, word_len_var

print("Обогащение тестового датасета признаками root_ttr и word_len_var...")
extra_feats = df_test['text'].apply(extract_robust_features)
df_test['root_ttr'] = [f[0] for f in extra_feats]
df_test['word_len_var'] = [f[1] for f in extra_feats]

FEATURES = [
    'bino_min', 'bino_mean', 'bino_var',
    'avg_sent_len', 'sent_len_var', 'short_sent_ratio',
    'root_ttr', 'word_len_var'
]

# Точные калибровочные параметры из train-скрипта
CONFIG = {
    "ru": {
        "weights": np.array([4.46094, -11.41842, 8.33193, -0.00539, -0.00318, -3.62979, 0.34151, 0.02739], dtype=np.float64),
        "bias": 5.09906,
        "t_strict": 0.8472,
        "t_soft": 0.7236
    },
    "eng": {
        "weights": np.array([4.41917, -3.59224, 7.92443, -0.03285, -0.00617, -3.05382, 0.22090, 0.33094], dtype=np.float64),
        "bias": -3.36463,
        "t_strict": 0.8730,
        "t_soft": 0.7628
    }
}

for lang in ['ru', 'eng']:
    sub = df_test[df_test['lang'] == lang].copy()
    X = sub[FEATURES].fillna(0).values
    y_true = (sub['label'] == 'ai').astype(int).values

    cfg = CONFIG[lang]
    logits = np.dot(X, cfg["weights"]) + cfg["bias"]
    logits = np.clip(logits, -20.0, 20.0)
    y_prob = 1.0 / (1.0 + np.exp(-logits))

    auc = roc_auc_score(y_true, y_prob)

    # Жесткий порог
    y_pred_s = (y_prob >= cfg["t_strict"]).astype(int)
    tn_s, fp_s, fn_s, tp_s = confusion_matrix(y_true, y_pred_s).ravel()

    # Мягкий порог
    y_pred_m = (y_prob >= cfg["t_soft"]).astype(int)
    tn_m, fp_m, fn_m, tp_m = confusion_matrix(y_true, y_pred_m).ravel()

    print(f"\n==================== [TEST] ИТОГОВАЯ ВАЛИДАЦИЯ {lang.upper()} ====================")
    print(f"ROC-AUC на тесте: {auc:.4f}")
    print(f"--- ЖЕСТКИЙ ПОРОГ (T = {cfg['t_strict']:.4f}) ---")
    print(f"FPR: {fp_s/(fp_s+tn_s):.2%} (защита почты) | Recall: {tp_s/(tp_s+fn_s):.2%} | Precision: {tp_s/(tp_s+fp_s):.2%}")
    print(f"Матрица: TN={tn_s}, FP={fp_s}, FN={fn_s}, TP={tp_s}")

    print(f"--- МЯГКИЙ ПОРОГ   (T = {cfg['t_soft']:.4f}) ---")
    print(f"FPR: {fp_m/(fp_m+tn_m):.2%} | Recall: {tp_m/(tp_m+fn_m):.2%} | Precision: {tp_m/(tp_m+fp_m):.2%}")
    print(f"Матрица: TN={tn_m}, FP={fp_m}, FN={fn_m}, TP={tp_m}")

    # Построение и сохранение графика
    plt.figure(figsize=(8, 4))
    plt.hist(y_prob[y_true == 0], bins=30, alpha=0.6, color='tab:blue', label='Human (Test)', density=True)
    plt.hist(y_prob[y_true == 1], bins=30, alpha=0.6, color='tab:red', label='AI (Test)', density=True)
    plt.axvline(cfg["t_strict"], color='black', linestyle='--', linewidth=2, label=f'Strict (T={cfg["t_strict"]:.2f})')
    plt.axvline(cfg["t_soft"], color='orange', linestyle='--', linewidth=2, label=f'Soft (T={cfg["t_soft"]:.2f})')
    plt.title(f"Test P(AI) [{lang.upper()}] | 8 Features | ROC-AUC: {auc:.3f}")
    plt.xlabel("P(AI)")
    plt.ylabel("Плотность")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_img = f"test_dist_8feat_{lang}.png"
    plt.savefig(out_img, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"График сохранен в файл: {out_img}")