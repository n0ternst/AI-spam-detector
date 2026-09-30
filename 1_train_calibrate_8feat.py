import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

TRAIN_PATH = "C:\\Users\\Bogdan\\Documents\\AI-spam-detector\\text_train_features.jsonl"
df_train = pd.read_json(TRAIN_PATH, lines=True)

# 1. Быстрый CPU-экстрактор 2 устойчивых стилометрических признаков
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

print("Обогащение обучающего датасета признаками root_ttr и word_len_var...")
extra_feats = df_train['text'].apply(extract_robust_features)
df_train['root_ttr'] = [f[0] for f in extra_feats]
df_train['word_len_var'] = [f[1] for f in extra_feats]

# 8 итоговых признаков без утечек форматирования
FEATURES = [
    'bino_min', 'bino_mean', 'bino_var',
    'avg_sent_len', 'sent_len_var', 'short_sent_ratio',
    'root_ttr', 'word_len_var'
]

for lang in ['ru', 'eng']:
    sub = df_train[df_train['lang'] == lang].copy()
    X = sub[FEATURES].fillna(0).values
    y = (sub['label'] == 'ai').astype(int).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_scaled, y)

    y_prob = clf.predict_proba(X_scaled)[:, 1]
    auc = roc_auc_score(y, y_prob)

    fpr, tpr, thresholds = roc_curve(y, y_prob)

    # 1. Жесткий порог (FPR <= 1%)
    idx_strict = np.where(fpr <= 0.01)[0][-1]
    t_strict = thresholds[idx_strict]
    y_pred_s = (y_prob >= t_strict).astype(int)
    tn_s, fp_s, fn_s, tp_s = confusion_matrix(y, y_pred_s).ravel()

    # 2. Мягкий порог (FPR <= 5%)
    idx_soft = np.where(fpr <= 0.05)[0][-1]
    t_soft = thresholds[idx_soft]
    y_pred_m = (y_prob >= t_soft).astype(int)
    tn_m, fp_m, fn_m, tp_m = confusion_matrix(y, y_pred_m).ravel()

    # Денормализация весов под сырые данные
    raw_w = clf.coef_[0] / scaler.scale_
    raw_b = clf.intercept_[0] - np.sum((clf.coef_[0] * scaler.mean_) / scaler.scale_)

    print(f"\n==================== [TRAIN] КАЛИБРОВКА {lang.upper()} (8 ФИЧЕЙ) ====================")
    print(f"ROC-AUC: {auc:.4f}")
    print(f"--- ЖЕСТКИЙ ПОРОГ (T = {t_strict:.4f}) ---")
    print(f"FPR: {fp_s/(fp_s+tn_s):.2%} | Recall: {tp_s/(tp_s+fn_s):.2%} | TN={tn_s}, FP={fp_s}, FN={fn_s}, TP={tp_s}")
    print(f"--- МЯГКИЙ ПОРОГ   (T = {t_soft:.4f}) ---")
    print(f"FPR: {fp_m/(fp_m+tn_m):.2%} | Recall: {tp_m/(tp_m+fn_m):.2%} | TN={tn_m}, FP={fp_m}, FN={fn_m}, TP={tp_m}")

    print(f"\nСкопируй эти веса во второй скрипт:")
    print(f"W_{lang.upper()} = np.array({list(np.round(raw_w, 5))})")
    print(f"B_{lang.upper()} = {round(float(raw_b), 5)}")
    print(f"T_STRICT_{lang.upper()} = {round(float(t_strict), 4)}")
    print(f"T_SOFT_{lang.upper()} = {round(float(t_soft), 4)}")

    # Построение графика распределения на трейне
    plt.figure(figsize=(8, 4))
    plt.hist(y_prob[y == 0], bins=35, alpha=0.6, color='tab:blue', label='Human', density=True)
    plt.hist(y_prob[y == 1], bins=35, alpha=0.6, color='tab:red', label='AI', density=True)
    plt.axvline(t_strict, color='black', linestyle='--', linewidth=2, label=f'Strict (T={t_strict:.2f})')
    plt.axvline(t_soft, color='orange', linestyle='--', linewidth=2, label=f'Soft (T={t_soft:.2f})')
    plt.title(f"Train P(AI) [{lang.upper()}] | 8 Features | ROC-AUC: {auc:.3f}")
    plt.xlabel("P(AI)")
    plt.ylabel("Плотность")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"train_dist_8feat_{lang}.png", dpi=300, bbox_inches='tight')
    plt.show()