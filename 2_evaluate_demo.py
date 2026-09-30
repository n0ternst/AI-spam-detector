import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix

TEST_PATH = r"C:\Users\Bogdan\Documents\AI-spam-detector\drweb_13_enriched.jsonl"
df_test = pd.read_json(TEST_PATH, lines=True)

# 1. CPU-экстрактор устойчивых признаков (root_ttr и word_len_var)
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

print(f"Загружено записей: {len(df_test)}")
print("Обогащение признаками root_ttr и word_len_var...")
extra_feats = df_test['text'].apply(extract_robust_features)
df_test['root_ttr'] = [f[0] for f in extra_feats]
df_test['word_len_var'] = [f[1] for f in extra_feats]

FEATURES = [
    'bino_min', 'bino_mean', 'bino_var',
    'avg_sent_len', 'sent_len_var', 'short_sent_ratio',
    'root_ttr', 'word_len_var'
]

# Точные калибровочные параметры логистической регрессии
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
    if len(sub) == 0:
        print(f"\n[!] Нет сэмплов для языка: {lang.upper()}")
        continue

    X = sub[FEATURES].fillna(0).values

    # Проверка имени целевой колонки
    if 'is_ai_ground_truth' in sub.columns:
        y_true = sub['is_ai_ground_truth'].astype(int).values
    elif 'label' in sub.columns:
        y_true = (sub['label'] == 'ai').astype(int).values
    elif 'category' in sub.columns:
        y_true = (sub['category'] == 'samples_prompt').astype(int).values
    else:
        y_true = np.zeros(len(sub), dtype=int)

    cfg = CONFIG[lang]
    logits = np.dot(X, cfg["weights"]) + cfg["bias"]
    logits = np.clip(logits, -20.0, 20.0)
    y_prob = 1.0 / (1.0 + np.exp(-logits))
    sub['pred_prob'] = y_prob

    print(f"\n==================== [TEST] ВАЛИДАЦИЯ {lang.upper()} (N = {len(sub)}) ====================")

    # Защищенный расчет AUC
    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_prob)
        print(f"ROC-AUC: {auc:.4f}")
    else:
        auc = float('nan')
        print(f"ROC-AUC: N/A (в выборке только один класс: {np.unique(y_true)})")

    # Безопасный расчет матрицы ошибок для малых выборок
    def calc_metrics(y_t, y_p, thresh):
        pred = (y_p >= thresh).astype(int)
        cm = confusion_matrix(y_t, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        return fpr, recall, precision, (tn, fp, fn, tp)

    fpr_s, rec_s, prec_s, (tn_s, fp_s, fn_s, tp_s) = calc_metrics(y_true, y_prob, cfg["t_strict"])
    fpr_m, rec_m, prec_m, (tn_m, fp_m, fn_m, tp_m) = calc_metrics(y_true, y_prob, cfg["t_soft"])

    print(f"--- ЖЕСТКИЙ ПОРОГ (T = {cfg['t_strict']:.4f}) ---")
    print(f"FPR: {fpr_s:.2%} | Recall: {rec_s:.2%} | Precision: {prec_s:.2%}")
    print(f"Матрица: TN={tn_s}, FP={fp_s}, FN={fn_s}, TP={tp_s}")

    print(f"--- МЯГКИЙ ПОРОГ   (T = {cfg['t_soft']:.4f}) ---")
    print(f"FPR: {fpr_m:.2%} | Recall: {rec_m:.2%} | Precision: {prec_m:.2%}")
    print(f"Матрица: TN={tn_m}, FP={fp_m}, FN={fn_m}, TP={tp_m}")

    # Печать индивидуальных результатов для каждого письма
    print("\nДетализация по письмам:")
    for idx, row in sub.iterrows():
        name = row.get('file_name', f"row_{idx}")
        prob = row['pred_prob']
        label = "AI" if y_true[len(sub) - len(sub.loc[idx:])] == 1 else "HUMAN/OTHER"
        status = "STRICT_TRIGGER" if prob >= cfg["t_strict"] else ("SOFT_SUSPICIOUS" if prob >= cfg["t_soft"] else "CLEAN")
        print(f"  • {name[:12]}... | P(AI) = {prob:.4f} | Истина: {label:<10} | Вердикт: {status}")

    # Построение графика: полосы порогов + точки сэмплов (Scatter/KDE замена для малых выборок)
    plt.figure(figsize=(9, 4.5))
    bins = np.linspace(0, 1, 15)
    
    if np.any(y_true == 0):
        plt.hist(y_prob[y_true == 0], bins=bins, alpha=0.5, color='tab:blue', label=f'Human/Ham (N={(y_true==0).sum()})')
    if np.any(y_true == 1):
        plt.hist(y_prob[y_true == 1], bins=bins, alpha=0.5, color='tab:red', label=f'AI Spam (N={(y_true==1).sum()})')

    # Наложение точек отдельных писем на ось X
    plt.scatter(y_prob, np.zeros_like(y_prob), color='black', s=40, zorder=5, label='Письма выборки')
    
    plt.axvline(cfg["t_strict"], color='black', linestyle='--', linewidth=2, label=f'Strict (T={cfg["t_strict"]:.2f})')
    plt.axvline(cfg["t_soft"], color='orange', linestyle='--', linewidth=2, label=f'Soft (T={cfg["t_soft"]:.2f})')
    
    auc_str = f"ROC-AUC: {auc:.3f}" if not np.isnan(auc) else "Small Sample Run"
    plt.title(f"Dr.Web Test Validation [{lang.upper()}] | {len(sub)} samples | {auc_str}")
    plt.xlabel("Вероятность P(AI)")
    plt.ylabel("Количество писем")
    plt.xlim(-0.05, 1.05)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    out_img = f"drweb_dist_8feat_{lang}.png"
    plt.savefig(out_img, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ График сохранен в файл: {out_img}")