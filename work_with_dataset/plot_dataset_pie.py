import json
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path

# Укажи путь к обучающему или тестовому файлу
INPUT_FILE = "text_train_calibration.jsonl"  # или text_test_features.jsonl
OUTPUT_IMG = "dataset_distribution_pie.png"

def main():
    if not Path(INPUT_FILE).exists():
        print(f"[Ошибка] Файл не найден: {INPUT_FILE}")
        return

    categories = Counter()
    total = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            
            lang = str(data.get("lang", "unknown")).upper()
            label = str(data.get("label", "unknown")).lower()
            
            # Приведение к читаемым названиям для легенды
            if label in ("human", "0", 0):
                tag = f"{lang} Human"
            elif label in ("ai", "1", 1):
                tag = f"{lang} AI"
            else:
                tag = f"{lang} Other"
                
            categories[tag] += 1
            total += 1

    if total == 0:
        print("[!] В файле нет данных.")
        return

    print(f"Всего писем: {total}")
    for cat, count in categories.items():
        print(f"  • {cat}: {count} ({count/total:.1%})")

    # Сортировка для аккуратного отображения
    labels = list(categories.keys())
    counts = [categories[l] for l in labels]

    # Цветовая палитра: холодные оттенки для Human, теплые/акцентные для AI
    color_map = {
        "RU Human": "#2b5c8f",
        "ENG Human": "#4a90e2",
        "RU AI": "#d9534f",
        "ENG AI": "#f0ad4e"
    }
    colors = [color_map.get(l, "#95a5a6") for l in labels]

    # Форматирование подписей: проценты + количество
    def make_autopct(values):
        def my_autopct(pct):
            val = int(round(pct * total / 100.0))
            return f"{pct:.1f}%\n({val} шт.)"
        return my_autopct

    # Отрисовка бублика (Donut chart)
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(aspect="equal"))

    wedges, texts, autotexts = ax.pie(
        counts,
        labels=labels,
        autopct=make_autopct(counts),
        pctdistance=0.75,
        startangle=140,
        colors=colors,
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2),
        textprops=dict(color="#222222", fontsize=11, fontweight="medium")
    )

    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_weight("bold")
        autotext.set_color("white")

    ax.set_title(
        f"Распределение датасета (Всего: {total} сэмплов)",
        fontsize=14,
        fontweight="bold",
        pad=20
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_IMG, dpi=300)
    plt.close()
    print(f"\n✓ График сохранён в: {OUTPUT_IMG}")

if __name__ == "__main__":
    main()