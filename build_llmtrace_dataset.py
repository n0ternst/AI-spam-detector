import os
import random
import pandas as pd
from datasets import load_dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLASSIFICATION_DS = "iitolstykh/LLMTrace_classification"
DETECTION_DS = "iitolstykh/LLMTrace_detection"

# Квоты на каждую из 4 корзин: (ru_human, ru_ai, eng_human, eng_ai)
QUOTAS = {
    "train": 1500,  # 1500 * 4 = 6000 писем
    "val": 300,     # 300 * 4  = 1200 писем
    "test": 300     # 300 * 4  = 1200 писем
}

def sample_balanced_split(hf_split, quota_per_bucket: int, used_topics: set) -> list[dict]:
    buckets = {
        ("ru", "human"): [],
        ("ru", "ai"): [],
        ("eng", "human"): [],
        ("eng", "ai"): [],
    }
    
    shuffled = hf_split.shuffle(seed=42)
    for row in shuffled:
        lang = row.get("lang")
        label = row.get("label")
        text = row.get("text")
        topic_id = row.get("topic_id")

        if (lang, label) not in buckets:
            continue
        if not isinstance(text, str) or len(text.strip()) < 60:
            continue
        # Защита от пересечения топиков между сплитами
        if topic_id and topic_id in used_topics:
            continue

        if len(buckets[(lang, label)]) < quota_per_bucket:
            buckets[(lang, label)].append({
                "text": text.strip(),
                "label": label,
                "lang": lang,
                "data_type": row.get("data_type", ""),
                "model": row.get("model", ""),
                "topic_id": topic_id or ""
            })
            if topic_id:
                used_topics.add(topic_id)

        if all(len(v) >= quota_per_bucket for v in buckets.values()):
            break

    flat = [item for sublist in buckets.values() for item in sublist]
    random.shuffle(flat)
    for k, v in buckets.items():
        print(f"    Корзина {k[0]}_{k[1]}: собрано {len(v)} шт.")
    return flat

def main():
    random.seed(42)
    os.makedirs(BASE_DIR, exist_ok=True)

    print(f"1. Загрузка {CLASSIFICATION_DS}...")
    ds_cls = load_dataset(CLASSIFICATION_DS)
    splits_available = list(ds_cls.keys())
    print(f"Доступные сплиты: {splits_available}")

    used_topics = set()

    # Определяем имена сплитов в датасете
    val_split_name = next((s for s in ["validation", "val", "dev"] if s in splits_available), None)
    test_split_name = "test" if "test" in splits_available else None

    if val_split_name and test_split_name:
        split_mapping = [
            ("train", ds_cls["train"], QUOTAS["train"]),
            ("val", ds_cls[val_split_name], QUOTAS["val"]),
            ("test", ds_cls[test_split_name], QUOTAS["test"]),
        ]
        for name, hf_sp, quota in split_mapping:
            print(f"\nСборка [{name.upper()}] (цель: {quota * 4} шт.)...")
            data = sample_balanced_split(hf_sp, quota, used_topics)
            out_path = os.path.join(BASE_DIR, f"raw_{name}.jsonl")
            pd.DataFrame(data).to_json(out_path, orient="records", lines=True, force_ascii=False)
            print(f"✓ Сохранено {len(data)} записей в {out_path}")
    else:
        # Если в датасете вдруг только один сплит — режем строго без пересечения topic_id
        print("\nНарезаем train / val / test из общего пула без пересечения topic_id...")
        for name, quota in QUOTAS.items():
            print(f"\nСборка [{name.upper()}] (цель: {quota * 4} шт.)...")
            data = sample_balanced_split(ds_cls["train"], quota, used_topics)
            out_path = os.path.join(BASE_DIR, f"raw_{name}.jsonl")
            pd.DataFrame(data).to_json(out_path, orient="records", lines=True, force_ascii=False)
            print(f"✓ Сохранено {len(data)} записей в {out_path}")

    # 2. Собираем отдельный тестовый сет смешанных текстов из LLMTrace_detection
    print(f"\n2. Загрузка {DETECTION_DS} для теста смешанных текстов (mixed)...")
    ds_det = load_dataset(DETECTION_DS)
    det_split = ds_det["test"] if "test" in ds_det else ds_det["train"]
    
    mixed_buckets = {"ru": [], "eng": []}
    for row in det_split.shuffle(seed=42):
        lang = row.get("lang")
        label = row.get("label")
        intervals = row.get("ai_char_intervals")
        text = row.get("text")

        if lang in mixed_buckets and label == "mixed" and intervals and isinstance(text, str) and len(text) > 150:
            if len(mixed_buckets[lang]) < 100:
                mixed_buckets[lang].append({
                    "text": text,
                    "lang": lang,
                    "label": "mixed",
                    "ai_char_intervals": intervals,
                    "prompt_type": row.get("prompt_type", ""),
                    "model": row.get("model", "")
                })
        if len(mixed_buckets["ru"]) >= 100 and len(mixed_buckets["eng"]) >= 100:
            break

    mixed_flat = mixed_buckets["ru"] + mixed_buckets["eng"]
    mixed_path = os.path.join(BASE_DIR, "raw_mixed_test.jsonl")
    pd.DataFrame(mixed_flat).to_json(mixed_path, orient="records", lines=True, force_ascii=False)
    print(f"✓ Сохранено {len(mixed_flat)} mixed-записей с интервалами -> {mixed_path}")

if __name__ == "__main__":
    main()