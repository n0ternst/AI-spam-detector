import json
import re
from pathlib import Path

INPUT_FILE = "drweb_test_dataset.jsonl"
OUTPUT_FILE = "drweb_test_dataset_dedup.jsonl"

def extract_first_word(subject: str) -> str:
    """Извлекает первое значимое слово темы, отсекая спецсимволы и эмодзи."""
    if not subject:
        return "__empty__"
    words = re.findall(r"\w+", subject.strip().lower())
    return words[0] if words else "__empty__"

def main():
    src_path = Path(INPUT_FILE)
    if not src_path.exists():
        print(f"[Ошибка] Файл {INPUT_FILE} не найден.")
        return

    seen_signatures = set()
    total_count = 0
    kept_count = 0
    duplicates_by_category = {}

    with open(src_path, "r", encoding="utf-8") as src, open(OUTPUT_FILE, "w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            
            total_count += 1
            record = json.loads(line)
            
            category = record.get("category", "unknown")
            first_word = extract_first_word(record.get("subject", ""))
            word_count = record.get("word_count", 0)
            
            # Составная сигнатура: первое слово темы + точное количество слов
            signature = (first_word, word_count)
            
            if signature in seen_signatures:
                duplicates_by_category[category] = duplicates_by_category.get(category, 0) + 1
                continue
            
            seen_signatures.add(signature)
            dst.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept_count += 1

    print("================ ИТОГИ ДЕДУПЛИКАЦИИ ================")
    print(f"Всего записей на входе:   {total_count}")
    print(f"Уникальных сохранено:      {kept_count}")
    print(f"Выкошено дубликатов:       {total_count - kept_count}")
    print("Удалено по категориям:")
    for cat, count in duplicates_by_category.items():
        print(f"  - {cat}: {count}")
    print(f"Результат сохранен в:      {OUTPUT_FILE}")

if __name__ == "__main__":
    main()