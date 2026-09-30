import json
import re

INPUT_FILE = "drweb_test_dataset_dedup.jsonl"
OUTPUT_FILE = "drweb_clean_100.jsonl"

# Базовые маркеры немецкого/французского, чтобы не путать с английским
NON_TARGET_STOPWORDS = {
    "und", "der", "die", "das", "nicht", "sie", "ich", "mit", "für", "auf", "ein", "eine", # DE
    "les", "des", "pour", "dans", "avec", "sur", "une", "est", "sont"                      # FR
}

def is_clean_sample(record: dict) -> bool:
    text = record.get("text", "")
    words = text.lower().split()
    
    # 1. Проверка на мусорные символы кодировок
    if "\ufffd" in text:
        return False
        
    # 2. Отсев немецкого/французского спама (если более 3% стоп-слов из других языков)
    if record.get("lang") == "eng":
        non_target_hits = sum(1 for w in words if w in NON_TARGET_STOPWORDS)
        if len(words) > 0 and (non_target_hits / len(words)) > 0.03:
            return False
            
    # 3. Базовый фильтр по длине: от 35 до 800 слов (отсекаем огромные дампы)
    if not (35 <= len(words) <= 800):
        return False
        
    return True

def main():
    cleaned_rows = []
    
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if is_clean_sample(row):
                cleaned_rows.append(row)
                
    print(f"После дополнительной фильтрации осталось: {len(cleaned_rows)} строк.")
    
    # Сохраняем итоговый датасет (до 100-110 штук)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for row in cleaned_rows[:110]:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            
    print(f"Финальный файл сохранён в: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()