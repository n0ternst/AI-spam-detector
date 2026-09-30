import os
import re
import json
from pathlib import Path
from core.parser import parse_eml  # либо from parser import parse_eml, если лежит рядом

INPUT_DIR = r"C:\Users\Bogdan\Documents\AI-spam-detector\AI_generated_emails"
OUTPUT_FILE = r"C:\Users\Bogdan\Documents\AI-spam-detector\drweb_test_dataset.jsonl"

def validate_language(text: str) -> str | None:
    """Проверяет язык. Допускаются только RU и ENG."""
    cyrillic = len(re.findall(r"[а-яА-ЯёЁ]", text))
    latin = len(re.findall(r"[a-zA-Z]", text))
    total_letters = len(re.findall(r"[^\W\d_]", text))

    if total_letters == 0:
        return None

    # Отсечение китайских/арабских символов и битых кодировок
    if (cyrillic + latin) / total_letters < 0.70:
        return None

    return "ru" if cyrillic >= latin else "eng"

def main():
    base_path = Path(INPUT_DIR)
    if not base_path.exists():
        print(f"[Ошибка] Папка не найдена: {base_path}")
        return

    subfolders = ["samples_gray", "samples_prompt", "samples_spam"]
    dataset = []
    stats = {"total": 0, "accepted": 0, "dropped_short": 0, "dropped_lang": 0, "dropped_corrupt": 0}

    for folder_name in subfolders:
        folder_path = base_path / folder_name
        if not folder_path.exists():
            continue

        eml_files = [p for p in folder_path.rglob("*") if p.is_file()]
        print(f"Сканирование {folder_name}: найдено {len(eml_files)} файлов...")

        for eml_file in eml_files:
            stats["total"] += 1
            try:
                # Используем парсер сокомандника
                parsed = parse_eml(str(eml_file))
            except Exception as e:
                # Выводим подробный лог для первых 3 упавших файлов
                if stats["dropped_corrupt"] < 3:
                    print(f"\n[!] ОШИБКА В ПАРСЕРЕ НА ФАЙЛЕ: {eml_file.name}")
                    import traceback
                    traceback.print_exc()
                stats["dropped_corrupt"] += 1
                continue

            text = parsed["clean_text"]

            # Проверка на кракозябры после фоллбека
            if "\ufffd\ufffd\ufffd" in text:
                stats["dropped_corrupt"] += 1
                continue

            words = text.split()
            # Guardrail: письма короче 30 слов отбрасываем
            if len(words) < 30:
                stats["dropped_short"] += 1
                continue

            lang = validate_language(text)
            if not lang:
                stats["dropped_lang"] += 1
                continue

            # samples_prompt = 1 (ИИ-генерация), samples_spam = 1 (спам), samples_gray = 0 (условно легитимные)
            is_ai_sample = 1 if folder_name == "samples_prompt" else 0

            dataset.append({
                "file_name": eml_file.name,
                "category": folder_name,
                "is_ai_ground_truth": is_ai_sample,
                "lang": lang,
                "word_count": len(words),
                "subject": parsed["subject"],
                "text": text,
                "raw_html": parsed["raw_html"],
                "attachments": parsed["attachments"],
                "has_images": parsed["media_flags"]["has_images"],
                "eml_path": str(eml_file)
            })
            stats["accepted"] += 1

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("\n================= ИТОГИ ФИЛЬТРАЦИИ =================")
    print(f"Всего проверено EML:     {stats['total']}")
    print(f"Отсеяно (короткие <30 сл): {stats['dropped_short']}")
    print(f"Отсеяно (не RU/EN):       {stats['dropped_lang']}")
    print(f"Отсеяно (битые/сбойные):  {stats['dropped_corrupt']}")
    print(f"Отобрано в датасет:       {stats['accepted']}")
    print(f"Итоговый файл:            {OUTPUT_FILE}")

if __name__ == "__main__":
    main()