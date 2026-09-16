import os
import email
from email import policy
from bs4 import BeautifulSoup
import pandas as pd

# Укажи точный абсолютный путь к папке со скриншота
DATASET_ROOT = r"C:\Users\Bogdan\Documents\AI-spam-detector\AI_generated_emails (1)"  # Проверь, чтобы внутри лежали samples_prompt и т.д.

def parse_eml_summary(file_path):
    try:
        with open(file_path, "rb") as f:
            msg = email.message_from_binary_file(f, policy=policy.default)
        
        subject = str(msg.get("Subject", ""))
        body_text = ""
        has_html = False
        images_count = 0

        for part in msg.walk():
            content_type = part.get_content_type()
            if content_type == "text/plain":
                try:
                    body_text += str(part.get_content())
                except Exception:
                    pass
            elif content_type == "text/html":
                has_html = True
                if not body_text:
                    try:
                        soup = BeautifulSoup(part.get_content(), "html.parser")
                        body_text = soup.get_text(separator=" ").strip()
                    except Exception:
                        pass
            elif "image" in content_type:
                images_count += 1

        return {
            "file_name": os.path.basename(file_path),
            "subject": subject[:60],
            "text_preview": body_text[:100].replace("\n", " ").strip(),
            "text_len": len(body_text),
            "has_html": has_html,
            "images_count": images_count
        }
    except Exception as e:
        return {
            "file_name": os.path.basename(file_path),
            "subject": f"[Ошибка парсинга: {e}]",
            "text_preview": "",
            "text_len": 0,
            "has_html": False,
            "images_count": 0
        }

rows = []
target_folders = ["samples_prompt", "samples_spam", "samples_gray"]

for folder in target_folders:
    folder_path = os.path.join(DATASET_ROOT, folder)
    if not os.path.exists(folder_path):
        print(f"[Пропуск] Папка не найдена: {folder_path}")
        continue
    
    # Забираем ВСЕ файлы, игнорируя расширение (пропускаем только подпапки)
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    print(f"Папка {folder}: найдено {len(files)} файлов")
    
    # Берем первые 5 файлов для превью
    for f in files[:5]:
        meta = parse_eml_summary(os.path.join(folder_path, f))
        meta["folder_label"] = folder
        meta["is_ai"] = 1 if folder == "samples_prompt" else 0
        rows.append(meta)

if not rows:
    print("\n[Внимание] Не найдено ни одного файла! Проверь переменную DATASET_ROOT.")
else:
    df_preview = pd.DataFrame(rows)
    print("\nПример содержимого:")
    print(df_preview[["folder_label", "is_ai", "subject", "text_len", "images_count", "text_preview"]])