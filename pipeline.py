import os
import json
from parser import parse_eml
from htmlModule import predict_html_features_and_score

MODEL_PATH = "html_ai_detector.joblib"

def extract_reason_codes(html_res: dict, clean_text: str) -> list[str]:
    reasons = []
    feats = html_res.get("features", {})
    proba = html_res.get("html_ai_proba", 0.0)

    # 1. Срабатывание ML модели верстки
    if proba >= 0.70:
        reasons.append(f"ML_HTML_SUSPICIOUS: Модель верстки классифицировала HTML как подозрительный (p={proba:.2%})")

    # 2. Структурные аномалии DOM и ссылок
    if feats.get("dummy_link_count", 0) > 0:
        reasons.append(f"SUSPICIOUS_LINKS: Найдено ссылок-заглушек: {feats['dummy_link_count']}")

    if feats.get("placeholder_hits", 0) > 0:
        reasons.append(f"TEMPLATE_ARTIFACTS: Нескомпилированные теги шаблонизатора: {feats['placeholder_hits']}")

    if feats.get("tracking_pixel_count", 0) > 0:
        reasons.append(f"TRACKING_PIXELS: Найдено скрытых трекеров: {feats['tracking_pixel_count']}")

    if feats.get("dom_max_depth", 0) > 12:
        reasons.append(f"DEEP_DOM: Аномальная глубина вложенности DOM: {feats['dom_max_depth']}")

    # 3. Эвристика по соотношению разметки и текста
    if feats.get("tag_to_text_ratio", 0) > 0.5 and len(clean_text) < 100:
        reasons.append("HIGH_TAG_DENSITY: Избыток тегов верстки при минимуме полезного текста")

    return reasons

def scan_email(eml_path: str) -> dict:
    if not os.path.exists(eml_path):
        return {"error": f"Файл {eml_path} не найден"}

    # 1. Разбор MIME структуры
    email_data = parse_eml(eml_path)
    raw_html = email_data.get("raw_html", "")
    clean_text = email_data.get("clean_text", "")

    # 2. Анализ HTML (если есть разметка)
    if raw_html and os.path.exists(MODEL_PATH):
        html_analysis = predict_html_features_and_score(raw_html, MODEL_PATH)
    else:
        html_analysis = {"html_ai_proba": 0.0, "features": {}}

    # 3. Генерация Reason Codes
    reasons = extract_reason_codes(html_analysis, clean_text)

    # 4. Итоговый скор риска
    html_proba = html_analysis.get("html_ai_proba", 0.0)
    final_score = max(html_proba, min(1.0, len(reasons) * 0.25))

    return {
        "message_id": email_data.get("id"),
        "subject": email_data.get("subject"),
        "verdict": "SUSPICIOUS" if final_score >= 0.5 else "CLEAN",
        "risk_score": round(final_score, 4),
        "reason_codes": reasons,
        "media_flags": email_data.get("media_flags"),
        "metrics": {
            "html_ai_proba": round(html_proba, 4),
            "text_length": len(clean_text),
            "dom_depth": html_analysis["features"].get("dom_max_depth", 0)
        }
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        result = scan_email(sys.argv[1])
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("Использование: python pipeline.py <путь_к_файлу.eml>")