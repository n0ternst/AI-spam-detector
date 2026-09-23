import os
import sys
import json
from pathlib import Path

from core.parser import parse_eml
from core.htmlModule import predict_html_features_and_score
from core.text_detector import BinocularsDetector

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "html_ai_detector.joblib"

# Инициализируем модель один раз при старте
text_detector = BinocularsDetector()


def extract_reason_codes(html_res: dict, text_res: dict, clean_text: str) -> list[str]:
    reasons = []
    h_feats = html_res.get("features", {})
    h_proba = html_res.get("html_ai_proba", 0.0)
    t_proba = text_res.get("text_ai_proba", 0.0)

    # 1. Текстовые Reason Codes (Binoculars & синтаксис)
    if t_proba >= 0.65:
        reasons.append(
            f"AI_GENERATED_TEXT: Высокая вероятность генерации LLM "
            f"(p={t_proba:.2%}, bino_min={text_res.get('bino_min')})"
        )
    if text_res.get("sent_len_var", 10.0) < 3.0 and len(clean_text.split()) > 30:
        reasons.append(
            f"SYNTAX_MONOTONY: Низкая дисперсия длины предложений "
            f"({text_res.get('sent_len_var')})"
        )

    # 2. HTML / DOM Reason Codes
    if h_proba >= 0.70:
        reasons.append(f"ML_HTML_SUSPICIOUS: Классификатор верстки определил аномалию (p={h_proba:.2%})")
    if h_feats.get("dummy_link_count", 0) > 0:
        reasons.append(f"SUSPICIOUS_LINKS: Найдено ссылок-заглушек/плейсхолдеров: {h_feats['dummy_link_count']}")
    if h_feats.get("placeholder_hits", 0) > 0:
        reasons.append(f"TEMPLATE_ARTIFACTS: Нескомпилированные теги шаблона: {h_feats['placeholder_hits']}")
    if h_feats.get("tracking_pixel_count", 0) > 0:
        reasons.append(f"TRACKING_PIXELS: Найдено скрытых трекеров: {h_feats['tracking_pixel_count']}")
    if h_feats.get("dom_max_depth", 0) > 12:
        reasons.append(f"DEEP_DOM: Аномальная глубина вложенности DOM: {h_feats['dom_max_depth']}")

    return reasons


def scan_email(eml_path: str) -> dict:
    if not os.path.exists(eml_path):
        return {"error": f"Файл {eml_path} не найден"}

    # 1. Парсинг MIME структуры
    email_data = parse_eml(eml_path)
    raw_html = email_data.get("raw_html", "")
    clean_text = email_data.get("clean_text", "")

    # 2. Анализ HTML верстки (если в письме был HTML)
    if raw_html and os.path.exists(MODEL_PATH):
        html_res = predict_html_features_and_score(raw_html, str(MODEL_PATH))
    else:
        html_res = {"html_ai_proba": 0.0, "features": {}}

    # 3. Анализ текста 
    text_res = text_detector.analyze_text(clean_text)

    # 4. Формирование Reason Codes
    reasons = extract_reason_codes(html_res, text_res, clean_text)

    # 5. Итоговый скор риска 
    h_score = html_res.get("html_ai_proba", 0.0)
    t_score = text_res.get("text_ai_proba", 0.0)
    risk_score = max(h_score, t_score)

    # Штраф за множественные подозрительные триггеры
    if len(reasons) >= 2:
        risk_score = min(1.0, risk_score + 0.2)

    return {
        "message_id": email_data.get("id"),
        "subject": email_data.get("subject"),
        "verdict": "SUSPICIOUS" if risk_score >= 0.5 else "CLEAN",
        "risk_score": round(risk_score, 4),
        "reason_codes": reasons,
        "metrics": {
            "text_ai_proba": text_res.get("text_ai_proba"),
            "html_ai_proba": html_res.get("html_ai_proba"),
            "bino_min": text_res.get("bino_min"),
            "dom_depth": html_res.get("features", {}).get("dom_max_depth", 0)
        }
    }


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(json.dumps(scan_email(sys.argv[1]), ensure_ascii=False, indent=2))
    else:
        print("Использование: python pipeline.py <путь_к_письму.eml>")