import os
import sys
import json
import shutil
from pathlib import Path

from core.parser import parse_eml
from core.htmlModule import predict_html_features_and_score
from core.text_detector import BinocularsDetector
from core.visual_detector import VisualDetector

BASE_DIR = Path(__file__).resolve().parent
HTML_MODEL_PATH = str(BASE_DIR / "models" / "html_ai_detector.joblib")

print("[Pipeline] Инициализация моделей (Text & Vision)...")
text_detector = BinocularsDetector()
visual_detector = VisualDetector()
print("[Pipeline] Модели успешно загружены в память.")


def scan_email(eml_path: str, cleanup_temp: bool = True) -> dict:
    if not os.path.exists(eml_path):
        return {"error": f"Файл {eml_path} не найден"}

    # 1. MIME-парсинг
    email_data = parse_eml(eml_path)
    raw_html = email_data.get("raw_html", "")
    clean_text = email_data.get("clean_text", "")
    attachments = email_data.get("attachments", [])
    extract_dir = email_data.get("extract_dir")

    try:
        # 2. Поток HTML
        html_res = predict_html_features_and_score(raw_html, HTML_MODEL_PATH)
        h_feats = html_res.get("features", {})
        h_risk = float(html_res.get("html_ai_proba", 0.0))

        # 3. Поток Текста (Binoculars + Синтаксис + Spans)
        text_res = text_detector.analyze_text(clean_text)
        t_risk = float(text_res.get("text_ai_proba", 0.0))

        # 4. Поток Визуальный (PaddleOCR + Quishing + SAFE)
        visual_res = visual_detector.analyze_attachments(attachments, text_detector=text_detector)
        v_risk = float(visual_res.get("visual_risk", 0.0))

        # 5. Диспетчер вердикта (Decoupled Late Fusion)
        reason_codes = []

        # Жесткие триггеры
        if h_feats.get("has_forms", 0) > 0:
            reason_codes.append("PHISHING_CREDENTIAL_FORM_DETECTED")
        if h_feats.get("has_iframes", 0) > 0:
            reason_codes.append("EMBEDDED_IFRAME_EXPLOIT")
        if h_feats.get("malicious_hidden_chars", 0) > 200:
            reason_codes.append("MALICIOUS_HIDDEN_TEXT_STUFFING")
        if h_feats.get("dummy_link_count", 0) > 0:
            reason_codes.append("EXTERNAL_SUSPICIOUS_REDIRECT")
        if visual_res.get("has_quishing_qr"):
            reason_codes.append("QUISHING_QR_CODE_DETECTED")

        # Текстовые и структурные маркеры ИИ
        if t_risk >= 0.80:
            reason_codes.append("HIGH_AI_SYNTACTIC_CONFIDENCE")
        if text_res.get("sent_len_var", 10.0) < 3.0 and len(clean_text.split()) >= 30:
            reason_codes.append("SYNTAX_MONOTONY")
        if h_feats.get("modern_html5_tags_count", 0) >= 2 and h_feats.get("has_mso_comments", 0) == 0:
            reason_codes.append("AI_GENERATED_HTML_STRUCTURE")

        # Визуальные спам-кнопки
        if visual_res.get("has_suspicious_ocr_keywords"):
            reason_codes.append("EMBEDDED_IMAGE_CALL_TO_ACTION")

        # Взвешенная сумма рисков
        weighted_risk = (0.5 * h_risk) + (0.3 * v_risk) + (0.2 * t_risk)

        # Амплификация
        if h_risk >= 0.40 and t_risk >= 0.70:
            weighted_risk = min(1.0, weighted_risk + 0.25)
            reason_codes.append("AI_AMPLIFIED_MALICIOUS_TEMPLATE")

        # Итоговый вердикт
        is_hard_blocked = (
            h_feats.get("has_forms", 0) > 0
            or h_feats.get("has_iframes", 0) > 0
            or h_feats.get("malicious_hidden_chars", 0) > 200
            or visual_res.get("has_quishing_qr", False)
        )

        if is_hard_blocked:
            final_risk = max(0.85, weighted_risk)
            verdict = "SUSPICIOUS_SPAM"
        elif weighted_risk >= 0.60:
            final_risk = weighted_risk
            verdict = "SUSPICIOUS_SPAM"
        elif t_risk >= 0.75 and weighted_risk < 0.60:
            final_risk = weighted_risk
            verdict = "AI_ASSISTED_HAM"
            reason_codes.append("AI_ASSISTED_CONTENT_CLEAN_ORIGIN")
        else:
            final_risk = weighted_risk
            verdict = "CLEAN_HAM"

        return {
            "verdict": verdict,
            "confidence_score": round(final_risk, 2),
            "reason_codes": reason_codes,
            "explainability": {
                "detected_ai_spans": text_res.get("detected_ai_spans", []),
                "html_evidence": html_res.get("evidence", {}),
                "visual_evidence": visual_res.get("details", []),
                "metrics": {
                    "text_ai_probability": t_risk,
                    "html_anomaly_score": h_risk,
                    "visual_anomaly_score": v_risk,
                    "sentence_length_variance": text_res.get("sent_len_var"),
                    "bino_min": text_res.get("bino_min")
                }
            },
            "meta": {
                "message_id": email_data.get("id"),
                "subject": email_data.get("subject"),
                "attachments_count": len(attachments)
            }
        }

    finally:
        if cleanup_temp and extract_dir and os.path.exists(extract_dir):
            try:
                shutil.rmtree(extract_dir)
            except Exception:
                pass


if __name__ == "__main__":
    if len(sys.argv) > 1:
        res = scan_email(sys.argv[1])
        print(json.dumps(res, ensure_ascii=False, indent=2))
    else:
        print("Использование: python pipeline.py <путь_к_письму.eml>")
