import os
import sys
import json
import argparse
from pathlib import Path

# Импорт парсера и детекторов
try:
    from core.parser import parse_eml
except ImportError:
    from parser import parse_eml

from core.html_detector import HTMLDetector
from core.text_detector import TextDetector


class SpamAiPipeline:
    def __init__(self):
        print("\n" + "=" * 60)
        print("ИНИЦИАЛИЗАЦИЯ СИСТЕМЫ ДЕТЕКЦИИ DR.WEB (LATE FUSION)")
        print("=" * 60)
        self.html_detector = HTMLDetector()
        self.text_detector = TextDetector()
        print("✓ Конвейер готов к работе.\n")

    def evaluate_html_risk(self, feats: dict, desync_score: float = 0.0) -> tuple[float, list[str]]:
        """Калиброванная оценка риска верстки по правилам фильтрации Dr.Web."""
        reasons = []
        risk = 0.0

        hidden = feats.get("hidden_elements_count", 0)
        dummy = feats.get("dummy_link_count", 0)
        forms = feats.get("has_forms", 0)
        iframes = feats.get("has_iframes", 0)
        zw = feats.get("zero_width_chars_count", 0)

        # 1. Жесткие триггеры фишинга и укрывательства (Hard Triggers)
        if forms == 1:
            risk = max(risk, 0.95)
            reasons.append("CREDENTIAL_PHISHING_FORM")

        if iframes == 1:
            risk = max(risk, 0.92)
            reasons.append("HIDDEN_IFRAME_PAYLOAD")

        # Отлов даже единичного скрытого блока (белый на белом, нулевой шрифт)
        if hidden >= 1:
            risk = max(risk, 0.90)
            reasons.append(f"CLOAKED_HIDDEN_CONTENT({hidden})")

        if hidden >= 2 and dummy >= 2:
            risk = max(risk, 0.90)
            reasons.append(f"PHISHING_LINK_MASKING(hidden={hidden}, dummy_links={dummy})")

        # Раздвоение личности между Plain text и HTML
        if desync_score >= 0.45:
            risk = max(risk, 0.85)
            reasons.append(f"MULTIPART_DESYNC_ATTACK(desync={desync_score})")

        # 2. Мягкие триггеры верстки (Soft Indicators)
        if dummy >= 3:
            risk = max(risk, 0.65)
            reasons.append(f"DUMMY_OR_TEMPLATE_LINKS({dummy})")

        if zw > 1000:
            risk = max(risk, 0.70)
            reasons.append(f"SUSPICIOUS_ZERO_WIDTH_PADDING({zw})")

        if feats.get("modern_html5_tags_count", 0) >= 3 and feats.get("has_mso_comments", 0) == 0:
            risk = max(risk, 0.40)
            reasons.append(f"LLM_SYNTHETIC_HTML5_LAYOUT({feats['modern_html5_tags_count']})")

        if feats.get("vendor_css_density", 1.0) == 0.0 and feats.get("modern_html5_tags_count", 0) > 0:
            risk = max(risk, 0.35)
            reasons.append("ZERO_VENDOR_CSS_ANOMALY")

        if feats.get("placeholder_hits", 0) > 0:
            risk = max(risk, 0.30)
            reasons.append(f"UNCOMPILED_TEMPLATE_TAGS({feats['placeholder_hits']})")

        if feats.get("data_uri_count", 0) >= 2:
            risk = max(risk, 0.50)
            reasons.append("EMBEDDED_BASE64_PAYLOAD")

        return round(risk, 2), reasons

    def aggregate_verdict(self, html_risk: float, html_reasons: list[str], text_res: dict) -> dict:
        """Иерархическая логика Late Fusion."""
        reasons = []
        p_ai = text_res.get("text_prob", 0.0)
        t_strict = text_res.get("t_strict", 0.85)

        # 1. Приоритет критических уязвимостей разметки (фишинг, формы, скрытый камуфляж)
        if html_risk >= 0.85:
            reasons.extend(html_reasons)
            if p_ai >= 0.30:
                reasons.append(f"SECONDARY_AI_CONFIDENCE(P={p_ai:.2f})")
            return {
                "verdict": "SPAM_BLOCKED",
                "final_risk": html_risk,
                "primary_module": "HTML_ANALYZER",
                "reasons": reasons
            }

        # 2. Высокая уверенность в синтетическом тексте (FPR <= 1%)
        if p_ai >= t_strict:
            reasons.append(f"HIGH_AI_SYNTACTIC_CONFIDENCE(P={p_ai:.2f})")
            if html_reasons:
                reasons.extend(html_reasons)
            return {
                "verdict": "AI_GENERATED_SPAM",
                "final_risk": round(p_ai, 2),
                "primary_module": "TEXT_BINOCULARS_LR",
                "reasons": reasons
            }

        # 3. Мягкая синергия Late Fusion (серая зона / гибриды)
        if p_ai >= 0.28 and html_risk < 0.85:
            reasons.append(f"SUSPICIOUS_AI_TEXT_STYLE(P={p_ai:.2f})")
            return {
                "verdict": "SUSPICIOUS_AI_TAGGED",
                "final_risk": round(p_ai, 2),
                "primary_module": "TEXT_BINOCULARS_LR",
                "reasons": reasons
            }

        # 4. Письмо признано чистым (Ham)
        return {
            "verdict": "CLEAN",
            "final_risk": max(html_risk, p_ai),
            "primary_module": "PASSED_ALL_CHECKS",
            "reasons": ["NO_CRITICAL_ANOMALIES"]
        }

    def process_email(self, eml_path: str) -> dict:
        """Полный сквозной прогон одного .eml файла."""
        parsed = parse_eml(eml_path)
        raw_text = parsed.get("clean_text", "")
        raw_html = parsed.get("raw_html", "")
        desync_score = parsed.get("multipart_desync_score", 0.0)

        # 1. Сначала запускаем текстовый модуль (получаем проценты и спаны ИИ)
        text_out = self.text_detector.predict(raw_text)
        ai_spans = text_out.get("detected_ai_spans", [])

        # 2. Затем передаем спаны в HTML-модуль для разметки AST-дерева
        html_out = self.html_detector.predict(raw_html, ai_spans=ai_spans)
        html_feats = html_out["features"]
        html_risk, html_reasons = self.evaluate_html_risk(html_feats, desync_score)

        # 3. Принятие итогового решения
        decision = self.aggregate_verdict(html_risk, html_reasons, text_out)

        return {
            "file": Path(eml_path).name,
            "subject": parsed.get("subject", ""),
            "verdict": decision["verdict"],
            "risk_score": decision["final_risk"],
            "primary_module": decision["primary_module"],
            "reasons": decision["reasons"],
            "details": {
                "html_risk": html_risk,
                "text_prob": text_out["text_prob"],
                "lang": text_out["lang"],
                "most_ai_chunk": text_out.get("most_ai_chunk", "")[:120],
                "hidden_css_tags": html_feats.get("hidden_elements_count", 0),
                "dummy_links": html_feats.get("dummy_link_count", 0),
                "desync_score": desync_score,
                "ast_tree": html_out["ast_tree"].to_dict() if html_out.get("ast_tree") else None
            }
        }

    def process_jsonl(self, jsonl_path: str):
        """Пакетный прогон тестовой выборки с наглядным отчетом."""
        with open(jsonl_path, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]

        print(f"\nЗапуск анализа выборки из {len(lines)} писем...")
        print("-" * 85)
        print(f"{'Файл':<14} | {'Вердикт':<20} | {'Риск':<5} | {'Модуль':<20} | {'Причины'}")
        print("-" * 85)

        for item in lines:
            raw_text = item.get("text", "")
            raw_html = item.get("raw_html", "")
            file_name = item.get("file_name", "sample")[:12]

            text_out = self.text_detector.predict(raw_text, force_lang=item.get("lang"))
            html_out = self.html_detector.predict(raw_html, ai_spans=text_out.get("detected_ai_spans", []))
            html_risk, html_reasons = self.evaluate_html_risk(html_out["features"])

            decision = self.aggregate_verdict(html_risk, html_reasons, text_out)

            reasons_str = ", ".join(decision["reasons"])
            print(f"{file_name:<14} | {decision['verdict']:<20} | {decision['final_risk']:<5.2f} | {decision['primary_module']:<20} | {reasons_str}")


# Адаптер вызова для Colab и внешних скриптов
_pipeline_instance = None

def scan_email(eml_path: str) -> dict:
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = SpamAiPipeline()
    return _pipeline_instance.process_email(eml_path)


def main():
    parser = argparse.ArgumentParser(description="Dr.Web Multi-Modal Spam & AI Detector")
    parser.add_argument("--eml", type=str, help="Путь к отдельному .eml файлу")
    parser.add_argument("--jsonl", type=str, help="Путь к тестовому .jsonl датасету")
    args = parser.parse_args()

    pipeline = SpamAiPipeline()

    if args.eml:
        res = pipeline.process_email(args.eml)
        print(json.dumps(res, ensure_ascii=False, indent=2))
    elif args.jsonl:
        pipeline.process_jsonl(args.jsonl)
    else:
        default_file = "drweb_13_enriched.jsonl"
        if os.path.exists(default_file):
            print(f"[Демо] Флаги не указаны, запуск на {default_file}...")
            pipeline.process_jsonl(default_file)
        else:
            print("Укажите параметр: python pipeline.py --eml <путь> или --jsonl <путь>")


if __name__ == "__main__":
    main()
