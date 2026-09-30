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

    def evaluate_html_risk(self, feats: dict) -> tuple[float, list[str]]:
        """
        Калиброванная оценка риска верстки по правилам фильтрации Dr.Web.
        """
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

        if hidden >= 4:
            risk = max(risk, 0.88)
            reasons.append(f"MASSIVE_HIDDEN_CSS({hidden})")

        if hidden >= 2 and dummy >= 2:
            risk = max(risk, 0.90)
            reasons.append(f"PHISHING_LINK_MASKING(hidden={hidden}, dummy_links={dummy})")

        # 2. Мягкие триггеры верстки (Soft Indicators)
        if dummy >= 3:
            risk = max(risk, 0.65)
            reasons.append(f"DUMMY_OR_TEMPLATE_LINKS({dummy})")

        if zw > 1000:
            risk = max(risk, 0.70)
            reasons.append(f"SUSPICIOUS_ZERO_WIDTH_PADDING({zw})")

        if feats.get("modern_html5_tags_count", 0) >= 3:
            risk = max(risk, 0.40)
            reasons.append(f"LLM_SYNTHETIC_HTML5_LAYOUT({feats['modern_html5_tags_count']})")

        if feats.get("data_uri_count", 0) >= 2:
            risk = max(risk, 0.50)
            reasons.append("EMBEDDED_BASE64_PAYLOAD")

        return round(risk, 2), reasons

    def aggregate_verdict(self, html_risk: float, html_reasons: list[str], text_res: dict) -> dict:
        """
        Иерархическая логика Late Fusion:
        1. Hard Trigger верстки (фишинг/формы) -> Блокировка
        2. Strict Trigger текста (P(AI) >= T_strict) -> AI Спам
        3. Composite Synergy (пред-пороговые значения обоих модулей) -> Серая зона
        4. Чистое письмо (Ham)
        """
        reasons = []
        p_ai = text_res.get("text_prob", 0.0)
        t_strict = text_res.get("t_strict", 0.85)
        t_soft = text_res.get("t_soft", 0.72)

        # 1. Приоритет критических уязвимостей разметки
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

        # 3. Мягкая синергия Late Fusion 
        # Если текст попал в диапазон 0.28 - 0.70 
        # не баним письмо, а пропускаем с предупреждающим флагом:
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

        # 1. HTML модуль
        html_out = self.html_detector.predict(raw_html)
        html_feats = html_out["features"]
        html_risk, html_reasons = self.evaluate_html_risk(html_feats)

        # 2. Text модуль
        text_out = self.text_detector.predict(raw_text)

        # 3. Late Fusion
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
                "dummy_links": html_feats.get("dummy_link_count", 0)
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

            html_out = self.html_detector.predict(raw_html)
            html_risk, html_reasons = self.evaluate_html_risk(html_out["features"])

            text_out = self.text_detector.predict(raw_text, force_lang=item.get("lang"))
            decision = self.aggregate_verdict(html_risk, html_reasons, text_out)

            reasons_str = ", ".join(decision["reasons"])
            print(f"{file_name:<14} | {decision['verdict']:<20} | {decision['final_risk']:<5.2f} | {decision['primary_module']:<20} | {reasons_str}")


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
        # Демонстрационный прогон по умолчанию на сохраненном срезе
        default_file = "drweb_13_enriched.jsonl"
        if os.path.exists(default_file):
            print(f"[Демо] Флаги не указаны, запуск на {default_file}...")
            pipeline.process_jsonl(default_file)
        else:
            print("Укажите параметр: python pipeline.py --eml <путь> или --jsonl <путь>")


if __name__ == "__main__":
    main()