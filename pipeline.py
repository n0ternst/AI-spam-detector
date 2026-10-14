import argparse
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any

# Импорт парсера и детекторов
try:
    from core.parser import parse_eml, cleanup_extracted_files
except ImportError:
    try:
        from parser import parse_eml, cleanup_extracted_files
    except ImportError:
        from eml_parser import parse_eml, cleanup_extracted_files

try:
    from core.html_detector import HTMLDetector
    from core.text_detector import TextDetector
except ImportError:
    from html_detector import HTMLDetector
    from text_detector import TextDetector

logger = logging.getLogger("DrWebSpamPipeline")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class SpamAiPipeline:

    def __init__(self):
        logger.info("=" * 60)
        logger.info(
            "ИНИЦИАЛИЗАЦИЯ СИСТЕМЫ ДЕТЕКЦИИ DR.WEB (MULTIMODAL LATE FUSION)"
        )
        logger.info("=" * 60)
        self.html_detector = HTMLDetector()
        self.text_detector = TextDetector()
        logger.info("✓ Конвейер готов к работе.")

    @staticmethod
    def _attach_ai_spans_to_ast(
        ast_dict: dict[str, Any] | None, ai_spans: list[dict]
    ) -> None:
        """Рекурсивно обогащает уже построенное AST-дерево спанами из текстового детектора."""
        if not ast_dict or not ai_spans:
            return

        snippet = ast_dict.get("text_snippet", "")
        if snippet and len(snippet.split()) >= 3:
            import re

            snip_w = set(re.findall(r"\b\w+\b", snippet.lower()))
            best_conf = 12.0
            for sp in ai_spans:
                target_w = set(
                    re.findall(r"\b\w+\b", sp.get("text_segment", "").lower())
                )
                if snip_w and len(snip_w & target_w) / len(snip_w) >= 0.40:
                    best_conf = float(sp.get("confidence_pct", 0.0))
                    break
            ast_dict["ai_confidence_pct"] = round(best_conf, 2)

        for child in ast_dict.get("children", []):
            SpamAiPipeline._attach_ai_spans_to_ast(child, ai_spans)

    def evaluate_html_risk(
        self,
        html_out: dict[str, Any],
        desync_score: float = 0.0,
        headers_meta: dict[str, Any] | None = None,
    ) -> tuple[float, float, list[str]]:
        """
        Раздельная оценка:
        1) threat_risk (фишинг, клоакинг, обход фильтров)
        2) html_ai_score (признаки генерации самой верстки через LLM / копипасту)
        """
        feats = html_out.get("features", {})
        reasons = list(html_out.get("top_risk_factors", []))

        # Базовые вероятности из нашего нового HTMLDetector
        html_ai_score = float(html_out.get("html_ai_proba", 0.0))
        threat_risk = float(html_out.get("cloaking_risk_score", 0.0))

        hidden = feats.get("hidden_elements_count", 0)
        cloaked_ratio = feats.get("cloaked_text_ratio", 0.0)
        dummy = feats.get("dummy_link_count", 0)
        forms = feats.get("has_forms", 0)
        iframes = feats.get("has_iframes", 0)
        zw = feats.get("zero_width_chars_count", 0)
        bidi = feats.get("bidi_spoofing_hits", 0)

        # Понижающий флаг для легитимных корпоративных рассылок (ESP: Mailchimp, HubSpot, Outlook)
        is_legit_esp = bool(
            feats.get("has_mso_comments", 0) > 0
            or feats.get("esp_signature_hits", 0) > 0
            and feats.get("has_unsubscribe_link", 0) > 0
        )

        # 1. Жесткие триггеры фишинга и атак на парсер (Hard Security Triggers)
        if forms >= 1:
            threat_risk = max(threat_risk, 0.95)
            reasons.append("CREDENTIAL_PHISHING_FORM")

        if iframes >= 1:
            threat_risk = max(threat_risk, 0.92)
            reasons.append("HIDDEN_IFRAME_PAYLOAD")

        if bidi >= 1:
            threat_risk = max(threat_risk, 0.92)
            reasons.append(f"BIDI_RTL_SPOOFING({bidi})")

        # Защита от False Positives: блокируем за скрытые блоки, только если скрыто >5% текста,
        # либо скрытых блоков >= 2 и это НЕ легитимный ESP-шаблон
        if hidden >= 1 and (
            cloaked_ratio >= 0.05 or (hidden >= 2 and not is_legit_esp)
        ):
            threat_risk = max(threat_risk, 0.90)
            reasons.append(
                f"CLOAKED_HIDDEN_CONTENT(blocks={hidden}, ratio={cloaked_ratio:.2f})"
            )

        if hidden >= 1 and dummy >= 2:
            threat_risk = max(threat_risk, 0.90)
            reasons.append(
                f"PHISHING_LINK_MASKING(hidden={hidden}, dummy_links={dummy})"
            )

        if desync_score >= 0.45:
            threat_risk = max(threat_risk, 0.88)
            reasons.append(f"MULTIPART_DESYNC_ATTACK(desync={desync_score})")

        if headers_meta and headers_meta.get("header_layout_mismatch"):
            threat_risk = max(threat_risk, 0.85)
            reasons.append("OUTLOOK_HEADER_SPOOFING_NO_MSO")

        # 2. Мягкие индикаторы и усиление скора ИИ-верстки
        if dummy >= 3:
            html_ai_score = max(html_ai_score, 0.65)
            reasons.append(f"DUMMY_OR_TEMPLATE_LINKS({dummy})")

        if zw > 500:
            threat_risk = max(threat_risk, 0.75)
            reasons.append(f"SUSPICIOUS_ZERO_WIDTH_PADDING({zw})")

        # Убираем дубликаты в причинах, сохраняя порядок
        unique_reasons = list(dict.fromkeys(reasons))
        return round(threat_risk, 2), round(html_ai_score, 2), unique_reasons

    def aggregate_verdict(
        self,
        threat_risk: float,
        html_ai_score: float,
        html_reasons: list[str],
        text_res: dict[str, Any],
    ) -> dict[str, Any]:
        """Иерархическая логика Multimodal Late Fusion (Текст + HTML-криминалистика)."""
        reasons: list[str] = []
        p_ai = float(text_res.get("text_prob", 0.0))
        t_strict = float(text_res.get("t_strict", 0.85))

        # 1. Приоритет критических угроз безопасности (фишинг, клоакинг, Bidi, подмена частей)
        if threat_risk >= 0.85:
            reasons.extend(html_reasons)
            if p_ai >= 0.30:
                reasons.append(f"SECONDARY_AI_CONFIDENCE(P={p_ai:.2f})")
            return {
                "verdict": "SPAM_BLOCKED",
                "final_risk": threat_risk,
                "primary_module": "HTML_SECURITY_ANALYZER",
                "reasons": reasons,
            }

        # 2. Настоящий Late Fusion (Синергия Текста и HTML-структуры)
        # Если оба детектора видят признаки ИИ (например, p_ai=0.60 и html_ai_score=0.70),
        # совместная вероятность пробивает порог блокировки!
        synergy_score = round(0.65 * p_ai + 0.35 * html_ai_score, 2)
        if p_ai >= 0.45 and html_ai_score >= 0.50:
            # Нелинейный буст при согласии двух независимых модальностей
            synergy_score = min(1.0, round(synergy_score + 0.18, 2))

        if p_ai >= t_strict or synergy_score >= t_strict:
            final_ai_score = max(p_ai, synergy_score)
            reasons.append(
                f"HIGH_MULTIMODAL_AI_CONFIDENCE(text={p_ai:.2f}, html_ai={html_ai_score:.2f})"
            )
            reasons.extend(html_reasons)
            return {
                "verdict": "AI_GENERATED_SPAM",
                "final_risk": round(final_ai_score, 2),
                "primary_module": (
                    "LATE_FUSION_ENSEMBLE"
                    if synergy_score > p_ai
                    else "TEXT_BINOCULARS_LR"
                ),
                "reasons": list(dict.fromkeys(reasons)),
            }

        # 3. Серая зона (Подозрение на ИИ по тексту ИЛИ явные следы копипасты ChatGPT в HTML)
        if p_ai >= 0.28 or html_ai_score >= 0.55 or threat_risk >= 0.60:
            combined_suspicion = max(
                p_ai, html_ai_score, threat_risk, synergy_score
            )
            if p_ai >= 0.28:
                reasons.append(f"SUSPICIOUS_AI_TEXT_STYLE(P={p_ai:.2f})")
            if html_reasons:
                reasons.extend(html_reasons)
            return {
                "verdict": "SUSPICIOUS_AI_TAGGED",
                "final_risk": round(combined_suspicion, 2),
                "primary_module": (
                    "HTML_FORENSICS"
                    if html_ai_score > p_ai
                    else "TEXT_BINOCULARS_LR"
                ),
                "reasons": list(dict.fromkeys(reasons)),
            }

        # 4. Письмо признано чистым (Ham)
        return {
            "verdict": "CLEAN",
            "final_risk": round(max(threat_risk, html_ai_score, p_ai), 2),
            "primary_module": "PASSED_ALL_CHECKS",
            "reasons": ["NO_CRITICAL_ANOMALIES"],
        }

    def process_email(self, eml_path: str) -> dict[str, Any]:
        """Полный сквозной прогон одного .eml файла с защитой от отравления скрытым текстом."""
        # 1. Парсер сразу вызывает html_detector, чтобы отсечь скрытый текст до запуска NLP!
        parsed = parse_eml(eml_path, html_detector=self.html_detector)
        try:
            clean_visible_text = parsed.get("clean_text", "")
            raw_html = parsed.get("raw_html", "")
            desync_score = parsed.get("multipart_desync_score", 0.0)
            headers_meta = parsed.get("headers_meta", {})

            # 2. Запускаем текстовый модуль по ЧИСТОМУ видимому тексту
            text_out = self.text_detector.predict(clean_visible_text)
            ai_spans = text_out.get("detected_ai_spans", [])

            # 3. Забираем результат HTML-детектора (или вызываем, если HTML был пуст)
            html_out = parsed.get(
                "html_forensics"
            ) or self.html_detector.predict(raw_html, ai_spans=ai_spans)
            html_feats = html_out.get("features", {})

            # Безопасная сериализация AST-дерева (поддержка и объекта, и dict)
            raw_ast = html_out.get("ast_tree")
            ast_dict = (
                raw_ast.to_dict() if hasattr(raw_ast, "to_dict") else raw_ast
            )
            if ai_spans and ast_dict:
                self._attach_ai_spans_to_ast(ast_dict, ai_spans)

            # 4. Оценка рисков верстки и Late Fusion
            threat_risk, html_ai_score, html_reasons = self.evaluate_html_risk(
                html_out, desync_score=desync_score, headers_meta=headers_meta
            )
            decision = self.aggregate_verdict(
                threat_risk, html_ai_score, html_reasons, text_out
            )

            return {
                "file": Path(eml_path).name,
                "subject": parsed.get("subject", ""),
                "verdict": decision["verdict"],
                "risk_score": decision["final_risk"],
                "primary_module": decision["primary_module"],
                "reasons": decision["reasons"],
                "details": {
                    "html_threat_risk": threat_risk,
                    "html_ai_proba": html_ai_score,
                    "text_prob": round(float(text_out.get("text_prob", 0.0)), 4),
                    "lang": text_out.get("lang", "unknown"),
                    "most_ai_chunk": text_out.get("most_ai_chunk", "")[:120],
                    "hidden_css_tags": html_feats.get(
                        "hidden_elements_count", 0
                    ),
                    "cloaked_text_ratio": html_feats.get(
                        "cloaked_text_ratio", 0.0
                    ),
                    "dummy_links": html_feats.get("dummy_link_count", 0),
                    "desync_score": desync_score,
                    "ast_tree": ast_dict,
                },
            }
        finally:
            # Обязательно очищаем временную папку с картинками, чтобы не забить диск сервера
            cleanup_extracted_files(parsed)

    def process_jsonl(self, jsonl_path: str):
        """Пакетный прогон тестовой выборки с наглядным отчетом."""
        with open(jsonl_path, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]

        print(f"\nЗапуск анализа выборки из {len(lines)} писем...")
        print("-" * 95)
        print(
            f"{'Файл':<14} | {'Вердикт':<20} | {'Риск':<5} | {'Модуль':<22} | {'Причины'}"
        )
        print("-" * 95)

        for item in lines:
            raw_html = item.get("raw_html", item.get("html", ""))
            fallback_text = item.get("text", item.get("clean_text", ""))
            file_name = str(item.get("file_name", item.get("id", "sample")))[:12]

            # Сначала HTML-детектор вычищает скрытый клоакинг из текста
            html_out = self.html_detector.predict(raw_html)
            clean_text = html_out.get("visible_text") or fallback_text

            # Затем текстовый детектор анализирует чистый текст
            text_out = self.text_detector.predict(
                clean_text, force_lang=item.get("lang")
            )
            threat_risk, html_ai_score, html_reasons = self.evaluate_html_risk(
                html_out
            )

            decision = self.aggregate_verdict(
                threat_risk, html_ai_score, html_reasons, text_out
            )

            reasons_str = ", ".join(decision["reasons"][:4])
            print(
                f"{file_name:<14} | {decision['verdict']:<20} | {decision['final_risk']:<5.2f} | {decision['primary_module']:<22} | {reasons_str}"
            )


_pipeline_instance = None


def scan_email(eml_path: str) -> dict[str, Any]:
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = SpamAiPipeline()
    return _pipeline_instance.process_email(eml_path)


def main():
    parser = argparse.ArgumentParser(
        description="Dr.Web Multi-Modal Spam & AI Detector"
    )
    parser.add_argument("--eml", type=str, help="Путь к отдельному .eml файлу")
    parser.add_argument(
        "--jsonl", type=str, help="Путь к тестовому .jsonl датасету"
    )
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
            logger.info(f"[Демо] Флаги не указаны, запуск на {default_file}...")
            pipeline.process_jsonl(default_file)
        else:
            print(
                "Укажите параметр: python pipeline.py --eml <путь> или --jsonl <путь>"
            )


if __name__ == "__main__":
    main()
