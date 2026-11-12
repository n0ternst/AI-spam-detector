import argparse
import json
import logging
import os
from pathlib import Path
import re
import time
from typing import Any

from catboost import Pool
import numpy as np

# Импорт парсера и детекторов (поддержка запуска и из корня, и из пакета core)
try:
  from core.parser import cleanup_extracted_files, parse_eml
except ImportError:
  try:
    from parser import cleanup_extracted_files, parse_eml
  except ImportError:
    from eml_parser import cleanup_extracted_files, parse_eml

try:
  from core.html_detector import ForensicResult, HTMLDetector
  from core.text_detector import FEATURE_ORDER, TextDetector
except ImportError:
  from html_detector import ForensicResult, HTMLDetector
  from text_detector import FEATURE_ORDER, TextDetector


# ==============================================================================
# ПАТЧ СОВМЕСТИМОСТИ ДЛЯ ForensicResult (чтобы parser.py не падал на .get())
# ==============================================================================
def _forensic_to_compat_dict(self: ForensicResult) -> dict[str, Any]:
  feats = self.evidence.get("features", {}) if self.evidence else {}
  compat_feats = dict(feats)
  # Маппинг старых имен ключей из parser.py / pipeline.py на 46 фичей HTMLDetector
  compat_feats.setdefault(
      "hidden_elements_count",
      int(
          feats.get("display_none_nodes", 0)
          + feats.get("zero_font_nodes", 0)
          + feats.get("zero_opacity_nodes", 0)
          + feats.get("offscreen_nodes", 0)
      ),
  )
  compat_feats.setdefault(
      "cloaked_text_ratio", float(feats.get("hidden_text_ratio", 0.0))
  )
  compat_feats.setdefault(
      "dummy_link_count", int(feats.get("deceptive_link_count", 0))
  )
  compat_feats.setdefault(
      "has_forms", 1 if feats.get("form_input_count", 0) > 0 else 0
  )
  compat_feats.setdefault(
      "has_iframes", 1 if feats.get("script_iframe_count", 0) > 0 else 0
  )
  compat_feats.setdefault(
      "zero_width_chars_count",
      int(feats.get("zero_width_char_density", 0.0) * 10),
  )
  compat_feats.setdefault(
      "bidi_spoofing_hits", int(feats.get("homoglyph_spoof_count", 0))
  )
  compat_feats.setdefault(
      "has_mso_comments", 1 if feats.get("mso_conditional_count", 0) > 0 else 0
  )
  compat_feats.setdefault(
      "mso_comment_count", int(feats.get("mso_conditional_count", 0))
  )
  compat_feats.setdefault(
      "esp_signature_hits", int(feats.get("esp_signature_count", 0))
  )
  compat_feats.setdefault(
      "has_unsubscribe_link",
      1 if feats.get("has_unsubscribe_pattern", 0) > 0 else 0,
  )
  compat_feats.setdefault(
      "image_count", int(feats.get("tracking_pixel_count", 0))
  )

  cloaking_risk = (
      min(
          0.95,
          0.45
          + 0.5 * compat_feats["cloaked_text_ratio"]
          + 0.08 * compat_feats["hidden_elements_count"],
      )
      if self.is_cloaking_detected
      else 0.0
  )

  return {
      "verdict": self.verdict,
      "html_ai_proba": self.global_risk_score,
      "global_risk_score": self.global_risk_score,
      "cloaking_risk_score": round(float(cloaking_risk), 4),
      "is_cloaking_detected": self.is_cloaking_detected,
      "top_risk_factors": self.top_risk_factors,
      "features": compat_feats,
      "evidence": self.evidence,
      "ast_tree": self.semantic_ast,
      "semantic_ast": self.semantic_ast,
      "visible_text": self.visible_text,
  }


ForensicResult.get = lambda self, k, default=None: _forensic_to_compat_dict(
    self
).get(k, default)
ForensicResult.__getitem__ = lambda self, k: _forensic_to_compat_dict(self)[k]
ForensicResult.__contains__ = lambda self, k: k in _forensic_to_compat_dict(
    self
)

# ==============================================================================
# СЛОВАРЬ РУССКИХ ОБЪЯСНЕНИЙ ПРИЗНАКОВ (П. 4.1.5.2 ТЗ DR.WEB)
# ==============================================================================
FEATURE_DESCRIPTIONS_RU = {
    "fd_global": (
        "высокая аналитическая кривизна вероятности текста (Fast-DetectGPT)"
    ),
    "lr_global": (
        "аномально предсказуемый средний ранг токенов по языковой модели"
        " Qwen-1.5B"
    ),
    "lrr_global": (
        "характерное для LLM отношение логарифма вероятности к рангу (LRR)"
    ),
    "lr_win_max": (
        "наличие локального участка с предельно шаблонным выбором слов"
    ),
    "lr_win_mean": "равномерная сглаженность токен-рангов по предложениям",
    "lr_win_std": "низкий разброс предсказуемости между соседними абзацами",
    "lr_win_q90": "высокий 90-й перцентиль вероятностного ранга окон",
    "lrr_win_max": "пиковое значение LRR в выделенном текстовом блоке",
    "lrr_win_mean": "стабильно высокий индекс LRR по окнам текста",
    "fd_win_max": (
        "выраженный локальный пик генеративности в отдельном фрагменте"
    ),
    "fd_win_mean": "повышенная средняя кривизна вероятности по абзацам",
    "merged_span_lr": (
        "высокая концентрация предсказуемых токенов внутри локализованного"
        " спана"
    ),
    "merged_span_ratio": (
        "значительная доля текста, охваченная генеративным паттерном"
    ),
    "sent_len_var": (
        "низкая вариативность длины предложений (равномерный машинный ритм)"
    ),
    "avg_sent_len": (
        "типичная для языковых моделей средняя длина предложения"
    ),
    "short_sent_ratio": (
        "отсутствие характерных для живой речи коротких предложений"
    ),
    "comma_rate": (
        "регулярная плотность расстановки запятых и вводных конструкций"
    ),
    "root_ttr": (
        "специфическое отношение словарного запаса к длине текста (Root-TTR)"
    ),
    "word_len_var": "сглаженный разброс длины используемых слов",
    "dash_rate": "характерная частота использования тире и дефисов",
    "excl_quest_rate": "специфическая частота восклицательных и вопросительных знаков",
    "colon_semi_rate": "структурная плотность двоеточий и точек с запятой",
}

logger = logging.getLogger("DrWebSpamPipeline")
if not logger.handlers:
  handler = logging.StreamHandler()
  handler.setFormatter(
      logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
  )
  logger.addHandler(handler)
  logger.setLevel(logging.INFO)


class SpamAiPipeline:

  def __init__(
      self,
      custom_threshold: float | None = None,
      llm_mode: str | None = None,
      api_url: str | None = None,
  ):
    logger.info("=" * 65)
    logger.info(
        "ИНИЦИАЛИЗАЦИЯ СИСТЕМЫ ДЕТЕКЦИИ DR.WEB (DW-1: MULTIMODAL LATE FUSION)"
    )
    logger.info("=" * 65)
    self.html_detector = HTMLDetector()
    self.text_detector = TextDetector(mode=llm_mode, api_url=api_url)
    self.custom_threshold = custom_threshold
    logger.info("✓ Конвейер (Qwen2.5-1.5B API + CatBoost + HTML Forensics) готов.")

  @staticmethod
  def _attach_ai_spans_to_ast(
      ast_nodes: list[dict[str, Any]] | dict[str, Any] | None,
      most_ai_chunk: str,
      ai_score: float,
  ) -> None:
    if not ast_nodes or not most_ai_chunk:
      return
    target_w = set(re.findall(r"\b\w{4,}\b", most_ai_chunk.lower()))
    if not target_w:
      return

    nodes_list = [ast_nodes] if isinstance(ast_nodes, dict) else ast_nodes
    for node in nodes_list:
      if not isinstance(node, dict):
        continue
      snippet = node.get("text", node.get("text_snippet", ""))
      if snippet:
        snip_w = set(re.findall(r"\b\w{4,}\b", snippet.lower()))
        if snip_w and len(snip_w & target_w) / len(snip_w) >= 0.35:
          node["ai_confidence"] = round(float(ai_score), 4)
      children = node.get("children", [])
      if children:
        SpamAiPipeline._attach_ai_spans_to_ast(
            children, most_ai_chunk, ai_score
        )

  def _extract_shap_top_reasons(
      self, lang: str, feats_dict: dict[str, float], top_k: int = 3
  ) -> list[dict[str, Any]]:
    """Вычисляет локальный вклад признаков (SHAP) для объяснения аналитику SOC (п. 4.1.5.3 ТЗ)."""
    if not feats_dict:
      return []
    try:
      llm_cols = FEATURE_ORDER[:15]
      stylo_cols = FEATURE_ORDER[15:]
      is_llm_empty = all(abs(feats_dict.get(c, 0.0)) < 1e-9 for c in llm_cols)

      # Если запуск без LLM (режим features или фолбэк) — считаем SHAP по cb_stylo (9 фичей)
      if is_llm_empty and hasattr(self.text_detector, "stylo_models") and lang in self.text_detector.stylo_models:
        active_cols = stylo_cols
        cb_model = self.text_detector.stylo_models[lang]
      elif lang in self.text_detector.models:
        active_cols = FEATURE_ORDER
        cb_model = self.text_detector.models[lang]
      else:
        return []

      x_vec = np.array(
          [[feats_dict.get(col, 0.0) for col in active_cols]],
          dtype=np.float64,
      )
      shap_vals = cb_model.get_feature_importance(
          Pool(x_vec), type="ShapValues"
      )[0][:-1]
      top_idx = np.argsort(shap_vals)[::-1][:top_k]

      reasons = []
      for i in top_idx:
        fname = active_cols[i]
        if is_llm_empty and fname in llm_cols:
          continue
        if shap_vals[i] > 0:
          reasons.append({
              "feature": fname,
              "value": round(float(x_vec[0, i]), 4),
              "shap_impact": round(float(shap_vals[i]), 4),
              "description_ru": FEATURE_DESCRIPTIONS_RU.get(
                  fname, f"статистический сигнал {fname}"
              ),
          })
      return reasons
    except Exception:
      return []

  def evaluate_html_risk(
      self,
      html_out: Any,
      desync_score: float = 0.0,
      headers_meta: dict[str, Any] | None = None,
  ) -> tuple[float, float, list[str]]:
    feats = html_out.get("features", {})
    reasons = list(html_out.get("top_risk_factors", []))

    html_ai_score = float(html_out.get("html_ai_proba", 0.0))
    threat_risk = float(html_out.get("cloaking_risk_score", 0.0))

    hidden = feats.get("hidden_elements_count", 0)
    cloaked_ratio = feats.get("cloaked_text_ratio", 0.0)
    dummy = feats.get("dummy_link_count", 0)
    forms = feats.get("has_forms", 0)
    iframes = feats.get("has_iframes", 0)
    zw = feats.get("zero_width_chars_count", 0)
    bidi = feats.get("bidi_spoofing_hits", 0)

    is_legit_esp = bool(
        feats.get("has_mso_comments", 0) > 0
        or (
            feats.get("esp_signature_hits", 0) > 0
            and feats.get("has_unsubscribe_link", 0) > 0
        )
    )

    if forms >= 1:
      threat_risk = max(threat_risk, 0.95)
      reasons.append("CREDENTIAL_PHISHING_FORM")
    if iframes >= 1:
      threat_risk = max(threat_risk, 0.92)
      reasons.append("HIDDEN_IFRAME_PAYLOAD")
    if bidi >= 1:
      threat_risk = max(threat_risk, 0.92)
      reasons.append(f"HOMOGLYPH_OR_BIDI_SPOOFING({bidi})")
    if hidden >= 1 and (
        cloaked_ratio >= 0.05 or (hidden >= 2 and not is_legit_esp)
    ):
      threat_risk = max(threat_risk, 0.90)
      reasons.append(
          f"CLOAKED_HIDDEN_CONTENT(blocks={hidden}, ratio={cloaked_ratio:.2f})"
      )
    if desync_score >= 0.45:
      threat_risk = max(threat_risk, 0.88)
      reasons.append(f"MULTIPART_DESYNC_ATTACK(desync={desync_score})")
    if headers_meta and headers_meta.get("header_layout_mismatch"):
      threat_risk = max(threat_risk, 0.85)
      reasons.append("OUTLOOK_HEADER_SPOOFING_NO_MSO")
    if dummy >= 2:
      html_ai_score = max(html_ai_score, 0.65)
      reasons.append(f"DECEPTIVE_OR_TEMPLATE_LINKS({dummy})")
    if zw > 50:
      threat_risk = max(threat_risk, 0.75)
      reasons.append("ZERO_WIDTH_OBFUSCATION")

    unique_reasons = list(dict.fromkeys(reasons))
    return round(threat_risk, 4), round(html_ai_score, 4), unique_reasons

  def aggregate_verdict(
      self,
      threat_risk: float,
      html_ai_score: float,
      html_reasons: list[str],
      text_res: dict[str, Any],
      has_html: bool = True,
  ) -> dict[str, Any]:
    """Late Fusion: объединение вероятности текстового ядра (Qwen-1.5B + Stylo) и HTML-форензики."""
    p_text = float(text_res.get("score", text_res.get("text_prob", 0.0)))
    lang = text_res.get("lang", "ru")
    feats_dict = text_res.get("features", {})
    is_llm_empty = all(abs(feats_dict.get(c, 0.0)) < 1e-9 for c in FEATURE_ORDER[:15])

    th_key = f"{lang}_stylo" if (is_llm_empty and f"{lang}_stylo" in self.text_detector.thresholds) else lang
    lang_th = self.text_detector.thresholds.get(th_key, self.text_detector.thresholds.get(lang, {}))
    t_soft = (
        self.custom_threshold
        if self.custom_threshold is not None
        else float(lang_th.get("t_soft", 0.60))
    )
    t_strict = float(lang_th.get("t_strict", 0.81))

    # Расчет итогового индекса AI-генерации (Late Fusion с настраиваемыми весами п. 4.1.3.4 ТЗ)
    if not has_html or html_ai_score <= 0.02:
      ai_index = p_text
      primary_mod = "TEXT_QWEN15B_CATBOOST"
    else:
      linear_comb = 0.72 * p_text + 0.28 * html_ai_score
      max_comb = max(p_text, 0.88 * html_ai_score)
      ai_index = 0.65 * max_comb + 0.35 * linear_comb
      if p_text >= 0.50 and html_ai_score >= 0.55:
        ai_index = min(1.0, ai_index + 0.12)
      primary_mod = (
          "LATE_FUSION_ENSEMBLE"
          if html_ai_score > 0.45
          else "TEXT_QWEN15B_CATBOOST"
      )

    ai_index = round(float(np.clip(ai_index, 0.0, 1.0)), 4)

    # Официальный бинарный вердикт по п. 4.1.4.2 ТЗ и А-6 ПМИ
    official_verdict = (
        "AI-Generated" if ai_index >= t_soft else "Human-Written"
    )

    # Детальный статус безопасности для SOC
    reasons: list[str] = []
    if threat_risk >= 0.85:
      sec_status = "SPAM_BLOCKED"
      reasons.extend(html_reasons)
    elif ai_index >= t_strict:
      sec_status = "STRICT_AI_GENERATED"
      reasons.append(
          f"HIGH_AI_CONFIDENCE(text={p_text:.2f}, html={html_ai_score:.2f})"
      )
      reasons.extend(html_reasons)
    elif ai_index >= t_soft:
      sec_status = "SOFT_AI_SUSPICIOUS"
      reasons.append(f"AI_ASSISTED_ORIGIN(score={ai_index:.2f})")
      reasons.extend(html_reasons)
    else:
      sec_status = "CLEAN_HUMAN"
      reasons.append("NO_CRITICAL_ANOMALIES")

    shap_top = self._extract_shap_top_reasons(
        lang, text_res.get("features", {})
    )
    shap_str = ", ".join(r["description_ru"] for r in shap_top)

    flagged_zones = []
    if p_text >= t_soft:
      flagged_zones.append("body_text")
    if has_html and html_ai_score >= 0.52:
      flagged_zones.append("html_structure")

    span_chars = text_res.get("pred_span_chars", [0, 0])
    if official_verdict == "AI-Generated":
      explanation_ru = (
          f"Итоговый индекс AI-генерации: {ai_index:.2f} (порог: {t_soft:.2f},"
          " вердикт: Сгенерировано ИИ / AI-Generated). Признаки генерации"
          " выявлены в зонах:"
          f" {', '.join(flagged_zones) if flagged_zones else 'основной текст'}."
          f" Ключевые сигналы модели: {shap_str or 'статистический профиль LLM'}."
          " Наиболее вероятный сгенерированный фрагмент локализован в"
          f" символьном интервале [{span_chars[0]}, {span_chars[1]}]."
      )
    else:
      explanation_ru = (
          f"Итоговый индекс AI-генерации: {ai_index:.2f} (ниже порога"
          f" {t_soft:.2f}, вердикт: Написано человеком / Human-Written)."
          " Выраженных паттернов машинной генерации в тексте и HTML-разметке"
          " не обнаружено."
      )

    return {
        "verdict": official_verdict,
        "security_status": sec_status,
        "ai_score": ai_index,
        "final_risk": round(max(ai_index, threat_risk), 4),
        "threshold_used": round(t_soft, 4),
        "primary_module": primary_mod,
        "flagged_zones": flagged_zones,
        "explanation": explanation_ru,
        "key_signals_shap": shap_top,
        "reasons": list(dict.fromkeys(reasons)),
    }

  def process_email(self, eml_path: str) -> dict[str, Any]:
    """Полный сквозной прогон одного .eml файла с выдачей JSON по п. 4.1.6.1 ТЗ."""
    t0 = time.perf_counter()
    parsed = parse_eml(eml_path, html_detector=self.html_detector)
    try:
      clean_visible_text = parsed.get("clean_text", "")
      raw_html = parsed.get("raw_html", "")
      desync_score = parsed.get("multipart_desync_score", 0.0)
      headers_meta = parsed.get("headers_meta", {})

      text_out = self.text_detector.predict(clean_visible_text)
      most_ai_chunk = text_out.get("most_ai_chunk", "")
      span_chars = text_out.get("pred_span_chars", [0, 0])

      html_out = parsed.get("html_forensics") or self.html_detector.predict(
          raw_html
      )
      html_feats = html_out.get("features", {})

      raw_ast = html_out.get("ast_tree")
      ast_list = (
          raw_ast.to_dict() if hasattr(raw_ast, "to_dict") else raw_ast
      )
      if most_ai_chunk and ast_list:
        self._attach_ai_spans_to_ast(
            ast_list, most_ai_chunk, text_out.get("score", 0.0)
        )

      threat_risk, html_ai_score, html_reasons = self.evaluate_html_risk(
          html_out, desync_score=desync_score, headers_meta=headers_meta
      )
      decision = self.aggregate_verdict(
          threat_risk,
          html_ai_score,
          html_reasons,
          text_out,
          has_html=bool(raw_html.strip()),
      )
      elapsed_sec = round(time.perf_counter() - t0, 3)

      return {
          "file_id": Path(eml_path).name,
          "file": Path(eml_path).name,
          "email_info": {
              "message_id": parsed.get("id", ""),
              "subject": parsed.get("subject", ""),
              "lang": text_out.get("lang", "ru"),
              "headers_meta": headers_meta,
          },
          "subject": parsed.get("subject", ""),
          "ai_score": decision["ai_score"],
          "risk_score": decision["final_risk"],
          "verdict": decision["verdict"],
          "security_status": decision["security_status"],
          "threshold": decision["threshold_used"],
          "method": "hybrid_qwen15b_catboost_late_fusion",
          "primary_module": decision["primary_module"],
          "flagged_zones": decision["flagged_zones"],
          "zones": {
              "body_text": {
                  "ai_probability": round(
                      float(text_out.get("score", 0.0)), 4
                  ),
                  "status": text_out.get("status", "CLEAN"),
                  "char_interval": span_chars,
                  "most_ai_chunk": most_ai_chunk[:250],
              },
              "html_structure": {
                  "ai_probability": html_ai_score,
                  "cloaking_risk": threat_risk,
                  "is_cloaking_detected": bool(
                      html_out.get("is_cloaking_detected", False)
                  ),
              },
          },
          "explanation": decision["explanation"],
          "key_signals": decision["key_signals_shap"],
          "reasons": decision["reasons"],
          "processing_time_sec": elapsed_sec,
          "details": {
              "html_threat_risk": threat_risk,
              "html_ai_proba": html_ai_score,
              "text_prob": round(float(text_out.get("score", 0.0)), 4),
              "lang": text_out.get("lang", "ru"),
              "most_ai_chunk": most_ai_chunk[:160],
              "pred_span_chars": span_chars,
              "hidden_css_tags": html_feats.get("hidden_elements_count", 0),
              "cloaked_text_ratio": html_feats.get("cloaked_text_ratio", 0.0),
              "dummy_links": html_feats.get("dummy_link_count", 0),
              "desync_score": desync_score,
              "ast_tree": ast_list,
          },
      }
    finally:
      cleanup_extracted_files(parsed)

  @staticmethod
  def _is_valid_email_candidate(file_path: Path) -> bool:
    """Отсекает служебные файлы/картинки, но пропускает письма с любыми именами (хэши, без .eml)."""
    if not file_path.is_file() or file_path.name.startswith("."):
      return False
    ignore_exts = {
        ".json", ".jsonl", ".csv", ".md", ".py", ".cbm", ".joblib",
        ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".zip", ".7z", ".log"
    }
    if file_path.suffix.lower() in ignore_exts:
      return False
    return True

  @staticmethod
  def _validate_parsed_email(file_path: str, parsed: dict[str, Any]) -> None:
    """Проверка на поврежденное письмо (Тест А-11 ПМИ)."""
    with open(file_path, "rb") as f:
      head_bytes = f.read(4096)

    if not head_bytes or not head_bytes.strip():
      raise ValueError("Пустой файл письма")

    # Если в начале файла бинарные нули (например, подсунули архив или картинку под видом письма)
    if b"\x00\x00" in head_bytes[:512]:
      raise ValueError("Бинарный или поврежденный файл (не текст RFC 5322)")

    head_text = head_bytes.decode("utf-8", errors="ignore")
    has_rfc_header = bool(
        re.search(
            r"^(From|To|Subject|Date|Message-ID|MIME-Version|Content-Type|Received|Return-Path)\s*:",
            head_text,
            flags=re.IGNORECASE | re.MULTILINE,
        )
    )
    clean_text = parsed.get("clean_text", "").strip()
    raw_html = parsed.get("raw_html", "").strip()

    if not has_rfc_header and not raw_html:
      raise ValueError("Отсутствуют заголовки RFC 5322 и структура письма (поврежденный файл)")
    if not clean_text and not raw_html and not parsed.get("subject"):
      raise ValueError("Письмо не содержит текстовой или HTML части")

  def process_directory(
      self, dir_path: str, out_dir: str | None = None
  ) -> dict[str, Any]:
    """Пакетный анализ каталога писем (поддерживает и .eml, и файлы-хэши без расширения; тест Ф-2, Н-1 ПМИ)."""
    root = Path(dir_path)
    # Сначала ищем все подходящие файлы (включая хэши без .eml)
    eml_files = sorted(p for p in root.rglob("*") if self._is_valid_email_candidate(p))
    logger.info(
        f"Найдено {len(eml_files)} файлов писем в каталоге: {dir_path}"
    )

    if out_dir:
      os.makedirs(out_dir, exist_ok=True)

    results = []
    errors = []

    for eml_file in eml_files:
      try:

        out_f = Path(out_dir) / f"{eml_file.name}.json" if out_dir else None
        # Поддержка возобновления после прерывания (тест Н-2 / А-12 п. 2 ПМИ)
        if out_f and out_f.exists():
          with open(out_f, "r", encoding="utf-8") as cached_f:
            res = json.load(cached_f)
          results.append(res)
          logger.info(f"[CACHED] {eml_file.name:<25} | Verdict: {res['verdict']:<14} | AI Score: {res['ai_score']:.3f}")
          continue

        # Внутри process_email сначала парсим и проверяем целостность
        parsed_check = parse_eml(str(eml_file), html_detector=None)
        cleanup_extracted_files(parsed_check)
        self._validate_parsed_email(str(eml_file), parsed_check)

        res = self.process_email(str(eml_file))
        results.append(res)
        if out_dir:
          out_f = Path(out_dir) / f"{eml_file.name}.json"
          with open(out_f, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        logger.info(
            f"[OK] {eml_file.name:<25} | Verdict: {res['verdict']:<14} | AI"
            f" Score: {res['ai_score']:.3f}"
        )
      except Exception as e:
        logger.warning(
            f"[SKIP] Поврежденное или неразбираемое письмо {eml_file.name}: {e}"
        )
        errors.append({"file": eml_file.name, "error": str(e)})

    ai_count = sum(1 for r in results if r["verdict"] == "AI-Generated")
    summary = {
        "total_found": len(eml_files),
        "processed_ok": len(results),
        "skipped_corrupted": len(errors),
        "ai_generated_count": ai_count,
        "human_written_count": len(results) - ai_count,
        "errors": errors,
        "results": results,
    }
    if out_dir:
      sum_path = Path(out_dir) / "summary.json"
      with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
      logger.info(f"✓ Сводный JSON сохранен в: {sum_path}")
    return summary

  def process_jsonl(self, jsonl_path: str):
    """Пакетный прогон .jsonl выборки с консольной таблицей."""
    with open(jsonl_path, "r", encoding="utf-8") as f:
      lines = [json.loads(line) for line in f if line.strip()]

    print(f"\nЗапуск анализа выборки из {len(lines)} записей...")
    print("-" * 105)
    print(
        f"{'Файл/ID':<14} | {'Вердикт ТЗ':<15} | {'AI Индекс':<9} | {'Модуль':<24} |"
        " {'Краткое объяснение'}"
    )
    print("-" * 105)

    for idx, item in enumerate(lines):
      raw_html = item.get("raw_html", item.get("html", ""))
      fallback_text = item.get("text", item.get("clean_text", ""))
      file_name = str(item.get("file_name", item.get("id", f"row_{idx}")))[:14]

      html_out = self.html_detector.predict(raw_html)
      clean_text = (
          html_out.get("visible_text")
          if raw_html.strip()
          else fallback_text
      ) or fallback_text

      text_out = self.text_detector.predict(
          clean_text, force_lang=item.get("lang")
      )
      threat_risk, html_ai_score, html_reasons = self.evaluate_html_risk(
          html_out
      )

      decision = self.aggregate_verdict(
          threat_risk,
          html_ai_score,
          html_reasons,
          text_out,
          has_html=bool(raw_html.strip()),
      )
      short_exp = decision["explanation"][:55] + "..."
      print(
          f"{file_name:<14} | {decision['verdict']:<15} |"
          f" {decision['ai_score']:<9.4f} | {decision['primary_module']:<24} |"
          f" {short_exp}"
      )


_pipeline_instance = None


def scan_email(eml_path: str) -> dict[str, Any]:
  global _pipeline_instance
  if _pipeline_instance is None:
    _pipeline_instance = SpamAiPipeline()
  return _pipeline_instance.process_email(eml_path)


def main():

  
  parser = argparse.ArgumentParser(
      description="Dr.Web Email AI Detector (DW-1)"
  )
  parser.add_argument("--eml", type=str, help="Путь к отдельному .eml файлу")
  parser.add_argument(
      "--dir", type=str, help="Путь к каталогу с .eml письмами (пакетный режим)"
  )
  parser.add_argument("--jsonl", type=str, help="Путь к .jsonl датасету")
  parser.add_argument(
      "--output",
      "-o",
      type=str,
      help="Путь для сохранения выходного JSON (или папки для --dir)",
  )
  parser.add_argument(
      "--threshold",
      type=float,
      default=None,
      help="Переопределить порог классификации (0..1)",
  )

  parser.add_argument(
      "--mode",
      type=str,
      choices=["remote", "features", "local"],
      default=None,
      help="Режим оценки: remote (API в контуре разработчика), features (только CPU-признаки), local",
  )
  parser.add_argument(
      "--api-url",
      type=str,
      default=None,
      help="Переопределить URL OpenAI-совместимого сервера LLM",
  )

  args = parser.parse_args()

  pipeline = SpamAiPipeline(
      custom_threshold=args.threshold,
      llm_mode=args.mode,
      api_url=args.api_url,
  )

  if args.eml:
    res = pipeline.process_email(args.eml)
    print("\n" + "=" * 75)
    print(f"ФАЙЛ      : {res['file_id']}")
    print(f"ИНДЕКС ИИ : {res['ai_score']:.4f} (порог: {res['threshold']:.4f})")
    print(f"ВЕРДИКТ   : {res['verdict']} ({res['security_status']})")
    print(f"ОБЪЯСНЕНИЕ: {res['explanation']}")
    print("=" * 75)
    if args.output:
      with open(args.output, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
      logger.info(f"✓ Полный результат сохранен в {args.output}")
    else:
      print(json.dumps(res, ensure_ascii=False, indent=2))
  elif args.dir:
    pipeline.process_directory(args.dir, out_dir=args.output or "results_out")
  elif args.jsonl:
    pipeline.process_jsonl(args.jsonl)
  else:
    parser.print_help()

if __name__ == "__main__":
  main()