"""
EML Parser & MIME Forensics Extractor.
Provides multipart desync detection (R_PARTS_DIFFER), anti-poisoning text
extraction, inline Base64 image decoding for OCR, and SMTP header forensics.
"""

import base64
from email import policy
from email.parser import BytesParser
from email.utils import parseaddr
import hashlib
import os
import re
import shutil
import tempfile
from typing import Any
from bs4 import BeautifulSoup

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff", ".gif"}
MAX_IMAGE_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB лимит на вложение
MIN_OCR_IMAGE_BYTES = 450  # Отсекаем 1x1 трекинг-пиксели перед отправкой в PaddleOCR
MAX_INLINE_B64_IMAGES = 5

_ZW_REGEX = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff\u00ad]")
_DATA_IMG_REGEX = re.compile(
    r"data:image/(png|jpeg|jpg|webp|gif);base64,([A-Za-z0-9+/=\s]+)", re.I
)


def decode_payload_safely(part) -> str:
    """Безопасное декодирование тела письма с каскадным перебором кодировок."""
    raw = part.get_payload(decode=True)
    if not raw:
        return ""

    charset = part.get_content_charset()
    encodings_to_try = [charset] if charset else []
    encodings_to_try.extend(
        ["utf-8", "windows-1251", "koi8-r", "iso-8859-5", "cp866", "latin-1"]
    )

    for enc in encodings_to_try:
        if not enc:
            continue
        try:
            return raw.decode(enc)
        except (LookupError, UnicodeDecodeError):
            continue

    return raw.decode("utf-8", errors="replace")


def calculate_jaccard_desync(text_plain: str, text_html: str) -> float:
    """Расчет коэффициента нечеткого расхождения тел письма (Rspamd R_PARTS_DIFFER)."""
    words_plain = set(re.findall(r"\b\w{3,}\b", text_plain.lower()))
    words_html = set(re.findall(r"\b\w{3,}\b", text_html.lower()))

    if not words_plain or not words_html:
        return 0.0

    intersection = len(words_plain & words_html)
    union = len(words_plain | words_html)
    similarity = intersection / union if union > 0 else 1.0
    return round(1.0 - similarity, 4)


def _normalize_preserving_newlines(text: str) -> str:
    """ОчищаетZero-Width символы и лишние пробелы, сохраняя переносы строк \\n для TextDetector."""
    if not text:
        return ""
    cleaned = _ZW_REGEX.sub("", text)
    lines = [" ".join(line.split()) for line in cleaned.splitlines()]
    non_empty_lines = [line for line in lines if line]
    return "\n".join(non_empty_lines).strip()


def _extract_domain(header_val: str) -> str:
    """Извлекает домен из email-заголовка From / Return-Path / Reply-To."""
    _, addr = parseaddr(header_val or "")
    if "@" in addr:
        return addr.split("@", 1)[1].strip().lower().rstrip(">")
    return ""


def parse_eml(
    path: str,
    extract_dir: str | None = None,
    html_detector: Any | None = None,
) -> dict[str, Any]:
    """
    Разбор MIME-структуры письма с изоляцией вложений, защитой от отравления NLP
    скрытым HTML/Plain-текстом и криминалистикой заголовков (X-Mailer / Domain Spoofing).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"EML файл не найден: {path}")

    with open(path, "rb") as f:
        msg = BytesParser(policy=policy.default).parse(f)

    msg_id = str(msg.get("Message-ID", os.path.basename(path))).strip("<> ")
    subj = _normalize_preserving_newlines(str(msg.get("Subject", "")))

    # Извлечение заголовков для криминалистики SMTP
    x_mailer = str(msg.get("X-Mailer", msg.get("User-Agent", ""))).strip()
    has_list_unsub_header = msg.get("List-Unsubscribe") is not None
    auth_results = str(msg.get("Authentication-Results", "")).lower()

    from_domain = _extract_domain(str(msg.get("From", "")))
    return_path_domain = _extract_domain(str(msg.get("Return-Path", "")))
    reply_to_domain = _extract_domain(str(msg.get("Reply-To", "")))

    # Проверка несовпадения домена Reply-To / From (классический признак фишинга/BEC)
    reply_to_mismatch = bool(
        from_domain and reply_to_domain and (from_domain != reply_to_domain)
    )

    txt_parts: list[str] = []
    html_parts: list[str] = []
    image_paths: list[str] = []
    tracking_pixels_skipped = 0

    if extract_dir is None:
        id_hash = hashlib.md5(
            msg_id.encode("utf-8", errors="ignore")
        ).hexdigest()[:12]
        extract_dir = os.path.join(tempfile.gettempdir(), f"eml_scan_{id_hash}")
    os.makedirs(extract_dir, exist_ok=True)

    parts_to_walk = msg.walk() if msg.is_multipart() else [msg]

    for idx, p in enumerate(parts_to_walk):
        if p.is_multipart():
            continue

        ctype = p.get_content_type().lower()
        disp = str(p.get("Content-Disposition", "")).lower()
        fname = p.get_filename()

        is_image_type = ctype.startswith("image/")
        has_image_ext = bool(
            fname
            and any(fname.lower().endswith(ext) for ext in IMAGE_EXTENSIONS)
        )

        if is_image_type or has_image_ext:
            data = p.get_payload(decode=True)
            if data:
                if len(data) < MIN_OCR_IMAGE_BYTES:
                    # Игнорируем 1x1 пиксели слежения, чтобы не нагружать PaddleOCR
                    tracking_pixels_skipped += 1
                elif len(data) <= MAX_IMAGE_SIZE_BYTES:
                    base_fname = (
                        os.path.basename(fname) if fname else f"img_{idx}.png"
                    )
                    safe_name = (
                        f"{idx}_{re.sub(r'[^a-zA-Z0-9._-]', '_', base_fname)}"
                    )
                    out_path = os.path.join(extract_dir, safe_name)
                    with open(out_path, "wb") as img_f:
                        img_f.write(data)
                    image_paths.append(out_path)
            continue

        if "attachment" in disp:
            continue

        if ctype == "text/plain":
            txt_parts.append(decode_payload_safely(p))
        elif ctype == "text/html":
            html_parts.append(decode_payload_safely(p))

    raw_plain_text = "\n".join(txt_parts).strip()
    raw_html = "\n".join(html_parts).strip()

    # Декодируем встроенные в HTML data:image/...;base64 картинки для OCR
    if raw_html and len(image_paths) < MAX_INLINE_B64_IMAGES:
        for b_idx, match in enumerate(_DATA_IMG_REGEX.finditer(raw_html)):
            if len(image_paths) >= MAX_INLINE_B64_IMAGES:
                break
            ext, b64_payload = match.group(1).lower(), match.group(2)
            try:
                img_bytes = base64.b64decode(b64_payload)
                if MIN_OCR_IMAGE_BYTES <= len(img_bytes) <= MAX_IMAGE_SIZE_BYTES:
                    out_path = os.path.join(extract_dir, f"inline_b64_{b_idx}.{ext}")
                    with open(out_path, "wb") as img_f:
                        img_f.write(img_bytes)
                    image_paths.append(out_path)
            except Exception:
                continue

    img_tags_count = 0
    html_extracted_text = ""
    html_forensics: dict[str, Any] | None = None
    html_features: dict[str, float] = {}

    if raw_html:
        img_tags_count = len(re.findall(r"<img\b", raw_html, flags=re.I))
        if html_detector is not None:
            # Используем адаптер evaluate_html_risk (или predict) нашего 46-фичевого HTMLDetector
            if hasattr(html_detector, "evaluate_html_risk"):
                html_forensics = html_detector.evaluate_html_risk(raw_html)
                html_extracted_text = html_forensics.get("visible_text", "")
                html_features = (
                    html_forensics.get("evidence", {}).get("features", {})
                )
            else:
                raw_res = html_detector.predict(raw_html)
                if isinstance(raw_res, dict):
                    html_forensics = raw_res
                    html_extracted_text = raw_res.get("visible_text", "")
                    html_features = raw_res.get(
                        "features", raw_res.get("evidence", {}).get("features", {})
                    )
                else:
                    html_extracted_text = getattr(raw_res, "visible_text", "")
                    evidence = getattr(raw_res, "evidence", {}) or {}
                    html_features = evidence.get("features", {})
                    html_forensics = {
                        "verdict": getattr(raw_res, "verdict", "HUMAN_LEGITIMATE"),
                        "html_ai_score": getattr(raw_res, "global_risk_score", 0.0),
                        "global_risk_score": getattr(raw_res, "global_risk_score", 0.0),
                        "is_cloaking_detected": getattr(raw_res, "is_cloaking_detected", False),
                        "forensic_flags": getattr(raw_res, "top_risk_factors", []),
                        "top_risk_factors": getattr(raw_res, "top_risk_factors", []),
                        "evidence": evidence,
                        "ast": getattr(raw_res, "semantic_ast", []),
                        "semantic_ast": getattr(raw_res, "semantic_ast", []),
                        "visible_text": html_extracted_text,
                        "features": html_features,
                    }
        else:
            try:
                soup = BeautifulSoup(raw_html, "lxml")
            except Exception:
                soup = BeautifulSoup(raw_html, "html.parser")
            for tag in soup(["script", "style", "noscript", "head", "title"]):
                tag.decompose()
            html_extracted_text = soup.get_text(separator="\n", strip=True)

    desync_score = calculate_jaccard_desync(raw_plain_text, html_extracted_text)
    is_multipart_cloaking = bool(
        raw_plain_text and html_extracted_text and desync_score > 0.35
    )

    # Защита от Multipart Perplexity Dilution:
    # Пользователь в почтовом клиенте видит ТОЛЬКО HTML-версию (если она не пустая).
    # Не позволяем фейковому text/plain размывать статистику токенов в TextDetector!
    if html_extracted_text and len(html_extracted_text.strip()) >= 25:
        primary_visible_text = html_extracted_text
    else:
        primary_visible_text = raw_plain_text or html_extracted_text

    final_clean_text = _normalize_preserving_newlines(primary_visible_text)

    # Проверка на подделку заголовка почтового клиента (X-Mailer Spoofing)
    header_layout_mismatch = False
    if html_forensics is not None:
        mso_count = html_features.get(
            "mso_conditional_count", html_features.get("mso_comment_count", 0.0)
        )
        ai_clip = html_features.get("ai_clipboard_markers", 0.0)
        if "outlook" in x_mailer.lower() and mso_count == 0:
            header_layout_mismatch = True
        elif any(m in x_mailer.lower() for m in ("apple mail", "iphone mail")) and (
            mso_count > 0 or ai_clip > 0
        ):
            header_layout_mismatch = True

    return {
        "id": msg_id,
        "subject": subj,
        "clean_text": final_clean_text,
        "raw_html": raw_html,
        "raw_plain_text": raw_plain_text,
        "html_extracted_text": html_extracted_text,
        "multipart_desync_score": desync_score,
        "is_multipart_cloaking": is_multipart_cloaking,
        "headers_meta": {
            "x_mailer": x_mailer,
            "from_domain": from_domain,
            "return_path_domain": return_path_domain,
            "reply_to_mismatch": reply_to_mismatch,
            "has_list_unsubscribe_header": has_list_unsub_header,
            "dkim_pass": "dkim=pass" in auth_results,
            "spf_pass": "spf=pass" in auth_results,
            "dmarc_pass": "dmarc=pass" in auth_results,
            "header_layout_mismatch": header_layout_mismatch,
        },
        "html_forensics": html_forensics,
        "attachments": image_paths,
        "extract_dir": extract_dir,
        "media_flags": {
            "has_images": len(image_paths) > 0 or img_tags_count > 0,
            "images_count": max(len(image_paths), img_tags_count),
            "tracking_pixels_skipped": tracking_pixels_skipped,
        },
    }


def cleanup_extracted_files(parsed_eml: dict[str, Any]) -> None:
    """Удаляет временную директорию с вложениями после завершения работы OCR/Vision."""
    ext_dir = parsed_eml.get("extract_dir")
    if ext_dir and os.path.exists(ext_dir):
        shutil.rmtree(ext_dir, ignore_errors=True)
