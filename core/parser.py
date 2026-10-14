"""
Enterprise EML Parser & MIME Forensics Extractor.
"""

from email import policy
from email.parser import BytesParser
import hashlib
import os
import re
import shutil
import tempfile
from typing import Any
from bs4 import BeautifulSoup

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff", ".gif"}
MAX_IMAGE_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB лимит на вложение


def decode_payload_safely(part) -> str:
    """Безопасное декодирование тела письма с каскадным перебором кодировок."""
    raw = part.get_payload(decode=True)
    if not raw:
        return ""

    charset = part.get_content_charset()
    encodings_to_try = [charset] if charset else []
    encodings_to_try.extend(
        ["utf-8", "windows-1251", "koi8-r", "iso-8859-5", "cp866"]
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


def parse_eml(
    path: str,
    extract_dir: str | None = None,
    html_detector: Any | None = None,
) -> dict[str, Any]:
    """
    Разбор MIME-структуры письма с изоляцией вложений, защитой от отравления NLP
    скрытым HTML-текстом и детектом подделки почтового клиента (X-Mailer spoofing).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"EML файл не найден: {path}")

    with open(path, "rb") as f:
        msg = BytesParser(policy=policy.default).parse(f)

    msg_id = str(msg.get("Message-ID", os.path.basename(path))).strip("<> ")
    subj = " ".join(str(msg.get("Subject", "")).split())

    # Извлечение заголовков для криминалистики
    x_mailer = str(msg.get("X-Mailer", msg.get("User-Agent", ""))).strip()
    has_list_unsub_header = msg.get("List-Unsubscribe") is not None
    auth_results = str(msg.get("Authentication-Results", "")).lower()

    txt_parts: list[str] = []
    html_parts: list[str] = []
    image_paths: list[str] = []

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
            if data and len(data) <= MAX_IMAGE_SIZE_BYTES:
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

    img_tags_count = 0
    html_extracted_text = ""
    html_forensics = None

    if raw_html:
        if html_detector is not None:
            # Единый проход: получаем и форензику, и гарантированно ВИДИМЫЙ текст (без клоакинга)
            html_forensics = html_detector.predict(raw_html)
            html_extracted_text = html_forensics.get("visible_text", "")
            img_tags_count = int(
                html_forensics.get("features", {}).get("image_count", 0)
            )
        else:
            try:
                soup = BeautifulSoup(raw_html, "lxml")
            except Exception:
                soup = BeautifulSoup(raw_html, "html.parser")
            for tag in soup(["script", "style", "noscript", "head", "title"]):
                tag.decompose()
            img_tags_count = len(soup.find_all("img"))
            html_extracted_text = soup.get_text(separator=" ", strip=True)

    desync_score = calculate_jaccard_desync(raw_plain_text, html_extracted_text)

    # Защита от дублирования текста для NLP: склеиваем только при реальном расхождении частей
    if raw_plain_text and html_extracted_text:
        if desync_score > 0.35:
            combined_text = f"{raw_plain_text}\n\n{html_extracted_text}"
        else:
            combined_text = (
                html_extracted_text
                if len(html_extracted_text) >= len(raw_plain_text)
                else raw_plain_text
            )
    else:
        combined_text = html_extracted_text or raw_plain_text

    final_clean_text = " ".join(combined_text.split())

    # Проверка на подделку заголовка Outlook (заявлен Outlook, но нет MSO-разметки)
    header_layout_mismatch = False
    if "outlook" in x_mailer.lower() and html_forensics is not None:
        mso_count = html_forensics.get("features", {}).get(
            "mso_comment_count", 0
        )
        if mso_count == 0:
            header_layout_mismatch = True

    return {
        "id": msg_id,
        "subject": subj,
        "clean_text": final_clean_text,
        "raw_html": raw_html,
        "raw_plain_text": raw_plain_text,
        "html_extracted_text": html_extracted_text,
        "multipart_desync_score": desync_score,
        "headers_meta": {
            "x_mailer": x_mailer,
            "has_list_unsubscribe_header": has_list_unsub_header,
            "dkim_pass": "dkim=pass" in auth_results,
            "spf_pass": "spf=pass" in auth_results,
            "header_layout_mismatch": header_layout_mismatch,
        },
        "html_forensics": html_forensics,
        "attachments": image_paths,
        "extract_dir": extract_dir,
        "media_flags": {
            "has_images": len(image_paths) > 0 or img_tags_count > 0,
            "images_count": max(len(image_paths), img_tags_count),
        },
    }


def cleanup_extracted_files(parsed_eml: dict[str, Any]) -> None:
    """Удаляет временную директорию с вложениями после завершения работы OCR/Vision."""
    ext_dir = parsed_eml.get("extract_dir")
    if ext_dir and os.path.exists(ext_dir):
        shutil.rmtree(ext_dir, ignore_errors=True)
