import os
import re
import tempfile
from email import policy
from email.parser import BytesParser
from bs4 import BeautifulSoup

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"}


def decode_payload_safely(part) -> str:
    """Безопасное декодирование тела письма с каскадным перебором кодировок."""
    raw = part.get_payload(decode=True)
    if not raw:
        return ""

    charset = part.get_content_charset()
    encodings_to_try = [charset] if charset else []
    encodings_to_try.extend(["utf-8", "windows-1251", "koi8-r", "iso-8859-5", "cp866"])

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


def parse_eml(path: str, extract_dir: str | None = None) -> dict:
    """Разбор MIME-структуры письма с защитой от уязвимостей и изоляцией вложений."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"EML файл не найден: {path}")

    with open(path, "rb") as f:
        msg = BytesParser(policy=policy.default).parse(f)

    msg_id = str(msg.get("Message-ID", os.path.basename(path))).strip("<> ")
    subj = " ".join(str(msg.get("Subject", "")).split())

    txt_parts = []
    html_parts = []
    image_paths = []

    if extract_dir is None:
        clean_id = "".join(c for c in msg_id if c.isalnum())[:16] or "temp_mail"
        extract_dir = os.path.join(tempfile.gettempdir(), f"eml_scan_{clean_id}")
    os.makedirs(extract_dir, exist_ok=True)

    if msg.is_multipart():
        for p in msg.walk():
            ctype = p.get_content_type().lower()
            disp = str(p.get("Content-Disposition", "")).lower()
            fname = p.get_filename()

            is_image_type = ctype.startswith("image/")
            has_image_ext = bool(fname and any(fname.lower().endswith(ext) for ext in IMAGE_EXTENSIONS))

            if is_image_type or has_image_ext:
                data = p.get_payload(decode=True)
                if data:
                    safe_name = os.path.basename(fname) if fname else f"image_{len(image_paths)+1}.png"
                    safe_name = re.sub(r"[^a-zA-Z0-9._-]", "_", safe_name)
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
    else:
        ctype = msg.get_content_type().lower()
        if ctype == "text/plain":
            txt_parts.append(decode_payload_safely(msg))
        elif ctype == "text/html":
            html_parts.append(decode_payload_safely(msg))

    raw_plain_text = "\n".join(txt_parts).strip()
    raw_html = "\n".join(html_parts).strip()

    img_tags_count = 0
    html_extracted_text = ""
    if raw_html:
        soup = BeautifulSoup(raw_html, "lxml")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        img_tags_count = len(soup.find_all("img"))
        html_extracted_text = soup.get_text(separator=" ", strip=True)

    desync_score = calculate_jaccard_desync(raw_plain_text, html_extracted_text)

    if raw_plain_text and html_extracted_text and (raw_plain_text != html_extracted_text):
        combined_text = f"{raw_plain_text}\n\n{html_extracted_text}"
    else:
        combined_text = raw_plain_text or html_extracted_text

    final_clean_text = " ".join(combined_text.split())

    return {
        "id": msg_id,
        "subject": subj,
        "clean_text": final_clean_text,
        "raw_html": raw_html,
        "raw_plain_text": raw_plain_text,
        "html_extracted_text": html_extracted_text,
        "multipart_desync_score": desync_score,
        "attachments": image_paths,
        "extract_dir": extract_dir,
        "media_flags": {
            "has_images": len(image_paths) > 0 or img_tags_count > 0,
            "images_count": max(len(image_paths), img_tags_count)
        }
    }
