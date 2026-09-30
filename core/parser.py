import os
import re
import shutil
import tempfile
from email import policy
from email.parser import BytesParser
from bs4 import BeautifulSoup

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"}

def decode_payload_safely(part) -> str:
    raw = part.get_payload(decode=True)
    if not raw:
        return ""
    
    charset = part.get_content_charset()
    encodings = [charset] if charset else []
    encodings.extend(["utf-8", "windows-1251", "koi8-r", "iso-8859-5"])

    for enc in encodings:
        if not enc:
            continue
        try:
            return raw.decode(enc)
        except (LookupError, UnicodeDecodeError):
            continue

    return raw.decode("utf-8", errors="replace")


def parse_eml(path: str, extract_dir: str | None = None) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"EML файл не найден: {path}")

    with open(path, "rb") as f:
        msg = BytesParser(policy=policy.default).parse(f)

    msg_id = msg.get("Message-ID", os.path.basename(path)).strip("<> ")
    subj = " ".join(str(msg.get("Subject", "")).split())

    txt_parts = []
    html_parts = []
    image_paths = []

    clean_id = "".join(c for c in msg_id if c.isalnum())[:12] or "temp"
    if extract_dir is None:
        extract_dir = os.path.join(tempfile.gettempdir(), f"eml_{clean_id}")
    os.makedirs(extract_dir, exist_ok=True)

    if msg.is_multipart():
        for p in msg.walk():
            ctype = p.get_content_type().lower()
            disp = str(p.get("Content-Disposition", "")).lower()
            fname = p.get_filename()

            # Санитизация имени файла (защита от Path Traversal / ../../)
            if fname:
                fname = os.path.basename(fname)
                fname = re.sub(r'[^a-zA-Z0-9._-]', '_', fname)

            if ctype.startswith("image/") or (fname and any(fname.lower().endswith(ext) for ext in IMAGE_EXTENSIONS)):
                data = p.get_payload(decode=True)
                if data:
                    out_name = fname or f"image_{len(image_paths)+1}.png"
                    out_path = os.path.join(extract_dir, out_name)
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

    raw_text = "\n".join(txt_parts).strip()
    raw_html = "\n".join(html_parts).strip()

    img_tags_count = 0
    html_clean_text = ""
    if raw_html:
        soup = BeautifulSoup(raw_html, "lxml")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        img_tags_count = len(soup.find_all("img"))
        html_clean_text = soup.get_text(separator=" ", strip=True)

    # Защита от трюка с раздвоением: объединяем, если оба содержат значимый текст
    if raw_text and html_clean_text and raw_text != html_clean_text:
        combined_text = f"{raw_text}\n\n{html_clean_text}"
    else:
        combined_text = raw_text or html_clean_text

    combined_text = " ".join(combined_text.split())

    return {
        "id": msg_id,
        "subject": subj,
        "clean_text": combined_text,
        "raw_html": raw_html,
        "attachments": image_paths,
        "extract_dir": extract_dir, 
        "media_flags": {
            "has_images": len(image_paths) > 0 or img_tags_count > 0,
            "images_count": max(len(image_paths), img_tags_count)
        }
    }