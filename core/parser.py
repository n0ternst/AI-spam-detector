import os
from email import policy
from email.parser import BytesParser
from bs4 import BeautifulSoup


def parse_eml(path: str):
    with open(path, "rb") as f:
        msg = BytesParser(policy=policy.default).parse(f)

    alt_id = os.path.basename(path)
    return get_data(msg, alt_id)


def get_text(part):
    try:
        content = part.get_content()
        return content if content else ""
    except (LookupError, UnicodeDecodeError, ValueError):
        raw_b = part.get_payload(decode=True)
        if raw_b:
            return raw_b.decode("utf-8", errors="replace")
        return ""


def get_data(msg, alt_id: str):
    msg_id = msg.get("Message-ID")
    if msg_id:
        msg_id = msg_id.strip("<> ")
    else:
        msg_id = alt_id

    subj = " ".join(str(msg.get("Subject", "")).split())

    txt = ""
    html = ""

    if msg.is_multipart():
        for p in msg.walk():
            disp = str(p.get("Content-Disposition", "")).lower()

            if "attachment" in disp:
                continue

            ctype = p.get_content_type()

            if ctype == "text/plain":
                txt += get_text(p)
            elif ctype == "text/html":
                html += get_text(p)

    else:
        ctype = msg.get_content_type()

        if ctype == "text/plain":
            txt = get_text(msg)
        elif ctype == "text/html":
            html = get_text(msg)

    img_count = 0

    if html:
        soup = BeautifulSoup(html, "lxml")

        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        img_count = len(soup.find_all("img"))

        clean_txt = soup.get_text(separator=" ", strip=True)

        if not clean_txt:
            clean_txt = txt.strip()
    else:
        clean_txt = txt.strip()

    clean_txt = " ".join(clean_txt.split())

    return {
        "id": msg_id,
        "subject": subj,
        "media_flags": {
            "has_images": img_count > 0,
            "images_count": img_count
        },
        "clean_text": clean_txt,
        "raw_html": html
    }


