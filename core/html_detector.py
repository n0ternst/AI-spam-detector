import os
import re
import json
import pandas as pd
from bs4 import BeautifulSoup, Comment

PLACEHOLDER_REGEX = [
    re.compile(r"\[[A-Za-zА-Яа-яЁё _]{2,30}\]"),
    re.compile(r"\{\{[a-zA-Z0-9_]+\}\}"),
    re.compile(r"\{%[^%]+%\}"),
]

MARKDOWN_REGEX = [
    re.compile(r"\*\*[^*]+\*\*"),
    re.compile(r"(?<!\w)#{1,3}\s+\S"),
    re.compile(r"(?m)^\s*[-*]\s+\S"),
]

HIDDEN_CSS_REGEX = [
    re.compile(r"display\s*:\s*none", re.IGNORECASE),
    re.compile(r"visibility\s*:\s*hidden", re.IGNORECASE),
    re.compile(r"opacity\s*:\s*0(\.0)?", re.IGNORECASE),
    re.compile(r"font-size\s*:\s*0(px|pt|em)?", re.IGNORECASE),
    re.compile(r"text-indent\s*:\s*-[0-9]{3,5}px", re.IGNORECASE),
    re.compile(r"left\s*:\s*-[0-9]{3,5}px", re.IGNORECASE),
    re.compile(r"color\s*:\s*transparent", re.IGNORECASE),
]

ZERO_WIDTH_REGEX = re.compile(r"[\u200B\u200C\u200D\uFEFF\u00AD]")
ESP_CLASS_HINTS = ("mcntextcontent", "mcnbutton", "mj-", "sg-", "hubspot", "mailchimp")
MODERN_HTML5_TAGS = {"section", "article", "header", "footer", "nav", "main", "aside"}


def _dom_max_depth_safe(soup, max_limit: int = 40) -> int:
    """Безопасный итеративный подсчет глубины DOM."""
    max_d = 0
    stack = [(soup.body or soup, 1)]
    while stack:
        tag, depth = stack.pop()
        if depth > max_d:
            max_d = depth
        if depth >= max_limit:
            continue
        for child in tag.find_all(recursive=False):
            if getattr(child, "name", None):
                stack.append((child, depth + 1))
    return max_d


class HTMLDetector:
    def __init__(self):
        pass

    def extract_features(self, html: str) -> dict:
        """Извлечение признаков из верстки письма."""
        if not html or not html.strip():
            return self._empty_features()

        try:
            soup = BeautifulSoup(html, "html.parser")
        except Exception:
            return self._empty_features()

        text = soup.get_text(separator=" ", strip=True)
        text_len = max(len(text), 1)

        all_tags = soup.find_all(True)
        tables = soup.find_all("table")
        images = soup.find_all("img")
        links = soup.find_all("a")
        forms = soup.find_all("form")
        iframes = soup.find_all("iframe")

        # 1. Скрытые CSS-элементы
        hidden_elements_count = 0
        for tag in all_tags:
            style = tag.get("style", "")
            if style and any(p.search(str(style).lower()) for p in HIDDEN_CSS_REGEX):
                hidden_elements_count += 1
            if tag.get("hidden") is not None:
                hidden_elements_count += 1

        # 2. Невидимые zero-width символы
        zero_width_hits = len(ZERO_WIDTH_REGEX.findall(html))

        # 3. MSO комментарии (маркер профессиональной почтовой вёрстки)
        comments = soup.find_all(string=lambda s: isinstance(s, Comment))
        mso_comments = sum(1 for c in comments if "mso" in c.lower() or "[if" in c.lower())

        # 4. Современные теги HTML5
        modern_tag_hits = sum(1 for tag in all_tags if tag.name.lower() in MODERN_HTML5_TAGS)
        doctype_present = int(html.strip().lower().startswith("<!doctype"))

        # 5. Трекинговые пиксели и Base64 Data URI
        tracking_pixels = 0
        data_uri_count = 0
        for img in images:
            w, h = str(img.get("width", "")), str(img.get("height", ""))
            src = str(img.get("src", "")).lower()
            if (w in ("1", "0") and h in ("1", "0")) or any(k in src for k in ("track", "open", "pixel")):
                tracking_pixels += 1
            if src.startswith("data:image"):
                data_uri_count += 1

        # 6. Подозрительные ссылки
        dummy_links = sum(
            1 for a in links
            if str(a.get("href", "")).strip() in ("#", "", "javascript:void(0)")
            or "example.com" in str(a.get("href", "")).lower()
            or "[" in str(a.get("href", "")) or "{{" in str(a.get("href", ""))
        )

        list_unsubscribe_link = any(
            "unsubscribe" in str(a.get("href", "")).lower() or "отписаться" in a.get_text().lower()
            for a in links
        )

        inline_styles = len(soup.find_all(attrs={"style": True}))
        class_attrs = len(soup.find_all(attrs={"class": True}))

        return {
            "tag_count": len(all_tags),
            "text_len": text_len,
            "dom_max_depth": _dom_max_depth_safe(soup),
            "table_count": len(tables),
            "has_table_layout": int(len(tables) > 0),
            "mso_comment_count": mso_comments,
            "has_mso_comments": int(mso_comments > 0),
            "doctype_present": doctype_present,
            "has_unsubscribe_link": int(list_unsubscribe_link),
            "modern_html5_tags_count": modern_tag_hits,
            "hidden_elements_count": hidden_elements_count,
            "zero_width_chars_count": zero_width_hits,
            "tracking_pixel_count": tracking_pixels,
            "dummy_link_count": dummy_links,
            "has_forms": int(len(forms) > 0),
            "has_iframes": int(len(iframes) > 0),
            "data_uri_count": data_uri_count,
            "image_count": len(images),
            "link_count": len(links),
            "inline_style_to_class_ratio": round(inline_styles / max(class_attrs, 1), 4)
        }

    def _empty_features(self) -> dict:
        keys = [
            "tag_count", "text_len", "dom_max_depth", "table_count", "has_table_layout",
            "mso_comment_count", "has_mso_comments", "doctype_present", "has_unsubscribe_link",
            "modern_html5_tags_count", "hidden_elements_count", "zero_width_chars_count",
            "tracking_pixel_count", "dummy_link_count", "has_forms", "has_iframes",
            "data_uri_count", "image_count", "link_count", "inline_style_to_class_ratio"
        ]
        return {k: 0 for k in keys}

    def predict(self, html: str) -> dict:
        """Возвращает извлеченные признаки верстки без принятия финального решения."""
        feats = self.extract_features(html)
        return {"features": feats}

    def batch_enrich(self, input_jsonl: str, output_jsonl: str):
        """Обогащает jsonl признаками верстки."""
        df = pd.read_json(input_jsonl, lines=True)
        results = []
        for _, row in df.iterrows():
            item = row.to_dict()
            html_feats = self.extract_features(item.get("raw_html", ""))
            # Добавляем с префиксом html_ для однозначности
            for k, v in html_feats.items():
                item[f"html_{k}"] = v
            results.append(item)

        out_df = pd.DataFrame(results)
        out_df.to_json(output_jsonl, orient="records", lines=True, force_ascii=False)
        print(f"[HTMLDetector] ✓ Размечено {len(out_df)} строк в {output_jsonl}")