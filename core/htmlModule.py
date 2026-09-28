import os
import re
import json
import joblib
import pandas as pd
from bs4 import BeautifulSoup, Comment
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
# 1. ПРЕДКОМПИЛИРОВАННЫЕ РЕГУЛЯРНЫЕ ВЫРАЖЕНИЯ
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

# Паттерны сокрытия текста в инлайн-стилях
HIDDEN_CSS_PATTERNS = [
    re.compile(r"display\s*:\s*none", re.IGNORECASE),
    re.compile(r"visibility\s*:\s*hidden", re.IGNORECASE),
    re.compile(r"opacity\s*:\s*0(\.0)?", re.IGNORECASE),
    re.compile(r"font-size\s*:\s*0(px|pt|em)?", re.IGNORECASE),
    re.compile(r"text-indent\s*:\s*-[0-9]{3,5}px", re.IGNORECASE),
    re.compile(r"left\s*:\s*-[0-9]{3,5}px", re.IGNORECASE),
    re.compile(r"color\s*:\s*transparent", re.IGNORECASE),
    re.compile(r"height\s*:\s*0(px)?;.*overflow\s*:\s*hidden", re.IGNORECASE),
]

ZERO_WIDTH_REGEX = re.compile(r"[\u200B\u200C\u200D\uFEFF\u00AD]")
ESP_CLASS_HINTS = ("mcntextcontent", "mcnbutton", "mj-", "sg-", "hubspot", "mailchimp")
MODERN_HTML5_TAGS = {"section", "article", "header", "footer", "nav", "main", "aside"}


def _dom_max_depth_safe(soup, max_limit: int = 40) -> int:
    """Безопасный итеративный обход DOM без риска RecursionError."""
    max_d = 0
    stack = [(soup.body or soup, 1)]
    while stack:
        tag, depth = stack.pop()
        if depth > max_d:
            max_d = depth
        if depth >= max_limit:
            continue
        for child in tag.find_all(recursive=False):
            if child.name:
                stack.append((child, depth + 1))
    return max_d


def extract_hidden_classes_from_style_tags(soup) -> set:
    """
    Извлекает классы, которые скрывают текст, но ИГНОРИРУЕТ @media queries
    (чтобы не браковать легитимную адаптивную верстку под смартфоны).
    """
    hidden_classes = set()
    for s in soup.find_all("style"):
        css_text = s.get_text()
        # Вырезаем блоки @media (...), чтобы не наказывать за мобильные стили
        clean_css = re.sub(r'@media[^{]+{(?:[^{}]+|{[^{}]*})*}', '', css_text, flags=re.IGNORECASE)

        # Ищем классы, где жестко задан display:none или font-size:0
        matches = re.findall(
            r'\.([a-zA-Z0-9_-]+)\s*\{[^}]*(?:display\s*:\s*none|font-size\s*:\s*0|opacity\s*:\s*0|visibility\s*:\s*hidden)[^}]*\}',
            clean_css,
            re.IGNORECASE
        )
        for cls_name in matches:
            hidden_classes.add(cls_name.lower())
    return hidden_classes


def extract_html_analysis(html: str) -> tuple[dict, dict]:
    """
    Анализирует верстку с защитой от ложных срабатываний на прехедерах и мобильных стилях.
    """
    if not html or not html.strip():
        return _empty_features(), _empty_evidence()

    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(separator=" ", strip=True)
    text_len = max(len(text), 1)

    all_tags = soup.find_all(True)
    tag_count = len(all_tags)

    tables = soup.find_all("table")
    images = soup.find_all("img")
    links = soup.find_all("a")
    forms = soup.find_all("form")
    iframes = soup.find_all("iframe")

    # 1. Извлекаем подозрительные классы из тегов <style>
    hidden_css_classes = extract_hidden_classes_from_style_tags(soup)

    # 2. Поиск скрытых элементов с АМНИСТИЕЙ ПРЕХЕДЕРА
    hidden_snippets = []
    malicious_hidden_chars = 0
    has_legitimate_preheader = False

    for idx, tag in enumerate(all_tags):
        style = tag.get("style", "")
        style_str = str(style).lower() if style else ""
        tag_classes = [c.lower() for c in tag.get("class", [])] if isinstance(tag.get("class"), list) else []

        is_hidden_style = any(p.search(style_str) for p in HIDDEN_CSS_PATTERNS)
        is_hidden_class = any(c in hidden_css_classes for c in tag_classes)
        is_hidden_attr = tag.get("hidden") is not None

        if is_hidden_style or is_hidden_class or is_hidden_attr:
            snippet = tag.get_text(strip=True)
            snippet_len = len(snippet)

            # АМНИСТИЯ ПРЕХЕДЕРА:
            # Если скрытый блок находится в первых 5 тегах письма,
            # его длина меньше 180 символов, и он не содержит форм/ссылок — это легитимный прехедер!
            if idx < 6 and snippet_len > 0 and snippet_len <= 180 and not tag.find(["a", "form", "input"]):
                has_legitimate_preheader = True
                continue  # НЕ считаем его вредоносным!

            # Если скрытый блок пустой (например, технический spacer) — не паникуем
            if snippet_len == 0 and not tag.find(["form", "iframe", "input"]):
                continue

            # В противном случае — это РЕАЛЬНЫЙ скрытый текст спамера
            malicious_hidden_chars += snippet_len
            hidden_snippets.append({
                "tag": tag.name,
                "style": style_str if is_hidden_style else f"class:{','.join(tag_classes)}",
                "text_snippet": snippet[:120] or "[Скрытый блок]"
            })

    # 3. Анализ ссылок
    suspicious_links = []
    utm_links_count = 0
    has_unsubscribe = False
    for a in links:
        href = str(a.get("href", "")).strip()
        href_lower = href.lower()
        link_text = a.get_text(strip=True).lower()

        if "utm_" in href_lower:
            utm_links_count += 1
        if "unsubscribe" in href_lower or "отписаться" in link_text:
            has_unsubscribe = True

        # Ссылки-пустышки и инъекции (исключаем безопасные якорные #top / #main)
        if href in ("", "javascript:void(0)") or "example.com" in href_lower or "[" in href or "{{" in href:
            suspicious_links.append({
                "url": href,
                "anchor_text": a.get_text(strip=True)[:50] or "[Без текста]"
            })

    # 4. Формы ввода и фреймы (100% фишинг)
    form_actions = [f"{f.get('method', 'GET').upper()} -> {f.get('action', '')}" for f in forms]
    iframe_sources = [str(ifr.get("src", "[inline iframe]")) for ifr in iframes]

    # 5. Артефакты шаблонов и сырой Markdown
    found_placeholders = []
    for p in PLACEHOLDER_REGEX:
        found_placeholders.extend(p.findall(text))

    found_markdown = []
    for p in MARKDOWN_REGEX:
        found_markdown.extend(p.findall(html))

    # 6. Трекеры
    tracking_pixels = []
    data_uri_count = 0
    for img in images:
        w, h = str(img.get("width", "")), str(img.get("height", ""))
        src = str(img.get("src", ""))
        src_lower = src.lower()
        if (w in ("1", "0") and h in ("1", "0")) or any(k in src_lower for k in ("track", "open", "pixel")):
            tracking_pixels.append(src[:100])
        if src_lower.startswith("data:image"):
            data_uri_count += 1

    comments = soup.find_all(string=lambda s: isinstance(s, Comment))
    mso_comments = sum(1 for c in comments if "mso" in c.lower() or "[if" in c.lower())
    modern_tags = [t.name for t in all_tags if t.name.lower() in MODERN_HTML5_TAGS]
    zero_width_hits = len(ZERO_WIDTH_REGEX.findall(html))

    esp_hits = 0
    for tag in all_tags:
        cls_str = " ".join(tag.get("class", [])).lower() if isinstance(tag.get("class"), list) else ""
        if any(hint in cls_str for hint in ESP_CLASS_HINTS):
            esp_hits += 1

    inline_styles = len(soup.find_all(attrs={"style": True}))
    class_attrs = len(soup.find_all(attrs={"class": True}))

    # ЧИСЛОВОЙ ВЕКТОР ДЛЯ ML (Очищенный от ложных срабатываний)
    features = {
        "tag_count": tag_count,
        "tag_to_text_ratio": round(tag_count / text_len, 4),
        "text_len": text_len,
        "dom_max_depth": _dom_max_depth_safe(soup),
        "table_count": len(tables),
        "has_table_layout": int(len(tables) > 0),
        "mso_comment_count": mso_comments,
        "has_mso_comments": int(mso_comments > 0),
        "doctype_present": int(html.strip().lower().startswith("<!doctype")),
        "esp_hint_hits": esp_hits,
        "has_unsubscribe_link": int(has_unsubscribe),
        "has_legitimate_preheader": int(has_legitimate_preheader),
        "modern_html5_tags_count": len(modern_tags),
        "placeholder_hits": len(found_placeholders),
        "markdown_leftover_hits": len(found_markdown),
        "hidden_elements_count": len(hidden_snippets),          # Только РЕАЛЬНО подозрительные!
        "malicious_hidden_chars": malicious_hidden_chars,       # Объем скрытого текста
        "zero_width_chars_count": zero_width_hits,
        "tracking_pixel_count": len(tracking_pixels),
        "dummy_link_count": len(suspicious_links),
        "utm_link_count": utm_links_count,
        "has_forms": int(len(forms) > 0),
        "has_iframes": int(len(iframes) > 0),
        "data_uri_count": data_uri_count,
        "image_count": len(images),
        "link_count": len(links),
        "inline_style_to_class_ratio": round(inline_styles / max(class_attrs, 1), 4),
    }

    evidence = {
        "hidden_elements": hidden_snippets[:5],
        "suspicious_links": suspicious_links[:5],
        "phishing_forms": form_actions[:3],
        "embedded_iframes": iframe_sources[:3],
        "template_placeholders": list(set(found_placeholders))[:5],
        "markdown_artifacts": list(set(found_markdown))[:5],
        "tracking_pixel_urls": tracking_pixels[:3],
        "modern_html5_tags": list(set(modern_tags))[:5]
    }

    return features, evidence


def _empty_features() -> dict:
    dummy_html = "<html><body><p>x</p></body></html>"
    f, _ = extract_html_analysis(dummy_html)
    return {k: 0 for k in f.keys()}

def _empty_evidence() -> dict:
    return {
        "hidden_elements": [], "suspicious_links": [], "phishing_forms": [],
        "embedded_iframes": [], "template_placeholders": [], "markdown_artifacts": [],
        "tracking_pixel_urls": [], "modern_html5_tags": []
    }

def extract_html_features(html: str) -> dict:
    feats, _ = extract_html_analysis(html)
    return feats


# 2. КЭШИРОВАННЫЙ ИНФЕРЕНС С КАЛИБРОВАННЫМИ ЭВРИСТИКАМИ
class HTMLDetector:
    def __init__(self, model_path: str = "models/html_ai_detector.joblib"):
        self.model_path = model_path
        self.model = None
        self.feature_names = None
        self._load()

    def _load(self):
        if os.path.exists(self.model_path):
            try:
                bundle = joblib.load(self.model_path)
                self.model = bundle["model"]
                self.feature_names = bundle["feature_names"]
            except Exception as e:
                print(f"[HTMLDetector] Загрузка модели не удалась, переход на откалиброванные правила: {e}")

    def predict(self, html: str) -> dict:
        if not html or not html.strip():
            return {"html_ai_proba": 0.0, "features": _empty_features(), "evidence": _empty_evidence()}

        feats, evidence = extract_html_analysis(html)

        # Если файл модели есть — используем обученный ML
        if self.model is not None and self.feature_names is not None:
            X = pd.DataFrame([feats])
            for col in self.feature_names:
                if col not in X.columns:
                    X[col] = 0
            X = X[self.feature_names]
            proba = float(self.model.predict_proba(X)[0, 1])
        else:
            # ОТКАЛИБРОВАННАЯ ЭВРИСТИЧЕСКАЯ ШКАЛА (Без ложных срабатываний!)
            risk = 0.0
            if feats["has_forms"] > 0:
                risk += 0.85
            if feats["has_iframes"] > 0:
                risk += 0.80
            if feats["malicious_hidden_chars"] > 150:
                risk += 0.50
            if feats["dummy_link_count"] > 0:
                risk += 0.30
            if feats["placeholder_hits"] > 0:
                risk += 0.35
            if feats["modern_html5_tags_count"] >= 2 and feats["has_mso_comments"] == 0:
                risk += 0.25
            
            # Если это белая рассылка с отпиской и таблицами — СНИЖАЕМ риск!
            if feats["has_unsubscribe_link"] and feats["has_table_layout"]:
                risk = max(0.0, risk - 0.20)

            proba = min(1.0, round(risk, 4))

        return {
            "html_ai_proba": proba,
            "features": feats,
            "evidence": evidence
        }


_detector_instance = None

def predict_html_features_and_score(html: str, model_path: str = "models/html_ai_detector.joblib") -> dict:
    global _detector_instance
    if _detector_instance is None or _detector_instance.model_path != model_path:
        _detector_instance = HTMLDetector(model_path)
    return _detector_instance.predict(html)
