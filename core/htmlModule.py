import os
import re
import json
import joblib
import pandas as pd
from bs4 import BeautifulSoup, Comment
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# =========================================================
# 1. ПРЕДКОМПИЛИРОВАННЫЕ РЕГУЛЯРКИ (Ускорение в 30+ раз)
# =========================================================
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
    """Безопасный итеративный подсчет глубины DOM без рекурсии (защита от краша воркера)."""
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


def extract_html_features(html: str) -> dict:
    """Извлекает полный вектор структурных, фишинговых и AI-признаков из верстки."""
    if not html or not html.strip():
        return _empty_features()

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

    # 1. Поиск скрытых элементов (Hard Block триггеры)
    hidden_elements_count = 0
    for tag in all_tags:
        style = tag.get("style", "")
        if style:
            style_str = str(style).lower()
            if any(p.search(style_str) for p in HIDDEN_CSS_REGEX):
                hidden_elements_count += 1
        if tag.get("hidden") is not None:
            hidden_elements_count += 1

    # 2. Невидимые символы обфускации (Zero-Width bypass)
    zero_width_hits = len(ZERO_WIDTH_REGEX.findall(html))

    # 3. MSO комментарии (Маркер профессионального ESP шаблона)
    comments = soup.find_all(string=lambda s: isinstance(s, Comment))
    mso_comments = sum(1 for c in comments if "mso" in c.lower() or "[if" in c.lower())

    # 4. Современные теги HTML5 (Маркер LLM-генерации: ИИ забывает про Outlook)
    modern_tag_hits = sum(1 for tag in all_tags if tag.name.lower() in MODERN_HTML5_TAGS)

    doctype_present = int(html.strip().lower().startswith("<!doctype"))

    # Стилистика
    inline_styles = len(soup.find_all(attrs={"style": True}))
    class_attrs = len(soup.find_all(attrs={"class": True}))

    # 5. Трекеры и Data URI
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

    utm_links = sum(1 for a in links if "utm_" in str(a.get("href", "")).lower())

    list_unsubscribe_link = any(
        "unsubscribe" in str(a.get("href", "")).lower() or "отписаться" in a.get_text().lower()
        for a in links
    )

    # 7. Нескомпилированные артефакты шаблонов и сырой Markdown
    placeholder_hits = sum(len(p.findall(text)) for p in PLACEHOLDER_REGEX)
    markdown_hits = sum(len(p.findall(html)) for p in MARKDOWN_REGEX)

    # ESP сигнатуры
    esp_hits = 0
    for tag in all_tags:
        classes = tag.get("class", [])
        cls_str = " ".join(classes).lower() if isinstance(classes, list) else str(classes).lower()
        if any(hint in cls_str for hint in ESP_CLASS_HINTS):
            esp_hits += 1

    return {
        # Размеры и пропорции
        "tag_count": tag_count,
        "tag_to_text_ratio": round(tag_count / text_len, 4),
        "text_len": text_len,
        "dom_max_depth": _dom_max_depth_safe(soup),
        
        # Индикаторы профессиональной верстки (человек)
        "table_count": len(tables),
        "has_table_layout": int(len(tables) > 0),
        "mso_comment_count": mso_comments,
        "has_mso_comments": int(mso_comments > 0),
        "doctype_present": doctype_present,
        "esp_hint_hits": esp_hits,
        "has_unsubscribe_link": int(list_unsubscribe_link),
        
        # Маркеры ИИ (ошибки генерации разметки)
        "modern_html5_tags_count": modern_tag_hits,
        "placeholder_hits": placeholder_hits,
        "markdown_leftover_hits": markdown_hits,
        
        # Маркеры фишинга / скрытого спама (Hard & Soft Blocks)
        "hidden_elements_count": hidden_elements_count,
        "zero_width_chars_count": zero_width_hits,
        "tracking_pixel_count": tracking_pixels,
        "dummy_link_count": dummy_links,
        "utm_link_count": utm_links,
        "has_forms": int(len(forms) > 0),
        "has_iframes": int(len(iframes) > 0),
        "data_uri_count": data_uri_count,
        "image_count": len(images),
        "link_count": len(links),
        "inline_style_to_class_ratio": round(inline_styles / max(class_attrs, 1), 4),
    }


def _empty_features() -> dict:
    """Шаблон пустых фичей для фоллбека."""
    keys = [
        "tag_count", "tag_to_text_ratio", "text_len", "dom_max_depth",
        "table_count", "has_table_layout", "mso_comment_count", "has_mso_comments",
        "doctype_present", "esp_hint_hits", "has_unsubscribe_link",
        "modern_html5_tags_count", "placeholder_hits", "markdown_leftover_hits",
        "hidden_elements_count", "zero_width_chars_count", "tracking_pixel_count",
        "dummy_link_count", "utm_link_count", "has_forms", "has_iframes",
        "data_uri_count", "image_count", "link_count", "inline_style_to_class_ratio"
    ]
    return {k: 0 for k in keys}


# =========================================================
# 2. КЭШИРОВАННЫЙ КЛАСС ИНФЕРЕНСА (Загрузка в память 1 раз)
# =========================================================
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
                print(f"[HTMLDetector] Ошибка загрузки модели {self.model_path}: {e}")

    def predict(self, html: str) -> dict:
        if not html or not html.strip():
            return {"html_ai_proba": 0.0, "features": _empty_features()}

        feats = extract_html_features(html)
        if self.model is None or self.feature_names is None:
            # Если обученной модели нет, отдаем фичи с нулевым риском
            return {"html_ai_proba": 0.0, "features": feats}

        # Выравнивание признаков строго под формат обучения
        X = pd.DataFrame([feats])
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_names]

        proba = float(self.model.predict_proba(X)[0, 1])
        return {
            "html_ai_proba": round(proba, 4),
            "features": feats
        }


# Синглтон для вызова из pipeline.py
_detector_instance = None

def predict_html_features_and_score(html: str, model_path: str = "models/html_ai_detector.joblib") -> dict:
    global _detector_instance
    if _detector_instance is None or _detector_instance.model_path != model_path:
        _detector_instance = HTMLDetector(model_path)
    return _detector_instance.predict(html)


# =========================================================
# 3. СКРИПТ ОБУЧЕНИЯ (Запуск при вызове напрямую)
# =========================================================
def train_html_model(human_dataset_path: str, ai_dataset_path: str, model_out: str = "models/html_ai_detector.joblib"):
    def load_records(path: str, default_label: int) -> pd.DataFrame:
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        df = pd.DataFrame(rows)
        if "label" not in df.columns:
            df["label"] = default_label
        return df[["html", "label"]]

    print("Загрузка обучающих выборок...")
    human_df = load_records(human_dataset_path, 0)
    ai_df = load_records(ai_dataset_path, 1)
    combined = pd.concat([human_df, ai_df], ignore_index=True)

    print(f"Всего: {len(combined)} записей (Человек: {(combined['label'] == 0).sum()}, ИИ: {(combined['label'] == 1).sum()})")
    print("Извлечение расширенного вектора HTML-признаков...")
    
    feature_rows = [extract_html_features(h) for h in combined["html"].fillna("")]
    X = pd.DataFrame(feature_rows)
    y = combined["label"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05, max_iter=350, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n" + classification_report(y_test, y_pred, target_names=["Human", "AI"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

    os.makedirs(os.path.dirname(model_out) or ".", exist_ok=True)
    joblib.dump({"model": model, "feature_names": list(X.columns)}, model_out)
    print(f"Модель успешно сохранена в {model_out}")


if __name__ == "__main__":
    human_file = "human_html_dataset.jsonl"
    ai_file = "ai_html_dataset.jsonl"

    if os.path.exists(human_file) and os.path.exists(ai_file):
        train_html_model(human_file, ai_file)
    else:
        print(f"Файлы {human_file} / {ai_file} не найдены. Модуль готов к инференсу.")
