import json
import joblib
import pandas as pd
import re
from bs4 import BeautifulSoup, Comment
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.inspection import permutation_importance

MODE = "train"  #train обучить модель, predict проверить одно письмо

HUMAN_DATASET_PATH = "human_html_dataset.jsonl"
HUMAN_DEFAULT_LABEL = 0  # человек

AI_DATASET_PATH = "ai_html_dataset.jsonl"
AI_DEFAULT_LABEL = 1  # ИИ

MODEL_OUT_PATH = "html_ai_detector.joblib"
TEST_SIZE = 0.2

# --- для MODE = "predict" ---
PREDICT_HTML_PATH = "new_email.html"
PREDICT_MODEL_PATH = "html_ai_detector.joblib"


#извлечение признаков из HTML
PLACEHOLDER_PATTERNS = [
    r"\[[A-Za-zА-Яа-яЁё _]{2,30}\]",
    r"\{\{[a-zA-Z_]+\}\}",
    r"\{%[^%]+%\}",
]

MARKDOWN_LEFTOVER_PATTERNS = [r"\*\*[^*]+\*\*", r"(?<!\w)#{1,3}\s+\S", r"(?m)^\s*[-*]\s+\S",
]

ESP_CLASS_HINTS = [
    "mcnTextContent", "mcnButton", "mj-", "sg-", "hubspot", "mailchimp",
]


def _dom_max_depth(tag, depth=0):
    children = [c for c in tag.find_all(recursive=False)]
    if not children:
        return depth
    return max(_dom_max_depth(c, depth + 1) for c in children)


def extract_html_features(html: str) -> dict:
    #возвращает словарь числовых признаков по одному html-письму
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

    comments = soup.find_all(string=lambda s: isinstance(s, Comment))
    mso_comments = sum(1 for c in comments if "mso" in c.lower() or "[if" in c.lower())

    doctype_present = html.strip().lower().startswith("<!doctype")

    legacy_font = len(soup.find_all("font"))
    legacy_bgcolor = len(soup.find_all(attrs={"bgcolor": True}))
    legacy_align_attr = len(soup.find_all(attrs={"align": True}))

    inline_styles = len(soup.find_all(attrs={"style": True}))
    class_attrs = len(soup.find_all(attrs={"class": True}))

    tracking_pixels = 0
    for img in images:
        w, h = str(img.get("width", "")), str(img.get("height", ""))
        src = str(img.get("src", "")).lower()
        if (w in ("1", "0") and h in ("1", "0")) or "track" in src or "open" in src or "pixel" in src:
            tracking_pixels += 1

    dummy_links = sum(
        1 for a in links
        if str(a.get("href", "")).strip() in ("#", "", "javascript:void(0)")
        or "example.com" in str(a.get("href", "")).lower()
        or re.search(r"\[.*\]|\{\{.*\}\}", str(a.get("href", "")))
    )

    utm_links = sum(1 for a in links if "utm_" in str(a.get("href", "")).lower())

    list_unsubscribe_link = any(
        "unsubscribe" in str(a.get("href", "")).lower() or "отписаться" in a.get_text().lower()
        for a in links
    )

    placeholder_hits = sum(
        len(re.findall(p, text)) for p in PLACEHOLDER_PATTERNS
    )
    markdown_leftover_hits = sum(
        len(re.findall(p, html)) for p in MARKDOWN_LEFTOVER_PATTERNS
    )

    placeholder_alt = sum(
        1 for img in images
        if str(img.get("alt", "")).strip().lower() in ("image", "placeholder", "img", "")
    )

    esp_hint_hits = 0
    for tag in all_tags:
        cls = " ".join(tag.get("class", []))
        for hint in ESP_CLASS_HINTS:
            if hint.lower() in cls.lower():
                esp_hint_hits += 1

    dom_depth = _dom_max_depth(soup) if soup.find() else 0

    return {
        "tag_count": tag_count,
        "tag_to_text_ratio": tag_count / text_len,
        "text_len": text_len,
        "dom_max_depth": dom_depth,
        "table_count": len(tables),
        "has_table_layout": int(len(tables) > 0),
        "image_count": len(images),
        "link_count": len(links),
        "mso_comment_count": mso_comments,
        "has_mso_comments": int(mso_comments > 0),
        "doctype_present": int(doctype_present),
        "legacy_font_count": legacy_font,
        "legacy_bgcolor_count": legacy_bgcolor,
        "legacy_align_count": legacy_align_attr,
        "inline_style_count": inline_styles,
        "class_attr_count": class_attrs,
        "inline_style_to_class_ratio": inline_styles / max(class_attrs, 1),
        "tracking_pixel_count": tracking_pixels,
        "has_tracking_pixel": int(tracking_pixels > 0),
        "dummy_link_count": dummy_links,
        "utm_link_count": utm_links,
        "has_unsubscribe_link": int(list_unsubscribe_link),
        "placeholder_hits": placeholder_hits,
        "markdown_leftover_hits": markdown_leftover_hits,
        "placeholder_alt_count": placeholder_alt,
        "esp_hint_hits": esp_hint_hits,
    }


def _empty_features() -> dict:
    keys = extract_html_features("<html><body><p>x</p></body></html>").keys()
    return {k: 0 for k in keys}

#обьеденение датасетов
def load_records(path: str) -> pd.DataFrame:
    if path.endswith(".jsonl"):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        df = pd.DataFrame(rows)
    else:
        df = pd.read_csv(path)
    return df


def load_dataset_with_label(path: str, default_label: int) -> pd.DataFrame:
    df = load_records(path)
    if "label" not in df.columns:
        df["label"] = default_label
    else:
        df["label"] = df["label"].fillna(default_label).astype(int)
    return df[["html", "label"]]


def load_combined_dataset(
    human_path: str, human_default_label: int,
    ai_path: str, ai_default_label: int,
) -> pd.DataFrame:
    human_df = load_dataset_with_label(human_path, human_default_label)
    ai_df = load_dataset_with_label(ai_path, ai_default_label)
    combined = pd.concat([human_df, ai_df], ignore_index=True)
    print(f"всего: {len(combined)} записей "
          f"(label=0 / человек: {(combined['label'] == 0).sum()}, "
          f"label=1 / ИИ: {(combined['label'] == 1).sum()})")
    return combined


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    feature_rows = [extract_html_features(html) for html in df["html"].fillna("")]
    X = pd.DataFrame(feature_rows)
    X["label"] = df["label"].values
    return X

#обучение
def train_model(df: pd.DataFrame, model_out: str, test_size: float = 0.2):
    feat_df = build_feature_matrix(df)
    y = feat_df.pop("label")
    X = feat_df

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    model = HistGradientBoostingClassifier(
        max_depth=4,
        learning_rate=0.08,
        max_iter=300,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred, target_names=["человек", "ИИ"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

    perm = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance": perm.importances_mean,
    }).sort_values("importance", ascending=False)

    print("\nтоп 10 признаков")
    print(importance_df.head(10).to_string(index=False))

    joblib.dump({"model": model, "feature_names": list(X.columns)}, model_out)
    print(f"\nмодель сохранена: {model_out}")



def predict_html_file(html_path: str, model_path: str):
    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_names = bundle["feature_names"]

    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    feats = extract_html_features(html)
    X = pd.DataFrame([feats])[feature_names]

    proba_ai = model.predict_proba(X)[0, 1]
    print(f"Вероятность: {proba_ai:.2%}")



if __name__ == "__main__":
    if MODE == "train":
        dataset = load_combined_dataset(
            HUMAN_DATASET_PATH, HUMAN_DEFAULT_LABEL,
            AI_DATASET_PATH, AI_DEFAULT_LABEL,
        )
        train_model(dataset, MODEL_OUT_PATH, TEST_SIZE)
    elif MODE == "predict":
        predict_html_file(PREDICT_HTML_PATH, PREDICT_MODEL_PATH)
