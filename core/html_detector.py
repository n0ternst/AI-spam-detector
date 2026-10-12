"""
HTML Layout Forensics & Semantic AST Detector.

This module conducts structural, semantic, and adversarial cloaking analysis
on raw email HTML markup. 

Key capabilities:
- WCAG 2.1 relative luminance contrast cloaking detection (e.g., white-on-white text).
- ReDoS-resistant linear CSS parser for safe style extraction.
- ChatGPT Web-UI paste scars detection (Söhne font, specific RGBA backgrounds).
- Semantic Abstract Syntax Tree (AST) construction with non-binary AI confidence mapping.
- ML-driven layout evaluation via HistGradientBoosting (Zero Data Leakage).
"""

import os
import re
import json
import joblib
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup, Tag, Comment
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

HEX_COLOR_RE = re.compile(r"#([0-9a-fA-F]{3,6})")
RGB_COLOR_RE = re.compile(r"rgba?\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)")

NAMED_COLORS = {
    "white": (255, 255, 255), "black": (0, 0, 0), "gray": (128, 128, 128),
    "grey": (128, 128, 128), "transparent": None
}

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

HIDDEN_CSS_PATTERNS = [
    re.compile(r"display\s*:\s*none", re.IGNORECASE),
    re.compile(r"visibility\s*:\s*hidden", re.IGNORECASE),
    re.compile(r"opacity\s*:\s*0(\.0+)?(?![0-9])", re.IGNORECASE),
    re.compile(r"font-size\s*:\s*0(px|pt|em|rem)?(?![0-9])", re.IGNORECASE),
    re.compile(r"text-indent\s*:\s*-[0-9]{3,5}px", re.IGNORECASE),
    re.compile(r"left\s*:\s*-[0-9]{3,5}px", re.IGNORECASE),
    re.compile(r"height\s*:\s*0(px)?;.*overflow\s*:\s*hidden", re.IGNORECASE),
    re.compile(r"transform\s*:\s*scale\s*\(\s*0?(\.0+)?(?![1-9])\s*\)", re.IGNORECASE),
    re.compile(r"zoom\s*:\s*0?(\.0+)?(?![1-9])\d*", re.IGNORECASE),
]

# Следы копипасты из веб-интерфейсов LLM (ChatGPT, Claude)
LLM_COPYPASTE_FONTS = re.compile(r"(Söhne|ui-sans-serif|Signifier|Proxima Nova)", re.IGNORECASE)
LLM_COPYPASTE_BG = re.compile(r"rgb\(\s*247\s*,\s*247\s*,\s*248\s*\)", re.IGNORECASE)

VENDOR_CSS_RE = re.compile(r"-(?:ms|webkit|moz)-|mso-", re.IGNORECASE)
ZERO_WIDTH_REGEX = re.compile(r"[\u200B\u200C\u200D\uFEFF\u00AD]")
ESP_CLASS_HINTS = ("mcntextcontent", "mcnbutton", "mj-", "sg-", "hubspot", "mailchimp")
MODERN_HTML5_TAGS = {"section", "article", "header", "footer", "nav", "main", "aside"}

MAX_CHILDREN_PER_NODE = 35


def parse_color_to_rgb(color_str: str) -> tuple[int, int, int] | None:
    if not color_str: return None
    c = color_str.strip().lower()
    if c in NAMED_COLORS: return NAMED_COLORS[c]
    hex_m = HEX_COLOR_RE.search(c)
    if hex_m:
        h = hex_m.group(1)
        if len(h) == 3: h = "".join([x * 2 for x in h])
        if len(h) == 6:
            try: return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
            except ValueError: return None
    rgb_m = RGB_COLOR_RE.search(c)
    if rgb_m:
        try: return (int(rgb_m.group(1)), int(rgb_m.group(2)), int(rgb_m.group(3)))
        except ValueError: return None
    return None


def calculate_relative_luminance(rgb: tuple[int, int, int]) -> float:
    def channel_lum(val: int) -> float:
        v = val / 255.0
        return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4
    return 0.2126 * channel_lum(rgb[0]) + 0.7152 * channel_lum(rgb[1]) + 0.0722 * channel_lum(rgb[2])


def calculate_contrast_ratio(rgb1: tuple[int, int, int], rgb2: tuple[int, int, int]) -> float:
    l1 = calculate_relative_luminance(rgb1)
    l2 = calculate_relative_luminance(rgb2)
    return (max(l1, l2) + 0.05) / (min(l1, l2) + 0.05)


def strip_media_queries_linear(css: str) -> str:
    """Безопасный линейный O(N) алгоритм парсинга CSS без риска ReDoS."""
    if "@media" not in css.lower():
        return css
    res, i, n = [], 0, len(css)
    while i < n:
        if css[i:i+6].lower() == "@media":
            while i < n and css[i] != '{': i += 1
            braces = 1
            i += 1
            while i < n and braces > 0:
                if css[i] == '{': braces += 1
                elif css[i] == '}': braces -= 1
                i += 1
        else:
            res.append(css[i])
            i += 1
    return "".join(res)


class SemanticASTNode:
    """Узел семантического дерева с атрибуцией вероятности ИИ-генерации."""
    def __init__(self, tag_name: str, role: str, text: str = "", is_hidden: bool = False,
                 hidden_reason: str | None = None, ai_confidence: float = 0.0, contrast_ratio: float | None = None):
        self.tag_name = tag_name
        self.role = role
        self.text = text.strip()[:100]
        self.is_hidden = is_hidden
        self.hidden_reason = hidden_reason
        self.ai_confidence = round(ai_confidence, 2)
        self.contrast_ratio = round(contrast_ratio, 2) if contrast_ratio else None
        self.children: list['SemanticASTNode'] = []

    def to_dict(self) -> dict:
        d = {"role": self.role, "tag": self.tag_name, "ai_confidence_pct": self.ai_confidence, "is_hidden": self.is_hidden}
        if self.text: d["text_snippet"] = self.text
        if self.is_hidden: d["hidden_reason"] = self.hidden_reason
        if self.contrast_ratio is not None: d["contrast_ratio"] = self.contrast_ratio
        if self.children: d["children"] = [c.to_dict() for c in self.children]
        return d


class HTMLDetector:
    def __init__(self, model_path: str = "models/html_ai_detector.joblib", default_bg: tuple[int, int, int] = (255, 255, 255)):
        self.model_path = model_path
        self.default_bg = default_bg
        self.model = None
        self.feature_names = None
        self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path):
            try:
                bundle = joblib.load(self.model_path)
                self.model = bundle["model"]
                self.feature_names = bundle["feature_names"]
            except Exception as e:
                print(f"[HTMLDetector] Ошибка загрузки ML модели: {e}")

    def _extract_css_context(self, soup: BeautifulSoup) -> tuple[dict[str, str], float]:
        class_rules = {}
        total_props = 0
        vendor_props = 0

        for s in soup.find_all("style"):
            css = s.get_text()[:40000]
            clean_css = strip_media_queries_linear(css)

            for decl in re.findall(r"\{([^}]+)\}", clean_css):
                props = [p.strip().split(":")[0].strip() for p in decl.split(";") if ":" in p]
                total_props += len(props)
                vendor_props += sum(1 for p in props if VENDOR_CSS_RE.search(p))

            for cls_name, decl in re.findall(r"\.([a-zA-Z0-9_-]+)\s*\{([^}]+)\}", clean_css):
                class_rules[cls_name.lower()] = decl.lower()

        vendor_density = round(vendor_props / max(1, total_props), 4) if total_props > 0 else 0.0
        return class_rules, vendor_density

    def _check_hidden_status(self, style_str: str, parent_bg: tuple[int, int, int], depth: int, text: str, is_hidden_attr: bool):
        bg_match = re.search(r"(?:background-color|background)\s*:\s*([^;!]+)", style_str)
        effective_bg = parse_color_to_rgb(bg_match.group(1)) if bg_match else parent_bg
        if effective_bg is None: effective_bg = parent_bg

        if is_hidden_attr:
            return True, "HTML_HIDDEN_ATTR", effective_bg, None

        for p in HIDDEN_CSS_PATTERNS:
            if p.search(style_str):
                if "display" in p.pattern and depth <= 2 and 0 < len(text) <= 180:
                    return False, "PREHEADER", effective_bg, None
                return True, "GEOMETRIC_OR_TRANSFORM_HIDDEN", effective_bg, None

        has_bg_image = bool(re.search(r"background(?:-image)?\s*:\s*url\(", style_str))

        color_match = re.search(r"(?<![a-z-])color\s*:\s*([^;!]+)", style_str)
        if color_match:
            raw_c = color_match.group(1).strip()
            if raw_c == "transparent":
                return True, "COLOR_TRANSPARENT", effective_bg, 1.0

            text_rgb = parse_color_to_rgb(raw_c)
            if text_rgb and effective_bg:
                contrast = calculate_contrast_ratio(text_rgb, effective_bg)
                if contrast < 1.25:
                    return True, f"LOW_CONTRAST_{contrast:.2f}:1", effective_bg, contrast
                if has_bg_image and contrast < 2.0:
                    return True, "BG_IMAGE_CLOAKING_RISK", effective_bg, contrast

        return False, None, effective_bg, None

    def extract_features(self, html: str, ai_spans: list[dict] | None = None) -> tuple[dict, SemanticASTNode, dict]:
        if not html or not html.strip():
            return self._empty_features(), SemanticASTNode("root", "EMPTY"), {"cloaked_snippets": []}

        if len(html) > 300000:
            html = html[:300000]

        try:
            soup = BeautifulSoup(html, "lxml")
        except Exception:
            soup = BeautifulSoup(html, "html.parser")

        class_styles, vendor_css_density = self._extract_css_context(soup)
        body = soup.body or soup

        all_tags = soup.find_all(True)
        tag_count = len(all_tags)
        text_full = soup.get_text(separator=" ", strip=True)
        text_len = max(len(text_full), 1)

        tables = soup.find_all("table")
        images = soup.find_all("img")
        links = soup.find_all("a")
        forms = soup.find_all("form")
        iframes = soup.find_all("iframe")

        # Новые анти-ИИ фичи: Следы копипасты из браузера и маркированные списки
        chatgpt_copypaste_hits = 0
        inline_styles = 0
        class_attrs = 0
        for tag in all_tags:
            st = str(tag.get("style", ""))
            if st:
                inline_styles += 1
                if LLM_COPYPASTE_FONTS.search(st) or LLM_COPYPASTE_BG.search(st):
                    chatgpt_copypaste_hits += 1
            if tag.get("class"):
                class_attrs += 1

        bold_li_hits = sum(1 for li in soup.find_all("li") if li.find(["strong", "b"]))

        tracking_pixels = sum(1 for img in images if str(img.get("width", "")) in ("0", "1") or "track" in str(img.get("src", "")).lower())
        dummy_links = sum(1 for a in links if str(a.get("href", "")).strip() in ("", "#", "javascript:void(0)") or "example.com" in str(a.get("href", "")).lower())
        list_unsubscribe_link = any("unsubscribe" in str(a.get("href", "")).lower() or "отписаться" in a.get_text().lower() for a in links)

        comments = soup.find_all(string=lambda s: isinstance(s, Comment))
        mso_comments = sum(1 for c in comments if "mso" in c.lower() or "[if" in c.lower())
        modern_tag_hits = sum(1 for tag in all_tags if tag.name.lower() in MODERN_HTML5_TAGS)
        doctype_present = int(html.strip().lower().startswith("<!doctype"))
        zero_width_hits = len(ZERO_WIDTH_REGEX.findall(html))

        hidden_elements_count = 0
        cloaked_snippets = []
        max_dom_depth = 0
        leaf_depths = []
        has_legitimate_preheader = False
        root = SemanticASTNode("email", "DOCUMENT_ROOT")

        def match_ai_conf(snippet: str) -> float:
            if not ai_spans or not snippet or len(snippet.split()) < 3:
                return 0.0
            snip_w = set(re.findall(r"\b\w+\b", snippet.lower()))
            for sp in ai_spans:
                target_w = set(re.findall(r"\b\w+\b", sp["text_segment"].lower()))
                if snip_w and len(snip_w & target_w) / len(snip_w) >= 0.40:
                    return float(sp["confidence_pct"])
            return 12.0

        def _traverse(tag: Tag, parent_node: SemanticASTNode, parent_bg: tuple[int, int, int], depth: int):
            nonlocal hidden_elements_count, max_dom_depth, has_legitimate_preheader
            if depth > max_dom_depth: max_dom_depth = depth
            if depth > 12: return

            direct_children = [c for c in tag.find_all(recursive=False) if isinstance(c, Tag)]
            if not direct_children:
                leaf_depths.append(depth)

            for child in direct_children[:MAX_CHILDREN_PER_NODE]:
                if child.name in ("script", "noscript", "style"): continue

                style_str = str(child.get("style", "")).lower()
                classes = [c.lower() for c in child.get("class", [])] if isinstance(child.get("class"), list) else []
                for c in classes:
                    if c in class_styles: style_str += ";" + class_styles[c]

                direct_text = "".join(child.find_all(string=True, recursive=False)).strip()
                is_hidden_attr = child.get("hidden") is not None

                is_hidden, reason, eff_bg, contrast = self._check_hidden_status(
                    style_str, parent_bg, depth, direct_text, is_hidden_attr
                )

                if reason == "PREHEADER":
                    has_legitimate_preheader = True

                if is_hidden:
                    hidden_elements_count += 1
                    cloaked_snippets.append({
                        "tag": child.name, "reason": reason, "text": direct_text[:60] or "[Скрытый блок]"
                    })

                role = "CLOAKED" if is_hidden else ("HEADER" if child.name in ("h1", "h2", "h3") else ("CONTAINER" if direct_children else "PARAGRAPH"))
                ai_conf = match_ai_conf(direct_text)

                children_tags = [c for c in child.find_all(recursive=False) if isinstance(c, Tag)]
                if role == "CONTAINER" and not direct_text and len(children_tags) == 1 and not is_hidden:
                    _traverse(child, parent_node, eff_bg, depth)
                    continue

                node = SemanticASTNode(child.name, role, direct_text, is_hidden, reason, ai_conf, contrast)
                parent_node.children.append(node)
                if children_tags:
                    _traverse(child, node, eff_bg, depth + 1)

        _traverse(body, root, self.default_bg, depth=1)

        leaf_variance = float(np.var(leaf_depths)) if len(leaf_depths) > 1 else 0.0

        features = {
            "tag_count": tag_count,
            "text_to_tag_ratio": round(text_len / max(1, tag_count), 2),
            "dom_max_depth": max_dom_depth,
            "dom_leaf_depth_variance": round(leaf_variance, 2),
            "table_count": len(tables),
            "has_table_layout": int(len(tables) > 0),
            "mso_comment_count": mso_comments,
            "has_mso_comments": int(mso_comments > 0),
            "vendor_css_density": vendor_css_density,
            "doctype_present": doctype_present,
            "has_unsubscribe_link": int(list_unsubscribe_link),
            "has_legitimate_preheader": int(has_legitimate_preheader),
            "modern_html5_tags_count": modern_tag_hits,
            "placeholder_hits": sum(len(p.findall(text_full)) for p in PLACEHOLDER_REGEX),
            "markdown_leftover_hits": sum(len(p.findall(html)) for p in MARKDOWN_REGEX),
            "hidden_elements_count": hidden_elements_count,
            "zero_width_chars_count": zero_width_hits,
            "tracking_pixel_count": tracking_pixels,
            "dummy_link_count": dummy_links,
            "has_forms": int(len(forms) > 0),
            "has_iframes": int(len(iframes) > 0),
            "image_count": len(images),
            "link_count": len(links),
            "inline_style_to_class_ratio": round(inline_styles / max(class_attrs, 1), 4),
            "chatgpt_copypaste_hits": chatgpt_copypaste_hits,
            "bold_list_anomalies": bold_li_hits
        }

        evidence = {
            "cloaked_snippets": cloaked_snippets[:5],
            "placeholders": list(set(re.findall(r"\[[A-Za-zА-Яа-яЁё _]{2,30}\]|\{\{[a-zA-Z0-9_]+\}\}", text_full)))[:5],
            "modern_tags_used": list(set(modern_tags))[:5]
        }

        return features, root, evidence

    def _empty_features(self) -> dict:
        keys = [
            "tag_count", "text_to_tag_ratio", "dom_max_depth", "dom_leaf_depth_variance",
            "table_count", "has_table_layout", "mso_comment_count", "has_mso_comments",
            "vendor_css_density", "doctype_present", "has_unsubscribe_link", "has_legitimate_preheader",
            "modern_html5_tags_count", "placeholder_hits", "markdown_leftover_hits",
            "hidden_elements_count", "zero_width_chars_count", "tracking_pixel_count",
            "dummy_link_count", "has_forms", "has_iframes", "image_count", "link_count",
            "inline_style_to_class_ratio", "chatgpt_copypaste_hits", "bold_list_anomalies"
        ]
        return {k: 0 for k in keys}

    def predict(self, html: str, ai_spans: list[dict] | None = None) -> dict:
        feats, ast_tree, evidence = self.extract_features(html, ai_spans=ai_spans)
        
        if self.model is not None and self.feature_names is not None:
            # Исключаем сырые текстовые и языковые маркеры из инференса ML, чтобы модель смотрела только на структуру
            clean_feats = {k: v for k, v in feats.items() if k not in ["lang", "label"]}
            X = pd.DataFrame([clean_feats])
            for col in self.feature_names:
                if col not in X.columns: X[col] = 0
            X = X[self.feature_names]
            proba = float(self.model.predict_proba(X)[0, 1])
        else:
            risk = 0.0
            if feats["has_forms"] > 0: risk += 0.85
            if feats["hidden_elements_count"] >= 1: risk += 0.90
            if feats["placeholder_hits"] > 0: risk += 0.35
            if feats["modern_html5_tags_count"] >= 2 and feats["has_mso_comments"] == 0: risk += 0.30
            if feats["chatgpt_copypaste_hits"] > 0: risk += 0.40
            if feats["bold_list_anomalies"] >= 2: risk += 0.20
            if feats["has_unsubscribe_link"] and feats["has_table_layout"]: risk = max(0.0, risk - 0.25)
            proba = min(1.0, round(risk, 4))

        return {
            "html_ai_proba": proba,
            "features": feats,
            "ast_tree": ast_tree,
            "evidence": evidence
        }

    def batch_enrich(self, input_jsonl: str, output_jsonl: str):
        df = pd.read_json(input_jsonl, lines=True)
        results = []
        for _, row in df.iterrows():
            item = row.to_dict()
            html_feats, _, _ = self.extract_features(item.get("raw_html", ""))
            for k, v in html_feats.items():
                item[f"html_{k}"] = v
            results.append(item)
        out_df = pd.DataFrame(results)
        out_df.to_json(output_jsonl, orient="records", lines=True, force_ascii=False)
        print(f"[HTMLDetector] Обогащено {len(out_df)} строк -> {output_jsonl}")


# Обучение модели вынесено из базового класса
def train_html_model(human_dataset_path: str, ai_dataset_path: str, model_out: str = "models/html_ai_detector.joblib"):
    def load_records(path: str, default_label: int) -> pd.DataFrame:
        rows = [json.loads(line) for line in open(path, "r", encoding="utf-8") if line.strip()]
        df = pd.DataFrame(rows)
        if "label" not in df.columns: df["label"] = default_label
        return df[["html", "label"]]

    print("Загрузка датасетов...")
    combined = pd.concat([load_records(human_dataset_path, 0), load_records(ai_dataset_path, 1)], ignore_index=True)

    det = HTMLDetector()
    feature_rows = [det.extract_features(h)[0] for h in combined["html"].fillna("")]
    df_features = pd.DataFrame(feature_rows)

    # Исключаем lang из обучения (защита от Data Leakage)
    X = df_features.drop(columns=["label", "lang"], errors="ignore")
    y = df_features["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05, max_iter=350, random_state=42)
    model.fit(X_train, y_train)

    print(f"ROC-AUC: {roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]):.4f}")

    os.makedirs(os.path.dirname(model_out) or ".", exist_ok=True)
    joblib.dump({"model": model, "feature_names": list(X.columns)}, model_out)
    print(f"Модель сохранена в: {model_out}")


if __name__ == "__main__":
    if os.path.exists("human_html_dataset.jsonl") and os.path.exists("ai_html_dataset.jsonl"):
        train_html_model("human_html_dataset.jsonl", "ai_html_dataset.jsonl")
    else:
        print("Датасеты не найдены. Готов к инференсу.")
