"""
HTML Layout Forensics & Semantic AST Detector
"""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import json
import logging
import os
import re
from typing import Any

from bs4 import BeautifulSoup, Comment, Tag
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

logger = logging.getLogger("HTMLForensicsDetector")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

HEX_COLOR_RE = re.compile(r"#([0-9a-fA-F]{3,6})\b")
RGBA_COLOR_RE = re.compile(
    r"rgba?\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)(?:\s*,\s*([0-9.]+))?\s*\)"
)

NAMED_COLORS = {
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "gray": (128, 128, 128),
    "grey": (128, 128, 128),
    "red": (255, 0, 0),
    "blue": (0, 0, 255),
    "green": (0, 128, 0),
    "transparent": None,
}

PLACEHOLDER_REGEX = [
    re.compile(r"\[[A-Za-zА-Яа-яЁё0-9 _/-]{2,35}\]"),
    re.compile(r"\{\{[a-zA-Z0-9_.-]+\}\}"),
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
    re.compile(r"font-size\s*:\s*0(px|pt|em|rem|%)?(?![0-9])", re.IGNORECASE),
    re.compile(r"text-indent\s*:\s*-[0-9]{3,6}px", re.IGNORECASE),
    re.compile(r"(?:left|top)\s*:\s*-[0-9]{3,6}px", re.IGNORECASE),
    re.compile(
        r"(?:height|max-height)\s*:\s*0(px)?;.*overflow\s*:\s*hidden",
        re.IGNORECASE,
    ),
    re.compile(
        r"transform\s*:\s*scale\s*\(\s*0?(\.0+)?(?![1-9])\s*\)", re.IGNORECASE
    ),
    re.compile(r"zoom\s*:\s*0?(\.0+)?(?![1-9])\d*", re.IGNORECASE),
    re.compile(r"clip\s*:\s*rect\(\s*0", re.IGNORECASE),
    re.compile(r"clip-path\s*:\s*inset\(\s*(?:50|100)%", re.IGNORECASE),
]

LLM_COPYPASTE_FONTS = re.compile(
    r"(Söhne|ui-sans-serif|Signifier|Proxima Nova|Tiempos)", re.IGNORECASE
)
LLM_COPYPASTE_BG = re.compile(
    r"rgb\(\s*24[5-9]\s*,\s*24[5-9]\s*,\s*24[6-9]\s*\)", re.IGNORECASE
)
TAILWIND_VARS_RE = re.compile(
    r"--tw-(?:border-spacing|translate|rotate|skew|scale|ring|shadow)",
    re.IGNORECASE,
)
GDOCS_GUID_RE = re.compile(r"docs-internal-guid-[0-9a-f-]+", re.IGNORECASE)

AI_BOILERPLATE_CSS_RE = re.compile(
    r"font-family\s*:\s*['\"]?Arial['\"]?,\s*sans-serif.*?(?:line-height\s*:\s*1\.[56]|color\s*:\s*#333(?:333)?)",
    re.IGNORECASE | re.DOTALL,
)
AI_ALT_PLACEHOLDERS_RE = re.compile(
    r"^(?:company\s+logo|logo|hero\s+image|product\s+image|banner|image\s+placeholder|icon)$",
    re.IGNORECASE,
)

BIDI_SPOOFING_RE = re.compile(
    r"unicode-bidi\s*:\s*bidi-override|direction\s*:\s*rtl", re.IGNORECASE
)
BIDI_CONTROL_CHARS_RE = re.compile(r"[\u202A-\u202E\u2066-\u2069]")

VENDOR_CSS_RE = re.compile(r"-(?:ms|webkit|moz)-|mso-", re.IGNORECASE)
ZERO_WIDTH_REGEX = re.compile(r"[\u200B\u200C\u200D\uFEFF\u00AD]")
ESP_CLASS_HINTS = (
    "mcntextcontent",
    "mcnbutton",
    "mj-",
    "sg-",
    "hubspot",
    "mailchimp",
    "klaviyo",
)
MODERN_HTML5_TAGS = {
    "section",
    "article",
    "header",
    "footer",
    "nav",
    "main",
    "aside",
}

MAX_CHILDREN_PER_NODE = 35
MAX_DOM_TRAVERSAL_DEPTH = 15


def parse_color_to_rgb(color_str: str) -> tuple[int, int, int] | None:
    if not color_str:
        return None
    c = color_str.strip().lower()
    if c in NAMED_COLORS:
        return NAMED_COLORS[c]

    rgb_m = RGBA_COLOR_RE.search(c)
    if rgb_m:
        try:
            if rgb_m.group(4) is not None and float(rgb_m.group(4)) < 0.12:
                return None
            r = min(255, max(0, int(rgb_m.group(1))))
            g = min(255, max(0, int(rgb_m.group(2))))
            b = min(255, max(0, int(rgb_m.group(3))))
            return (r, g, b)
        except ValueError:
            return None

    hex_m = HEX_COLOR_RE.search(c)
    if hex_m:
        h = hex_m.group(1)
        if len(h) == 3:
            h = "".join([x * 2 for x in h])
        if len(h) == 6:
            try:
                return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
            except ValueError:
                return None
    return None


def calculate_relative_luminance(rgb: tuple[int, int, int]) -> float:
    def channel_lum(val: int) -> float:
        v = val / 255.0
        return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4

    return (
        0.2126 * channel_lum(rgb[0])
        + 0.7152 * channel_lum(rgb[1])
        + 0.0722 * channel_lum(rgb[2])
    )


def calculate_contrast_ratio(
    rgb1: tuple[int, int, int], rgb2: tuple[int, int, int]
) -> float:
    l1 = calculate_relative_luminance(rgb1)
    l2 = calculate_relative_luminance(rgb2)
    return (max(l1, l2) + 0.05) / (min(l1, l2) + 0.05)


def strip_media_queries_linear(css: str) -> str:
    if "@media" not in css.lower():
        return css
    res, i, n = [], 0, len(css)
    while i < n:
        if css[i : i + 6].lower() == "@media":
            while i < n and css[i] != "{":
                i += 1
            braces = 1
            i += 1
            while i < n and braces > 0:
                if css[i] == "{":
                    braces += 1
                elif css[i] == "}":
                    braces -= 1
                i += 1
        else:
            res.append(css[i])
            i += 1
    return "".join(res)


class SemanticASTNode:

    def __init__(
        self,
        tag_name: str,
        role: str,
        text: str = "",
        is_hidden: bool = False,
        hidden_reason: str | None = None,
        ai_confidence: float = 0.0,
        contrast_ratio: float | None = None,
    ):
        self.tag_name = tag_name
        self.role = role
        self.text = text.strip()[:100]
        self.is_hidden = is_hidden
        self.hidden_reason = hidden_reason
        self.ai_confidence = round(ai_confidence, 2)
        self.contrast_ratio = (
            round(contrast_ratio, 2) if contrast_ratio is not None else None
        )
        self.children: list["SemanticASTNode"] = []

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "role": self.role,
            "tag": self.tag_name,
            "ai_confidence_pct": self.ai_confidence,
            "is_hidden": self.is_hidden,
        }
        if self.text:
            d["text_snippet"] = self.text
        if self.is_hidden:
            d["hidden_reason"] = self.hidden_reason
        if self.contrast_ratio is not None:
            d["contrast_ratio"] = self.contrast_ratio
        if self.children:
            d["children"] = [c.to_dict() for c in self.children]
        return d


@dataclass
class ForensicResult:
    html_ai_proba: float
    cloaking_risk_score: float
    top_risk_factors: list[str]
    features: dict[str, float | int]
    ast_tree: SemanticASTNode
    evidence: dict[str, Any]
    visible_text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "html_ai_proba": self.html_ai_proba,
            "cloaking_risk_score": self.cloaking_risk_score,
            "top_risk_factors": self.top_risk_factors,
            "features": self.features,
            "ast_tree": self.ast_tree.to_dict(),
            "evidence": self.evidence,
            "visible_text": self.visible_text,
        }


class HTMLDetector:
    FEATURE_KEYS = [
        "tag_count",
        "text_to_tag_ratio",
        "dom_max_depth",
        "dom_leaf_depth_variance",
        "table_count",
        "has_table_layout",
        "mso_comment_count",
        "has_mso_comments",
        "vendor_css_density",
        "important_css_density",
        "ai_boilerplate_css",
        "doctype_present",
        "has_unsubscribe_link",
        "has_legitimate_preheader",
        "esp_signature_hits",
        "modern_html5_tags_count",
        "placeholder_hits",
        "markdown_leftover_hits",
        "hidden_elements_count",
        "cloaked_text_ratio",
        "zero_width_chars_count",
        "bidi_spoofing_hits",
        "tracking_pixel_count",
        "dummy_link_count",
        "has_forms",
        "has_iframes",
        "image_count",
        "ai_alt_placeholder_hits",
        "link_count",
        "inline_style_to_class_ratio",
        "chatgpt_copypaste_hits",
        "gdocs_copypaste_hits",
        "tailwind_css_hits",
        "bold_list_anomalies",
        "list_length_uniformity",
    ]

    def __init__(
        self,
        model_path: str = "models/html_ai_detector.joblib",
        default_bg: tuple[int, int, int] = (255, 255, 255),
    ):
        self.model_path = model_path
        self.default_bg = default_bg
        self.model = None
        self.feature_names: list[str] | None = None
        self._load_model()

    def _load_model(self):
        if self.model_path and os.path.exists(self.model_path):
            try:
                bundle = joblib.load(self.model_path)
                self.model = bundle["model"]
                self.feature_names = bundle["feature_names"]
                logger.info(f"ML-модель загружена: {self.model_path}")
            except Exception as e:
                logger.error(f"Ошибка загрузки ML модели: {e}")

    def _extract_css_context(
        self, soup: BeautifulSoup
    ) -> tuple[dict[str, str], float, float, int]:
        class_rules = {}
        total_props = 0
        vendor_props = 0
        important_props = 0
        ai_boilerplate_hits = 0

        for s in soup.find_all("style"):
            css = s.get_text()[:40000]
            if AI_BOILERPLATE_CSS_RE.search(css):
                ai_boilerplate_hits = 1

            clean_css = strip_media_queries_linear(css)

            for decl in re.findall(r"\{([^}]+)\}", clean_css):
                raw_props = [p.strip() for p in decl.split(";") if ":" in p]
                total_props += len(raw_props)
                for p in raw_props:
                    prop_name = p.split(":")[0].strip()
                    if VENDOR_CSS_RE.search(prop_name):
                        vendor_props += 1
                    if "!important" in p.lower():
                        important_props += 1

            for cls_name, decl in re.findall(
                r"\.([a-zA-Z0-9_-]+)\s*\{([^}]+)\}", clean_css
            ):
                class_rules[cls_name.lower()] = decl.lower()

        denom = max(1, total_props)
        vendor_density = (
            round(vendor_props / denom, 4) if total_props > 0 else 0.0
        )
        important_density = (
            round(important_props / denom, 4) if total_props > 0 else 0.0
        )
        return (
            class_rules,
            vendor_density,
            important_density,
            ai_boilerplate_hits,
        )

    def _check_hidden_status(
        self,
        style_str: str,
        parent_bg: tuple[int, int, int],
        depth: int,
        text: str,
        is_hidden_attr: bool,
    ) -> tuple[bool, str | None, tuple[int, int, int], float | None]:
        bg_match = re.search(
            r"(?<![a-z-])(?:background-color|background)\s*:\s*([^;!]+)",
            style_str,
        )
        effective_bg = (
            parse_color_to_rgb(bg_match.group(1)) if bg_match else parent_bg
        )
        if effective_bg is None:
            effective_bg = parent_bg

        if is_hidden_attr:
            return True, "HTML_HIDDEN_ATTR", effective_bg, None

        for p in HIDDEN_CSS_PATTERNS:
            if p.search(style_str):
                if (
                    "display" in p.pattern
                    and depth <= 3
                    and 0 < len(text) <= 200
                ):
                    return False, "PREHEADER", effective_bg, None
                return True, "GEOMETRIC_OR_CSS_HIDDEN", effective_bg, None

        has_bg_image = bool(
            re.search(r"background(?:-image)?\s*:\s*url\(", style_str)
        )

        color_match = re.search(r"(?<![a-z-])color\s*:\s*([^;!]+)", style_str)
        if color_match:
            raw_c = color_match.group(1).strip().lower()
            if raw_c == "transparent":
                return True, "COLOR_TRANSPARENT", effective_bg, 1.0

            if "rgba" in raw_c and parse_color_to_rgb(raw_c) is None:
                return True, "RGBA_ALPHA_ZERO_CLOAKING", effective_bg, 1.0

            text_rgb = parse_color_to_rgb(raw_c)
            if text_rgb and effective_bg:
                contrast = calculate_contrast_ratio(text_rgb, effective_bg)
                if contrast < 1.25:
                    return (
                        True,
                        f"LOW_CONTRAST_{contrast:.2f}:1",
                        effective_bg,
                        contrast,
                    )
                if has_bg_image and contrast < 2.0:
                    return (
                        True,
                        "BG_IMAGE_CLOAKING_RISK",
                        effective_bg,
                        contrast,
                    )

        return False, None, effective_bg, None

    def extract_features(
        self, html: str, ai_spans: list[dict] | None = None
    ) -> tuple[dict[str, float | int], SemanticASTNode, dict[str, Any]]:
        if not html or not html.strip():
            return (
                self._empty_features(),
                SemanticASTNode("root", "EMPTY"),
                {
                    "cloaked_snippets": [],
                    "placeholders": [],
                    "modern_tags_used": [],
                    "visible_text": "",
                },
            )

        if len(html) > 300000:
            html = html[:300000]

        try:
            soup = BeautifulSoup(html, "lxml")
        except Exception:
            soup = BeautifulSoup(html, "html.parser")

        # 1. Сначала извлекаем CSS-правила из <style>
        (
            class_styles,
            vendor_css_density,
            important_css_density,
            ai_boilerplate_css,
        ) = self._extract_css_context(soup)

        # 2. Удаляем служебные теги, чтобы CSS-селекторы не попали в text_full и плейсхолдеры
        for s_tag in soup(["script", "style", "noscript", "head", "title"]):
            s_tag.decompose()

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

        chatgpt_copypaste_hits = 0
        gdocs_copypaste_hits = len(GDOCS_GUID_RE.findall(html))
        tailwind_css_hits = len(TAILWIND_VARS_RE.findall(html))
        esp_signature_hits = 0
        bidi_spoofing_hits = len(BIDI_CONTROL_CHARS_RE.findall(html))
        inline_styles = 0
        class_attrs = 0

        for tag in all_tags:
            if tag.name == "bdo" or str(tag.get("dir", "")).lower() == "rtl":
                bidi_spoofing_hits += 1

            st = str(tag.get("style", ""))
            if st:
                inline_styles += 1
                if LLM_COPYPASTE_FONTS.search(st) or LLM_COPYPASTE_BG.search(
                    st
                ):
                    chatgpt_copypaste_hits += 1
                if BIDI_SPOOFING_RE.search(st):
                    bidi_spoofing_hits += 1
                if not ai_boilerplate_css and AI_BOILERPLATE_CSS_RE.search(st):
                    ai_boilerplate_css = 1

            cls_list = tag.get("class")
            if cls_list:
                class_attrs += 1
                cls_joined = (
                    " ".join(cls_list).lower()
                    if isinstance(cls_list, list)
                    else str(cls_list).lower()
                )
                if any(hint in cls_joined for hint in ESP_CLASS_HINTS):
                    esp_signature_hits += 1

        li_tags = soup.find_all("li")
        bold_li_hits = sum(1 for li in li_tags if li.find(["strong", "b"]))
        li_word_counts = [
            len(li.get_text(strip=True).split())
            for li in li_tags
            if len(li.get_text(strip=True)) > 0
        ]
        if len(li_word_counts) >= 3:
            mean_wc = max(float(np.mean(li_word_counts)), 1.0)
            cv_wc = float(np.std(li_word_counts)) / mean_wc
            list_length_uniformity = round(1.0 / (1.0 + cv_wc), 4)
        else:
            list_length_uniformity = 0.0

        tracking_pixels = 0
        ai_alt_placeholder_hits = 0
        for img in images:
            w, h = str(img.get("width", "")), str(img.get("height", ""))
            src = str(img.get("src", "")).lower()
            if w in ("0", "1") or h in ("0", "1") or "track" in src:
                tracking_pixels += 1
            alt_text = str(img.get("alt", "")).strip()
            if alt_text and AI_ALT_PLACEHOLDERS_RE.match(alt_text):
                ai_alt_placeholder_hits += 1

        dummy_links = sum(
            1
            for a in links
            if str(a.get("href", "")).strip() in ("", "#", "javascript:void(0)")
            or "example.com" in str(a.get("href", "")).lower()
        )
        list_unsubscribe_link = any(
            "unsubscribe" in str(a.get("href", "")).lower()
            or "отписаться" in a.get_text().lower()
            for a in links
        )

        comments = soup.find_all(string=lambda s: isinstance(s, Comment))
        mso_comments = sum(
            1 for c in comments if "mso" in c.lower() or "[if" in c.lower()
        )

        modern_tags = [
            tag.name.lower()
            for tag in all_tags
            if tag.name.lower() in MODERN_HTML5_TAGS
        ]
        modern_tag_hits = len(modern_tags)

        doctype_present = int(html.strip().lower().startswith("<!doctype"))
        zero_width_hits = len(ZERO_WIDTH_REGEX.findall(html))

        hidden_elements_count = 0
        hidden_text_chars = 0
        cloaked_snippets = []
        visible_text_chunks: list[str] = []
        max_dom_depth = 0
        leaf_depths = []
        has_legitimate_preheader = False
        root = SemanticASTNode("email", "DOCUMENT_ROOT")

        def match_ai_conf(snippet: str) -> float:
            if not ai_spans or not snippet or len(snippet.split()) < 3:
                return 0.0
            snip_w = set(re.findall(r"\b\w+\b", snippet.lower()))
            for sp in ai_spans:
                target_w = set(
                    re.findall(r"\b\w+\b", sp.get("text_segment", "").lower())
                )
                if snip_w and len(snip_w & target_w) / len(snip_w) >= 0.40:
                    return float(sp.get("confidence_pct", 0.0))
            return 12.0

        def _traverse(
            tag: Tag,
            parent_node: SemanticASTNode,
            parent_bg: tuple[int, int, int],
            logical_depth: int,
            raw_depth: int,
        ):
            nonlocal hidden_elements_count, hidden_text_chars, max_dom_depth, has_legitimate_preheader
            if logical_depth > max_dom_depth:
                max_dom_depth = logical_depth
            if logical_depth > MAX_DOM_TRAVERSAL_DEPTH or raw_depth > 45:
                return

            direct_children = [
                c for c in tag.find_all(recursive=False) if isinstance(c, Tag)
            ]
            if not direct_children:
                leaf_depths.append(logical_depth)

            for child in direct_children[:MAX_CHILDREN_PER_NODE]:
                style_str = str(child.get("style", "")).lower()
                classes = (
                    [c.lower() for c in child.get("class", [])]
                    if isinstance(child.get("class"), list)
                    else []
                )
                for c in classes:
                    if c in class_styles:
                        style_str += ";" + class_styles[c]

                direct_text = " ".join(
                    t.strip()
                    for t in child.find_all(string=True, recursive=False)
                    if not isinstance(t, Comment) and t.strip()
                )
                is_hidden_attr = child.get("hidden") is not None

                is_hidden, reason, eff_bg, contrast = self._check_hidden_status(
                    style_str,
                    parent_bg,
                    logical_depth,
                    direct_text,
                    is_hidden_attr,
                )

                if reason == "PREHEADER":
                    has_legitimate_preheader = True

                if is_hidden:
                    hidden_elements_count += 1
                    subtree_text = child.get_text(separator=" ", strip=True)
                    hidden_text_chars += len(subtree_text)
                    cloaked_snippets.append({
                        "tag": child.name,
                        "reason": reason,
                        "text": (direct_text or subtree_text)[:60]
                        or "[Скрытый блок]",
                    })
                    # Не идем вглубь скрытого узла для сбора видимого текста (защита NLP от отравления)
                    node = SemanticASTNode(
                        child.name,
                        "CLOAKED",
                        direct_text,
                        True,
                        reason,
                        0.0,
                        contrast,
                    )
                    parent_node.children.append(node)
                    continue

                if direct_text:
                    visible_text_chunks.append(direct_text)

                children_tags = [
                    c
                    for c in child.find_all(recursive=False)
                    if isinstance(c, Tag)
                ]
                role = (
                    "HEADER"
                    if child.name in ("h1", "h2", "h3")
                    else ("CONTAINER" if children_tags else "PARAGRAPH")
                )
                ai_conf = match_ai_conf(direct_text)

                if (
                    role == "CONTAINER"
                    and not direct_text
                    and len(children_tags) == 1
                ):
                    _traverse(
                        child,
                        parent_node,
                        eff_bg,
                        logical_depth,
                        raw_depth + 1,
                    )
                    continue

                node = SemanticASTNode(
                    child.name,
                    role,
                    direct_text,
                    False,
                    reason,
                    ai_conf,
                    contrast,
                )
                parent_node.children.append(node)
                if children_tags:
                    _traverse(
                        child, node, eff_bg, logical_depth + 1, raw_depth + 1
                    )

        # Если текст лежит прямо в body без тегов-оберток
        root_direct_text = " ".join(
            t.strip()
            for t in body.find_all(string=True, recursive=False)
            if not isinstance(t, Comment) and t.strip()
        )
        if root_direct_text:
            visible_text_chunks.append(root_direct_text)

        _traverse(body, root, self.default_bg, logical_depth=1, raw_depth=1)

        leaf_variance = (
            float(np.var(leaf_depths)) if len(leaf_depths) > 1 else 0.0
        )
        cloaked_text_ratio = min(1.0, round(hidden_text_chars / text_len, 4))
        clean_visible_text = (
            " ".join(visible_text_chunks).strip() or text_full
        )

        features: dict[str, float | int] = {
            "tag_count": tag_count,
            "text_to_tag_ratio": round(text_len / max(1, tag_count), 2),
            "dom_max_depth": max_dom_depth,
            "dom_leaf_depth_variance": round(leaf_variance, 2),
            "table_count": len(tables),
            "has_table_layout": int(len(tables) > 0),
            "mso_comment_count": mso_comments,
            "has_mso_comments": int(mso_comments > 0),
            "vendor_css_density": vendor_css_density,
            "important_css_density": important_css_density,
            "ai_boilerplate_css": ai_boilerplate_css,
            "doctype_present": doctype_present,
            "has_unsubscribe_link": int(list_unsubscribe_link),
            "has_legitimate_preheader": int(has_legitimate_preheader),
            "esp_signature_hits": esp_signature_hits,
            "modern_html5_tags_count": modern_tag_hits,
            "placeholder_hits": sum(
                len(p.findall(clean_visible_text)) for p in PLACEHOLDER_REGEX
            ),
            "markdown_leftover_hits": sum(
                len(p.findall(clean_visible_text)) for p in MARKDOWN_REGEX
            ),
            "hidden_elements_count": hidden_elements_count,
            "cloaked_text_ratio": cloaked_text_ratio,
            "zero_width_chars_count": zero_width_hits,
            "bidi_spoofing_hits": bidi_spoofing_hits,
            "tracking_pixel_count": tracking_pixels,
            "dummy_link_count": dummy_links,
            "has_forms": int(len(forms) > 0),
            "has_iframes": int(len(iframes) > 0),
            "image_count": len(images),
            "ai_alt_placeholder_hits": ai_alt_placeholder_hits,
            "link_count": len(links),
            "inline_style_to_class_ratio": round(
                inline_styles / max(class_attrs, 1), 4
            ),
            "chatgpt_copypaste_hits": chatgpt_copypaste_hits,
            "gdocs_copypaste_hits": gdocs_copypaste_hits,
            "tailwind_css_hits": tailwind_css_hits,
            "bold_list_anomalies": bold_li_hits,
            "list_length_uniformity": list_length_uniformity,
        }

        evidence = {
            "cloaked_snippets": cloaked_snippets[:5],
            "placeholders": list(
                set(
                    re.findall(
                        r"\[[A-Za-zА-Яа-яЁё0-9 _/-]{2,35}\]|\{\{[a-zA-Z0-9_.-]+\}\}",
                        clean_visible_text,
                    )
                )
            )[:5],
            "modern_tags_used": list(set(modern_tags))[:5],
            "visible_text": clean_visible_text,
        }

        return features, root, evidence

    def _empty_features(self) -> dict[str, float | int]:
        return {k: 0 for k in self.FEATURE_KEYS}

    def _explain_triggers(self, feats: dict[str, float | int]) -> list[str]:
        triggers = []
        if feats["chatgpt_copypaste_hits"] > 0:
            triggers.append(
                f"LLM_UI_PASTE_SCARS ({feats['chatgpt_copypaste_hits']})"
            )
        if feats["tailwind_css_hits"] > 0:
            triggers.append(
                f"TAILWIND_ARTIFACT_VARS ({feats['tailwind_css_hits']})"
            )
        if feats["gdocs_copypaste_hits"] > 0:
            triggers.append("GDOCS_INTERMEDIATE_PASTE")
        if feats["markdown_leftover_hits"] > 0:
            triggers.append(
                f"UNPARSED_MARKDOWN ({feats['markdown_leftover_hits']})"
            )
        if feats["placeholder_hits"] > 0:
            triggers.append(
                f"UNFILLED_PLACEHOLDERS ({feats['placeholder_hits']})"
            )
        if feats["ai_alt_placeholder_hits"] > 0:
            triggers.append("GENERIC_AI_ALT_TAGS")
        if feats["ai_boilerplate_css"] > 0 and feats["has_mso_comments"] == 0:
            triggers.append("STERILE_AI_CSS_BOILERPLATE")
        if (
            feats["modern_html5_tags_count"] >= 2
            and feats["has_table_layout"] == 0
        ):
            triggers.append("WEB_HTML5_INSTEAD_OF_EMAIL_TABLES")
        if (
            feats["bold_list_anomalies"] >= 2
            and feats["list_length_uniformity"] > 0.65
        ):
            triggers.append("UNIFORM_BOLD_LIST_PATTERN")
        if (
            feats["hidden_elements_count"] > 0
            or feats["cloaked_text_ratio"] > 0.05
        ):
            triggers.append(
                f"ADVERSARIAL_CLOAKING (ratio={feats['cloaked_text_ratio']})"
            )
        if feats["bidi_spoofing_hits"] > 0:
            triggers.append("BIDI_RTL_TEXT_SPOOFING")
        return triggers

    def predict(
        self, html: str, ai_spans: list[dict] | None = None
    ) -> dict[str, Any]:
        feats, ast_tree, evidence = self.extract_features(
            html, ai_spans=ai_spans
        )
        visible_text = evidence.pop("visible_text", "")

        cloaking_risk = 0.0
        if feats["hidden_elements_count"] >= 1:
            cloaking_risk += 0.55
        if feats["cloaked_text_ratio"] > 0.15:
            cloaking_risk += 0.35
        if feats["bidi_spoofing_hits"] > 0:
            cloaking_risk += 0.60
        if feats["zero_width_chars_count"] > 3:
            cloaking_risk += 0.25
        if feats["has_forms"] > 0 or feats["has_iframes"] > 0:
            cloaking_risk += 0.40
        cloaking_risk_score = min(1.0, round(cloaking_risk, 4))

        if self.model is not None and self.feature_names is not None:
            x_vec = np.array(
                [[float(feats.get(col, 0.0)) for col in self.feature_names]],
                dtype=np.float64,
            )
            proba = round(float(self.model.predict_proba(x_vec)[0, 1]), 4)
        else:
            ai_risk = 0.0
            if feats["chatgpt_copypaste_hits"] > 0:
                ai_risk += 0.45
            if feats["tailwind_css_hits"] > 0:
                ai_risk += 0.30
            if feats["markdown_leftover_hits"] > 0:
                ai_risk += 0.35
            if feats["placeholder_hits"] > 0:
                ai_risk += 0.35
            if feats["ai_boilerplate_css"] > 0:
                ai_risk += 0.25
            if feats["ai_alt_placeholder_hits"] > 0:
                ai_risk += 0.20
            if (
                feats["modern_html5_tags_count"] >= 2
                and feats["has_mso_comments"] == 0
            ):
                ai_risk += 0.25
            if feats["bold_list_anomalies"] >= 2:
                ai_risk += 0.20
            if feats["list_length_uniformity"] > 0.70:
                ai_risk += 0.15

            if feats["has_mso_comments"] > 0 or feats["esp_signature_hits"] > 0:
                ai_risk -= 0.35
            if feats["has_unsubscribe_link"] and feats["has_table_layout"]:
                ai_risk -= 0.20

            proba = min(1.0, max(0.0, round(ai_risk, 4)))

        result = ForensicResult(
            html_ai_proba=proba,
            cloaking_risk_score=cloaking_risk_score,
            top_risk_factors=self._explain_triggers(feats),
            features=feats,
            ast_tree=ast_tree,
            evidence=evidence,
            visible_text=visible_text,
        )
        return result.to_dict()


def _extract_single_worker(html_str: str) -> dict[str, float | int]:
    det = HTMLDetector(model_path="")
    return det.extract_features(html_str)[0]


def train_html_model(
    human_dataset_path: str,
    ai_dataset_path: str,
    model_out: str = "models/html_ai_detector.joblib",
):
    def load_records(path: str, default_label: int) -> pd.DataFrame:
        with open(path, "r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        df = pd.DataFrame(rows)
        if "label" not in df.columns:
            df["label"] = default_label
        html_col = "html" if "html" in df.columns else "raw_html"
        return df[[html_col, "label"]].rename(columns={html_col: "html"})

    logger.info("Загрузка датасетов...")
    combined = pd.concat(
        [
            load_records(human_dataset_path, 0),
            load_records(ai_dataset_path, 1),
        ],
        ignore_index=True,
    )

    html_list = combined["html"].fillna("").tolist()
    logger.info(f"Извлечение признаков из {len(html_list)} писем...")

    with ProcessPoolExecutor() as executor:
        feature_rows = list(
            executor.map(_extract_single_worker, html_list, chunksize=50)
        )

    df_features = pd.DataFrame(feature_rows)
    X = df_features.drop(columns=["label", "lang"], errors="ignore")
    y = combined["label"].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=0.2, random_state=42, stratify=y
    )

    base_model = HistGradientBoostingClassifier(
        max_depth=6, learning_rate=0.05, max_iter=350, random_state=42
    )
    calibrated_model = CalibratedClassifierCV(
        estimator=base_model, method="isotonic", cv=5
    )
    calibrated_model.fit(X_train, y_train)

    y_probs = calibrated_model.predict_proba(X_test)[:, 1]
    y_preds = calibrated_model.predict(X_test)

    logger.info(f"ROC-AUC: {roc_auc_score(y_test, y_probs):.4f}")
    logger.info(
        "\n"
        + classification_report(y_test, y_preds, target_names=["Human", "AI"])
    )

    os.makedirs(os.path.dirname(model_out) or ".", exist_ok=True)
    joblib.dump(
        {"model": calibrated_model, "feature_names": list(X.columns)}, model_out
    )
    logger.info(f"Откалиброванная модель сохранена в: {model_out}")
