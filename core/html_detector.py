import os
import re
import json
import pandas as pd
from bs4 import BeautifulSoup, Tag, Comment

# Распознавание цветов и палитры
HEX_COLOR_RE = re.compile(r"#([0-9a-fA-F]{3,6})")
RGB_COLOR_RE = re.compile(r"rgba?\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)")

NAMED_COLORS = {
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "gray": (128, 128, 128),
    "grey": (128, 128, 128),
    "transparent": None
}

# Поиск артефактов генерации ИИ и шаблонизаторов
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

# Геометрическое, масштабное и пограничное сокрытие
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

ZERO_WIDTH_REGEX = re.compile(r"[\u200B\u200C\u200D\uFEFF\u00AD]")
ESP_CLASS_HINTS = ("mcntextcontent", "mcnbutton", "mj-", "sg-", "hubspot", "mailchimp")
MODERN_HTML5_TAGS = {"section", "article", "header", "footer", "nav", "main", "aside"}

MAX_CHILDREN_PER_NODE = 40


def parse_color_to_rgb(color_str: str) -> tuple[int, int, int] | None:
    """Парсит hex, rgb или строковые имена цветов в кортеж (R, G, B)."""
    if not color_str:
        return None
    c = color_str.strip().lower()
    if c in NAMED_COLORS:
        return NAMED_COLORS[c]

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

    rgb_m = RGB_COLOR_RE.search(c)
    if rgb_m:
        try:
            return (int(rgb_m.group(1)), int(rgb_m.group(2)), int(rgb_m.group(3)))
        except ValueError:
            return None
    return None


def calculate_relative_luminance(rgb: tuple[int, int, int]) -> float:
    """Вычисляет относительную яркость по формуле стандарта WCAG 2.1."""
    def channel_lum(val: int) -> float:
        v = val / 255.0
        return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4
    return 0.2126 * channel_lum(rgb[0]) + 0.7152 * channel_lum(rgb[1]) + 0.0722 * channel_lum(rgb[2])


def calculate_contrast_ratio(rgb1: tuple[int, int, int], rgb2: tuple[int, int, int]) -> float:
    """Вычисляет коэффициент контрастности между двумя цветами."""
    l1 = calculate_relative_luminance(rgb1)
    l2 = calculate_relative_luminance(rgb2)
    lighter = max(l1, l2)
    darker = min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


class SemanticASTNode:
    """Узел семантического AST-дерева с небинарной вероятностью ИИ."""
    def __init__(
        self,
        tag_name: str,
        role: str,
        text: str = "",
        is_hidden: bool = False,
        hidden_reason: str | None = None,
        ai_confidence: float = 0.0,
        contrast_ratio: float | None = None
    ):
        self.tag_name = tag_name
        self.role = role
        self.text = text.strip()[:100]
        self.is_hidden = is_hidden
        self.hidden_reason = hidden_reason
        self.ai_confidence = round(ai_confidence, 2)
        self.contrast_ratio = round(contrast_ratio, 2) if contrast_ratio is not None else None
        self.children: list['SemanticASTNode'] = []

    def to_dict(self) -> dict:
        """Чистый экспорт в JSON для API и веб-дашборда заказчицы."""
        d = {
            "tag": self.tag_name,
            "role": self.role,
            "ai_confidence_pct": self.ai_confidence,
            "is_hidden": self.is_hidden
        }
        if self.text:
            d["text"] = self.text
        if self.is_hidden:
            d["hidden_reason"] = self.hidden_reason
        if self.contrast_ratio is not None:
            d["contrast_ratio"] = self.contrast_ratio
        if self.children:
            d["children"] = [c.to_dict() for c in self.children]
        return d


def _dom_max_depth_safe(soup: BeautifulSoup, max_limit: int = 40) -> int:
    """Безопасный итеративный подсчет глубины вложенности DOM."""
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
    """
    Основной аналитический модуль:
    - Извлекает признаки верстки и скрытого текста для pipeline.py
    - Строит семантическое AST-дерево
    - Рассчитывает контрастность WCAG 2.1 (белый текст на белом фоне)
    - Поддерживает batch_enrich для датасетов
    """
    def __init__(self, default_bg: tuple[int, int, int] = (255, 255, 255)):
        self.default_bg = default_bg

    def _extract_css_context(self, soup: BeautifulSoup) -> tuple[dict[str, str], dict[str, str]]:
        class_rules = {}
        css_variables = {}
        for s in soup.find_all("style"):
            css = s.get_text()
            clean_css = re.sub(r'@media[^{]+{(?:[^{}]+|{[^{}]*})*}', '', css, flags=re.IGNORECASE)

            # Извлечение CSS-переменных
            var_matches = re.findall(r'(--[a-zA-Z0-9_-]+)\s*:\s*([^;!}]+)', clean_css)
            for v_name, v_val in var_matches:
                css_variables[v_name.strip().lower()] = v_val.strip().lower()

            # Извлечение стилей классов
            rules = re.findall(r'\.([a-zA-Z0-9_-]+)\s*\{([^}]+)\}', clean_css)
            for cls_name, declarations in rules:
                class_rules[cls_name.lower()] = declarations.lower()

        return class_rules, css_variables

    def _resolve_css_variables(self, style_str: str, css_vars: dict[str, str]) -> str:
        if "var(" not in style_str:
            return style_str
        for var_name, var_val in css_vars.items():
            style_str = style_str.replace(f"var({var_name})", var_val)
        return style_str

    def _check_hidden_status(
        self,
        style_str: str,
        parent_bg: tuple[int, int, int],
        depth: int,
        text: str,
        is_hidden_attr: bool
    ) -> tuple[bool, str | None, tuple[int, int, int], float | None]:
        # Наследование фона
        bg_match = re.search(r'(?:background-color|background)\s*:\s*([^;!]+)', style_str)
        effective_bg = parse_color_to_rgb(bg_match.group(1)) if bg_match else parent_bg
        if effective_bg is None:
            effective_bg = parent_bg

        if is_hidden_attr:
            return True, "HTML_HIDDEN_ATTR", effective_bg, None

        # Проверка геометрического скрытия
        for p in HIDDEN_CSS_PATTERNS:
            if p.search(style_str):
                # Амнистия легитимного короткого прехедера в начале письма
                if "display" in p.pattern and depth <= 2 and 0 < len(text) <= 180:
                    return False, "PREHEADER", effective_bg, None
                return True, "GEOMETRIC_OR_TRANSFORM_HIDDEN", effective_bg, None

        has_bg_image = bool(re.search(r'background(?:-image)?\s*:\s*url\(', style_str))

        # Расчет цветового камуфляжа WCAG 2.1
        color_match = re.search(r'(?<![a-z-])color\s*:\s*([^;!]+)', style_str)
        if color_match:
            text_color_raw = color_match.group(1).strip()
            if text_color_raw == "transparent":
                return True, "COLOR_TRANSPARENT", effective_bg, 1.0

            text_rgb = parse_color_to_rgb(text_color_raw)
            if text_rgb and effective_bg:
                contrast = calculate_contrast_ratio(text_rgb, effective_bg)
                if contrast < 1.25:
                    return True, f"LOW_CONTRAST_{contrast:.2f}:1", effective_bg, contrast
                if has_bg_image and contrast < 2.0:
                    return True, "BG_IMAGE_CLOAKING_RISK", effective_bg, contrast

        return False, None, effective_bg, None

    def extract_features(self, html: str, ai_spans: list[dict] | None = None) -> tuple[dict, SemanticASTNode, dict]:
        """Единый проход: извлечение фичей, улик и семантического дерева."""
        if not html or not html.strip():
            root = SemanticASTNode("root", "EMPTY")
            return self._empty_features(), root, {"cloaked_snippets": []}

        try:
            soup = BeautifulSoup(html, "lxml")
        except Exception:
            soup = BeautifulSoup(html, "html.parser")

        class_styles, css_vars = self._extract_css_context(soup)
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

        tracking_pixels = 0
        data_uri_count = 0
        for img in images:
            w, h = str(img.get("width", "")), str(img.get("height", ""))
            src = str(img.get("src", "")).lower()
            if (w in ("1", "0") and h in ("1", "0")) or any(k in src for k in ("track", "open", "pixel")):
                tracking_pixels += 1
            if src.startswith("data:image"):
                data_uri_count += 1

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

        comments = soup.find_all(string=lambda s: isinstance(s, Comment))
        mso_comments = sum(1 for c in comments if "mso" in c.lower() or "[if" in c.lower())
        modern_tag_hits = sum(1 for tag in all_tags if tag.name.lower() in MODERN_HTML5_TAGS)
        doctype_present = int(html.strip().lower().startswith("<!doctype"))
        zero_width_hits = len(ZERO_WIDTH_REGEX.findall(html))

        inline_styles = len(soup.find_all(attrs={"style": True}))
        class_attrs = len(soup.find_all(attrs={"class": True}))

        hidden_elements_count = 0
        cloaked_snippets = []
        max_dom_depth = 0
        root = SemanticASTNode("email", "DOCUMENT_ROOT")

        def match_ai_conf(snippet: str) -> float:
            if not ai_spans or not snippet or len(snippet.split()) < 3:
                return 0.0
            snip_w = set(re.findall(r'\b\w+\b', snippet.lower()))
            for sp in ai_spans:
                target_w = set(re.findall(r'\b\w+\b', sp["text_segment"].lower()))
                if snip_w and len(snip_w & target_w) / len(snip_w) >= 0.40:
                    return float(sp["confidence_pct"])
            return 12.0

        def _traverse(tag: Tag, parent_node: SemanticASTNode, parent_bg: tuple[int, int, int], depth: int):
            nonlocal hidden_elements_count, max_dom_depth
            if depth > max_dom_depth:
                max_dom_depth = depth

            direct_children = [c for c in tag.find_all(recursive=False) if isinstance(c, Tag)]
            process_children = direct_children[:MAX_CHILDREN_PER_NODE]

            for child in process_children:
                if child.name in ("script", "noscript", "style"):
                    continue

                style_str = str(child.get("style", "")).lower()
                classes = [c.lower() for c in child.get("class", [])] if isinstance(child.get("class"), list) else []
                for c in classes:
                    if c in class_styles:
                        style_str += ";" + class_styles[c]

                style_str = self._resolve_css_variables(style_str, css_vars)
                text_content = child.get_text(separator=" ", strip=True)
                is_hidden_attr = child.get("hidden") is not None

                is_hidden, reason, eff_bg, contrast = self._check_hidden_status(
                    style_str, parent_bg, depth, text_content, is_hidden_attr
                )

                if is_hidden:
                    hidden_elements_count += 1
                    cloaked_snippets.append({
                        "tag": child.name,
                        "reason": reason,
                        "text": text_content[:80] or "[Скрытый блок]"
                    })

                if is_hidden:
                    role = "CLOAKED_HIDDEN_BLOCK"
                elif reason == "PREHEADER":
                    role = "PREHEADER"
                elif child.name in ("h1", "h2", "h3"):
                    role = "HEADER"
                elif child.name == "a":
                    role = "BUTTON_OR_LINK"
                elif child.name == "form":
                    role = "FORM_CONTAINER"
                elif child.name in ("p", "span", "td"):
                    role = "PARAGRAPH"
                else:
                    role = "CONTAINER"

                ai_conf = match_ai_conf(text_content)

                children_tags = [c for c in child.find_all(recursive=False) if isinstance(c, Tag)]
                direct_text = "".join(child.find_all(string=True, recursive=False)).strip()
                if role == "CONTAINER" and not direct_text and len(children_tags) == 1 and not is_hidden:
                    _traverse(child, parent_node, eff_bg, depth)
                    continue

                node = SemanticASTNode(
                    tag_name=child.name,
                    role=role,
                    text=direct_text or (text_content[:60] if not children_tags else ""),
                    is_hidden=is_hidden,
                    hidden_reason=reason,
                    ai_confidence=ai_conf,
                    contrast_ratio=contrast
                )
                parent_node.children.append(node)
                _traverse(child, node, eff_bg, depth + 1)

        _traverse(body, root, self.default_bg, depth=1)

        features = {
            "tag_count": tag_count,
            "text_len": text_len,
            "dom_max_depth": max_dom_depth,
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

        evidence = {
            "cloaked_snippets": cloaked_snippets[:5]
        }

        return features, root, evidence

    def _empty_features(self) -> dict:
        keys = [
            "tag_count", "text_len", "dom_max_depth", "table_count", "has_table_layout",
            "mso_comment_count", "has_mso_comments", "doctype_present", "has_unsubscribe_link",
            "modern_html5_tags_count", "hidden_elements_count", "zero_width_chars_count",
            "tracking_pixel_count", "dummy_link_count", "has_forms", "has_iframes",
            "data_uri_count", "image_count", "link_count", "inline_style_to_class_ratio"
        ]
        return {k: 0 for k in keys}

    def predict(self, html: str, ai_spans: list[dict] | None = None) -> dict:
        """Метод инференса: полностью совместим с pipeline.py Богдана."""
        feats, ast_tree, evidence = self.extract_features(html, ai_spans=ai_spans)
        return {
            "features": feats,
            "ast_tree": ast_tree,
            "evidence": evidence
        }

    def batch_enrich(self, input_jsonl: str, output_jsonl: str):
        """Метод пакетной разметки датасетов Богдана."""
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
        print(f"[HTMLDetector] ✓ Размечено {len(out_df)} строк в {output_jsonl}")
