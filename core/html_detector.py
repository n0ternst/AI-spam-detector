import os
import re
import html
import math
import joblib
import numpy as np
from urllib.parse import unquote
from collections import Counter
from dataclasses import dataclass, field
from typing import Any
from bs4 import BeautifulSoup, Comment, Tag


@dataclass
class SemanticASTNode:
    tag: str
    text: str
    is_hidden: bool = False
    hide_reason: str = ""
    ai_confidence: float = 0.0
    children: list["SemanticASTNode"] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tag": self.tag,
            "text": self.text[:120] + ("..." if len(self.text) > 120 else ""),
            "is_hidden": self.is_hidden,
            "hide_reason": self.hide_reason,
            "ai_confidence": round(self.ai_confidence, 4),
            "children": [c.to_dict() for c in self.children],
        }


@dataclass
class ForensicResult:
    verdict: str
    global_risk_score: float
    is_cloaking_detected: bool
    top_risk_factors: list[str]
    evidence: dict[str, Any]
    semantic_ast: list[dict[str, Any]]
    visible_text: str


class HTMLDetector:
    """
    Enterprise HTML Forensic & DOM-Cloaking Detector (46 features).
    Provides calibrated probabilities, context-adaptive thresholding tau(x),
    WCAG 2.1 contrast analysis, CSS de-obfuscation, and Semantic AST extraction.
    """
    MAX_DEPTH = 35
    MAX_CHILDREN_PER_NODE = 250

    FEATURE_NAMES = [
        # --- Group 1: LLM / AI Provenance & Structural Formatting (12) ---
        "ai_clipboard_markers", "ai_css_vars_count", "markdown_artifact_count",
        "ai_unicode_typo_density", "paragraph_length_cv", "list_length_uniformity",
        "semantic_html5_ratio", "inline_formatting_density", "sentence_length_cv",
        "bullet_list_ratio", "bold_lead_in_count", "code_block_count",
        # --- Group 2: Human / Legacy / ESP / Corporate Clients (12) ---
        "mso_conditional_count", "table_Nesting_depth", "has_table_layout",
        "legacy_attr_count", "esp_signature_count", "client_quote_markers",
        "has_unsubscribe_pattern", "media_query_count", "vml_namespace_count",
        "meta_generator_present", "reply_forward_markers", "cid_attachment_count",
        # --- Group 3: DOM Topology & Visual Stealth Cloaking (12) ---
        "max_dom_depth", "avg_dom_branching", "tag_transition_entropy",
        "tag_diversity_ratio", "text_to_tag_ratio", "hidden_text_ratio",
        "zero_font_nodes", "low_contrast_nodes", "offscreen_nodes",
        "zero_opacity_nodes", "display_none_nodes", "comment_density",
        # --- Group 4: Phishing, Anti-Evasion & Prompt Injection (10) ---
        "deceptive_link_count", "external_link_ratio", "url_shortener_count",
        "suspicious_uri_count", "zero_width_char_density", "homoglyph_spoof_count",
        "prompt_injection_score", "tracking_pixel_count", "form_input_count",
        "script_iframe_count",
    ]

    _NAMED_COLORS = {
        "white": (255, 255, 255), "black": (0, 0, 0), "red": (255, 0, 0),
        "blue": (0, 0, 255), "green": (0, 128, 0), "gray": (128, 128, 128),
        "grey": (128, 128, 128), "silver": (192, 192, 192), "yellow": (255, 255, 0),
    }
    _ZW_REGEX = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff\u00ad]")
    _SHORTENER_REGEX = re.compile(r"(?:bit\.ly|tinyurl\.com|t\.co|goo\.gl|is\.gd|rb\.gy|ow\.ly)/", re.I)
    _PROMPT_INJ_REGEX = re.compile(
        r"(?:ignore\s+(?:all\s+)?(?:previous|prior)\s+instructions|"
        r"disregard\s+(?:all\s+)?(?:previous|prior)|system\s*prompt|"
        r"you\s+are\s+an?\s+ai|do\s+not\s+flag\s+as\s+spam|"
        r"classify\s+this\s+(?:email|message)\s+as\s+(?:safe|legitimate|ham))", re.I
    )

    def __init__(self, model_path: str | None = None, use_adaptive_threshold: bool = True):
        self.model = None
        self.threshold = 0.52
        self.use_adaptive_threshold = use_adaptive_threshold

        if model_path is None:
          local_dir = (
              os.path.dirname(os.path.abspath(__file__))
              if "__file__" in globals()
              else "."
          )
          root_dir = os.path.dirname(local_dir)
          filename = "html_forensic_classifier.joblib"

          candidates = [
              os.path.join(root_dir, "weights", filename),
              os.path.join("weights", filename),
              os.path.join(local_dir, filename),
              filename,
          ]
          for candidate in candidates:
            if os.path.exists(candidate):
              model_path = candidate
              print("Нашел")
              break


    @staticmethod
    def _deobfuscate_css(css_text: str) -> str:
        if not css_text:
            return ""
        res = html.unescape(css_text)
        res = re.sub(r"/\*.*?\*/", "", res, flags=re.S)
        res = re.sub(
            r"\\([0-9a-fA-F]{1,6})\s?",
            lambda m: chr(int(m.group(1), 16)) if int(m.group(1), 16) < 0x110000 else "",
            res,
        )
        return res.lower()

    @classmethod
    def _parse_style_blocks(cls, style_text: str) -> dict[str, str]:
        clean_css = cls._deobfuscate_css(style_text)
        clean_css = re.sub(r"@media[^{]+\{([\s\S]*?)\}\s*\}", r"\1}", clean_css)
        rules_map: dict[str, str] = {}
        for match in re.finditer(r"([^{}@]+)\{([^}]*)\}", clean_css):
            selectors_raw, props = match.group(1), match.group(2).strip()
            if not props:
                continue
            for sel in selectors_raw.split(","):
                parts = sel.strip().split()
                if parts:
                    target = parts[-1]
                    rules_map[target] = rules_map.get(target, "") + ";" + props
        return rules_map

    @classmethod
    def _parse_color(cls, val: str) -> tuple[tuple[int, int, int] | None, float]:
        val = val.strip().lower()
        if "transparent" in val:
            return (255, 255, 255), 0.0
        if val in cls._NAMED_COLORS:
            return cls._NAMED_COLORS[val], 1.0
        m_hex = re.search(r"#([0-9a-f]{3,8})\b", val)
        if m_hex:
            h = m_hex.group(1)
            if len(h) == 3:
                return tuple(int(c * 2, 16) for c in h), 1.0
            elif len(h) == 6:
                return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)), 1.0
            elif len(h) == 8:
                return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)), int(h[6:8], 16) / 255.0
        m_rgba = re.search(
            r"rgba?\s*\(\s*(\d+)[\s,]+(\d+)[\s,]+(\d+)(?:\s*[,/]\s*([0-9.]+)(%?))?\s*\)", val
        )
        if m_rgba:
            rgb = (min(255, int(m_rgba.group(1))), min(255, int(m_rgba.group(2))), min(255, int(m_rgba.group(3))))
            if m_rgba.group(4) is not None:
                alpha = float(m_rgba.group(4)) / (100.0 if m_rgba.group(5) == "%" else 1.0)
            else:
                alpha = 1.0
            return rgb, alpha
        return None, 1.0

    @staticmethod
    def _wcag_contrast(rgb1: tuple[int, int, int], rgb2: tuple[int, int, int]) -> float:
        def _lum(rgb):
            c_lin = [(c / 255.0) / 12.92 if (c / 255.0) <= 0.03928 else (((c / 255.0) + 0.055) / 1.055) ** 2.4 for c in rgb]
            return 0.2126 * c_lin[0] + 0.7152 * c_lin[1] + 0.0722 * c_lin[2]
        l1, l2 = _lum(rgb1), _lum(rgb2)
        return (max(l1, l2) + 0.05) / (min(l1, l2) + 0.05)

    def _get_effective_style(self, tag: Tag, css_rules_map: dict[str, str]) -> str:
        collected = []
        t_name = tag.name.lower()
        if t_name in css_rules_map:
            collected.append(css_rules_map[t_name])
        for cls_name in (tag.get("class") or []):
            for key in (f".{cls_name.lower()}", f"{t_name}.{cls_name.lower()}"):
                if key in css_rules_map:
                    collected.append(css_rules_map[key])
        t_id = tag.get("id")
        if t_id and f"#{t_id.lower()}" in css_rules_map:
            collected.append(css_rules_map[f"#{t_id.lower()}"])
        if tag.get("style"):
            collected.append(self._deobfuscate_css(tag.get("style")))
        return ";".join(collected)

    def _check_node_stealth(
        self, tag: Tag, css_rules_map: dict[str, str], parent_bg: tuple[int, int, int]
    ) -> tuple[bool, str, dict[str, int], tuple[int, int, int]]:
        flags = {"zero_font": 0, "low_contrast": 0, "offscreen": 0, "zero_opacity": 0, "display_none": 0}
        reasons = []
        style = self._get_effective_style(tag, css_rules_map)

        if tag.has_attr("hidden") or tag.get("aria-hidden") == "true":
            reasons.append("hidden_attr")
            flags["display_none"] = 1
        if re.search(r"display\s*:\s*none|visibility\s*:\s*(?:hidden|collapse)", style):
            reasons.append("display_none")
            flags["display_none"] = 1
        if re.search(r"opacity\s*:\s*0(?:\.0+)?\b|opacity\s*:\s*0\.0[1-4]\b", style):
            reasons.append("zero_opacity")
            flags["zero_opacity"] = 1
            flags["offscreen"] = 1

        m_fs = re.search(r"font-size\s*:\s*(\d+(?:\.\d+)?)\s*(px|pt|em|rem|%)?", style)
        if m_fs and float(m_fs.group(1)) <= 1.0:
            reasons.append("zero_font")
            flags["zero_font"] = 1

        if (
            re.search(r"(?:left|top|text-indent|margin-left|margin-top)\s*:\s*-\d{3,5}px", style)
            or re.search(r"max-height\s*:\s*0(?:px)?\b|height\s*:\s*0(?:px)?\b.*overflow\s*:\s*hidden", style)
            or re.search(r"clip\s*:\s*rect\(\s*0|transform\s*:\s*scale\(\s*0\s*\)", style)
        ):
            reasons.append("offscreen_positioning")
            flags["offscreen"] = 1

        cur_bg = parent_bg
        bg_cand = tag.get("bgcolor") or ""
        m_bg = re.search(r"background(?:-color)?\s*:\s*([^;]+)", style)
        if m_bg:
            bg_cand = m_bg.group(1)
        if bg_cand:
            p_bg, a_bg = self._parse_color(bg_cand)
            if p_bg and a_bg > 0.1:
                cur_bg = p_bg

        fg_cand = tag.get("color") or ""
        m_fg = re.search(r"(?<![a-z-])color\s*:\s*([^;]+)", style)
        if m_fg:
            fg_cand = m_fg.group(1)
        if fg_cand:
            p_fg, a_fg = self._parse_color(fg_cand)
            if a_fg <= 0.05:
                reasons.append("transparent_color")
                flags["zero_opacity"] = 1
                flags["low_contrast"] = 1
            elif p_fg:
                ratio = self._wcag_contrast(p_fg, cur_bg)
                if ratio < 1.30:
                    reasons.append(f"low_contrast({ratio:.2f})")
                    flags["low_contrast"] = 1

        return (len(reasons) > 0), ",".join(reasons), flags, cur_bg

    def extract_features(
        self, raw_html: str, ai_spans: list[dict] | None = None
    ) -> tuple[dict[str, float], list[SemanticASTNode], str, dict[str, Any]]:
        raw_html = raw_html or ""
        soup = BeautifulSoup(raw_html, "html.parser")

        style_blocks = "\n".join(s.get_text() for s in soup.find_all("style"))
        css_rules_map = self._parse_style_blocks(style_blocks)
        media_query_count = len(re.findall(r"@media\b", style_blocks, flags=re.I))

        comments = [str(c) for c in soup.find_all(string=lambda t: isinstance(t, Comment))]
        comment_chars = sum(len(c) for c in comments)
        comment_density = comment_chars / max(1, len(raw_html))
        mso_count = sum(1 for c in comments if re.search(r"\[if\s+[^\]]*mso", c, re.I))
        mso_count += len(re.findall(r"\bmso-[a-z-]+:", raw_html, flags=re.I))
        vml_count = len(re.findall(r"<v:[a-z]+|urn:schemas-microsoft-com:vml", raw_html, flags=re.I))
        meta_gen = 1.0 if soup.find("meta", attrs={"name": re.compile(r"^generator$", re.I)}) else 0.0
        cid_count = len(re.findall(r"src\s*=\s*['\"]cid:", raw_html, flags=re.I))
        script_iframe_count = len(soup.find_all(["script", "iframe", "object", "embed"]))
        form_input_count = len(soup.find_all(["form", "input", "button", "select"]))

        prompt_inj_hits = len(self._PROMPT_INJ_REGEX.findall(" ".join(comments)))

        tracking_pixels = 0
        for img in soup.find_all("img"):
            w, h = str(img.get("width", "")).strip(), str(img.get("height", "")).strip()
            st = (img.get("style") or "").lower().replace(" ", "")
            if w in {"0", "1", "1px", "0px"} or h in {"0", "1", "1px", "0px"} or "width:1px" in st:
                tracking_pixels += 1

        for noisy in soup.find_all(["script", "style", "meta", "link", "head", "title", "noscript"]):
            noisy.decompose()

        ai_clip_markers = len(re.findall(
            r"(?:docs-internal-guid|data-start=|data-end=|data-message-author-role|Söhne|font-claude-message|prose\s+w-full|markdown\s+prose)",
            raw_html, flags=re.I
        ))
        ai_css_vars = len(re.findall(r"--tw-[a-z0-9-]+|--cib-[a-z0-9-]+", raw_html, flags=re.I))
        esp_sigs = len(re.findall(
            r"(?:mc:edit|mcnTextContent|sendgrid|mailchimp|hubspot|hs_cos_wrapper|list-manage\.com|ConstantContact|mj-column-per-100|stripo)",
            raw_html, flags=re.I
        ))
        client_quotes = len(re.findall(
            r"(?:gmail_quote|gmail_attr|WordSection1|AppleMailSignature|yahoo_quoted|outlook-mobile-signature)",
            raw_html, flags=re.I
        ))
        reply_fwd = len(re.findall(r"(?:Original Message|Forwarded message|Начало пересылаемого сообщения|От:\s|Кому:\s)", raw_html, flags=re.I))
        has_unsub = 1.0 if re.search(r"unsubscribe|отписаться|opt-out|manage\s+preferences", raw_html, flags=re.I) else 0.0

        all_tags = soup.find_all(True)
        total_tags = len(all_tags)
        tag_names = [t.name.lower() for t in all_tags]
        tag_diversity_ratio = len(set(tag_names)) / max(1, total_tags)

        if len(tag_names) > 1:
            bigrams = list(zip(tag_names[:-1], tag_names[1:]))
            bg_counts = Counter(bigrams)
            tag_entropy = -sum((c / len(bigrams)) * math.log2(c / len(bigrams)) for c in bg_counts.values())
        else:
            tag_entropy = 0.0

        sem_tags = sum(1 for n in tag_names if n in {"article", "section", "header", "footer", "main", "nav"})
        semantic_html5_ratio = sem_tags / max(1, total_tags)
        bullet_list_ratio = sum(1 for n in tag_names if n in {"ul", "ol", "li"}) / max(1, total_tags)
        code_block_count = len(soup.find_all(["pre", "code"]))

        bold_lead_in_count = 0
        for block in soup.find_all(["li", "p"]):
            first_child = next((c for c in block.children if not isinstance(c, Comment) and str(c).strip()), None)
            if isinstance(first_child, Tag) and first_child.name.lower() in {"strong", "b"}:
                bold_lead_in_count += 1

        legacy_attrs = sum(
            1 for t in all_tags
            if any(t.has_attr(a) for a in ("cellpadding", "cellspacing", "bgcolor", "valign", "border"))
            or t.name in {"font", "center"}
        )

        max_table_depth = max((1 + len(tbl.find_parents("table")) for tbl in soup.find_all("table")), default=0)
        has_table_layout = 1.0 if (max_table_depth >= 1 and len(soup.find_all("td")) >= 2) else 0.0

        deceptive_links, ext_links, shortener_links, susp_uris = 0, 0, 0, 0
        all_hrefs = []
        for a in soup.find_all("a", href=True):
            href = unquote(html.unescape(a["href"].strip())).lower()
            all_hrefs.append(href)
            vis_a = a.get_text(strip=True).lower()
            if href.startswith(("http://", "https://")):
                ext_links += 1
            if self._SHORTENER_REGEX.search(href):
                shortener_links += 1
            if href.startswith(("javascript:", "data:", "vbscript:")):
                susp_uris += 1
            m_vis = re.search(r"https?://([a-z0-9.-]+\.[a-z]{2,})", vis_a)
            m_hrf = re.search(r"https?://([a-z0-9.-]+\.[a-z]{2,})", href)
            if m_vis and m_hrf and (m_vis.group(1) != m_hrf.group(1)):
                deceptive_links += 1
        external_link_ratio = ext_links / max(1, len(all_hrefs))

        visible_chars, hidden_chars = 0, 0
        zero_font_nodes, low_contrast_nodes, offscreen_nodes = 0, 0, 0
        zero_opacity_nodes, display_none_nodes = 0, 0
        max_dom_depth = 0
        branch_counts = []
        cloaked_snippets = []
        BLOCK_TAGS = {"p", "div", "td", "li", "h1", "h2", "h3", "h4", "article", "section", "blockquote", "tr", "table"}

        def _traverse(el: Tag, depth: int, inh_hid: bool, inh_reason: str, p_bg: tuple[int, int, int]) -> list[SemanticASTNode]:
            nonlocal visible_chars, hidden_chars, zero_font_nodes, low_contrast_nodes, offscreen_nodes
            nonlocal zero_opacity_nodes, display_none_nodes, max_dom_depth
            if depth > max_dom_depth:
                max_dom_depth = depth
            if depth > self.MAX_DEPTH:
                return []

            nodes: list[SemanticASTNode] = []
            direct_children = [c for c in el.children if isinstance(c, Tag)]
            if direct_children:
                branch_counts.append(len(direct_children))

            for child in direct_children[: self.MAX_CHILDREN_PER_NODE]:
                is_hid, reason, flags, cur_bg = self._check_node_stealth(child, css_rules_map, p_bg)
                eff_hid = inh_hid or is_hid
                eff_reason = inh_reason if inh_hid else reason

                zero_font_nodes += flags["zero_font"]
                low_contrast_nodes += flags["low_contrast"]
                offscreen_nodes += flags["offscreen"]
                zero_opacity_nodes += flags["zero_opacity"]
                display_none_nodes += flags["display_none"]

                child_subtags = [c for c in child.children if isinstance(c, Tag)]
                has_block_children = any(c.name.lower() in BLOCK_TAGS for c in child_subtags)

                if not has_block_children:
                    full_node_text = " ".join(t.strip() for t in child.find_all(string=True) if not isinstance(t, Comment) and t.strip())
                    if full_node_text:
                        if eff_hid:
                            hidden_chars += len(full_node_text)
                            if len(cloaked_snippets) < 5:
                                cloaked_snippets.append(f"[{eff_reason}] {full_node_text[:100]}")
                        else:
                            visible_chars += len(full_node_text)
                    child_ast = []
                else:
                    full_node_text = " ".join(t.strip() for t in child.find_all(string=True, recursive=False) if not isinstance(t, Comment) and t.strip())
                    if full_node_text:
                        if eff_hid:
                            hidden_chars += len(full_node_text)
                        else:
                            visible_chars += len(full_node_text)
                    child_ast = _traverse(child, depth + 1, eff_hid, eff_reason, cur_bg)

                if len(child_ast) == 1 and not full_node_text and not eff_hid:
                    nodes.extend(child_ast)
                elif full_node_text or child_ast or eff_hid:
                    nodes.append(SemanticASTNode(tag=child.name.lower(), text=full_node_text, is_hidden=eff_hid, hide_reason=eff_reason, children=child_ast))
            return nodes

        root_el = soup.body if soup.body else soup
        ast_roots = _traverse(root_el, 1, False, "", (255, 255, 255))
        avg_dom_branching = float(np.mean(branch_counts)) if branch_counts else 0.0

        vis_chunks, hid_chunks = [], []
        def _collect(nodes: list[SemanticASTNode]):
            for n in nodes:
                if n.text:
                    (hid_chunks if n.is_hidden else vis_chunks).append(n.text)
                if n.children:
                    _collect(n.children)
        _collect(ast_roots)

        visible_text = "\n".join(vis_chunks).strip() or soup.get_text(" ", strip=True)
        visible_chars = max(visible_chars, len(visible_text))
        hidden_full_text = "\n".join(hid_chunks).strip()

        zw_matches = len(self._ZW_REGEX.findall(raw_html))
        zero_width_char_density = (zw_matches * 100.0) / max(1, len(visible_text))
        clean_visible_text = self._ZW_REGEX.sub("", visible_text)

        homoglyph_spoofs = sum(1 for h in all_hrefs if "xn--" in h)
        for w in re.findall(r"\b\w{3,}\b", clean_visible_text):
            if re.search(r"[a-zA-Z]", w) and re.search(r"[а-яА-ЯёЁ]", w):
                homoglyph_spoofs += 1

        prompt_inj_hits += len(self._PROMPT_INJ_REGEX.findall(hidden_full_text))
        prompt_inj_hits += len(self._PROMPT_INJ_REGEX.findall(clean_visible_text))

        if ai_spans:
            self._attach_ai_spans_to_ast(ast_roots, ai_spans)

        p_lengths = [len(p.get_text(strip=True)) for p in soup.find_all(["p", "div"]) if len(p.get_text(strip=True)) > 25]
        paragraph_length_cv = float(np.std(p_lengths) / (np.mean(p_lengths) + 1e-5)) if len(p_lengths) >= 3 else 0.65

        sents = [len(s.strip()) for s in re.split(r"[.!?]+", clean_visible_text) if len(s.strip()) > 10]
        sentence_length_cv = float(np.std(sents) / (np.mean(sents) + 1e-5)) if len(sents) >= 3 else 0.60

        li_lengths = [len(li.get_text(strip=True)) for li in soup.find_all("li") if len(li.get_text(strip=True)) > 5]
        list_length_uniformity = float(1.0 / (1.0 + (np.std(li_lengths) / (np.mean(li_lengths) + 1e-5)))) if len(li_lengths) >= 3 else 0.0

        md_artifacts = len(re.findall(r"(\*\*[^*]+\*\*|##+\s+\w+|```)", clean_visible_text))
        ai_typo_chars = len(re.findall(r"[—–“”‘’•…]", clean_visible_text))
        ai_unicode_typo_density = (ai_typo_chars * 100.0) / max(1, len(clean_visible_text))
        inline_bolds = len(soup.find_all(["strong", "b", "em", "code"]))
        inline_formatting_density = (inline_bolds * 100.0) / max(1, len(clean_visible_text.split()))

        hidden_text_ratio = hidden_chars / max(1, visible_chars + hidden_chars)
        text_to_tag_ratio = visible_chars / max(1, len(raw_html))

        features = {
            "ai_clipboard_markers": float(ai_clip_markers),
            "ai_css_vars_count": float(ai_css_vars),
            "markdown_artifact_count": float(md_artifacts),
            "ai_unicode_typo_density": float(ai_unicode_typo_density),
            "paragraph_length_cv": float(paragraph_length_cv),
            "list_length_uniformity": float(list_length_uniformity),
            "semantic_html5_ratio": float(semantic_html5_ratio),
            "inline_formatting_density": float(inline_formatting_density),
            "sentence_length_cv": float(sentence_length_cv),
            "bullet_list_ratio": float(bullet_list_ratio),
            "bold_lead_in_count": float(bold_lead_in_count),
            "code_block_count": float(code_block_count),
            "mso_conditional_count": float(mso_count),
            "table_Nesting_depth": float(max_table_depth),
            "has_table_layout": float(has_table_layout),
            "legacy_attr_count": float(legacy_attrs),
            "esp_signature_count": float(esp_sigs),
            "client_quote_markers": float(client_quotes),
            "has_unsubscribe_pattern": float(has_unsub),
            "media_query_count": float(media_query_count),
            "vml_namespace_count": float(vml_count),
            "meta_generator_present": float(meta_gen),
            "reply_forward_markers": float(reply_fwd),
            "cid_attachment_count": float(cid_count),
            "max_dom_depth": float(max_dom_depth),
            "avg_dom_branching": float(avg_dom_branching),
            "tag_transition_entropy": float(tag_entropy),
            "tag_diversity_ratio": float(tag_diversity_ratio),
            "text_to_tag_ratio": float(text_to_tag_ratio),
            "hidden_text_ratio": float(hidden_text_ratio),
            "zero_font_nodes": float(zero_font_nodes),
            "low_contrast_nodes": float(low_contrast_nodes),
            "offscreen_nodes": float(offscreen_nodes),
            "zero_opacity_nodes": float(zero_opacity_nodes),
            "display_none_nodes": float(display_none_nodes),
            "comment_density": float(comment_density),
            "deceptive_link_count": float(deceptive_links),
            "external_link_ratio": float(external_link_ratio),
            "url_shortener_count": float(shortener_links),
            "suspicious_uri_count": float(susp_uris),
            "zero_width_char_density": float(zero_width_char_density),
            "homoglyph_spoof_count": float(homoglyph_spoofs),
            "prompt_injection_score": float(prompt_inj_hits),
            "tracking_pixel_count": float(tracking_pixels),
            "form_input_count": float(form_input_count),
            "script_iframe_count": float(script_iframe_count),
        }

        meta = {"cloaked_snippets": cloaked_snippets, "visible_chars": visible_chars, "hidden_chars": hidden_chars}
        return features, ast_roots, clean_visible_text, meta

    def _attach_ai_spans_to_ast(self, nodes: list[Any], ai_spans: list[dict]) -> list[Any]:
        """Supports both list[SemanticASTNode] and serialized list[dict] for SpamAiPipeline."""
        if not nodes or not ai_spans:
            return nodes
        for node in nodes:
            is_dict = isinstance(node, dict)
            node_text = node.get("text", "") if is_dict else node.text
            if node_text:
                best_conf = 0.0
                for span in ai_spans:
                    span_txt = span.get("text", "").strip()
                    conf = float(span.get("confidence", span.get("score", 0.0)))
                    if span_txt and (span_txt in node_text or node_text in span_txt):
                        best_conf = max(best_conf, conf)
                if is_dict:
                    node["ai_confidence"] = round(best_conf, 4)
                else:
                    node.ai_confidence = best_conf
            children = node.get("children", []) if is_dict else node.children
            if children:
                self._attach_ai_spans_to_ast(children, ai_spans)
        return nodes

    @staticmethod
    def score_rule_based_legacy(f: dict[str, float]) -> float:
        score = 0.0
        if f["ai_clipboard_markers"] > 0: score += 0.45
        if f["markdown_artifact_count"] > 0 or f["bold_lead_in_count"] >= 2: score += 0.30
        if f["hidden_text_ratio"] > 0.05 or f["zero_font_nodes"] > 0 or f["low_contrast_nodes"] > 0: score += 0.40
        if f["prompt_injection_score"] > 0 or f["zero_width_char_density"] > 0.1 or f["homoglyph_spoof_count"] > 0: score += 0.45
        if f["ai_unicode_typo_density"] > 0.35: score += 0.25
        if f["mso_conditional_count"] > 0 or f["has_table_layout"] > 0: score -= 0.20
        return float(np.clip(score, 0.01, 0.99))

    @staticmethod
    def get_adaptive_threshold(f: dict[str, float], base_th: float) -> float:
        hard_ai = (
            f["prompt_injection_score"] + f["ai_clipboard_markers"]
            + f["markdown_artifact_count"] + f["ai_css_vars_count"]
            + f["homoglyph_spoof_count"] + (1.0 if f["zero_width_char_density"] > 0.05 else 0.0)
        )
        strong_typo_list = (f["ai_unicode_typo_density"] > 0.35) and (
            f["list_length_uniformity"] > 0.65 or f["bold_lead_in_count"] >= 2
        )
        clean_human_legacy = (
            (f["mso_conditional_count"] > 0 or f["reply_forward_markers"] > 0 or f["legacy_attr_count"] >= 2)
            and f["hidden_text_ratio"] < 0.01
            and f["ai_unicode_typo_density"] < 0.18
            and hard_ai == 0
        )
        if hard_ai > 0 or strong_typo_list:
            return max(0.30, base_th - 0.16)
        if clean_human_legacy:
            return min(0.68, base_th + 0.12)
        return base_th

    def predict(self, raw_html: str, ai_spans: list[dict] | None = None) -> ForensicResult:
        features, ast_roots, vis_text, meta = self.extract_features(raw_html, ai_spans)
        x_vec = np.array([[features[k] for k in self.FEATURE_NAMES]], dtype=np.float32)

        prob = float(self.model.predict_proba(x_vec)[0, 1]) if self.model is not None else self.score_rule_based_legacy(features)
        eff_th = self.get_adaptive_threshold(features, self.threshold) if self.use_adaptive_threshold else self.threshold

        is_cloaking = bool(
            features["hidden_text_ratio"] > 0.03
            or features["zero_font_nodes"] > 0
            or features["low_contrast_nodes"] > 0
            or features["offscreen_nodes"] > 0
            or features["zero_opacity_nodes"] > 0
        )

        risk_factors = []
        if features["ai_clipboard_markers"] > 0: risk_factors.append("AI Clipboard Signatures (ChatGPT/Claude/GDocs)")
        if features["markdown_artifact_count"] > 0: risk_factors.append("Unparsed LLM Markdown Artifacts")
        if features["bold_lead_in_count"] >= 2: risk_factors.append("LLM Bold Lead-In List Pattern (<li><strong>...)")
        if is_cloaking: risk_factors.append("DOM Stealth Cloaking / Hidden Text")
        if features["prompt_injection_score"] > 0: risk_factors.append("Indirect Prompt Injection Attempt")
        if features["zero_width_char_density"] > 0.05: risk_factors.append("Zero-Width Character Obfuscation")
        if features["homoglyph_spoof_count"] > 0: risk_factors.append("IDN / Homoglyph Spoofing Detected")
        if features["deceptive_link_count"] > 0: risk_factors.append("Deceptive Phishing Links (href != text)")

        verdict = "AI_GENERATED_OR_CLOAKED_SPAM" if prob >= eff_th else "HUMAN_LEGITIMATE"
        return ForensicResult(
            verdict=verdict,
            global_risk_score=round(prob, 4),
            is_cloaking_detected=is_cloaking,
            top_risk_factors=risk_factors,
            evidence={
                "features": {k: round(v, 4) for k, v in features.items()},
                "cloaked_snippets": meta["cloaked_snippets"],
                "visible_chars": meta["visible_chars"],
                "hidden_chars": meta["hidden_chars"],
                "base_threshold": round(self.threshold, 4),
                "effective_adaptive_threshold": round(eff_th, 4),
            },
            semantic_ast=[n.to_dict() for n in ast_roots],
            visible_text=vis_text,
        )

    def evaluate_html_risk(self, html_body: str, ai_spans: list[dict] | None = None) -> dict[str, Any]:
        """Adapter method for seamless integration with SpamAiPipeline."""
        res = self.predict(html_body, ai_spans=ai_spans)
        feats = res.evidence["features"]
        cloaking_penalty = 0.0
        if res.is_cloaking_detected:
            cloaking_penalty = min(
                0.35,
                0.15 + 0.5 * feats["hidden_text_ratio"] + 0.05 * (feats["zero_font_nodes"] + feats["low_contrast_nodes"])
            )
        return {
            "verdict": res.verdict,
            "html_ai_score": res.global_risk_score,
            "is_cloaking_detected": res.is_cloaking_detected,
            "cloaking_penalty": round(float(cloaking_penalty), 4),
            "forensic_flags": res.top_risk_factors,
            "ast": res.semantic_ast,
            "visible_text": res.visible_text,
            "evidence": res.evidence,
        }
