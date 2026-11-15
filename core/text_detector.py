import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any

from catboost import CatBoostClassifier
import numpy as np

# Автопоиск папки, где лежат cb_ru.cbm и cb_eng.cbm (в корне или в core/)
_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_CUR_DIR)

if os.path.exists(os.path.join(_ROOT_DIR, "weights", "cb_ru.cbm")):
  BASE_DIR = os.path.join(_ROOT_DIR, "weights")
elif os.path.exists(os.path.join(_CUR_DIR, "cb_ru.cbm")):
  BASE_DIR = _CUR_DIR
else:
  BASE_DIR = _ROOT_DIR


MODEL_NAME = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")

MAX_TOKENS_CAP = 1024
MIN_WINDOW_TOKENS = 32
FALLBACK_WIN = 96
FALLBACK_STRIDE = 32

FEATURE_ORDER = [
    "lr_global", "lrr_global", "fd_global",
    "lr_win_max", "lr_win_mean", "lr_win_std", "lr_win_q90",
    "lrr_win_max", "lrr_win_mean", "lrr_win_std",
    "fd_win_max", "fd_win_mean", "fd_win_std",
    "merged_span_lr", "merged_span_ratio",
    "avg_sent_len", "sent_len_var", "short_sent_ratio",
    "root_ttr", "word_len_var", "comma_rate",
    "dash_rate", "excl_quest_rate", "colon_semi_rate",
]

logger = logging.getLogger("DrWebSpamPipeline")


def detect_language(text: str) -> str:
    cyrillic_count = len(re.findall(r"[а-яА-ЯёЁ]", text))
    latin_count = len(re.findall(r"[a-zA-Z]", text))
    return "ru" if cyrillic_count >= latin_count else "eng"


def sanitize_text_for_nlp(text: str) -> str:
    """Удаляет URL, email, телефоны и длинные хэши, сохраняя абзацы \\n."""
    if not isinstance(text, str) or not text:
        return ""
    # 1. URL-ссылки и домены
    cleaned = re.sub(r"<?https?://\S+>?|www\.\S+", " ", text, flags=re.I)
    # 2. Email-адреса
    cleaned = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", " ", cleaned)
    # 3. Номера телефонов (+7 (999) 123-45-67, 8-800-555-35-35 и т.д.)
    cleaned = re.sub(r"(?:\+?\d[\d\-\(\) ]{7,}\d)", " ", cleaned)
    # 4. Длинные технические токены/хэши (> 40 символов без пробелов) и разделители
    cleaned = re.sub(r"\S{40,}", " ", cleaned)
    cleaned = re.sub(r"[-=_*#~|]{3,}", " ", cleaned)
    # 5. Сохраняем границы абзацев (\n\n), но убираем одиночные переносы строк внутри предложений
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r" *\n *", "\n", cleaned)
    cleaned = re.sub(r"(?<!\n)\n(?!\n)", " ", cleaned)
    cleaned = re.sub(r"\n{2,}", "\n", cleaned)
    return cleaned.strip()


def extract_stylometric_features(text: str) -> dict[str, float]:
    zero_stylo = {k: 0.0 for k in FEATURE_ORDER[15:]}
    if not isinstance(text, str) or len(text.strip()) < 5:
        return zero_stylo

    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if len(s.strip()) > 0]
    sent_lengths = [len(re.findall(r"\b[^\W\d_]+\b", s)) for s in sentences]
    sent_lengths = [l for l in sent_lengths if l > 0]

    if len(sent_lengths) > 0:
        avg_sent_len = float(np.mean(sent_lengths))
        sent_len_var = float(np.var(sent_lengths)) if len(sent_lengths) > 1 else 0.0
        short_sent_ratio = float(sum(1 for l in sent_lengths if l < 4) / len(sent_lengths))
    else:
        avg_sent_len, sent_len_var, short_sent_ratio = 0.0, 0.0, 0.0

    words = re.findall(r"\b[^\W\d_]+\b", text.lower())
    n_words = max(len(words), 1)

    root_ttr = float(len(set(words)) / np.sqrt(n_words))
    word_lengths = [len(w) for w in words]
    word_len_var = float(np.var(word_lengths)) if len(word_lengths) > 1 else 0.0

    return {
        "avg_sent_len": avg_sent_len,
        "sent_len_var": sent_len_var,
        "short_sent_ratio": short_sent_ratio,
        "root_ttr": root_ttr,
        "word_len_var": word_len_var,
        "comma_rate": float(text.count(",") / n_words),
        "dash_rate": float(sum(text.count(c) for c in ["-", "—", "–"]) / n_words),
        "excl_quest_rate": float((text.count("!") + text.count("?")) / n_words),
        "colon_semi_rate": float((text.count(":") + text.count(";")) / n_words),
    }


def build_sentence_windows(
    text: str, offsets: list[tuple[int, int]], n_metric_tokens: int
) -> list[tuple[int, int]]:
    if n_metric_tokens <= MIN_WINDOW_TOKENS:
        return [(0, n_metric_tokens)]

    sent_char_spans = [
        (m.start(), m.end())
        for m in re.finditer(r"[^.!?\n]+[.!?\n]*", text)
        if len(m.group(0).strip()) > 2
    ]

    token_ends = np.array([offsets[i + 1][1] for i in range(n_metric_tokens)])
    sent_tok_bounds = []
    cur_tok = 0
    for _, c_end in sent_char_spans:
        next_tok = int(np.searchsorted(token_ends, c_end, side="right"))
        if next_tok > cur_tok:
            sent_tok_bounds.append((cur_tok, min(next_tok, n_metric_tokens)))
            cur_tok = next_tok
        if cur_tok >= n_metric_tokens:
            break

    if len(sent_tok_bounds) < 3:
        bounds = [
            (s, s + FALLBACK_WIN)
            for s in range(0, max(1, n_metric_tokens - FALLBACK_WIN + 1), FALLBACK_STRIDE)
        ]
        if bounds[-1][1] < n_metric_tokens:
            bounds.append((max(0, n_metric_tokens - FALLBACK_WIN), n_metric_tokens))
        return bounds

    windows = []
    num_sents = len(sent_tok_bounds)
    for i in range(num_sents):
        s_tok = sent_tok_bounds[i][0]
        j = min(i + 2, num_sents - 1)
        while j < num_sents - 1 and (sent_tok_bounds[j][1] - s_tok) < MIN_WINDOW_TOKENS:
            j += 1
        e_tok = sent_tok_bounds[j][1]
        if (e_tok - s_tok) < MIN_WINDOW_TOKENS:
            s_tok = max(0, e_tok - MIN_WINDOW_TOKENS)
        if not windows or windows[-1] != (s_tok, e_tok):
            windows.append((s_tok, e_tok))
        if e_tok >= n_metric_tokens:
            break

    return windows


def _heuristic_span_fallback(text: str) -> tuple[str, list[int]]:
    sents = list(re.finditer(r"[^.!?\n]+[.!?\n]*", text))
    if len(sents) <= 3:
        return text.strip(), [0, len(text)]
    best_score = -1.0
    best_span = (0, len(text))
    for i in range(len(sents) - 2):
        s_char, e_char = sents[i].start(), sents[i + 2].end()
        window_txt = text[s_char:e_char]
        words = re.findall(r"\b[^\W\d_]+\b", window_txt.lower())
        if len(words) < 10:
            continue
        ttr = len(set(words)) / np.sqrt(len(words))
        uniformity = 1.0 / (1.0 + abs(len(words) / 3.0 - 18.0) + 0.5 * ttr)
        if uniformity > best_score:
            best_score = uniformity
            best_span = (s_char, e_char)
    return text[best_span[0]:best_span[1]].strip(), [best_span[0], best_span[1]]


class TextDetector:
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: str | None = None,
        mode: str | None = None,
        api_url: str | None = None,
    ):
        self.model_name = model_name
        # По умолчанию 'local', если не задан явно 'remote' или 'features'
        self.mode = (mode or os.getenv("LLM_MODE", "local")).lower()
        self.api_url = (api_url or os.getenv("LLM_API_URL", "http://127.0.0.1:8000")).rstrip("/")
        self.api_key = os.getenv("LLM_API_KEY", "")
        self.timeout = float(os.getenv("LLM_TIMEOUT", "180"))
        self.max_retries = int(os.getenv("LLM_MAX_RETRIES", "3"))
        self.retry_delay = float(os.getenv("LLM_RETRY_DELAY", "1.5"))

        self.tokenizer = None
        self.model = None
        self.device = device or "cpu"

        if self.mode == "local":
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = (
                AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=self.dtype)
                .to(self.device)
                .eval()
            )

        self.models: dict[str, CatBoostClassifier] = {}
        self.stylo_models: dict[str, CatBoostClassifier] = {}
        for lang in ["ru", "eng"]:
            m_path = os.path.join(BASE_DIR, f"cb_{lang}.cbm")
            if os.path.exists(m_path):
                cb = CatBoostClassifier()
                cb.load_model(m_path)
                self.models[lang] = cb
            sm_path = os.path.join(BASE_DIR, f"cb_stylo_{lang}.cbm")
            if os.path.exists(sm_path):
                cb_s = CatBoostClassifier()
                cb_s.load_model(sm_path)
                self.stylo_models[lang] = cb_s

        thresh_path = os.path.join(BASE_DIR, "cb_thresholds.json")
        if os.path.exists(thresh_path):
            with open(thresh_path, "r", encoding="utf-8") as f:
                self.thresholds = json.load(f)
        else:
            self.thresholds = {
                "ru": {"t_strict": 0.828, "t_soft": 0.558},
                "eng": {"t_strict": 0.807, "t_soft": 0.658},
            }

    def _extract_llm_features_remote(self, text: str) -> tuple[dict[str, float], str, list[int]]:
        endpoint = (
            f"{self.api_url}/chat/completions"
            if self.api_url.endswith("/v1")
            else f"{self.api_url}/v1/chat/completions"
        )
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": text}],
            "temperature": 0.0,
        }
        data_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        for attempt in range(1, self.max_retries + 1):
            try:
                req = urllib.request.Request(endpoint, data=data_bytes, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    resp_json = json.loads(resp.read().decode("utf-8"))
                forensics = resp_json.get("dw1_forensics") or json.loads(resp_json["choices"][0]["message"]["content"])
                feats = {k: float(forensics.get("features", {}).get(k, 0.0)) for k in FEATURE_ORDER[:15]}
                return feats, str(forensics.get("most_ai_chunk", "")), list(forensics.get("pred_span_chars", [0, len(text)]))
            except Exception as e:
                logger.warning(f"[LLM API] Попытка {attempt}/{self.max_retries} не удалась ({endpoint}): {e}")
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)

        zero_llm = {k: 0.0 for k in FEATURE_ORDER[:15]}
        chunk, span = _heuristic_span_fallback(text)
        return zero_llm, chunk, span

    def extract_llm_features(self, text: str) -> tuple[dict[str, float], str, list[int]]:
        if self.mode == "features":
            zero_llm = {k: 0.0 for k in FEATURE_ORDER[:15]}
            chunk, span = _heuristic_span_fallback(text)
            return zero_llm, chunk, span

        if self.mode == "remote":
            return self._extract_llm_features_remote(text)

        import torch
        with torch.no_grad():
            enc = self.tokenizer(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=MAX_TOKENS_CAP,
                return_offsets_mapping=True,
            )
            input_ids = enc["input_ids"]
            offsets = enc["offset_mapping"]

            if len(input_ids) < 8:
                zero_llm = {k: 0.0 for k in FEATURE_ORDER[:15]}
                return zero_llm, text, [0, len(text)]

            ids_tensor = torch.tensor([input_ids], device=self.device)
            logits = self.model(ids_tensor).logits[0, :-1, :].float()
            labels = ids_tensor[0, 1:]

            log_probs = torch.log_softmax(logits, dim=-1)
            probs = torch.exp(log_probs)
            token_lp = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

            exp_lp = (probs * log_probs).sum(dim=-1)
            var_lp = (probs * (log_probs - exp_lp.unsqueeze(-1)) ** 2).sum(dim=-1)

            target_logits = logits.gather(dim=-1, index=labels.unsqueeze(-1))
            ranks = (logits > target_logits).sum(dim=-1).float() + 1.0
            log_ranks = torch.log(ranks)

            tlp_np = token_lp.cpu().numpy()
            lr_np = log_ranks.cpu().numpy()
            exp_np = exp_lp.cpu().numpy()
            var_np = var_lp.cpu().numpy()
            n_tok = len(tlp_np)

            lr_global = -float(np.mean(lr_np))
            lrr_global = -float(np.sum(tlp_np) / (np.sum(lr_np) + 1e-8))
            fd_global = float(np.sum(tlp_np - exp_np) / np.sqrt(np.sum(var_np) + 1e-8))

            win_bounds = build_sentence_windows(text, offsets, n_tok)
            lr_w, lrr_w, fd_w = [], [], []
            for s, e in win_bounds:
                w_lp = tlp_np[s:e]
                w_lr = lr_np[s:e]
                w_diff = w_lp - exp_np[s:e]
                w_var = var_np[s:e]
                lr_w.append(-float(np.mean(w_lr)))
                lrr_w.append(-float(np.sum(w_lp) / (np.sum(w_lr) + 1e-8)))
                fd_w.append(float(np.sum(w_diff) / np.sqrt(np.sum(w_var) + 1e-8)))

            lr_w, lrr_w, fd_w = np.array(lr_w), np.array(lrr_w), np.array(fd_w)
            best_idx = int(np.argmax(lr_w))
            peak_lr = lr_w[best_idx]
            mean_lr = float(np.mean(lr_w))
            std_lr = float(np.std(lr_w)) if len(lr_w) > 1 else 0.0

            expand_thresh = max(mean_lr + 0.5 * std_lr, peak_lr - 0.12)

            left_idx = best_idx
            while left_idx > 0 and lr_w[left_idx - 1] >= expand_thresh:
                left_idx -= 1
            right_idx = best_idx
            while right_idx < len(win_bounds) - 1 and lr_w[right_idx + 1] >= expand_thresh:
                right_idx += 1

            span_tok_s, span_tok_e = win_bounds[left_idx][0], win_bounds[right_idx][1]
            merged_span_lr = -float(np.mean(lr_np[span_tok_s:span_tok_e]))
            merged_span_ratio = float((span_tok_e - span_tok_s) / max(n_tok, 1))

            # s_tok + 1 — это первый токен внутри нашего окна (так как метрики сдвинуты на 1 токен из-за labels = ids[1:])
            char_start = int(offsets[min(span_tok_s + 1, len(offsets) - 1)][0])
            char_end = int(offsets[min(span_tok_e, len(offsets) - 1)][1])

            # Дотягиваем влево до начала текущего предложения или абзаца
            while char_start > 0 and text[char_start - 1] not in ".!?\n":
                char_start -= 1
            # Убираем ведущие пробелы/кавычки после точки предыдущего предложения
            while char_start < char_end and text[char_start] in " \t\n\"«'":
                char_start += 1

            # Дотягиваем вправо до конца текущего предложения или абзаца
            while char_end < len(text) and text[char_end - 1] not in ".!?\n" and text[char_end] not in "\n":
                char_end += 1

            most_ai_chunk = text[char_start:char_end].strip()

            llm_feats = {
                "lr_global": lr_global,
                "lrr_global": lrr_global,
                "fd_global": fd_global,
                "lr_win_max": float(np.max(lr_w)),
                "lr_win_mean": mean_lr,
                "lr_win_std": std_lr,
                "lr_win_q90": float(np.percentile(lr_w, 90)),
                "lrr_win_max": float(np.max(lrr_w)),
                "lrr_win_mean": float(np.mean(lrr_w)),
                "lrr_win_std": float(np.std(lrr_w)) if len(lrr_w) > 1 else 0.0,
                "fd_win_max": float(np.max(fd_w)),
                "fd_win_mean": float(np.mean(fd_w)),
                "fd_win_std": float(np.std(fd_w)) if len(fd_w) > 1 else 0.0,
                "merged_span_lr": merged_span_lr,
                "merged_span_ratio": merged_span_ratio,
            }
            return llm_feats, most_ai_chunk, [char_start, char_end]

    def predict(self, text: str, force_lang: str | None = None) -> dict[str, Any]:
        clean_nlp_text = sanitize_text_for_nlp(text)

        if len(clean_nlp_text) < 15:
            return {
                "score": 0.0,
                "status": "CLEAN",
                "most_ai_chunk": "",
                "pred_span_chars": [0, 0],
                "lang": force_lang or "ru",
                "features": {},
            }

        lang = force_lang or detect_language(clean_nlp_text)
        llm_feats, most_ai_chunk, span_chars = self.extract_llm_features(clean_nlp_text)
        stylo_feats = extract_stylometric_features(clean_nlp_text)

        if most_ai_chunk:
            most_ai_chunk = " ".join(most_ai_chunk.split())

        is_llm_empty = all(abs(v) < 1e-9 for v in llm_feats.values())

        all_feats = {**llm_feats, **stylo_feats}

        if is_llm_empty and lang in self.stylo_models:
            x_stylo = np.array([[stylo_feats[col] for col in FEATURE_ORDER[15:]]], dtype=np.float64)
            prob = float(self.stylo_models[lang].predict_proba(x_stylo)[0, 1])
            th_key = f"{lang}_stylo" if f"{lang}_stylo" in self.thresholds else lang
            t_strict = self.thresholds[th_key]["t_strict"]
            t_soft = self.thresholds[th_key]["t_soft"]
        else:
            x_vec = np.array([[all_feats[col] for col in FEATURE_ORDER]], dtype=np.float64)
            prob = float(self.models[lang].predict_proba(x_vec)[0, 1]) if lang in self.models else 0.0
            t_strict = self.thresholds[lang]["t_strict"]
            t_soft = self.thresholds[lang]["t_soft"]

        if prob >= t_strict:
            status = "STRICT_AI"
        elif prob >= t_soft:
            status = "SOFT_AI"
        else:
            status = "CLEAN"

        return {
            "score": round(prob, 4),
            "status": status,
            "most_ai_chunk": most_ai_chunk,
            "pred_span_chars": span_chars,
            "lang": lang,
            "features": all_feats,
        }