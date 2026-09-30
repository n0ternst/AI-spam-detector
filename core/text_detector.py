import os
import re
import gc
import json
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Коэффициенты логистической регрессии (8 признаков)
CONFIG = {
    "ru": {
        "weights": np.array([4.46094, -11.41842, 8.33193, -0.00539, -0.00318, -3.62979, 0.34151, 0.02739], dtype=np.float64),
        "bias": 5.09906,
        "t_strict": 0.8472,
        "t_soft": 0.7236
    },
    "eng": {
        "weights": np.array([4.41917, -3.59224, 7.92443, -0.03285, -0.00617, -3.05382, 0.22090, 0.33094], dtype=np.float64),
        "bias": -3.36463,
        "t_strict": 0.8730,
        "t_soft": 0.7628
    }
}

FEATURE_NAMES = [
    "bino_min", "bino_mean", "bino_var",
    "avg_sent_len", "sent_len_var", "short_sent_ratio",
    "root_ttr", "word_len_var"
]


def clean_text_for_scoring(text: str) -> str:
    """Глубокая очистка текста от технических артефактов перед токенизацией."""
    if not text:
        return ""

    # 1. Замена URL и веб-адресов
    text = re.sub(r'https?://\S+|www\.\S+', ' [ссылка] ', text)

    # 2. Маскирование номеров телефонов (RU/международные)
    phone_pattern = r'(\+?\d{1,3}[\s\-]?)?(\(?\d{2,4}\)?[\s\-]?)?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}'
    text = re.sub(phone_pattern, ' [телефон] ', text)

    # 3. Вырезание остатков CSS, MSO и шрифтовых директив
    text = re.sub(r'@import\s+url\([^)]*\);?', ' ', text)
    text = re.sub(r'@[a-zA-Z\-]+\s*\{[^}]*\}', ' ', text)
    text = re.sub(r'(\b(color|background|font|margin|padding|border|width|display|mso-[a-z\-]+)\s*:[^;]+;)+', ' ', text, flags=re.I)

    # 4. Удаление шаблонных тегов Jinja / ESP
    text = re.sub(r'\{\{[^}]+\}\}|\{%\s*[^%]+\s*%\}|\[[A-Z_]{3,20}\]', ' ', text)

    # 5. Схлопывание пробелов и неразрывных пробелов
    text = text.replace('\xa0', ' ')
    return " ".join(text.split()).strip()


def detect_language(text: str) -> str:
    """Определяет доминирующий язык текста (ru или eng)."""
    cyrillic = len(re.findall(r'[а-яА-ЯёЁ]', text))
    latin = len(re.findall(r'[a-zA-Z]', text))
    return "ru" if cyrillic >= latin else "eng"


def extract_sentence_stats(text: str) -> tuple[float, float, float]:
    """Синтаксические фичи вариативности длины предложений."""
    if not text or len(text.strip()) < 10:
        return 0.0, 0.0, 0.0

    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    if not sentences:
        return 0.0, 0.0, 0.0

    lengths = [len(s.split()) for s in sentences]
    avg_len = float(np.mean(lengths))
    len_var = float(np.var(lengths)) if len(lengths) > 1 else 0.0
    short_ratio = float(sum(1 for l in lengths if l < 4) / len(sentences))
    return avg_len, len_var, short_ratio


def extract_robust_word_features(text: str) -> tuple[float, float]:
    """Лексические фичи разнообразия (Root TTR) и вариативности длины слов."""
    if not isinstance(text, str) or len(text.strip()) < 10:
        return 0.0, 0.0

    words = re.findall(r'\b[^\W\d_]+\b', text.lower())
    n_words = len(words)
    if n_words == 0:
        return 0.0, 0.0

    root_ttr = float(len(set(words)) / np.sqrt(n_words))
    word_lengths = [len(w) for w in words]
    word_len_var = float(np.var(word_lengths)) if len(word_lengths) > 1 else 0.0
    return root_ttr, word_len_var


class TextDetector:
    def __init__(
        self,
        observer_name: str = "Qwen/Qwen2.5-3B",
        performer_name: str = "Qwen/Qwen2.5-3B-Instruct",
        device: torch.device = DEVICE
    ):
        self.device = device
        print(f"[TextDetector] Загрузка {observer_name} и {performer_name} на {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(observer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        self.observer = AutoModelForCausalLM.from_pretrained(
            observer_name, torch_dtype=dtype, low_cpu_mem_usage=True
        ).to(self.device)
        self.performer = AutoModelForCausalLM.from_pretrained(
            performer_name, torch_dtype=dtype, low_cpu_mem_usage=True
        ).to(self.device)

        self.observer.eval()
        self.performer.eval()
        self.observer.config.use_cache = False
        self.performer.config.use_cache = False

        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        print("[TextDetector] ✓ Модели инициализированы.")

    @torch.inference_mode()
    def _score_tokens(self, input_ids: torch.Tensor) -> float:
        attention_mask = torch.ones_like(input_ids)

        obs_out = self.observer(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        obs_logits = obs_out.logits[..., :-1, :].float()
        shift_labels = input_ids[..., 1:]

        obs_log_probs = torch.log_softmax(obs_logits, dim=-1)
        obs_gathered = torch.gather(obs_log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
        log_ppl = -obs_gathered.mean().item()
        del obs_out, obs_logits, shift_labels

        perf_out = self.performer(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        perf_logits = perf_out.logits[..., :-1, :].float()

        perf_log_probs = torch.log_softmax(perf_logits, dim=-1)
        perf_probs = torch.exp(perf_log_probs)
        cross_entropy = -(perf_probs * obs_log_probs).sum(dim=-1)
        log_x_ppl = cross_entropy.mean().item()
        del perf_out, perf_logits, perf_log_probs, perf_probs, obs_log_probs, cross_entropy, attention_mask

        return float(log_ppl / log_x_ppl if log_x_ppl != 0 else 1.0)

    @torch.inference_mode()
    def compute_bino_features(self, clean_text: str, window_size: int = 50, step: int = 15, max_tokens: int = 1024) -> dict:
        if not clean_text or len(clean_text) < 10:
            return {"bino_min": 1.0, "bino_mean": 1.0, "bino_var": 0.0, "most_ai_chunk": ""}

        try:
            inputs = self.tokenizer(clean_text, return_tensors="pt", truncation=False)
            input_ids = inputs["input_ids"][:, :max_tokens].to(self.device)
            total_tokens = input_ids.shape[1]

            if total_tokens <= window_size:
                score = self._score_tokens(input_ids)
                chunk_str = self.tokenizer.decode(input_ids[0], skip_special_tokens=True).strip()
                del input_ids
                return {
                    "bino_min": round(float(score), 4),
                    "bino_mean": round(float(score), 4),
                    "bino_var": 0.0,
                    "most_ai_chunk": chunk_str
                }

            scores, windows = [], []
            for start in range(0, total_tokens - window_size + 1, step):
                w_ids = input_ids[:, start:start + window_size]
                scores.append(self._score_tokens(w_ids))
                windows.append(w_ids)

            del input_ids
            scores_np = np.asarray(scores, dtype=np.float32)
            min_idx = int(np.argmin(scores_np))
            best_chunk = self.tokenizer.decode(windows[min_idx][0], skip_special_tokens=True).strip()

            return {
                "bino_min": round(float(np.min(scores_np)), 4),
                "bino_mean": round(float(np.mean(scores_np)), 4),
                "bino_var": round(float(np.var(scores_np)), 6),
                "most_ai_chunk": best_chunk
            }
        except Exception as e:
            print(f"[TextDetector] Ошибка токенизации: {e}")
            return {"bino_min": 1.0, "bino_mean": 1.0, "bino_var": 0.0, "most_ai_chunk": ""}

    def predict(self, raw_text: str, force_lang: str | None = None) -> dict:
        """Полный цикл: предобработка -> 8 признаков -> логистическая регрессия -> вероятность."""
        clean_text = clean_text_for_scoring(raw_text)
        words = clean_text.split()

        # на очень коротких текстах стилометрия не информативна
        if len(words) < 20:
            return {
                "text_prob": 0.0,
                "lang": force_lang or "ru",
                "clean_text": clean_text,
                "features": {f: 0.0 for f in FEATURE_NAMES},
                "most_ai_chunk": "",
                "status": "TOO_SHORT"
            }

        lang = force_lang or detect_language(clean_text)
        cfg = CONFIG.get(lang, CONFIG["ru"])

        # 1. Binoculars
        bino_res = self.compute_bino_features(clean_text)

        # 2. Синтаксис предложений
        avg_len, len_var, short_ratio = extract_sentence_stats(clean_text)

        # 3. Лексика слов
        root_ttr, word_len_var = extract_robust_word_features(clean_text)

        feats_dict = {
            "bino_min": bino_res["bino_min"],
            "bino_mean": bino_res["bino_mean"],
            "bino_var": bino_res["bino_var"],
            "avg_sent_len": round(avg_len, 2),
            "sent_len_var": round(len_var, 2),
            "short_sent_ratio": round(short_ratio, 2),
            "root_ttr": round(root_ttr, 4),
            "word_len_var": round(word_len_var, 2)
        }

        # 4. Логистическая регрессия
        x_vec = np.array([feats_dict[f] for f in FEATURE_NAMES], dtype=np.float64)
        logit = float(np.dot(x_vec, cfg["weights"]) + cfg["bias"])
        logit = np.clip(logit, -20.0, 20.0)
        prob = float(1.0 / (1.0 + np.exp(-logit)))

        return {
            "text_prob": round(prob, 4),
            "lang": lang,
            "clean_text": clean_text,
            "features": feats_dict,
            "most_ai_chunk": bino_res["most_ai_chunk"],
            "t_strict": cfg["t_strict"],
            "t_soft": cfg["t_soft"],
            "status": "STRICT_AI" if prob >= cfg["t_strict"] else ("SOFT_AI" if prob >= cfg["t_soft"] else "CLEAN")
        }

    def batch_enrich(self, input_jsonl: str, output_jsonl: str):
        """Пакетная разметка jsonl файла 8 признаками и P(AI)."""
        df = pd.read_json(input_jsonl, lines=True)
        results = []
        for _, row in df.iterrows():
            item = row.to_dict()
            res = self.predict(item.get("text", ""))
            item.update(res["features"])
            item["text_prob"] = res["text_prob"]
            item["most_ai_chunk"] = res["most_ai_chunk"]
            results.append(item)
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        out_df = pd.DataFrame(results)
        out_df.to_json(output_jsonl, orient="records", lines=True, force_ascii=False)
        print(f"[TextDetector] ✓ Размечено {len(out_df)} строк в {output_jsonl}")