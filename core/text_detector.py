import re
import math
import gc
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. СИНТАКСИС ПРЕДЛОЖЕНИЙ
def extract_sentence_stats(text: str) -> tuple[float, float, float]:
    if not text or len(text.split()) < 4:
        return 0.0, 0.0, 0.0

    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    if not sentences:
        return 0.0, 0.0, 0.0

    lengths = [len(s.split()) for s in sentences]
    avg_len = float(np.mean(lengths))
    len_var = float(np.var(lengths)) if len(lengths) > 1 else 0.0
    short_ratio = float(sum(1 for l in lengths if l < 4) / len(sentences))
    return avg_len, len_var, short_ratio


# 2. ЭВРИСТИКИ NO-AI-SLOP
SLOP_BUZZWORDS_EN = {"delve", "tapestry", "crucial", "pivotal", "testament", "beacon", "multifaceted", "paramount", "harness", "foster"}
SLOP_BUZZWORDS_RU = {"погрузиться", "гобелен", "неотъемлемый", "краеугольный", "симфония", "свидетельство", "многогранный", "бесшовный"}

THROAT_CLEARING_PATTERNS = [
    r"\bhere'?s\s+the\s+thing\b", r"\blet'?s\s+dive\s+in\b", r"\bin\s+today'?s\s+fast-paced\b",
    r"\bit'?s\s+important\s+to\s+note\b", r"\bважно\s+(?:понимать|отметить)\b", r"\bстоит\s+отметить\b"
]

BINARY_CONTRAST_PATTERNS = [
    r"\b(?:it'?s|this\s+is)\s+not\s+(?:just|only)\b.*?\b(?:it'?s|it\s+is)\b",
    r"\bnot\s+only\b.*?\bbut\s+also\b", r"\bне\s+просто\b.*?\bа\b", r"\bне\s+только\b.*?\bно\s+и\b"
]

def extract_ai_slop_features(text: str) -> dict:
    if not text or len(text.strip()) < 10:
        return {"slop_buzzwords": 0, "slop_throat_clearing": 0, "slop_binary_contrast": 0, "slop_em_dashes": 0.0}

    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    num_sentences = max(1, len(sentences))

    buzzwords_hit = sum(1 for w in words if w in SLOP_BUZZWORDS_EN or w in SLOP_BUZZWORDS_RU)
    first_two_sent = " ".join(sentences[:2]).lower() if sentences else text_lower
    throat_hit = int(any(re.search(p, first_two_sent) for p in THROAT_CLEARING_PATTERNS))
    contrast_hit = sum(len(re.findall(p, text_lower)) for p in BINARY_CONTRAST_PATTERNS)
    em_dashes_ratio = float(len(re.findall(r'[—–]|--', text)) / num_sentences)

    return {
        "slop_buzzwords": buzzwords_hit,
        "slop_throat_clearing": throat_hit,
        "slop_binary_contrast": contrast_hit,
        "slop_em_dashes": round(em_dashes_ratio, 4)
    }


# 3. BINOCULARS С МИКРО-ЛОКАЛИЗАЦИЕЙ (XAI SPAN HEATMAP)
class BinocularsDetector:
    def __init__(
        self,
        observer_name: str = "Qwen/Qwen2.5-1.5B",
        performer_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: torch.device = DEVICE,
        b_0: float = 0.9015,
        k: float = 35.0
    ):
        self.device = device
        self.b_0 = b_0
        self.k = k

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

    @torch.inference_mode()
    def _score_tokens(self, input_ids: torch.Tensor) -> float:
        attention_mask = torch.ones_like(input_ids)

        obs_out = self.observer(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        obs_logits = obs_out.logits[..., :-1, :].float()
        shift_labels = input_ids[..., 1:]

        obs_log_probs = torch.log_softmax(obs_logits, dim=-1)
        obs_gathered = torch.gather(obs_log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
        log_ppl = -obs_gathered.mean().item()

        perf_out = self.performer(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        perf_logits = perf_out.logits[..., :-1, :].float()

        perf_log_probs = torch.log_softmax(perf_logits, dim=-1)
        perf_probs = torch.exp(perf_log_probs)
        cross_entropy = -(perf_probs * obs_log_probs).sum(dim=-1)
        log_x_ppl = cross_entropy.mean().item()

        return float(log_ppl / log_x_ppl if log_x_ppl != 0 else 1.0)

    @torch.inference_mode()
    def analyze_text(self, text: str, window_size: int = 50, step: int = 15, confidence_threshold: float = 65.0) -> dict:
        """
        Возвращает скоры И ТОЧНУЮ КАРТУ СГЕНЕРИРОВАННЫХ ОТРЕЗКОВ:
        где началось, где закончилось, точная цитата и уверенность в %.
        """
        word_count = len(text.split())
        avg_len, len_var, short_ratio = extract_sentence_stats(text)
        slop = extract_ai_slop_features(text)

        # Защита ультракоротких сообщений
        if not text or word_count < 30:
            return {
                "text_ai_proba": 0.0,
                "is_ai": False,
                "bino_min": 1.0,
                "bino_mean": 1.0,
                "bino_var": 0.0,
                "avg_sent_len": round(avg_len, 2),
                "sent_len_var": round(len_var, 2),
                "short_sent_ratio": round(short_ratio, 2),
                "slop": slop,
                "most_ai_chunk": None,
                "detected_ai_spans": []  # Список найденных зон
            }

        # Включаем offset_mapping для точной привязки к символам исходного текста
        inputs = self.tokenizer(text, return_offsets_mapping=True, return_tensors="pt", truncation=False)
        offset_mapping = inputs["offset_mapping"][0].cpu().numpy()
        input_ids = inputs["input_ids"].to(self.device)
        total_tokens = input_ids.shape[1]

        def get_confidence_pct(score_val: float) -> float:
            return round((1.0 / (1.0 + math.exp(-self.k * (self.b_0 - score_val)))) * 100.0, 2)

        raw_spans = []
        scores = []

        if total_tokens <= window_size:
            score = self._score_tokens(input_ids)
            conf = get_confidence_pct(score)
            scores.append(score)
            if conf >= confidence_threshold:
                raw_spans.append({
                    "start_char": int(offset_mapping[0][0]),
                    "end_char": int(offset_mapping[-1][1]),
                    "text_segment": text.strip(),
                    "confidence_pct": conf,
                    "bino_score": round(score, 4)
                })
        else:
            for start in range(0, total_tokens - window_size + 1, step):
                end = start + window_size
                w_ids = input_ids[:, start:end]
                s = self._score_tokens(w_ids)
                scores.append(s)
                conf = get_confidence_pct(s)

                if conf >= confidence_threshold:
                    start_char = int(offset_mapping[start][0])
                    end_char = int(offset_mapping[end - 1][1])
                    segment_str = text[start_char:end_char].strip()

                    raw_spans.append({
                        "start_char": start_char,
                        "end_char": end_char,
                        "text_segment": segment_str,
                        "confidence_pct": conf,
                        "bino_score": round(s, 4)
                    })

        scores = np.asarray(scores, dtype=np.float32)
        min_idx = int(np.argmin(scores)) if len(scores) > 0 else 0
        b_min = float(scores[min_idx]) if len(scores) > 0 else 1.0
        b_mean = float(np.mean(scores)) if len(scores) > 0 else 1.0
        b_var = float(np.var(scores)) if len(scores) > 0 else 0.0

        # Склеиваем перекрывающиеся зоны генерации в сплошные абзацы
        merged_spans = []
        if raw_spans:
            raw_spans.sort(key=lambda x: x["start_char"])
            merged_spans = [raw_spans[0]]

            for cur in raw_spans[1:]:
                prev = merged_spans[-1]
                # Если зоны пересекаются или идут вплотную друг к другу
                if cur["start_char"] <= prev["end_char"] + 15:
                    prev["end_char"] = max(prev["end_char"], cur["end_char"])
                    prev["confidence_pct"] = max(prev["confidence_pct"], cur["confidence_pct"])
                    prev["bino_score"] = min(prev["bino_score"], cur["bino_score"])
                    prev["text_segment"] = text[prev["start_char"]:prev["end_char"]].strip()
                else:
                    merged_spans.append(cur)

        overall_proba = round(1.0 / (1.0 + math.exp(-self.k * (self.b_0 - b_min))), 4)
        most_ai = merged_spans[0]["text_segment"] if merged_spans else None

        return {
            "text_ai_proba": overall_proba,
            "is_ai": overall_proba >= 0.65,
            "bino_min": round(b_min, 4),
            "bino_mean": round(b_mean, 4),
            "bino_var": round(b_var, 6),
            "avg_sent_len": round(avg_len, 2),
            "sent_len_var": round(len_var, 2),
            "short_sent_ratio": round(short_ratio, 2),
            "slop": slop,
            "most_ai_chunk": most_ai,
            "detected_ai_spans": merged_spans  # ВСЕ ЗОНЫ С КООРДИНАТАМИ И % УВЕРЕННОСТИ!
        }
