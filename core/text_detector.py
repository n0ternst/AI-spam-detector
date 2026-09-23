import os
import re
import math
import gc
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_sentence_stats(text: str) -> tuple[float, float, float]:
    #Синтаксический анализ предложений
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


class BinocularsDetector:
    def __init__(
        self,
        observer_name: str = "Qwen/Qwen2.5-3B",
        performer_name: str = "Qwen/Qwen2.5-3B-Instruct",
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
    def analyze_text(self, text: str, window_size: int = 50, step: int = 15) -> dict:
        """Основной инференс: возвращает скоры и вероятность генерации."""
        if not text or len(text.strip()) < 15:
            return {
                "text_ai_proba": 0.0,
                "is_ai": False,
                "bino_min": 1.0,
                "bino_mean": 1.0,
                "bino_var": 0.0,
                "avg_sent_len": 0.0,
                "sent_len_var": 0.0
            }

        avg_len, len_var, short_ratio = extract_sentence_stats(text)

        inputs = self.tokenizer(text, return_tensors="pt", truncation=False)
        input_ids = inputs["input_ids"].to(self.device)
        total_tokens = input_ids.shape[1]

        if total_tokens <= window_size:
            score = self._score_tokens(input_ids)
            b_min, b_mean, b_var = score, score, 0.0
        else:
            scores = []
            for start in range(0, total_tokens - window_size + 1, step):
                w_ids = input_ids[:, start:start + window_size]
                scores.append(self._score_tokens(w_ids))
            scores = np.asarray(scores, dtype=np.float32)
            b_min = float(np.min(scores))
            b_mean = float(np.mean(scores))
            b_var = float(np.var(scores))

        # Расчет вероятности через порог Binoculars
        proba = 1.0 / (1.0 + math.exp(-self.k * (self.b_0 - b_min)))

        return {
            "text_ai_proba": round(proba, 4),
            "is_ai": proba >= 0.65,
            "bino_min": round(b_min, 4),
            "bino_mean": round(b_mean, 4),
            "bino_var": round(b_var, 6),
            "avg_sent_len": round(avg_len, 2),
            "sent_len_var": round(len_var, 2),
            "short_sent_ratio": round(short_ratio, 2)
        }