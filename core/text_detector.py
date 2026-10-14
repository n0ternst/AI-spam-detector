import gc
import json
import logging
import os
import re
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger("TextBinocularsDetector")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"
        )
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Финальные зафиксированные веса (8 признаков, окно 120/30)
CONFIG = {
    "ru": {
        "weights": np.array(
            [
                12.54760,
                -19.45794,
                94.53831,
                -0.00617,
                -0.00347,
                -3.46861,
                0.41326,
                0.01772,
            ],
            dtype=np.float64,
        ),
        "bias": 3.62764,
        "t_strict": 0.8501,
        "t_soft": 0.6605,
    },
    "eng": {
        "weights": np.array(
            [
                13.71354,
                -15.00183,
                137.82553,
                -0.03127,
                -0.00613,
                -2.99892,
                0.25012,
                0.33329,
            ],
            dtype=np.float64,
        ),
        "bias": -1.59887,
        "t_strict": 0.8806,
        "t_soft": 0.6687,
    },
}

FEATURE_NAMES = [
    "bino_min",
    "bino_mean",
    "bino_var",
    "avg_sent_len",
    "sent_len_var",
    "short_sent_ratio",
    "root_ttr",
    "word_len_var",
]


def clean_text_for_scoring(text: str) -> str:
    """Глубокая очистка текста от технических артефактов перед токенизацией."""
    if not text:
        return ""

    # Используем маркеры с подчеркиванием, чтобы они не искажали detect_language и word_features
    text = re.sub(r"https?://\S+|www\.\S+", " [URL_REF] ", text)
    phone_pattern = r"(\+?\d{1,3}[\s\-]?)?(\(?\d{2,4}\)?[\s\-]?)?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}"
    text = re.sub(phone_pattern, " [TEL_REF] ", text)
    text = re.sub(r"@import\s+url\([^)]*\);?", " ", text, flags=re.I)
    text = re.sub(r"@[a-zA-Z\-]+\s*\{[^}]*\}", " ", text)
    text = re.sub(
        r"(\b(color|background|font|margin|padding|border|width|display|mso-[a-z\-]+)\s*:[^;]+;)+",
        " ",
        text,
        flags=re.I,
    )
    text = re.sub(
        r"\{\{[^}]+\}\}|\{%\s*[^%]+\s*%\}|\[[A-Z_]{3,20}\]", " ", text
    )
    text = text.replace("\xa0", " ")
    return " ".join(text.split()).strip()


def detect_language(text: str) -> str:
    """Определяет доминирующий язык текста (ru или eng) без учета технических маркеров."""
    clean_for_lang = re.sub(r"\[(?:URL_REF|TEL_REF)\]", "", text)
    cyrillic = len(re.findall(r"[а-яА-ЯёЁ]", clean_for_lang))
    latin = len(re.findall(r"[a-zA-Z]", clean_for_lang))
    return "ru" if cyrillic >= latin else "eng"


def extract_sentence_stats(text: str) -> tuple[float, float, float]:
    """Синтаксические фичи вариативности длины предложений."""
    if not text or len(text.strip()) < 10:
        return 0.0, 0.0, 0.0

    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    if not sentences:
        return 0.0, 0.0, 0.0

    lengths = [len(s.split()) for s in sentences if len(s.split()) > 0]
    if not lengths:
        return 0.0, 0.0, 0.0

    avg_len = float(np.mean(lengths))
    len_var = float(np.var(lengths)) if len(lengths) > 1 else 0.0
    short_ratio = float(sum(1 for l in lengths if l < 4) / len(lengths))
    return avg_len, len_var, short_ratio


def extract_robust_word_features(text: str) -> tuple[float, float]:
    """Лексические фичи разнообразия (Root TTR) и вариативности длины слов."""
    if not isinstance(text, str) or len(text.strip()) < 10:
        return 0.0, 0.0

    # Исключаем технические плейсхолдеры с подчеркиванием (URL_REF, TEL_REF)
    words = re.findall(r"\b[^\W\d_]+\b", text.lower())
    n_words = len(words)
    if n_words == 0:
        return 0.0, 0.0

    root_ttr = float(len(set(words)) / np.sqrt(n_words))
    word_lengths = [len(w) for w in words]
    word_len_var = (
        float(np.var(word_lengths)) if len(word_lengths) > 1 else 0.0
    )
    return root_ttr, word_len_var


class TextDetector:

    def __init__(
        self,
        observer_name: str = "Qwen/Qwen2.5-0.5B",
        performer_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: torch.device = DEVICE,
        batch_size: int = 16,
    ):
        self.device = device
        self.batch_size = batch_size
        logger.info(
            f"Загрузка {observer_name} и {performer_name} на {self.device}..."
        )

        self.tokenizer = AutoTokenizer.from_pretrained(observer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.device.type == "cuda" and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        elif self.device.type == "cuda":
            dtype = torch.float16
        else:
            dtype = torch.float32

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
        logger.info("✓ Модели 0.5B инициализированы (Батч-режим активен).")

    @torch.inference_mode()
    def _score_windows_batch(
        self, windows_tensor: torch.Tensor
    ) -> list[float]:
        """
        Векторизованный расчет Binoculars сразу для пачки окон [B, W].
        Дает 10-15x ускорение по сравнению с поштучным циклом без изменения математики.
        """
        all_scores: list[float] = []
        n_wins = windows_tensor.shape[0]

        for i in range(0, n_wins, self.batch_size):
            batch_ids = windows_tensor[i : i + self.batch_size]
            attention_mask = torch.ones_like(batch_ids)

            obs_out = self.observer(
                input_ids=batch_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            obs_logits = obs_out.logits[..., :-1, :].float()
            shift_labels = batch_ids[..., 1:]

            obs_log_probs = torch.log_softmax(obs_logits, dim=-1)
            obs_gathered = torch.gather(
                obs_log_probs, dim=-1, index=shift_labels.unsqueeze(-1)
            ).squeeze(-1)
            # Средний log_ppl для каждого окна в батче: размерность [B]
            log_ppl = -obs_gathered.mean(dim=-1)
            del obs_out, obs_logits, shift_labels, obs_gathered

            perf_out = self.performer(
                input_ids=batch_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            perf_logits = perf_out.logits[..., :-1, :].float()

            perf_log_probs = torch.log_softmax(perf_logits, dim=-1)
            perf_probs = torch.exp(perf_log_probs)
            cross_entropy = -(perf_probs * obs_log_probs).sum(dim=-1)
            # Средний log_x_ppl для каждого окна в батче: размерность [B]
            log_x_ppl = cross_entropy.mean(dim=-1)
            del (
                perf_out,
                perf_logits,
                perf_log_probs,
                perf_probs,
                obs_log_probs,
                cross_entropy,
                attention_mask,
            )

            ratios = torch.where(
                log_x_ppl != 0,
                log_ppl / log_x_ppl,
                torch.ones_like(log_ppl),
            )
            all_scores.extend(ratios.cpu().numpy().tolist())

        return [float(s) for s in all_scores]

    @staticmethod
    def _bino_score_to_ai_confidence(bino_score: float) -> float:
        """
        Переводит сырой скор Binoculars отдельного окна в локальную уверенность ИИ (0..100%)
        для подсветки узлов в SemanticASTNode. Чем ниже bino_score (< 0.85), тем выше вероятность ИИ.
        """
        # При bino_score = 0.70 -> ~92% ИИ, при bino_score = 0.95 -> ~12% ИИ
        conf = 1.0 / (1.0 + np.exp((bino_score - 0.82) * 18.0))
        return round(float(conf * 100.0), 1)

    @torch.inference_mode()
    def compute_bino_features(
        self,
        clean_text: str,
        window_size: int = 120,
        step: int = 30,
        max_tokens: int = 1024,
    ) -> dict[str, Any]:
        if not clean_text or len(clean_text) < 10:
            return {
                "bino_min": 1.0,
                "bino_mean": 1.0,
                "bino_var": 0.0,
                "most_ai_chunk": "",
                "detected_ai_spans": [],
            }

        try:
            inputs = self.tokenizer(
                clean_text, return_tensors="pt", truncation=False
            )
            input_ids = inputs["input_ids"][:, :max_tokens].to(self.device)
            total_tokens = input_ids.shape[1]

            if total_tokens <= window_size:
                score = self._score_windows_batch(input_ids)[0]
                chunk_str = self.tokenizer.decode(
                    input_ids[0], skip_special_tokens=True
                ).strip()
                del input_ids
                span_conf = self._bino_score_to_ai_confidence(score)
                return {
                    "bino_min": round(float(score), 4),
                    "bino_mean": round(float(score), 4),
                    "bino_var": 0.0,
                    "most_ai_chunk": chunk_str,
                    "detected_ai_spans": [
                        {
                            "text_segment": chunk_str,
                            "confidence_pct": span_conf,
                            "bino_score": round(float(score), 4),
                        }
                    ],
                }

            # Нарезаем окна тензорной операцией unfold: [1, T] -> [N_windows, window_size]
            windows_tensor = input_ids[0].unfold(0, window_size, step)
            del input_ids

            scores = self._score_windows_batch(windows_tensor)
            scores_np = np.asarray(scores, dtype=np.float32)
            min_idx = int(np.argmin(scores_np))

            # Декодируем лучшее окно и формируем спаны для AST-дерева HTMLDetector
            best_chunk = self.tokenizer.decode(
                windows_tensor[min_idx], skip_special_tokens=True
            ).strip()

            detected_spans: list[dict[str, Any]] = []
            for idx, win_score in enumerate(scores):
                conf_pct = self._bino_score_to_ai_confidence(win_score)
                if conf_pct >= 35.0 or idx == min_idx:
                    win_text = self.tokenizer.decode(
                        windows_tensor[idx], skip_special_tokens=True
                    ).strip()
                    detected_spans.append({
                        "text_segment": win_text,
                        "confidence_pct": conf_pct,
                        "bino_score": round(float(win_score), 4),
                    })

            del windows_tensor

            return {
                "bino_min": round(float(np.min(scores_np)), 4),
                "bino_mean": round(float(np.mean(scores_np)), 4),
                "bino_var": round(float(np.var(scores_np)), 6),
                "most_ai_chunk": best_chunk,
                "detected_ai_spans": detected_spans,
            }
        except Exception as e:
            logger.error(f"Ошибка токенизации/скоринга: {e}")
            return {
                "bino_min": 1.0,
                "bino_mean": 1.0,
                "bino_var": 0.0,
                "most_ai_chunk": "",
                "detected_ai_spans": [],
            }

    def predict(
        self, raw_text: str, force_lang: str | None = None
    ) -> dict[str, Any]:
        """Полный цикл: предобработка -> 8 признаков -> логистическая регрессия -> вероятность и спаны."""
        clean_text = clean_text_for_scoring(raw_text)
        words = clean_text.split()

        if len(words) < 20:
            return {
                "text_prob": 0.0,
                "lang": force_lang or "ru",
                "clean_text": clean_text,
                "features": {f: 0.0 for f in FEATURE_NAMES},
                "most_ai_chunk": "",
                "detected_ai_spans": [],
                "t_strict": CONFIG.get(force_lang or "ru", CONFIG["ru"])[
                    "t_strict"
                ],
                "t_soft": CONFIG.get(force_lang or "ru", CONFIG["ru"])[
                    "t_soft"
                ],
                "status": "TOO_SHORT",
            }

        lang = force_lang or detect_language(clean_text)
        cfg = CONFIG.get(lang, CONFIG["ru"])

        # 1. Binoculars в батч-режиме (окно 120, шаг 30)
        bino_res = self.compute_bino_features(
            clean_text, window_size=120, step=30
        )

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
            "word_len_var": round(word_len_var, 2),
        }

        # 4. Логистическая регрессия
        x_vec = np.array(
            [feats_dict[f] for f in FEATURE_NAMES], dtype=np.float64
        )
        logit = float(np.dot(x_vec, cfg["weights"]) + cfg["bias"])
        logit = float(np.clip(logit, -20.0, 20.0))
        prob = float(1.0 / (1.0 + np.exp(-logit)))

        return {
            "text_prob": round(prob, 4),
            "lang": lang,
            "clean_text": clean_text,
            "features": feats_dict,
            "most_ai_chunk": bino_res["most_ai_chunk"],
            "detected_ai_spans": bino_res["detected_ai_spans"],
            "t_strict": cfg["t_strict"],
            "t_soft": cfg["t_soft"],
            "status": (
                "STRICT_AI"
                if prob >= cfg["t_strict"]
                else ("SOFT_AI" if prob >= cfg["t_soft"] else "CLEAN")
            ),
        }

    def batch_enrich(self, input_jsonl: str, output_jsonl: str):
        """Потоковая пакетная разметка jsonl файла 8 признаками и P(AI)."""
        if not os.path.exists(input_jsonl):
            logger.error(f"Файл не найден: {input_jsonl}")
            return

        count = 0
        with (
            open(input_jsonl, "r", encoding="utf-8") as fin,
            open(output_jsonl, "w", encoding="utf-8") as fout,
        ):
            for line in fin:
                if not line.strip():
                    continue
                item = json.loads(line)
                res = self.predict(
                    item.get("text", item.get("clean_text", "")),
                    force_lang=item.get("lang"),
                )
                item.update(res["features"])
                item["text_prob"] = res["text_prob"]
                item["most_ai_chunk"] = res["most_ai_chunk"]
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                count += 1

        logger.info(f"✓ Размечено {count} строк в {output_jsonl}")
