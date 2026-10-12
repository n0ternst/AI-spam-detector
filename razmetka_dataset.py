import os
import re
import math
import gc
import json
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используемое устройство: {DEVICE}")


# 1. Извлечение синтаксических признаков длины предложений
def extract_sentence_stats(text: str) -> tuple[float, float, float]:
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


# 2. Модель Binoculars на Qwen 2.5 0.5B
class BinocularsScorer:
    def __init__(
        self,
        observer_name: str = "Qwen/Qwen2.5-0.5B",
        performer_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: torch.device = DEVICE
    ):
        self.device = device
        print(f"Загрузка {observer_name} и {performer_name} на {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(observer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Адаптивный выбор типа данных: bfloat16 при аппаратной поддержке, иначе float16
        if self.device.type == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
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
        print("✓ Модели успешно инициализированы!")

    @torch.inference_mode()
    def _score_tokens(self, input_ids: torch.Tensor) -> float:
        attention_mask = torch.ones_like(input_ids)

        # 1. PPL от базовой модели Observer
        obs_out = self.observer(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        obs_logits = obs_out.logits[..., :-1, :].float()
        shift_labels = input_ids[..., 1:]

        obs_log_probs = torch.log_softmax(obs_logits, dim=-1)
        obs_gathered = torch.gather(obs_log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
        log_ppl = -obs_gathered.mean().item()
        del obs_out, obs_logits, shift_labels

        # 2. Кросс-перплексия (Performer -> Observer)
        perf_out = self.performer(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        perf_logits = perf_out.logits[..., :-1, :].float()

        perf_log_probs = torch.log_softmax(perf_logits, dim=-1)
        perf_probs = torch.exp(perf_log_probs)
        cross_entropy = -(perf_probs * obs_log_probs).sum(dim=-1)
        log_x_ppl = cross_entropy.mean().item()
        del perf_out, perf_logits, perf_log_probs, perf_probs, obs_log_probs, cross_entropy, attention_mask

        return float(log_ppl / log_x_ppl if log_x_ppl != 0 else 1.0)

    @torch.inference_mode()
    def process_text(self, text: str, window_size: int = 120, step: int = 30) -> dict:
        if not text or len(text.strip()) < 10:
            return {
                "bino_min": 1.0,
                "bino_mean": 1.0,
                "bino_median": 1.0,
                "bino_q10": 1.0,
                "bino_var": 0.0,
                "most_ai_chunk": ""
            }

        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=False)
            input_ids = inputs["input_ids"].to(self.device)
            total_tokens = input_ids.shape[1]

            # Короткий текст — одно окно целиком
            if total_tokens <= window_size:
                score = round(float(self._score_tokens(input_ids)), 4)
                chunk_str = self.tokenizer.decode(input_ids[0], skip_special_tokens=True).strip()
                del input_ids
                return {
                    "bino_min": score,
                    "bino_mean": score,
                    "bino_median": score,
                    "bino_q10": score,
                    "bino_var": 0.0,
                    "most_ai_chunk": chunk_str
                }

            scores = []
            windows = []
            for start in range(0, total_tokens - window_size + 1, step):
                w_ids = input_ids[:, start:start + window_size]
                scores.append(self._score_tokens(w_ids))
                windows.append(w_ids)

            del input_ids
            scores_np = np.asarray(scores, dtype=np.float32)
            min_idx = int(np.argmin(scores_np))

            # Текст окна с минимальным скором (наиболее похожий на ИИ)
            best_chunk = self.tokenizer.decode(windows[min_idx][0], skip_special_tokens=True).strip()

            return {
                "bino_min": round(float(np.min(scores_np)), 4),
                "bino_mean": round(float(np.mean(scores_np)), 4),
                "bino_median": round(float(np.median(scores_np)), 4),
                "bino_q10": round(float(np.percentile(scores_np, 10)), 4),
                "bino_var": round(float(np.var(scores_np)), 6),
                "most_ai_chunk": best_chunk
            }

        except Exception as e:
            print(f"Ошибка токенизации: {e}")
            return {
                "bino_min": 1.0,
                "bino_mean": 1.0,
                "bino_median": 1.0,
                "bino_q10": 1.0,
                "bino_var": 0.0,
                "most_ai_chunk": ""
            }


# 3. Функция пакетной разметки
def run_dataset_enrichment(input_path: str, output_path: str, scorer: BinocularsScorer):
    if not os.path.exists(input_path):
        print(f"[Ошибка] Файл не найден: {input_path}")
        return

    df = pd.read_json(input_path, lines=True)
    print(f"\nНачало обработки {len(df)} строк из {input_path}...")

    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Scoring {os.path.basename(input_path)}"):
        item = row.to_dict()
        raw_text = str(item.get("text", ""))

        # Расчет перплексии и чанка
        bino_feats = scorer.process_text(raw_text)
        item.update(bino_feats)

        # Расчет длины предложений
        avg_len, len_var, short_ratio = extract_sentence_stats(raw_text)
        item["avg_sent_len"] = round(avg_len, 2)
        item["sent_len_var"] = round(len_var, 2)
        item["short_sent_ratio"] = round(short_ratio, 2)

        records.append(item)

    # Очистка кэша видеопамяти строго один раз в конце обработки файла
    if scorer.device.type == "cuda":
        torch.cuda.empty_cache()

    out_df = pd.DataFrame(records)
    out_df.to_json(output_path, orient="records", lines=True, force_ascii=False)
    print(f"✓ Сохранено в {output_path}")


# 4. Запуск скоринга
scorer = BinocularsScorer()

# Размечаем обучающий/калибровочный сплит
run_dataset_enrichment("text_train_calibration.jsonl", "text_train_features10v2.jsonl", scorer)

# Если уже загружен и тестовый сплит — можно раскомментировать:
run_dataset_enrichment("text_test_evaluation.jsonl", "text_test_features10v2.jsonl", scorer)