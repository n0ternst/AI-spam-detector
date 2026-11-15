import json
import os
import re
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

MAX_TOKENS_CAP = 1024
MIN_WINDOW_TOKENS = 48   # Защита от шума: окно не может быть короче ~48 токенов
FALLBACK_WIN = 96        # Если в тексте нет точек/абзацев
FALLBACK_STRIDE = 32


def extract_clean_stylometry(text: str) -> dict:
    zero_stylo = {
        "avg_sent_len": 0.0, "sent_len_var": 0.0, "short_sent_ratio": 0.0,
        "root_ttr": 0.0, "word_len_var": 0.0, "comma_rate": 0.0,
        "dash_rate": 0.0, "excl_quest_rate": 0.0, "colon_semi_rate": 0.0
    }
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
    """
    Строит окна по 3 предложениям со сдвигом в 1 предложение.
    Если 3 предложения короче MIN_WINDOW_TOKENS (48 токенов), захватывает следующие,
    чтобы детектор никогда не шумел на коротких обрывках.
    """
    if n_metric_tokens <= MIN_WINDOW_TOKENS:
        return [(0, n_metric_tokens)]

    # Находим символьные границы предложений в тексте
    sent_char_spans = [
        (m.start(), m.end())
        for m in re.finditer(r"[^.!?\n]+[.!?\n]*", text)
        if len(m.group(0).strip()) > 2
    ]

    # Переводим символьные границы предложений в индексы токен-массива (длины n_metric_tokens)
    # Токен i в массиве метрик соответствует токену i+1 в исходном тексте
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

    # Если текст без пунктуации (одно гигантское предложение) — фолбэк на 96/32
    if len(sent_tok_bounds) < 3:
        bounds = [
            (s, s + FALLBACK_WIN)
            for s in range(0, max(1, n_metric_tokens - FALLBACK_WIN + 1), FALLBACK_STRIDE)
        ]
        if bounds[-1][1] < n_metric_tokens:
            bounds.append((max(0, n_metric_tokens - FALLBACK_WIN), n_metric_tokens))
        return bounds

    # Собираем окна: минимум 3 предложения И минимум MIN_WINDOW_TOKENS (48 токенов)
    windows = []
    num_sents = len(sent_tok_bounds)
    for i in range(num_sents):
        s_tok = sent_tok_bounds[i][0]
        j = min(i + 2, num_sents - 1)
        while j < num_sents - 1 and (sent_tok_bounds[j][1] - s_tok) < MIN_WINDOW_TOKENS:
            j += 1
        e_tok = sent_tok_bounds[j][1]

        # Если в самом конце текста остался короткий хвост, расширяем окно влево
        if (e_tok - s_tok) < MIN_WINDOW_TOKENS:
            s_tok = max(0, e_tok - MIN_WINDOW_TOKENS)

        if not windows or windows[-1] != (s_tok, e_tok):
            windows.append((s_tok, e_tok))
        if e_tok >= n_metric_tokens:
            break

    return windows


class Qwen15BFeatureExtractor:
    def __init__(self, model_name: str = MODEL_NAME):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        print(f"Загрузка {model_name} на {self.device} ({self.dtype})...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=self.dtype
        ).to(self.device).eval()

    @torch.no_grad()
    def extract_llm_features(self, text: str) -> dict:
        enc = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=MAX_TOKENS_CAP,
            return_offsets_mapping=True,
        )
        input_ids = enc["input_ids"]
        offsets = enc["offset_mapping"]

        zero_res = {
            "lr_global": 0.0, "lrr_global": 0.0, "fd_global": 0.0,
            "lr_win_max": 0.0, "lr_win_mean": 0.0, "lr_win_std": 0.0, "lr_win_q90": 0.0,
            "lrr_win_max": 0.0, "lrr_win_mean": 0.0, "lrr_win_std": 0.0,
            "fd_win_max": 0.0, "fd_win_mean": 0.0, "fd_win_std": 0.0,
            "merged_span_lr": 0.0, "merged_span_ratio": 0.0,
            "pred_span_chars": [0, 0]
        }
        if len(input_ids) < 8:
            return zero_res

        # 1. Один полный прогон текста через Qwen-1.5B
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

        # 2. Глобальные метрики всего письма
        lr_global = -float(np.mean(lr_np))
        lrr_global = -float(np.sum(tlp_np) / (np.sum(lr_np) + 1e-8))
        fd_global = float(np.sum(tlp_np - exp_np) / np.sqrt(np.sum(var_np) + 1e-8))

        # 3. Умные окна по 3 предложениям (с защитой >= 48 токенов)
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

        lr_w = np.array(lr_w)
        lrr_w = np.array(lrr_w)
        fd_w = np.array(fd_w)

        # 4. Адаптивное расширение спана (если вставка длинная, напр. 120-200 токенов)
        best_idx = int(np.argmax(lr_w))
        peak_lr = lr_w[best_idx]
        mean_lr = float(np.mean(lr_w))
        std_lr = float(np.std(lr_w)) if len(lr_w) > 1 else 0.0

        # Порог присоединения соседних окон: выше среднего + 0.5 ст. отклонения (и близко к пику)
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

        return {
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
            "pred_span_chars": [char_start, char_end],
        }


def process_split(extractor: Qwen15BFeatureExtractor, in_file: str, out_file: str):
    if not os.path.exists(in_file):
        print(f"Файл {in_file} не найден, пропуск.")
        return

    df = pd.read_json(in_file, lines=True)
    done_count = 0
    if os.path.exists(out_file):
        with open(out_file, "r", encoding="utf-8") as f:
            done_count = sum(1 for _ in f)
        print(f"В {out_file} уже есть {done_count}/{len(df)} строк. Докачиваем...")

    with open(out_file, "a", encoding="utf-8") as f_out:
        for idx in tqdm(range(done_count, len(df)), desc=os.path.basename(out_file)):
            row = df.iloc[idx]
            text = str(row["text"])

            llm_feats = extractor.extract_llm_features(text)
            stylo_feats = extract_clean_stylometry(text)

            record = {
                "text": text,
                "label": row["label"],
                "lang": row["lang"],
                "data_type": row.get("data_type", ""),
                "model": row.get("model", ""),
                **llm_feats,
                **stylo_feats,
            }
            if "ai_char_intervals" in row and row["ai_char_intervals"] is not None:
                record["ai_char_intervals"] = row["ai_char_intervals"]

            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            if idx % 50 == 0:
                f_out.flush()


if __name__ == "__main__":
    extractor = Qwen15BFeatureExtractor()
    for split in ["train", "val", "test", "mixed_test"]: 
        in_p = os.path.join(BASE_DIR, f"raw_{split}.jsonl")
        out_p = os.path.join(BASE_DIR, f"features_15b_{split}.jsonl")
        process_split(extractor, in_p, out_p)
    print("\n✓ Все датасеты успешно обработаны!")