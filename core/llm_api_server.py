"""
OpenAI-совместимый сервер инференса Qwen2.5-1.5B-Instruct (контур Разработчика).
Запускается на GPU-машине (RTX 4060 8GB):
    uvicorn llm_api_server:app --host 0.0.0.0 --port 8000
"""

import os
import re
import time
from typing import Any
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
API_KEY = os.getenv("LLM_API_KEY", "")  # Если задан в .env, проверяет Bearer токен

MAX_TOKENS_CAP = 1024
MIN_WINDOW_TOKENS = 48
FALLBACK_WIN = 96
FALLBACK_STRIDE = 32

LLM_FEATURE_KEYS = [
    "lr_global", "lrr_global", "fd_global",
    "lr_win_max", "lr_win_mean", "lr_win_std", "lr_win_q90",
    "lrr_win_max", "lrr_win_mean", "lrr_win_std",
    "fd_win_max", "fd_win_mean", "fd_win_std",
    "merged_span_lr", "merged_span_ratio",
]

app = FastAPI(title="Dr.Web DW-1 OpenAI-Compatible LLM Server", version="1.0.0")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

print(f"[LLM-Server] Загрузка {MODEL_NAME} на {device} ({dtype})...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype).to(device).eval()
print("[LLM-Server] Модель готова к приёму запросов.")


def _verify_auth(authorization: str | None) -> None:
    if API_KEY:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing Bearer token")
        token = authorization.split(" ", 1)[1].strip()
        if token != API_KEY:
            raise HTTPException(status_code=403, detail="Invalid API key")


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


@torch.no_grad()
def compute_qwen_forensics(text: str) -> tuple[dict[str, float], str, list[int]]:
    enc = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_TOKENS_CAP,
        return_offsets_mapping=True,
    )
    input_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]

    if len(input_ids) < 8:
        zero_llm = {k: 0.0 for k in LLM_FEATURE_KEYS}
        return zero_llm, text, [0, len(text)]

    ids_tensor = torch.tensor([input_ids], device=device)
    logits = model(ids_tensor).logits[0, :-1, :].float()
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

    lr_w = np.array(lr_w)
    lrr_w = np.array(lrr_w)
    fd_w = np.array(fd_w)

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

    span_tok_s = win_bounds[left_idx][0]
    span_tok_e = win_bounds[right_idx][1]

    merged_span_lr = -float(np.mean(lr_np[span_tok_s:span_tok_e]))
    merged_span_ratio = float((span_tok_e - span_tok_s) / max(n_tok, 1))

    char_start = int(offsets[span_tok_s + 1][0])
    char_end = int(offsets[min(span_tok_e, len(offsets) - 1)][1])
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


class ScoreRequest(BaseModel):
    text: str
    model: str = MODEL_NAME


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = MODEL_NAME
    messages: list[ChatMessage]
    temperature: float = 0.0


@app.get("/v1/models")
def list_models(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    _verify_auth(authorization)
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "developer-contour-nsu",
            }
        ],
    }


@app.post("/v1/score_text")
def score_text_endpoint(
    req: ScoreRequest, authorization: str | None = Header(default=None)
) -> dict[str, Any]:
    _verify_auth(authorization)
    t0 = time.perf_counter()
    feats, chunk, span = compute_qwen_forensics(req.text)
    return {
        "model": MODEL_NAME,
        "features": feats,
        "most_ai_chunk": chunk,
        "pred_span_chars": span,
        "inference_time_sec": round(time.perf_counter() - t0, 4),
    }


@app.post("/v1/chat/completions")
def chat_completions_endpoint(
    req: ChatCompletionRequest, authorization: str | None = Header(default=None)
) -> dict[str, Any]:
    """OpenAI-совместимый эндпоинт (п. 4.4.2, 4.5.1 ТЗ и Б-3 ПМИ)."""
    _verify_auth(authorization)
    user_text = "\n".join(m.content for m in req.messages if m.role == "user").strip()
    feats, chunk, span = compute_qwen_forensics(user_text)
    payload = {
        "features": feats,
        "most_ai_chunk": chunk,
        "pred_span_chars": span,
    }
    import json
    return {
        "id": f"chatcmpl-dw1-{int(time.time()*1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": json.dumps(payload, ensure_ascii=False),
                },
                "finish_reason": "stop",
            }
        ],
        "dw1_forensics": payload,
    }