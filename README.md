# AI-Assisted Spam & Phishing Detection System

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/🤗%20Transformers-HuggingFace-yellow?style=flat-square" alt="Transformers" />
  <img src="https://img.shields.io/badge/CatBoost-GBDT-FFCC00?style=flat-square" alt="CatBoost" />
  <img src="https://img.shields.io/badge/PaddleOCR-PP--OCRv4-00B050?style=flat-square" alt="PaddleOCR" />
  <img src="https://img.shields.io/badge/Google%20Colab-T4%20GPU-F9AB00?style=flat-square&logo=googlecolab&logoColor=white" alt="Colab" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License" />
</p>

A multi-signal analytical pipeline for detecting AI-generated and hybrid email spam campaigns with an ultra-low False Positive Rate (FPR $\le 0.01\% - 0.05\%$).

---

## Overview

Modern phishing attacks often use large language models (LLMs) for **partial text modification** (rewriting calls-to-action, subject lines, or instructions) and synthetic image generation. This system combines linguistic probability metrics, syntactic analysis, optical character recognition (OCR), and computer vision to identify AI intervention across text and attachments.

---

## Core Components

```text
Raw Email (.eml)
 ├── Text Body    ──> Sliding-Window Binoculars (Qwen 2.5) + Syntax Analysis ──┐
 └── Attachments  ──> PaddleOCR + SAFE (Synthetic Image Detection)  ────────────┴──> CatBoost GBDT ──> Verdict & XAI Report

  - Text Analysis (Binoculars): Token-level sliding-window scoring using an
    aligned model pair (Qwen/Qwen2.5-1.5B & Qwen/Qwen2.5-1.5B-Instruct) to
    detect partially rewritten paragraphs.
  - Syntactic Regularization: Sentence length variance and short-sentence ratios
    to prevent false alarms on rigid human-written templates.
  - Visual OCR (PaddleOCR): Multilingual (RU/EN) text extraction from banners
    and image attachments.
  - Synthetic Image Detection (SAFE KDD 2025): High-resolution patch sampling to
    detect AI-generated documents and stamps under heavy JPEG compression.
  - Decision Core (CatBoost): Gradient boosted decision trees aggregating
    heterogeneous signals into a single calibrated verdict.
  - Explainable AI (XAI): Localized suspicious text segments with
    sigmoid-calibrated confidence scores for SOC analysts.

Quick Start

1. Installation

pip install catboost paddleocr paddlepaddle sentencepiece protobuf transformers datasets

2. Inference Example

from src.pipeline import analyze_email

email_text = "Dear user, we detected an unauthorized login. Please check the attached invoice."
attachment = "attachments/invoice.png"

# Run multimodal inference
report = analyze_email(raw_text=email_text, image_path=attachment)
print(report)

Example Output

{
  "verdict": "AI Generated",
  "ai_probability": "94.80%",
  "features": {
    "bino_min": 0.8124,
    "bino_var": 0.0084,
    "avg_sent_len": 6.80
  },
  "visual_attachment": {
    "has_image_text": 1,
    "img_safe_score": 0.8840,
    "is_image_ai_generated": true
  },
  "suspicious_segments_count": 1,
  "segments": [
    {
      "start": 10,
      "end": 35,
      "confidence": 96.45,
      "text": "Please verify your credentials within 24 hours to retain remote access."
    }
  ]
}

Tech Stack

  - Language Models: Qwen 2.5 (1.5B / 1.5B-Instruct)
  - Computer Vision & OCR: PaddleOCR (PP-OCRv4), ResNet-50 / ViT (SAFE
    framework)
  - Tabular Classification: CatBoost
  - Frameworks: PyTorch, Hugging Face Transformers, Pandas, PyArrow

