# Email AI Detector (DW-1)

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.11%2B-blue?style=flat-square&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/🤗%20Transformers-Qwen2.5--1.5B-yellow?style=flat-square" alt="Transformers" />
  <img src="https://img.shields.io/badge/scikit--learn-HistGB%20%2B%20Isotonic-F7931E?style=flat-square&logo=scikitlearn&logoColor=white" alt="scikit-learn" />
  <img src="https://img.shields.io/badge/CatBoost-GBDT-FFCC00?style=flat-square" alt="CatBoost" />
  <img src="https://img.shields.io/badge/PaddleOCR-PP--OCRv4-00B050?style=flat-square" alt="PaddleOCR" />
  <img src="https://img.shields.io/badge/Kaggle%20%2F%20Colab-2x%20T4%20GPU-20BEFF?style=flat-square&logo=kaggle&logoColor=white" alt="Kaggle" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License" />
</p>

Прототип модуля выявления и объяснения признаков использования генеративного искусственного интеллекта в электронных письмах для антиспам-анализа и SOC компании «Доктор Веб». Принимает на вход письма в формате RFC 5322 (`.eml` или каталоги писем), выполняет очистку видимого текста от стелс-клоакинга, рассчитывает вероятностные сигналы токенов через локальную модель `Qwen2.5-1.5B-Instruct` и 9 стилометрических признаков, после чего выполняет позднее слияние (Late Fusion) с HTML-криминалистикой в двуязычном `CatBoost` и выдаёт структурированный JSON с локализацией сгенерированных фрагментов и русскоязычным объяснением.

## Содержание

* [Архитектура](#архитектура)
* [Состав системы](#состав-системы)
* [Быстрый старт](#быстрый-старт)
* [Тесты](#тесты)
* [Логирование](#логирование)

---

## Архитектура

Поток данных внутри конвейера от входного MIME-файла до формирования итогового вердикта по ТЗ:

```text
Входящее письмо (Файл .EML / Каталог / RFC 5322 MIME)
      │
      ▼
┌──────────────────────────────────────────────────────────┐
│  1. EML & MIME Parser (parser.py)                        │  CPU — декодирование кодировок (UTF-8, CP1251, KOI8-R),
│  • Анализ заголовков (X-Mailer mismatch, Reply-To)       │  расчёт расхождения R_PARTS_DIFFER (Jaccard desync),
│  • Изоляция и фильтрация вложений                        │  разделение text/plain и text/html частей
└────────────┬─────────────────────────────────────────────┘
             │  raw_html + clean_text + headers_meta
             ▼
┌──────────────────────────────────────────────────────────┐
│  2. HTMLDetector (html_detector.py)                      │  CPU (~4 мс) — DOM-форензика (46 признаков)
│  • Деобфускация CSS, WCAG 2.1 контрастность              │  • Выявление zero-font, offscreen, hidden-блоков
│  • Санитайзинг текста от клоакинга и Zero-Width (\u200b) │  • HistGradientBoosting + Isotonic + адаптивный порог
└────────────┬─────────────────────────────────────────────┘
             │  Санитизированный visible_text + HTML Risk + Semantic AST
             ▼
┌──────────────────────────────────────────────────────────┐
│  3. TextDetector (text_detector.py)                      │  GPU (bfloat16) / CPU — Single Forward Pass
│  • Qwen2.5-1.5B-Instruct: Log-Rank, LRR, Fast-DetectGPT  │  • Окна по 3 предложениям + Contiguous Merging
│  • 9 стилометрических признаков (Root-TTR, CV предл.)    │  • Двуязычный CatBoost (cb_ru.cbm / cb_eng.cbm)
└────────────┬─────────────────────────────────────────────┘
             │  Text AI Probability + most_ai_chunk [start_char, end_char]
             ▼
┌──────────────────────────────────────────────────────────┐
│  4. SpamAiPipeline (pipeline.py)                         │  Late Fusion & Explainable AI (XAI)
│  • Взвешенное нелинейное объединение оценок              │  • Локальный расчёт SHAP-вкладов ключевых сигналов
│  • Формирование вердикта: AI-Generated / Human-Written   │  • Генерация текстового объяснения на русском языке
└──────────────────────────────────────────────────────────┘
```

### Ключевые компоненты

**`parser.py` — `parse_eml`**

Модуль разбора RFC 5322 / MIME-структуры. Реализует каскадный перебор кодировок (`utf-8`, `windows-1251`, `koi8-r`, `iso-8859-5`, `cp866`), рассчитывает метрику расхождения между текстовой и HTML-частью письма (`calculate_jaccard_desync`) для защиты от атак *Multipart Perplexity Dilution*, детектирует расхождение между почтовым клиентом в `X-Mailer` и разметкой тела письма.

**`html_detector.py` — `HTMLDetector`**

Модуль структурной криминалистики HTML-разметки (46 признаков). Производит вычисление контрастности по WCAG 2.1, деобфускацию стилей CSS, выявление скрытых блоков (`display:none`, `opacity:0`, отрицательные отступы, `font-size: 0px`), символов нулевой ширины и маркеров копирования из интерфейсов генеративных моделей (`Söhne`, `--tw-`, утечки разметки Markdown). Строит `SemanticASTNode` с привязкой текстовых фрагментов.

**`text_detector.py` — `TextDetector`**

Модуль глубокого нейросетевого и статистического анализа видимого текста. Выполняет один проход (`Single Forward Pass`) через локальную модель `Qwen/Qwen2.5-1.5B-Instruct` в `bfloat16`, вычисляя аналитические метрики `Log-Rank`, `LRR` и `Fast-DetectGPT Curvature` глобально и по скользящим окнам из 3 предложений. Рассчитывает 9 стилометрических признаков и передаёт 24-мерный вектор в обученные модели `CatBoostClassifier` (`cb_ru.cbm` / `cb_eng.cbm`) с калиброванными порогами.

**`pipeline.py` — `SpamAiPipeline`**

Оркестратор мультимодального анализа и шлюз позднего слияния (Late Fusion). Агрегирует оценки текста и HTML-разметки, рассчитывает локальные вклады признаков (SHAP), формирует финальный вердикт (`AI-Generated` / `Human-Written`), вероятности по смысловым зонам с символьными границами интервала и русскоязычное объяснение для аналитика SOC.

### Особенности работы и режимы

* **Автоматический выбор устройства (GPU / CPU):** При старте проверяется наличие CUDA. На GPU инференс `Qwen2.5-1.5B-Instruct` выполняется в `torch.bfloat16` (~35–45 мс на письмо). При отсутствии GPU модуль автоматически выполняет переключение на CPU (`torch.float32`).
* **Двуязычная маршрутизация (`ru` / `eng`):** Язык определяется автоматически по морфологическому составу алфавитов (`detect_language`), подключая соответствующие веса `cb_ru.cbm` (пороги `t_soft=0.558`, `t_strict=0.828`) или `cb_eng.cbm` (пороги `t_soft=0.658`, `t_strict=0.807`).
* **Фолбэк при отсутствии HTML:** Если письмо содержит только `text/plain`, вес HTML-компонента в формуле Late Fusion обнуляется, и итоговый индекс рассчитывается исключительно на базе текстового ансамбля.

---

## Состав системы

| Компонент | Расположение | Описание |
| --- | --- | --- |
| **Pipeline CLI** | `./pipeline.py` | Главная точка входа: CLI-команды анализа одного письма, каталогов и JSONL |
| **MIME Parser** | `./parser.py` (или `./core/parser.py`) | Разбор структуры EML, вложений и проверка целостности заголовков |
| **HTML Detector** | `./html_detector.py` | Экстрактор 46 признаков HTML-форензики, детектор клоакинга и построитель AST |
| **HTML Model** | `./html_forensic_classifier.joblib` | Обученный калиброванный классификатор HTML-разметки |
| **Text Detector** | `./text_detector.py` | Экстрактор 15 признаков Qwen-1.5B и 9 признаков стилометрии |
| **CatBoost Models** | `./cb_ru.cbm`, `./cb_eng.cbm` | Обученные модели градиентного бустинга для русского и английского языков |
| **Thresholds Config** | `./cb_thresholds.json` | Файл калибровочных порогов принятия решений (`t_soft`, `t_strict`) |

---

## Быстрый старт

### Требования

* **Интерпретатор:** Python 3.11 или выше
* **ОС:** Linux (Ubuntu 22.04+), Windows 10/11, macOS
* **Оборудование:**
  * *Минимальное (только признаки и бустинг на CPU):* 2 ядра CPU x86-64, 8 ГБ ОЗУ
  * *Рекомендуемое (полный пайплайн с Qwen-1.5B):* GPU NVIDIA с объёмом видеопамяти от 6 ГБ (RTX 3060/4060, Tesla T4) с поддержкой CUDA 12+

### Конфигурация

Создайте файл `.env` в корневом каталоге проекта на основе шаблона `.env.example`:

```bash
cp .env.example .env
```

Пример содержимого `.env`:

```env
# URL сервера инференса языковой модели (контур Разработчика)
LLM_API_URL=http://localhost:8000
# Таймаут ожидания сетевых запросов (в секундах)
LLM_TIMEOUT=180
# Уровень журналирования (DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL=INFO
```

| Переменная | Обязательная | По умолчанию | Описание |
| --- | --- | --- | --- |
| `LLM_API_URL` | нет | `http://localhost:8000` | Адрес сервиса модели в контуре разработчика |
| `LLM_TIMEOUT` | нет | `180` | Таймаут ответа языковой модели по ТЗ |
| `LOG_LEVEL` | нет | `INFO` | Детализация вывода журналов обработки |

### Запуск

#### Вариант 1. Локальный запуск в виртуальном окружении

```bash
# 1. Клонирование репозитория
git clone <URL_РЕПОЗИТОРИЯ>
cd AI-spam-detector

# 2. Создание и активация виртуального окружения Python 3.11
python3.11 -m venv venv
source venv/bin/activate  # Для Linux/macOS
# .\venv\Scripts\activate # Для Windows PowerShell

# 3. Установка зависимостей
pip install --upgrade pip
pip install -r requirements.txt

# 4. Анализ одного письма
python pipeline.py --eml sample_test.eml --output result.json

# 5. Пакетный анализ каталога с письмами
python pipeline.py --dir /path/to/emails_dir --output results_dir
```

#### Вариант 2. Запуск в Docker-контейнере

Контейнер собирается с запуском процесса от непривилегированного пользователя (`appuser`) в соответствии с требованиями безопасности ТЗ (проверка Б-2):

```bash
# Сборка образа
docker build -t email-ai-detector:latest .

# Запуск анализа одного письма в контейнере
docker run --rm \
  -v $(pwd)/sample_test.eml:/app/sample_test.eml:ro \
  -v $(pwd)/results:/app/results \
  email-ai-detector:latest --eml /app/sample_test.eml --output /app/results/result.json

# Запуск пакетного анализа каталога
docker run --rm \
  -v /path/to/emails:/app/input_emails:ro \
  -v $(pwd)/results:/app/results \
  email-ai-detector:latest --dir /app/input_emails --output /app/results
```

### Запуск отдельных компонентов

```python
# Запуск только текстового детектора
from text_detector import TextDetector
td = TextDetector()
res = td.predict("Текст подозрительного фрагмента для анализа...")
print(f"Статус: {res['status']}, Вероятность: {res['score']}")

# Запуск только HTML-детектора
from html_detector import HTMLDetector
hd = HTMLDetector()
html_res = hd.predict("<html><body><p>Контент письма</p></body></html>")
print(f"Вердикт: {html_res.verdict}, Риск: {html_res.global_risk_score}")
```

---

## Тесты

Запуск базового набора верификационных тестов компонентов:

```bash
# Проверка извлечения признаков и инференса CatBoost
python -c "from text_detector import TextDetector; td = TextDetector(); r = td.predict('Тестовый текст письма для проверки конвейера.'); assert 'score' in r; print('TextDetector: OK')"

# Проверка детекции HTML-клоакинга
python -c "from html_detector import HTMLDetector; hd = HTMLDetector(); r = hd.predict('<div style=\"display:none\">Hidden</div><p>Visible</p>'); assert r.is_cloaking_detected; print('HTMLDetector: OK')"

# Проверка расчёта расхождения multipart-частей в EML-парсере
python -c "from parser import calculate_jaccard_desync; assert calculate_jaccard_desync('text body', 'completely different html text') > 0.4; print('Parser Desync: OK')"
```

### Вспомогательные скрипты

| Скрипт | Назначение | Когда использовать |
| --- | --- | --- |
| `build_llmtrace_dataset.py` | Сборка сбалансированных сплитов Train/Val/Test из LLMTrace | Подготовка обучающих данных без утечки `topic_id` |
| `extract_features_15b.py` | Одиночный прогон Qwen-1.5B и расчёт 24 признаков | Пересчёт признаков по новому корпусу текстов |
| `train_eval_15b_cb.py` | Обучение CatBoost и калибровка порогов на Validation | Обучение моделей и расчёт метрик по ТЗ |
| `make_presentation_plots.py` | Построение ROC-кривых с 95% ДИ (1000 итераций бутстрепа) | Формирование графических отчётов качества |

---

## Логирование

Логирование осуществляется через стандартный модуль Python `logging` с выводом структурированных записей в поток `stdout`/`stderr` (согласно п. 4.1.8.2 ТЗ и п. 6.3 ПМИ).

Пример строки журнала:

```text
[2026-10-22 01:15:32,104] [INFO] ИНИЦИАЛИЗАЦИЯ СИСТЕМЫ ДЕТЕКЦИИ DR.WEB (DW-1: MULTIMODAL LATE FUSION)
[2026-10-22 01:15:41,250] [INFO] ✓ Конвейер (Qwen2.5-1.5B + CatBoost + HTML Forensics) готов.
[2026-10-22 01:15:41,310] [INFO] [OK] message_01.eml  | Verdict: AI-Generated  | AI Score: 0.842
[2026-10-22 01:15:41,365] [WARNING] [SKIP] Поврежденное письмо broken.eml: Отсутствуют заголовки RFC 5322
```

* **Уровни:** `INFO` (ход обработки, статусы писем, итоговая сводка), `WARNING` (пропуск повреждённых файлов без падения конвейера согласно требованию Н-1 ПМИ), `ERROR` (критические ошибки инфраструктуры).
* **Ротация и хранение:** При работе в Docker-контейнере поток логов перехватывается стандартным драйвером Docker (`json-file`) с поддержкой внешнего сбора.
