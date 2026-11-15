# Email AI Detector (DW-1)

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.11%2B-blue?style=flat-square&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/🤗%20Transformers-Qwen2.5--1.5B-yellow?style=flat-square" alt="Transformers" />
  <img src="https://img.shields.io/badge/scikit--learn-HistGB%20%2B%20Isotonic-F7931E?style=flat-square&logo=scikitlearn&logoColor=white" alt="scikit-learn" />
  <img src="https://img.shields.io/badge/CatBoost-GBDT-FFCC00?style=flat-square" alt="CatBoost" />
  <img src="https://img.shields.io/badge/FastAPI-OpenAI%20API-009688?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/Docker-Container-2496ED?style=flat-square&logo=docker&logoColor=white" alt="Docker" />
</p>

Прототип модуля выявления и объяснения признаков использования генеративного искусственного интеллекта в электронных письмах для антиспам-анализа и SOC компании «Доктор Веб». Принимает на вход письма в формате RFC 5322 (`.eml` или каталоги писем), выполняет очистку видимого текста от стелс-клоакинга, рассчитывает вероятностные сигналы токенов через локальную модель `Qwen2.5-1.5B-Instruct` и 9 стилометрических признаков, после чего выполняет позднее слияние (Late Fusion) с HTML-криминалистикой в двуязычном `CatBoost` и выдаёт структурированный JSON с локализацией сгенерированных фрагментов и русскоязычным объяснением.

## Содержание

- [Архитектура](#архитектура)
- [Состав системы](#состав-системы)
- [Быстрый старт](#быстрый-старт)
- [Тесты](#тесты)
- [API](#api)
- [Логирование](#логирование)

---

## Архитектура

Поток данных внутри конвейера от входного MIME-файла до формирования итогового вердикта по ТЗ:

```text
Входящее письмо (Файл .EML / Каталог / RFC 5322 MIME)
      │
      ▼
┌──────────────────────────────────────────────────────────┐
│  1. EML & MIME Parser (core/parser.py)                   │  CPU — декодирование кодировок (UTF-8, CP1251, KOI8-R),
│  • Анализ заголовков (X-Mailer mismatch, Reply-To)       │  расчёт расхождения R_PARTS_DIFFER (Jaccard desync),
│  • Изоляция вложений и фильтрация трекинг-пикселей       │  разделение text/plain и text/html частей
└────────────┬─────────────────────────────────────────────┘
             │  raw_html + clean_text + headers_meta
             ▼
┌──────────────────────────────────────────────────────────┐
│  2. HTMLDetector (core/html_detector.py)                 │  CPU (~4 мс) — DOM-форензика (46 признаков)
│  • Деобфускация CSS, WCAG 2.1 контрастность              │  • Выявление zero-font, offscreen, hidden-блоков
│  • Санитайзинг текста от клоакинга и Zero-Width (\u200b) │  • HistGradientBoosting + Isotonic + адаптивный порог
└────────────┬─────────────────────────────────────────────┘
             │  Санитизированный visible_text + HTML Risk + Semantic AST
             ▼
┌──────────────────────────────────────────────────────────┐
│  3. TextDetector (core/text_detector.py)                 │  API (GPU bfloat16) / CPU Fallback
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

**`core/parser.py` — `parse_eml`**
Модуль разбора RFC 5322 / MIME-структуры. Реализует каскадный перебор кодировок (`utf-8`, `windows-1251`, `koi8-r`, `iso-8859-5`, `cp866`), рассчитывает метрику расхождения между текстовой и HTML-частью письма (`calculate_jaccard_desync`) для защиты от атак *Multipart Perplexity Dilution*, детектирует расхождение между почтовым клиентом в `X-Mailer` и разметкой тела письма.

**`core/html_detector.py` — `HTMLDetector`**
Модуль структурной криминалистики HTML-разметки (46 признаков). Производит вычисление контрастности по WCAG 2.1, деобфускацию стилей CSS, выявление скрытых блоков (`display:none`, `opacity:0`, отрицательные отступы, `font-size: 0px`), символов нулевой ширины и маркеров копирования из интерфейсов генеративных моделей (`Söhne`, `--tw-`, утечки разметки Markdown). Строит `SemanticASTNode` с привязкой текстовых фрагментов.

**`core/text_detector.py` — `TextDetector`**
Модуль глубокого нейросетевого и статистического анализа видимого текста. Получает 15 аналитических метрик (`Log-Rank`, `LRR`, `Fast-DetectGPT Curvature`) от сервера `Qwen/Qwen2.5-1.5B-Instruct` по OpenAI-совместимому API (или локально), рассчитывает 9 стилометрических признаков на CPU и передаёт 24-мерный вектор в обученные модели `CatBoostClassifier` (`cb_ru.cbm` / `cb_eng.cbm`) с калиброванными порогами.

**`core/llm_api_server.py` — FastAPI LLM Server**
Сервер инференса языковой модели `Qwen/Qwen2.5-1.5B-Instruct`, разворачиваемый в контуре Разработчика на GPU (`bfloat16`). Предоставляет OpenAI-совместимый интерфейс `/v1/chat/completions` с проверкой Bearer-токена.

**`pipeline.py` — `SpamAiPipeline`**
Оркестратор мультимодального анализа и шлюз позднего слияния (Late Fusion). Агрегирует оценки текста и HTML-разметки, рассчитывает локальные вклады признаков (SHAP), формирует финальный вердикт (`AI-Generated` / `Human-Written`), вероятности по смысловым зонам с символьными границами интервала и русскоязычное объяснение для аналитика SOC.

### Особенности работы и режимы

* **Разделение контуров (Client / Server):** клиентский модуль (`pipeline.py`) работает на CPU без видеокарты и обращается к серверу языковой модели по HTTP API (`LLM_MODE=remote`). Поддерживается также автономный CPU-режим (`LLM_MODE=features`, только стилометрия и HTML-анализ) и полностью локальный GPU-режим (`LLM_MODE=local`).
* **Отказоустойчивость (Fallback):** при недоступности или таймауте API языковой модели клиент выполняет 3 повторные попытки с задержкой и автоматически переключается на облегчённые стилометрические модели (`cb_stylo_ru.cbm` / `cb_stylo_eng.cbm`) без прерывания пакетной обработки.
* **Двуязычная маршрутизация (`ru` / `eng`):** язык определяется автоматически (`detect_language`), подключая соответствующие веса `cb_ru.cbm` (`t_soft=0.558`, `t_strict=0.828`) или `cb_eng.cbm` (`t_soft=0.658`, `t_strict=0.807`).

---

## Состав системы

| Компонент | Директория / файл | Порты | Описание |
| --- | --- | --- | --- |
| **Pipeline CLI** | `pipeline.py` | — | Главная точка входа: анализ одного письма, каталогов и JSONL |
| **MIME Parser** | `core/parser.py` | — | Разбор структуры EML, кодировок и проверка заголовков |
| **HTML Detector** | `core/html_detector.py` | — | Экстрактор 46 признаков HTML-форензики и детектор клоакинга |
| **Text Detector** | `core/text_detector.py` | — | Клиент LLM API, расчёт стилометрии и инференс CatBoost |
| **LLM API Server** | `core/llm_api_server.py` | `8000` | OpenAI-совместимый FastAPI-сервер инференса Qwen2.5-1.5B |
| **Веса и пороги** | `weights/` | — | `cb_ru.cbm`, `cb_eng.cbm`, `cb_stylo_*.cbm`, `html_forensic_classifier.joblib`, `cb_thresholds.json` |

---

## Быстрый старт

### Требования

* **Интерпретатор:** Python 3.11 или выше
* **ОС:** Linux (Ubuntu 22.04+), Windows 10/11, macOS
* **Оборудование:**
* *Клиентский стенд (CPU):* 2 ядра CPU x86-64, 8 ГБ ОЗУ, 100 ГБ диска (без GPU)
* *Сервер языковой модели (контур Разработчика):* GPU NVIDIA с объёмом видеопамяти от 6 ГБ (RTX 3060/4060, Tesla T4) с поддержкой CUDA 12+



### Конфигурация

Создайте файл `.env` в корневом каталоге проекта на основе шаблона `.env.example`:

```bash
cp .env.example .env

```

Пример содержимого `.env`:

```env
LLM_MODE=remote
LLM_API_URL=[http://127.0.0.1:8000/v1](http://127.0.0.1:8000/v1)
LLM_API_KEY=dw1-local-contour-key
LLM_TIMEOUT=180
LLM_MAX_RETRIES=3
LLM_RETRY_DELAY=1.5
LOG_LEVEL=INFO

```

| Переменная | Обязательная | По умолчанию | Описание |
| --- | --- | --- | --- |
| `LLM_MODE` | нет | `local` (в `.env.example`: `remote`) | Режим работы: `remote` (через API), `features` (только CPU), `local` (локальный GPU) |
| `LLM_API_URL` | нет | `http://127.0.0.1:8000` | Адрес OpenAI-совместимого сервера языковой модели |
| `LLM_API_KEY` | нет | `""` | Токен авторизации Bearer (если задан на сервере) |
| `LLM_TIMEOUT` | нет | `180` | Таймаут ожидания ответа языковой модели в секундах |
| `LLM_MAX_RETRIES` | нет | `3` | Число повторных попыток при сбое сети/API |
| `LOG_LEVEL` | нет | `INFO` | Детализация вывода журналов обработки |

### Запуск

#### Вариант 1. Запуск сервера языковой модели (в контуре Разработчика с GPU)

```bash
python3.11 -m venv venv_gpu
source venv_gpu/bin/activate  # Для Windows PowerShell: .\venv_gpu\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements-gpu.txt

uvicorn core.llm_api_server:app --host 0.0.0.0 --port 8000

```

#### Вариант 2. Локальный запуск клиента через командную строку (CLI на CPU)

```bash
# 1. Клонирование репозитория и установка лёгких зависимостей
git clone <URL_РЕПОЗИТОРИЯ>
cd AI-spam-detector
python3.11 -m venv venv
source venv/bin/activate  # Для Windows PowerShell: .\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

# 2. Анализ одного письма (вывод в консоль и сохранение в JSON)
python pipeline.py --eml sample_test.eml --output result.json

# 3. Пакетный анализ каталога с письмами
python pipeline.py --dir /path/to/emails_dir --output results_out

```

#### Вариант 3. Запуск в Docker-контейнере (на стенде Заказчика)

Контейнер собирается с запуском процесса от непривилегированного пользователя `appuser` (требование безопасности п. 4.5.2.4 ТЗ):

```bash
# Сборка образа
docker build -t email-ai-detector:latest .

# Анализ одного письма в контейнере
docker run --rm --env-file .env \
  -v "$(pwd)/sample_test.eml:/app/sample_test.eml:ro" \
  -v "$(pwd)/results_out:/app/results_out" \
  email-ai-detector:latest --eml /app/sample_test.eml --output /app/results_out/result.json

# Пакетный анализ каталога писем в контейнере
docker run --rm --env-file .env \
  -v "$(pwd)/emails_dir:/app/input:ro" \
  -v "$(pwd)/results_out:/app/output" \
  email-ai-detector:latest --dir /app/input --output /app/output

```

*(Примечание: на Linux при обращении из контейнера к серверу на том же хосте добавьте флаг `--add-host=host.docker.internal:host-gateway` при указании `http://host.docker.internal:8000/v1` в `.env`).*

### Запуск отдельных компонентов

```bash
# Автономный режим без LLM (только CPU-признаки и HTML-форензика)
python pipeline.py --eml sample_test.eml --mode features --output result_cpu.json

```

```python
# Вызов из библиотеки Python
from pipeline import scan_email

result = scan_email("sample_test.eml")
print(result["verdict"], result["ai_score"], result["explanation"])

```

### Остановка

* Остановка сервера `uvicorn` или локального прогона CLI: нажмите `Ctrl + C` в терминале.
* Контейнеры Docker запускаются с флагом `--rm` и удаляются автоматически после завершения работы.

---

## Тесты

```bash
# Проверка извлечения признаков и инференса CatBoost
python -c "from core.text_detector import TextDetector; td = TextDetector(mode='features'); r = td.predict('Тестовый текст письма для проверки конвейера.'); assert 'score' in r; print('TextDetector: OK')"

# Проверка детекции HTML-клоакинга
python -c "from core.html_detector import HTMLDetector; hd = HTMLDetector(); r = hd.predict('<div style=\"display:none\">Hidden</div><p>Visible</p>'); assert r.is_cloaking_detected; print('HTMLDetector: OK')"

# Проверка расчёта расхождения multipart-частей в EML-парсере
python -c "from core.parser import calculate_jaccard_desync; assert calculate_jaccard_desync('text body', 'completely different html text') > 0.4; print('Parser Desync: OK')"

```

### Вспомогательные скрипты

| Скрипт | Назначение | Когда использовать |
| --- | --- | --- |
| `build_llmtrace_dataset.py` | Сборка сбалансированных сплитов Train/Val/Test/Mixed из LLMTrace | Подготовка обучающих данных без утечки `topic_id` |
| `extract_features_15b.py` | Одиночный прогон Qwen-1.5B и расчёт 24 признаков | Пересчёт признаков по новому корпусу текстов |
| `train_eval_15b_cb.py` | Обучение CatBoost, калибровка порогов на Validation и расчёт метрик | Обучение моделей и формирование отчёта качества по ТЗ |

```bash
# Воспроизводимый расчёт метрик на отложенных тестовых выборках (Test и Mixed)
python train_eval_15b_cb.py

```

---

## API

Интерактивная документация Swagger UI доступна после запуска сервера по адресу `http://127.0.0.1:8000/docs`.

### Инференс языковой модели (`/v1`)

Сервер предоставляет OpenAI-совместимый интерфейс для расчёта вероятностных признаков текста моделью `Qwen/Qwen2.5-1.5B-Instruct`.

| Метод | Путь | Описание |
| --- | --- | --- |
| `GET` | `/v1/models` | Получение списка доступных моделей на сервере |
| `POST` | `/v1/chat/completions` | OpenAI-совместимый расчёт 15 признаков LLM и локализация фрагмента |
| `POST` | `/v1/score_text` | Прямой расчёт признаков текста с замером времени инференса |

**Тело запроса `POST /v1/chat/completions`:**

```json
{
  "model": "Qwen/Qwen2.5-1.5B-Instruct",
  "messages": [
    {
      "role": "user",
      "content": "Текст видимой части электронного письма для анализа..."
    }
  ],
  "temperature": 0.0
}

```

| Поле | Тип | Обязательное | Описание |
| --- | --- | --- | --- |
| `model` | `string` | нет | Идентификатор модели (`Qwen/Qwen2.5-1.5B-Instruct`) |
| `messages` | `array` | да | Список сообщений диалога; анализируется содержимое с ролью `user` |
| `temperature` | `float` | нет | Параметр детерминированности (по умолчанию `0.0`) |

**Коды ответа HTTP:**

| Код | Статус | Описание |
| --- | --- | --- |
| `200` | `OK` | Признаки и границы фрагмента успешно рассчитаны |
| `401` | `Unauthorized` | Отсутствует заголовок `Authorization: Bearer <token>` (если задан `LLM_API_KEY`) |
| `403` | `Forbidden` | Передан неверный ключ доступа API |

---

## Логирование

Формат: `[%(asctime)s] [%(levelname)s] %(message)s`. Пример:

```text
[2026-11-15 08:09:05,629] [INFO] ИНИЦИАЛИЗАЦИЯ СИСТЕМЫ ДЕТЕКЦИИ DR.WEB (DW-1: MULTIMODAL LATE FUSION)
[2026-11-15 08:09:07,130] [INFO] ✓ Конвейер (Qwen2.5-1.5B API + CatBoost + HTML Forensics) готов.
[2026-11-15 08:09:08,053] [INFO] [OK] 2_ai_text_only.eml         | Verdict: AI-Generated   | AI Score: 0.834
[2026-11-15 08:09:09,207] [INFO] ✓ ИТОГО ПО ДИРЕКТОРИИ: обработано 6/6 писем за 2.08 сек (среднее: 0.338 с/письмо, медиана: 0.255 с/письмо)

```

* **Конфигурация логгера:** настраивается в модуле `pipeline.py` (логгер `DrWebSpamPipeline`), уровень детализации задаётся через переменную `LOG_LEVEL` в `.env`.
* **Обработчики и хранение:** вывод осуществляется в стандартные потоки `stdout`/`stderr` (`logging.StreamHandler`); в Docker-окружении журналы перехватываются драйвером логирования Docker (`json-file`).
* **Уровни:** `INFO` (ход пакетной обработки, вердикты, итоговая статистика), `WARNING` (пропуск повреждённых файлов без остановки конвейера, сбои API с переходом на резервный режим), `ERROR` (критические ошибки запуска).
