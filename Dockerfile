FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Создаём непривилегированного пользователя (требование п. 4.5.2.4 ТЗ и проверки Б-2 / А-16 ПМИ)
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY parser.py html_detector.py text_detector.py pipeline.py /app/
COPY cb_ru.cbm cb_eng.cbm cb_thresholds.json html_forensic_classifier.joblib .env.example /app/

RUN chown -R appuser:appuser /app

USER appuser

ENTRYPOINT ["python", "pipeline.py"]