FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Непривилегированный пользователь (требование п. 4.5.2.4 ТЗ и проверки Б-2 / А-16 ПМИ)
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY core/ /app/core/
COPY weights/ /app/weights/
COPY pipeline.py .env.example /app/

RUN chown -R appuser:appuser /app

USER appuser

ENTRYPOINT ["python", "pipeline.py"]