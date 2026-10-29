FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_ENV=production \
    MODEL_DIR=/models

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-api.txt

COPY src/ /app/src/
COPY monitoring/ /app/monitoring/
COPY api/ /app/api/
COPY models/ /models/
COPY static/production.html /app/static/production.html
COPY reports/sample_sequence.json /app/static/sample_sequence.json

RUN useradd --uid 10001 --create-home appuser \
    && mkdir -p /models \
    && chown appuser:appuser /app

USER 10001:10001

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import os; from urllib.request import urlopen; urlopen('http://127.0.0.1:' + os.getenv('PORT', '7860') + '/health', timeout=3)"

CMD ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
