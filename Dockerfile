# Serving image: CPU only, no PyTorch.
FROM python:3.11-slim
WORKDIR /app
ENV BRAILLELM_HOME=/app PYTHONUTF8=1 PYTHONUNBUFFERED=1
COPY pyproject.toml README.md ./
COPY src ./src
COPY models/braillelm_v3/braillelm_v3.onnx models/braillelm_v3/model_card.json ./models/braillelm_v3/
COPY reports/summary.json ./reports/summary.json
RUN pip install --no-cache-dir ".[serve]"
EXPOSE 7861
HEALTHCHECK CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:7861/health')"
CMD ["uvicorn", "braillelm.serving.api:app", "--host", "0.0.0.0", "--port", "7861"]
