# Imagen única para la API (FastAPI) y la UI (Streamlit). Construir desde la raíz del repo:
#   docker compose -f app/docker-compose.yml up --build
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
# libgomp1: requerido por LightGBM (OpenMP).
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /srv
COPY app/requirements-app.txt app/requirements-app.txt
RUN pip install --no-cache-dir -r app/requirements-app.txt

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps .

COPY models ./models
COPY app ./app

ENV MODEL_PATH=/srv/models/model_final.joblib METADATA_PATH=/srv/models/metadata.json
EXPOSE 8000 8501
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
