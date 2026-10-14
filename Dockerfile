# Slim CPU image for the inference API and the dashboard. Training happens on the host (GPU) - this image only serves.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# CPU-only PyTorch keeps the image ~1 GB smaller than the default CUDA wheel
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps .
COPY configs ./configs
COPY .streamlit ./.streamlit

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000 8501
# default: the API. docker-compose overrides the command for the dashboard.
CMD ["uvicorn", "nids.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
