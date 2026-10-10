# syntax=docker/dockerfile:1
#
# Images for serving the committed model. Default target: the API.
#
#   docker build --target api -t fraud-api .
#   docker build --target streamlit --build-arg GROUP=app -t fraud-app .
#
# Only the runtime dependencies and one dependency group are installed, from uv.lock.
# Training and plotting packages (lightgbm, shap, matplotlib, ...) are not in the image.

FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
# libgomp is the OpenMP runtime that XGBoost and scikit-learn need.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# --- Build the virtual environment -------------------------------------------------------
FROM base AS builder
COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# Dependency group to install on top of the runtime dependencies: "api" or "app".
ARG GROUP=api
# Dependencies first, so this layer is reused until pyproject.toml or uv.lock change.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups --group ${GROUP} --no-install-project
COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups --group ${GROUP} --no-editable

# --- Common runtime: virtual environment, model, unprivileged user -----------------------
FROM base AS runtime
RUN useradd --create-home --uid 10001 appuser
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
# The package is installed as a wheel, so it is told where the model files are.
ENV PATH="/app/.venv/bin:${PATH}" \
    FRAUD_PROJECT_ROOT=/app
COPY models/final_pipeline.joblib models/final_config.json ./models/
COPY app/__init__.py ./app/

# --- Streamlit app (build with --build-arg GROUP=app) ------------------------------------
FROM runtime AS streamlit
COPY .streamlit ./.streamlit
COPY app/streamlit_app.py ./app/
COPY app/data ./app/data
COPY reports/results.csv reports/final_metrics.json ./reports/
COPY reports/figures/exp_stage2_cv_heatmap.png reports/figures/eval_test_pr_curves.png \
     reports/figures/eval_shap_summary.png ./reports/figures/
USER appuser
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)"]
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]

# --- API (default target) ----------------------------------------------------------------
FROM runtime AS api
COPY app/api.py ./app/
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
