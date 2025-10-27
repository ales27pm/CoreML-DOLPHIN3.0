# syntax=docker/dockerfile:1.6

############################################
# Builder stage
############################################
FROM python:3.11-slim AS builder

ARG APP_HOME=/app
WORKDIR ${APP_HOME}

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY tasks/systems_backend_engineering/requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir --target /opt/site -r requirements.txt

COPY tasks/systems_backend_engineering ./app
RUN python -m compileall ./app

############################################
# Test stage
############################################
FROM python:3.11-slim AS tester

WORKDIR /workspace

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY requirements-dev.txt ./requirements-dev.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements-dev.txt

COPY . ./
RUN pytest tests/systems_backend_engineering

############################################
# Runtime stage
############################################
FROM gcr.io/distroless/python3-debian12 AS runtime

ARG APP_HOME=/app
WORKDIR ${APP_HOME}

COPY --from=builder /opt/site /opt/site
COPY --from=builder /app/app ./app

ENV PYTHONPATH=/opt/site:/app/app

USER nonroot

EXPOSE 8080
CMD ["python", "-m", "uvicorn", "tasks.systems_backend_engineering.metrics_app:app", "--host", "0.0.0.0", "--port", "8080"]
