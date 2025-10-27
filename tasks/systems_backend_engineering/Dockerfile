# syntax=docker/dockerfile:1.6

############################################
# Builder stage
############################################
FROM python:3.11-slim AS builder

ARG APP_HOME=/app
WORKDIR ${APP_HOME}

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VERSION=1.8.3

COPY pyproject.toml poetry.lock ./
RUN pip install --upgrade pip \
    && pip install "poetry==${POETRY_VERSION}" \
    && poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

COPY . .
RUN poetry build -f wheel

############################################
# Test stage
############################################
FROM builder AS tester

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN poetry run pytest --maxfail=1 --disable-warnings

############################################
# Runtime stage
############################################
FROM gcr.io/distroless/python3-debian12 AS runtime

ARG APP_HOME=/app
WORKDIR ${APP_HOME}

COPY --from=builder /app/dist/*.whl ./
RUN python -m pip install --no-cache-dir ./$(ls *.whl | head -n1)

USER nonroot

ENV PYTHONPATH=${APP_HOME}

EXPOSE 8080
CMD ["python", "-m", "service"]
