FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[test]"
COPY tests ./tests
COPY scripts ./scripts
# Runs the test suite, then the SYNTHETIC experiment (writes to /app/results).
CMD ["sh", "-c", "pytest -q && python scripts/run_synthetic_experiment.py"]
