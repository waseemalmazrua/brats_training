FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./

RUN pip install uv
RUN uv sync --frozen

ENV PATH="/app/.venv/bin:$PATH"

COPY . .

CMD ["python", "train.py"]