FROM python:3.10-slim

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -U pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt \
 && pip cache purge

COPY . /app

# Default: show help
ENV PYTHONPATH=/app
CMD ["python", "dynamic_inference.py", "-h"]
