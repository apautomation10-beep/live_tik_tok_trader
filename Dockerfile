FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsndfile1 \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install CPU torch first (use pytorch cpu wheels)
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch || true
# Install matching torchaudio (CPU) to avoid CUDA-linked wheels
RUN pip install --index-url https://download.pytorch.org/whl/cpu torchaudio || true

# Copy requirements and install
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . /app/

# TTS models path (created at build and used at runtime)
ENV TTS_MODEL_PATH=/app/tts_models
RUN mkdir -p /app/tts_models

# Pre-download models during build so they are cached in the image and not redownloaded on every build.
# If you prefer a bind-mounted host folder, make sure to populate `./tts_models` on the host first,
# or use the named volume in docker-compose.yml (it will be initialized from the image on first run).
# Note: don't remove the TTS cache here; keep it intact to avoid partial downloads or library surprises.
#RUN if [ -f /app/scripts/download_models.py ]; then python /app/scripts/download_models.py || true; fi

# Add an entrypoint that downloads models at container start only when missing
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh
ENTRYPOINT ["/app/entrypoint.sh"]

# Collect static (optional)
RUN mkdir -p /app/media

EXPOSE 8000

ENV DJANGO_SETTINGS_MODULE=live_tts_project.settings

CMD ["/bin/bash", "-lc", "python manage.py migrate --noinput && python manage.py runserver 0.0.0.0:8000"]
