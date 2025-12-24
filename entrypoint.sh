#!/bin/sh
set -eu

TTS_DIR="${TTS_MODEL_PATH:-/app/tts_models}"

# Run the downloader if the script exists. The downloader is idempotent and will skip models
# that already have a completion marker file ('.download_complete').
if [ -f /app/scripts/download_models.py ]; then
  echo "Running /app/scripts/download_models.py to ensure TTS models are present in ${TTS_DIR}..."
  python /app/scripts/download_models.py || echo "Model download failed; continuing startup"
else
  echo "No download script found at /app/scripts/download_models.py; skipping model download"
fi

# Execute the container command (migrate + gunicorn provided via CMD)
exec "$@"
