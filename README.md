# Live TTS Django Project

This project provides a simple Django dashboard that generates live commentary text via OpenAI and converts it to speech using Coqui TTS models offline.

Quick start (Linux / Docker):

1. Copy `.env.example` to `.env` and set `OPENAI_API_KEY`.
2. Build and run with Docker Compose:

```
docker compose build --no-cache
docker compose up
```

The build step will pre-download the Coqui TTS models specified in `scripts/download_models.py` so they aren't downloaded at runtime. Note: if you use a host bind mount for `./tts_models` (as in the old default docker-compose setup), an empty host directory will override the image content and models will be downloaded at container start. To avoid repeated downloads, use the named volume (`tts_models`) in `docker-compose.yml` (recommended) or populate `./tts_models` on the host by running `python scripts/download_models.py` locally before `docker compose up`.

Notes:
- Models used:
  - `tts_models/en/ljspeech/tacotron2-DDC`
  - `tts_models/es/mai/tacotron2-DDC`
- OpenAI token limits may prevent generating extremely long single responses; you may need to chunk generation.
