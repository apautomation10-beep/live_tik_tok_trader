#!/usr/bin/env python3
"""Background TTS worker that reads tts_queue.json from MEDIA_ROOT and generates
one audio file at a time using Coqui TTS.

Behavior rules (as required):
- Run forever in a while True loop
- Load Coqui TTS model(s) ONCE at startup
- Process exactly one chunk per loop
- Sleep 1 second after a generated audio
- Sleep 2 seconds when no pending jobs
- Use atomic writes (tmp + os.replace)
- Be tolerant to restarts and idempotent
- Log with print

Run with: python /app/scripts/worker_tts.py
"""

import os
import sys
import time
import json
import gc
import traceback

# Ensure Django settings are available
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'live_tts_project.settings')

# Ensure the project root is on sys.path. When this script is executed directly
# (e.g. /app/scripts/worker_tts.py) sys.path[0] will be the scripts folder, so
# imports like `import live_tts_project` may fail unless the repo root is added.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import django
    django.setup()
except ModuleNotFoundError as e:
    print('Failed to setup Django (module not found):', e)
    print('Checked PROJECT_ROOT:', PROJECT_ROOT)
    print('sys.path (first entries):', sys.path[:5])
    raise
except Exception as e:
    print('Failed to setup Django:', e)
    raise

from django.conf import settings
from TTS.api import TTS
from openai import OpenAI

# Try to reuse helpers and model mapping from streamer.views if available
try:
    from streamer.views import (
        TTS_MODELS,
        generate_long_commentary,
        add_spanish_pauses,
        split_text_into_chunks,
        WORDS_PER_AUDIO,
    )
except Exception:
    TTS_MODELS = {
        'en': 'tts_models/en/ljspeech/tacotron2-DDC',
        'es': 'tts_models/es/css10/vits',
    }
    # Fallback helpers (small standins)
    def split_text_into_chunks(text, words_per_chunk=250):
        words = text.split()
        return [" ".join(words[i:i + words_per_chunk]) for i in range(0, len(words), words_per_chunk)]
    def add_spanish_pauses(text):
        return text
    def generate_long_commentary(client, model_name, base_prompt, max_tokens, parts=3):
        # Very small fallback -- in practice the worker will import the real function.
        return base_prompt
    WORDS_PER_AUDIO = 250

TTS_MODEL_ROOT = os.environ.get('TTS_MODEL_PATH', '/app/tts_models')
TTS_QUEUE_FILE = os.path.join(settings.MEDIA_ROOT, 'tts_queue.json')


def _ensure_tts_queue():
    if not os.path.isdir(settings.MEDIA_ROOT):
        os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
    if not os.path.exists(TTS_QUEUE_FILE):
        with open(TTS_QUEUE_FILE, 'w', encoding='utf-8') as fh:
            json.dump([], fh)


def _read_tts_queue():
    _ensure_tts_queue()
    with open(TTS_QUEUE_FILE, 'r', encoding='utf-8') as fh:
        try:
            return json.load(fh)
        except Exception:
            return []


def _write_tts_queue(q):
    tmp = TTS_QUEUE_FILE + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(q, fh)
    os.replace(tmp, TTS_QUEUE_FILE)


def atomic_update_tts_queue(update_fn, retries=5, backoff=0.05):
    for _ in range(retries):
        try:
            _ensure_tts_queue()
            with open(TTS_QUEUE_FILE, 'r', encoding='utf-8') as fh:
                q = json.load(fh)
            new_q = update_fn(q) or q
            tmp = TTS_QUEUE_FILE + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as fh:
                json.dump(new_q, fh)
            os.replace(tmp, TTS_QUEUE_FILE)
            return new_q
        except Exception:
            time.sleep(backoff)
    raise RuntimeError('Failed to update tts queue after retries')


def load_models():
    """Load TTS models once at startup for each language present in mapping."""
    tts_instances = {}

    for lang, model_name in TTS_MODELS.items():
        safe_name = model_name.replace('/', '_')
        local_model_path = os.path.join(TTS_MODEL_ROOT, safe_name)
        try:
            if os.path.isdir(local_model_path):
                print(f'Loading local TTS model for {lang} from {local_model_path}...')
                tts = TTS(model_name=local_model_path, progress_bar=False, gpu=False)
            else:
                print(f'Loading remote TTS model for {lang} {model_name}...')
                tts = TTS(model_name=model_name, progress_bar=False, gpu=False)
            tts_instances[lang] = tts
            print(f'Loaded model for lang={lang}')
        except Exception:
            print(f'Failed to load TTS model for lang={lang}', traceback.format_exc())
    return tts_instances


def find_next_job(queue):
    """Return the first TTS chunk job to process or None.

    Criteria: status == 'pending' OR status == 'processing' but the expected
    file is missing (allows recovery after crashes).
    Queue items are expected to have: session, index, text, status.
    """
    for job in queue:
        status = job.get('status')
        # support legacy keys (session_ts/idx) for backward compatibility
        session = job.get('session') if job.get('session') is not None else job.get('session_ts')
        index = job.get('index') if job.get('index') is not None else job.get('idx')
        if status == 'pending':
            return job
        if status == 'processing':
            # if file is missing, re-process it
            try:
                fn = f"live_{session}_{int(index):03d}.wav" if session is not None and index is not None else None
            except Exception:
                fn = None
            if fn:
                path = os.path.join(settings.MEDIA_ROOT, fn)
                if not os.path.exists(path):
                    return job
    return None


def set_job_status(session, index, status_val):
    def updater(q):
        for j in q:
            # support legacy keys
            if (j.get('session') == session and j.get('index') == index) or (j.get('session_ts') == session and j.get('idx') == index):
                j['status'] = status_val
                # update a timestamp to help debugging
                j['updated_at'] = int(time.time())
                break
        return q
    atomic_update_tts_queue(updater)


def main_loop():
    tts_models = load_models()
    print('Worker started; entering main loop')

    while True:
        try:
            q = _read_tts_queue()
            job = find_next_job(q)

            if not job:
                # nothing to do
                time.sleep(2)
                continue


            # get data from the job using the new schema
            session = job.get('session')
            index = int(job.get('index')) if job.get('index') is not None else 0
            text = job.get('text', '')
            language = job.get('language', 'en')

            # mark processing (atomic)
            set_job_status(session, index, 'processing')

            # reload job (in case something changed)
            q2 = _read_tts_queue()
            myjob = None
            for j in q2:
                if j.get('session') == session and int(j.get('index', -1)) == index:
                    myjob = j
                    break

            if not myjob:
                print('Job disappeared from queue, skipping')
                continue

            filename = f"live_{session}_{index:03d}.wav"
            final_path = os.path.join(settings.MEDIA_ROOT, filename)

            # If file already exists, mark done and continue
            if os.path.exists(final_path):
                print(f'File already exists {final_path}; marking done')
                set_job_status(session, index, 'done')
                continue

            tts = tts_models.get(language)
            if not tts:
                print(f'No TTS model loaded for language {language}; setting job back to pending')
                set_job_status(session, index, 'pending')
                time.sleep(1)
                continue

            temp_path = final_path + '.tmp'
            try:
                # generate audio to temp path
                print(f'Generating audio for session={session} index={index} to {temp_path} ...')

                if language == 'es':
                    tts.tts_to_file(
                        text=text,
                        file_path=temp_path,
                        length_scale=1.45,
                        noise_scale=0.6,
                        noise_scale_w=0.75
                    )
                else:
                    tts.tts_to_file(text=text, file_path=temp_path)

                # atomic move into place
                os.replace(temp_path, final_path)
                set_job_status(session, index, 'done')
                print(f'Job complete: {final_path}')

            except Exception:
                print('TTS generation failed', traceback.format_exc())
                # reset to pending so it can be retried later
                set_job_status(session, index, 'pending')

            # housekeeping
            gc.collect()
            time.sleep(0.3)
        except Exception:
            print('Worker main loop exception', traceback.format_exc())
            time.sleep(2)


if __name__ == '__main__':
    try:
        main_loop()
    except KeyboardInterrupt:
        print('Worker interrupted; exiting')
    except Exception:
        print('Worker failed', traceback.format_exc())
