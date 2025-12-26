import os

# 🔒 HARD limits to prevent OOM (must be before torch import)
os.environ["TORCH_DISABLE_NNPACK"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "0"


import time
import json
import collections
import traceback
import torch
import re
from datetime import datetime

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings

# External libs
from openai import OpenAI
from TTS.api import TTS
from TTS.utils.radam import RAdam

# ------------------------
# TTS SINGLETON CACHE (CRITICAL)
# ------------------------

_TTS_CACHE = {}

def get_tts(model_name: str) -> TTS:
    """
    Load exactly ONE TTS model into memory.
    Clears previous model to avoid OOM.
    """
    if model_name not in _TTS_CACHE:
        _TTS_CACHE.clear()
        _TTS_CACHE[model_name] = TTS(
            model_name=model_name,
            progress_bar=False,
            gpu=False,
        )
    return _TTS_CACHE[model_name]

# 🔥 PyTorch 2.6 compatibility for Coqui TTS
torch.serialization.add_safe_globals([
    RAdam,
    collections.defaultdict,
    dict,
])

# ------------------------
# PROMPTS
# ------------------------
PROMPT_EN = """
PROMPT — LIVE AI TRADER AGENT XAUUSD

You are a professional gold market participant with deep real world experience focused exclusively on XAUUSD gold

You host daily live market commentary sessions where you speak calmly and continuously about the gold market using the most recent market context and developments from the last twenty four hours

You are not a retail trader
You are not a signal provider
You are not a news reader
You are not an academic analyst

You speak like someone who has lived through many market cycles and understands how capital behaves during uncertainty transitions and stress

Your voice is calm grounded reflective and confident
Never rushed
Never emotional
Never promotional

Anyone listening should naturally feel that this is experience not theory
That there is no hype
That this is someone who understands markets at a deep structural level
That this is someone they could trust to learn how markets really work over time

PRIMARY PURPOSE OF THIS LIVE SESSION

This live session exists to organically attract aligned serious participants into a private long term trading community

The live is not meant to teach trading techniques
It is not meant to give signals
It is not meant to explain strategies step by step

It exists to position authority build trust and naturally filter people who care about context discipline and long term understanding rather than shortcuts

STRICT LANGUAGE AND FORMAT RULES FOR TTS

Use plain natural English words only
Do not use any symbols
Do not use punctuation
Do not use commas
Do not use dots
Do not use hyphens
Do not use quotation marks
Do not use question marks
Do not use emojis
Do not use bullet points
Do not use tables
Do not write numbers as digits always spell them out as words

Write in one continuous spoken style flow as if you are speaking live without stopping

ABOUT PERFORMANCE AND RESULTS STRICT RULES

You may only reference performance indirectly and elegantly

Never show numbers
Never promise results
Never imply easy or fast money
Never use performance as marketing

Performance must always be framed as
A private discussion
A byproduct of discipline risk control and time
Secondary to process and understanding

Acceptable tone examples include
Talking about performance does not really make sense in a public live
Results are something reviewed privately with proper context
Here we focus on market behavior not outcomes

STRICTLY FORBIDDEN NON NEGOTIABLE

You must never give trading advice
You must never suggest entries exits stops or direction
You must never use words such as buy sell long short setup trade position opportunity

If the discussion approaches operational thinking immediately reframe toward
Market behavior
Participation
Structure
Psychology of capital
Conceptual risk management

CORE CONTENT YOU MUST ALWAYS COVER

You must speak about XAUUSD through
The current macroeconomic environment
Central bank narratives credibility and contradictions
The relationship between gold the US dollar yields and risk sentiment
How price behaves through balance absorption hesitation and waiting phases
What the market is doing and equally what it is not doing
Why gold punishes impatience
How institutional capital behaves during uncertainty and transition phases

All explanations must feel like thinking out loud
Never teaching
Never lecturing
Never summarizing

RECENT NEWS DAILY CONTEXT REQUIREMENT

Because this is a daily live session you must strongly focus on developments from the last twenty four hours

Explain how recent news changes perception behavior and positioning rather than price targets
Separate headline reaction from deeper structural meaning
Avoid repeating old narratives unless they are still actively influencing capital behavior

You may reference professional news sources such as Investing or ForexFactory only as context never as signal drivers

COMMUNICATION STYLE MANDATORY

Continuous spoken style
Natural reflective phrasing
No structured sections inside the spoken content

Natural phrases you may use include
What really stands out to me here is
This is usually what happens when the market enters this phase
A lot of people misunderstand this because
This is exactly the kind of nuance we unpack inside the community

COMMUNITY CALL TO ACTION SOFT AND NATURAL

Approximately every three minutes of spoken content naturally include a light conversational line such as

If you want to follow this kind of thinking consistently you can join our channel using the link provided

This must feel natural
Never promotional
Never interrupt the flow

DEEPER COMMUNITY CONTEXT

At appropriate moments naturally mention that
The full process is discussed privately
Performance is reviewed privately with context
This is not a signals group
It is designed for long term understanding and discipline

ENDING FIXED CLOSING LINES ALWAYS USE EXACTLY

At the end of the live session always close with these exact lines

Thanks it is time for goodbye
We are twenty four by seven available for you on Telegram
Meet you soon on the next live

LENGTH REQUIREMENT

Generate approximately eight thousand words
Maintain coherence depth and flow throughout
Never summarize
Never oversimplify
It must feel like one long uninterrupted calm live broadcast

LANGUAGE

Write everything in natural professional English
Human conversational reflective and grounded
Like a real experienced market participant speaking live

"""

PROMPT_ES = """
PROMPT COMENTARISTA PROFESIONAL DE ORO XAUUSD EN VIVO

Eres un participante profesional del mercado del oro con experiencia real profunda enfocado exclusivamente en el oro XAUUSD

Realizas sesiones diarias en vivo donde hablas de forma calmada continua y reflexiva sobre el mercado del oro utilizando el contexto mas reciente y los desarrollos de las ultimas veinticuatro horas

No eres un trader minorista
No eres proveedor de senales
No eres lector de noticias
No eres analista academico

Hablas como alguien que ha vivido muchos ciclos de mercado y entiende como se comporta el capital durante periodos de incertidumbre transicion y estres

Tu voz es calmada solida reflexiva y confiable
Nunca apresurada
Nunca emocional
Nunca promocional

Cualquier persona que escuche debe sentir naturalmente que esto es experiencia y no teoria
Que no hay exageracion
Que es alguien que entiende los mercados a un nivel estructural profundo
Que es alguien en quien se puede confiar para aprender como funcionan realmente los mercados con el tiempo

PROPOSITO PRINCIPAL DE ESTA SESION EN VIVO

Esta sesion existe para atraer de forma organica a participantes serios y alineados hacia una comunidad privada de trading a largo plazo

La sesion no esta diseñada para ensenar tecnicas de trading
No esta diseñada para dar senales
No esta diseñada para explicar estrategias paso a paso

Existe para posicionar autoridad construir confianza y filtrar naturalmente a personas que valoran contexto disciplina y entendimiento a largo plazo en lugar de atajos

REGLAS ESTRICTAS DE LENGUAJE Y FORMATO PARA TTS

Usa solo palabras naturales en espanol
No uses simbolos
No uses signos de puntuacion
No uses comas
No uses puntos
No uses guiones
No uses comillas
No uses signos de pregunta
No uses emojis
No uses listas
No uses tablas
No escribas numeros como digitos siempre escribelos con palabras

Escribe en un flujo hablado continuo como si estuvieras hablando en vivo sin detenerte

REGLAS ESTRICTAS SOBRE RENDIMIENTO Y RESULTADOS

Solo puedes mencionar el rendimiento de forma indirecta y elegante

Nunca muestres numeros
Nunca prometas resultados
Nunca impliques dinero facil o rapido
Nunca uses el rendimiento como marketing

El rendimiento siempre debe presentarse como
Una conversacion privada
Un subproducto de disciplina control de riesgo y tiempo
Secundario al proceso y al entendimiento

Ejemplos de tono aceptable incluyen
Hablar de resultados no tiene sentido en una transmision publica
Los resultados se revisan de forma privada con el contexto adecuado
Aqui nos enfocamos en el comportamiento del mercado no en resultados

ESTRICTAMENTE PROHIBIDO SIN EXCEPCIONES

Nunca dar consejos de trading
Nunca sugerir entradas salidas stops o direccion
Nunca usar palabras como comprar vender largo corto operacion posicion oportunidad

Si la conversacion se acerca a pensamiento operativo debes redirigir inmediatamente hacia
Comportamiento del mercado
Participacion
Estructura
Psicologia del capital
Gestion conceptual del riesgo

CONTENIDO CENTRAL QUE SIEMPRE DEBES CUBRIR

Debes hablar sobre XAUUSD a traves de
El entorno macroeconomico actual
Las narrativas de los bancos centrales su credibilidad y contradicciones
La relacion entre el oro el dolar estadounidense los rendimientos y el sentimiento de riesgo
Como el precio se comporta durante fases de equilibrio absorcion duda y espera
Lo que el mercado esta haciendo y tambien lo que no esta haciendo
Por que el oro castiga la impaciencia
Como se comporta el capital institucional durante periodos de incertidumbre y transicion

Todas las explicaciones deben sentirse como pensamiento en voz alta
Nunca ensenar
Nunca dar lecciones
Nunca resumir

REQUISITO DE CONTEXTO DIARIO DE NOTICIAS RECIENTES

Debido a que esta es una sesion diaria debes enfocarte fuertemente en los desarrollos de las ultimas veinticuatro horas

Explica como las noticias recientes cambian la percepcion el comportamiento y el posicionamiento en lugar de objetivos de precio
Separa la reaccion a titulares del significado estructural profundo
Evita repetir narrativas antiguas a menos que sigan influyendo activamente en el comportamiento del capital

Puedes mencionar fuentes profesionales como Investing o ForexFactory solo como contexto nunca como generadores de senales

ESTILO DE COMUNICACION OBLIGATORIO

Estilo hablado continuo
Frases naturales y reflexivas
Sin secciones estructuradas dentro del contenido hablado

Frases naturales que puedes usar incluyen
Lo que realmente me llama la atencion aqui es
Esto suele ocurrir cuando el mercado entra en esta fase
Mucha gente malinterpreta esto porque
Este es exactamente el tipo de matiz que analizamos dentro de la comunidad

LLAMADO A LA COMUNIDAD SUAVE Y NATURAL

Aproximadamente cada tres minutos de contenido hablado incluye de forma natural una linea conversacional ligera como

Si quieres seguir este tipo de pensamiento de forma consistente puedes unirte a nuestro canal usando el enlace proporcionado

Debe sentirse natural
Nunca promocional
Nunca interrumpir el flujo

CONTEXTO PROFUNDO DE LA COMUNIDAD

En momentos apropiados menciona de forma natural que
El proceso completo se discute de forma privada
El rendimiento se revisa de forma privada con contexto
Este no es un grupo de senales
Esta diseñado para entendimiento y disciplina a largo plazo

CIERRE FINAL FIJO USAR EXACTAMENTE ESTAS LINEAS

Al final de la sesion siempre cierra con estas lineas exactas

Gracias es momento de despedirnos
Estamos disponibles veinticuatro horas siete dias en Telegram
Nos vemos pronto en la siguiente transmision en vivo

REQUISITO DE LONGITUD

Genera aproximadamente ocho mil palabras
Mantiene coherencia profundidad y fluidez en todo momento
Nunca resumir
Nunca simplificar en exceso
Debe sentirse como una sola transmision en vivo larga calmada e ininterrumpida

IDIOMA

Escribe todo en espanol profesional natural
Conversacional reflexivo y solido
Como un participante del mercado con experiencia real hablando en vivo
"""


# ------------------------
# TTS MODELS
# ------------------------
TTS_MODELS = {
    'en': 'tts_models/en/ljspeech/tacotron2-DDC',
    'es': 'tts_models/es/css10/vits',
}

# ~3 minutes per audio
WORDS_PER_AUDIO = 190


def dashboard(request):
    return render(request, 'streamer/dashboard.html')


@csrf_exempt
def list_recordings(request):
    """Return all recordings grouped by live session (based on filename pattern live_{ts}_{idx}.wav)."""
    media_root = settings.MEDIA_ROOT
    media_url = settings.MEDIA_URL
    if not os.path.isdir(media_root):
        return JsonResponse({'sessions': []})

    sessions = {}
    pattern = re.compile(r'^live_(\d+)_\d+\.(wav|mp3)$')

    for fn in os.listdir(media_root):
        m = pattern.match(fn)
        if not m:
            continue
        ts = m.group(1)
        url = media_url + fn
        sessions.setdefault(ts, []).append({'filename': fn, 'url': url})

    result = []
    for ts, files in sessions.items():
        try:
            ts_int = int(ts)
            created = datetime.utcfromtimestamp(ts_int).strftime('%Y-%m-%d %H:%M:%S UTC')
        except Exception:
            created = ts
        result.append({
            'session': ts,
            'created_at': created,
            'files': sorted(files, key=lambda x: x['filename'])
        })

    result = sorted(result, key=lambda x: x['created_at'], reverse=True)
    return JsonResponse({'sessions': result})


@csrf_exempt
def delete_all_recordings(request):
    """Delete all recordings that match the live_\d+_\d+.wav|mp3 pattern."""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=400)

    media_root = settings.MEDIA_ROOT
    if not os.path.isdir(media_root):
        return JsonResponse({'deleted': 0})

    deleted = 0
    errors = []
    pattern = re.compile(r'^live_\d+_\d+\.(wav|mp3)$')

    for fn in os.listdir(media_root):
        if pattern.match(fn):
            try:
                os.remove(os.path.join(media_root, fn))
                deleted += 1
            except Exception as e:
                errors.append(str(e))

    return JsonResponse({'deleted': deleted, 'errors': errors})


def split_text_into_chunks(text, words_per_chunk=WORDS_PER_AUDIO):
    words = text.split()
    return [
        " ".join(words[i:i + words_per_chunk])
        for i in range(0, len(words), words_per_chunk)
    ]

def generate_long_commentary(client, model_name, base_prompt, max_tokens, parts=3):
    """
    Generate a long XAUUSD live commentary in multiple parts.
    Handles both English and Spanish prompts.
    Fixed for OpenAI Responses API (safe attribute access).
    """
    full_text = ""
    last_context = ""

    # Detect Spanish from base prompt
    is_spanish = "espanol" in base_prompt.lower() or "español" in base_prompt.lower()

    for i in range(parts):
        if i == 0:
            prompt = base_prompt
        else:
            if is_spanish:
                prompt = f"""
Continua la misma transmision en vivo sobre el mercado del oro XAUUSD

No repitas ideas anteriores

Contexto previo
{last_context}

Escribe aproximadamente tres mil palabras mas
Sigue todas las reglas anteriores
Usa solo espanol
"""
            else:
                prompt = f"""
Continue the same XAUUSD gold market commentary

Do not repeat earlier ideas

Previous context
{last_context}

Write another three thousand words
Follow all earlier rules
Plain spoken English
"""

        # 🔹 OpenAI call
        response = client.responses.create(
            model=model_name,
            input=[
                {"role": "user", "content": prompt}
            ],
            max_output_tokens=max_tokens,
        )

        # 🔹 Extract text safely
        text = getattr(response, "output_text", "")
        if not text:
            fragments = []
            for item in getattr(response, "output", []) or []:
                contents = getattr(item, "content", []) or []
                for c in contents:
                    if getattr(c, "type", None) == "output_text":
                        fragments.append(getattr(c, "text", ""))
            text = " ".join(fragments)

        # 🔹 Safety check
        if not text.strip():
            raise ValueError("No text generated from OpenAI")

        # 🔹 Append to full text
        full_text += " " + text.strip()

        # 🔹 Save last context for next part
        words = full_text.split()
        last_context = " ".join(words[-400:])

        # 🔹 Small pause between parts
        time.sleep(1)

    return full_text.strip()



def add_spanish_pauses(text):
    """
    Add natural pauses for Spanish TTS (Coqui vits)
    Uses line breaks + soft pause words
    """
    pause_words = [
        "bien",
        "entonces",
        "ahora",
        "veamos esto",
        "con calma",
    ]

    sentences = text.split(" ")
    out = []
    counter = 0

    for word in sentences:
        out.append(word)
        counter += 1

        # roughly every 18–22 words → pause
        if counter >= 20:
            pause = pause_words[counter % len(pause_words)]
            out.append("\n\n" + pause + "\n\n")
            counter = 0

    return " ".join(out)


@csrf_exempt
def go_live(request):
    """Generate OpenAI text only, split into chunks, and enqueue TTS jobs on disk.

    Returns immediately with session_ts, total_chunks and status queued.
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=400)

    try:
        data = json.loads(request.body.decode('utf-8'))
    except Exception:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)

    language = data.get('language', 'en')

    # ------------------------
    # OPENAI (unchanged)
    # ------------------------
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    model_name = os.environ.get('OPENAI_MODEL', 'gpt-5-mini')
    max_tokens = int(os.environ.get('OPENAI_MAX_TOKENS', '16000'))

    prompt = PROMPT_ES if language == 'es' else PROMPT_EN

    # Instead of running the OpenAI generation inside the HTTP request (which may
    # time out on Hostinger), enqueue a lightweight 'generate' job which the
    # background worker will pick up and turn into TTS jobs.
    session_ts = int(time.time())

    def _adder(q):
        # q is a list of job dicts
        existing = {(j.get('session_ts'), j.get('type')) for j in q}
        key = (session_ts, 'generate')
        if key in existing:
            return q

        job = {
            'type': 'generate',
            'session_ts': session_ts,
            'idx': -1,
            'language': language,
            'prompt': prompt,
            'parts': 3,
            'max_tokens': max_tokens,
            'status': 'pending',
            'filename': ''
        }
        q.append(job)
        return q

    try:
        atomic_update_tts_queue(_adder)
    except Exception as e:
        traceback.print_exc()
        return JsonResponse({'error': f'Failed to enqueue generate job {e}'}, status=500)

    return JsonResponse({'session_ts': session_ts, 'total_chunks': 0, 'status': 'queued'})



# ------------------------
# List all audio files (live + reply + others)
@csrf_exempt
def list_all_audio(request):
    if request.method != 'GET':
        return JsonResponse({'error': 'GET required'}, status=400)

    media_root = settings.MEDIA_ROOT
    media_url = settings.MEDIA_URL
    if not os.path.isdir(media_root):
        return JsonResponse({'files': []})

    files = []
    for fn in os.listdir(media_root):
        if not fn.lower().endswith(('.wav', '.mp3')):
            continue
        path = os.path.join(media_root, fn)
        try:
            stat = os.stat(path)
            created = datetime.utcfromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S UTC')
            size = stat.st_size
        except Exception:
            created = None
            size = 0

        if fn.startswith('live_'):
            ftype = 'live'
        elif fn.startswith('reply_batch_'):
            ftype = 'reply'
        else:
            ftype = 'other'

        files.append({
            'filename': fn,
            'url': media_url + fn,
            'created': created,
            'size': size,
            'type': ftype
        })

    files = sorted(files, key=lambda x: x.get('created') or '', reverse=True)
    return JsonResponse({'files': files})


@csrf_exempt
def delete_file(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=400)

    try:
        data = json.loads(request.body.decode('utf-8'))
        filename = data.get('filename', '')
    except Exception:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)

    if not filename or '/' in filename or '..' in filename:
        return JsonResponse({'error': 'invalid filename'}, status=400)

    path = os.path.join(settings.MEDIA_ROOT, os.path.basename(filename))
    if not os.path.exists(path):
        return JsonResponse({'error': 'not found'}, status=404)

    try:
        os.remove(path)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'deleted': True})


# ------------------------
# TikTok comment -> 1-minute batch reply pipeline (concurrency-safe, atomic file handling)
# ------------------------

import logging
import uuid

logger = logging.getLogger(__name__)

QUEUE_FILE = os.path.join(settings.MEDIA_ROOT, 'tiktok_reply_queue.json')
MAX_BATCH_COMMENTS = int(os.environ.get('TIKTOK_MAX_BATCH_COMMENTS', '20'))

# Emoji regex
try:
    EMOJI_RE = re.compile(r'[\U0001F300-\U0001F6FF\U0001F900-\U0001F9FF\U0001F1E6-\U0001F1FF]', flags=re.UNICODE)
except re.error:
    EMOJI_RE = re.compile(r'[\u2600-\u26FF\u2700-\u27BF]', flags=re.UNICODE)


# ------------------------
# Simple file lock helpers
# ------------------------

def _read_queue():
    _ensure_queue()
    with open(QUEUE_FILE, 'r', encoding='utf-8') as fh:
        return json.load(fh)


def _write_queue(q):
    tmp = QUEUE_FILE + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(q, fh)
    os.replace(tmp, QUEUE_FILE)


def atomic_update_queue(update_fn, retries=5, backoff=0.05):
    """
    Atomically read/modify/write the queue file by applying update_fn on the
    current queue dict. Retries a few times to reduce race conditions.
    update_fn should return the modified queue dict.
    """
    for _ in range(retries):
        try:
            _ensure_queue()
            with open(QUEUE_FILE, 'r', encoding='utf-8') as fh:
                q = json.load(fh)
            new_q = update_fn(q) or q
            tmp = QUEUE_FILE + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as fh:
                json.dump(new_q, fh)
            os.replace(tmp, QUEUE_FILE)
            return new_q
        except Exception:
            time.sleep(backoff)
    raise RuntimeError('Failed to update queue after retries')


# ------------------------
# Queue helpers (atomic with lock)
# ------------------------

def _ensure_queue():
    if not os.path.isdir(settings.MEDIA_ROOT):
        os.makedirs(settings.MEDIA_ROOT, exist_ok=True)

    if not os.path.exists(QUEUE_FILE):
        with open(QUEUE_FILE, 'w', encoding='utf-8') as fh:
            json.dump({
                'window_start': 0,
                'comments': [],
                'audio_queue': []
            }, fh)


# ------------------------
# TTS QUEUE (disk-based JSON queue for background worker)
# ------------------------
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
    """
    Atomically read / modify / write the tts queue using a temp file + os.replace.
    update_fn should accept the current queue (list) and return the modified queue.
    """
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




# ------------------------
# Validation
# ------------------------

def _is_comment_valid(text):
    if not text or not text.strip():
        return False

    if re.search(r'http[s]?://', text):
        return False

    if EMOJI_RE.search(text):
        stripped = EMOJI_RE.sub('', text).strip()
        if not stripped:
            return False

    if len(text.split()) < 2:
        return False

    if len(text.strip()) < 4:
        return False

    if re.match(r'^(.)\1{5,}$', text.strip()):
        return False

    return True


# ------------------------
# OpenAI short reply (sanitized)
# ------------------------

def _sanitize_reply(text):
    # Remove punctuation and digits; collapse whitespace
    text = re.sub(r'[0-9]', '', text)
    text = re.sub(r'[\.,:;!\?\-\—"\'\(\)\[\]\{\}…]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
def _generate_short_reply(client, model_name, comment, user, language='en'):
    if language == 'es':
        system = (
            'Eres un anfitrion de TikTok en vivo calmado y profesional '
            'Lee el comentario y responde directamente a la persona por su nombre '
            'Habla como si estuvieras respondiendo en tiempo real en el chat '
            'Usa una o dos frases cortas en tono natural y conversacional '
            'No des consejos de trading '
            'No uses numeros ni signos de puntuacion'
        )
        user_prompt = f'{user} dijo {comment}'
    else:
        system = (
            'You are a calm professional TikTok live host '
            'Read the comment and reply directly to the person by name '
            'Speak like you are responding in real time in the live chat '
            'Use one or two short sentences in a natural conversational tone '
            'Do not give trading advice '
            'Do not use numbers or punctuation'
        )
        user_prompt = f'{user} said {comment}'

    response = client.responses.create(
        model=model_name,
        input=[
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': user_prompt},
        ],
        max_output_tokens=64,
    )

    # ✅ Preferred path (new SDK)
    text = getattr(response, 'output_text', '') or ''

    # ✅ Fallback for structured outputs
    if not text:
        fragments = []
        for item in getattr(response, 'output', []) or []:
            contents = getattr(item, 'content', []) or []
            for c in contents:
                c_type = getattr(c, 'type', None)
                if c_type == 'output_text':
                    fragments.append(getattr(c, 'text', ''))
        text = ' '.join(fragments)

    text = text.strip()
    text = _sanitize_reply(text)

    # hard safety limit
    if len(text.split()) > 30:
        text = ' '.join(text.split()[:30])

    return text


# ------------------------
# COMMENT INGEST ENDPOINT (locked writes)
# ------------------------
@csrf_exempt
def tiktok_comment(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=400)

    expected = os.environ.get('TIKTOK_SECRET', '')
    incoming = request.headers.get('X-TIKTOK-SECRET', '')
    if expected and incoming != expected:
        return JsonResponse({'error': 'unauthorized'}, status=403)

    try:
        data = json.loads(request.body.decode('utf-8'))
        comment = data.get('comment', '').strip()
        user = data.get('user', '').strip() or 'viewer'
        language = data.get('language', '').strip().lower() or 'en'
    except Exception:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)

    if not _is_comment_valid(comment):
        return JsonResponse({'skipped': 'filter'}, status=200)

    now = time.time()

    def updater(q):
        if not q.get('window_start'):
            q['window_start'] = now
        q.setdefault('comments', []).append({
            'user': user,
            'comment': comment,
            'time': now,
            'language': language,
        })
        return q

    try:
        atomic_update_queue(updater)
    except Exception:
        logger.exception('Failed to store tiktok comment')
        return JsonResponse({'error': 'server error'}, status=500)

    return JsonResponse({'stored': True})


def _window_ready():
    try:
        q = _read_queue()
        ws = q.get('window_start', 0)
        if not ws:
            return False
        return (time.time() - ws) >= 60
    except Exception:
        return False

# ------------------------
# PROCESS 1-MINUTE WINDOW
# ------------------------
def process_tiktok_comment_window():
    """
    Process comments collected during the one minute window without using an external lock.
    Uses an in-file 'processing' timestamp to ensure only one process handles a window at a time.
    """
    try:
        _ensure_queue()
        q = _read_queue()
        now = time.time()
        ws = q.get('window_start', 0)
        if not ws or (now - ws) < 60:
            return

        def claim(q_local):
            proc = q_local.get('processing', 0)
            if proc and (now - proc) < 120:
                return q_local
            q_local['processing'] = now
            return q_local

        q_after_claim = atomic_update_queue(claim)
        if q_after_claim.get('processing') != now:
            return

        q = _read_queue()
        comments = q.get('comments', []) or []
        if not comments:
            def clear_proc(qc):
                qc.pop('processing', None)
                qc['window_start'] = 0
                return qc
            atomic_update_queue(clear_proc)
            return

        batch = comments[:MAX_BATCH_COMMENTS]

        client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
        model_name = os.environ.get('OPENAI_MODEL', 'gpt-5-mini')

        spoken_entries = []
        for item in batch:
            reply = _generate_short_reply(
                client,
                model_name,
                item['comment'],
                item['user'],
                language=item.get('language', 'en')
            )

            entry = (
                f"We have a comment From {item['user']} "
                f"Comment says {item['comment']} "
                f"My response is {reply}"
            )
            spoken_entries.append(entry)

        combined_text = "\n".join(spoken_entries)

        lang_counts = {'en': 0, 'es': 0}
        for c in batch:
            lang_counts[c.get('language', 'en')] += 1
        tts_lang = 'es' if lang_counts['es'] > lang_counts['en'] else 'en'

        tts_model = TTS_MODELS.get(tts_lang, TTS_MODELS['en'])
        local_models_root = os.environ.get('TTS_MODEL_PATH', '/app/tts_models')
        safe_name = tts_model.replace('/', '_')
        local_model_path = os.path.join(local_models_root, safe_name)

        uid = uuid.uuid4().hex[:12]
        filename = f"reply_batch_{int(now)}_{uid}.wav"
        temp_path = os.path.join(settings.MEDIA_ROOT, filename.replace('.wav', '_tmp.wav'))
        final_path = os.path.join(settings.MEDIA_ROOT, filename)

        model_to_use = local_model_path if os.path.isdir(local_model_path) else tts_model
        tts = get_tts(model_to_use)



        tts.tts_to_file(text=combined_text, file_path=temp_path)
        os.replace(temp_path, final_path)

        def finish(q_finish):
            q_finish.setdefault('audio_queue', []).append({
                'filename': filename,
                'created': now
            })
            q_finish['window_start'] = 0
            q_finish['comments'] = comments[MAX_BATCH_COMMENTS:]
            q_finish.pop('processing', None)
            return q_finish

        atomic_update_queue(finish)

    except Exception:
        logger.exception('Error while processing tiktok comment window')

@csrf_exempt
def next_reply(request):
    if request.method != 'GET':
        return JsonResponse({'error': 'GET required'}, status=400)

    # ✅ FIX: only process when window is actually ready
    if _window_ready():
        try:
            process_tiktok_comment_window()
        except Exception:
            logger.exception('process_tiktok_comment_window failed')

    popped = {}

    def pop_audio(q):
        if not q.get('audio_queue'):
            return q
        popped['entry'] = q['audio_queue'].pop(0)
        return q

    try:
        atomic_update_queue(pop_audio)
    except Exception:
        logger.exception('Failed to pop next reply')
        return JsonResponse({'error': 'server error'}, status=500)

    entry = popped.get('entry')
    if not entry:
        return JsonResponse({'found': False})

    return JsonResponse({
        'found': True,
        'url': settings.MEDIA_URL + entry['filename'],
        'filename': entry['filename']
    })
