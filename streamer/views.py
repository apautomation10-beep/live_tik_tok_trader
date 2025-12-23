import os
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
WORDS_PER_AUDIO = 600


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
def split_text_into_chunks(text, words_per_chunk=600):
    words = text.split()
    return [
        " ".join(words[i:i + words_per_chunk])
        for i in range(0, len(words), words_per_chunk)
    ]


def generate_long_commentary(client, model_name, base_prompt, max_tokens, parts=3):
    full_text = ""
    last_context = ""

    # detect Spanish from base prompt
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

        response = client.responses.create(
            model=model_name,
            input=[
                {
                    "role": "system",
                    "content": prompt
                }
            ],
            max_output_tokens=max_tokens,
        )

        text = getattr(response, "output_text", "")
        if not text:
            fragments = []
            for item in response.output:
                for c in item.get("content", []):
                    if c.get("type") == "output_text":
                        fragments.append(c.get("text", ""))
            text = " ".join(fragments)

        if not text.strip():
            raise ValueError("No text generated")

        full_text += " " + text.strip()

        words = full_text.split()
        last_context = " ".join(words[-400:])

        time.sleep(1)

    return full_text.strip()


@csrf_exempt
def go_live(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=400)

    try:
        data = json.loads(request.body.decode('utf-8'))
    except Exception:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)

    language = data.get('language', 'en')

    # ------------------------
    # OPENAI
    # ------------------------
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    model_name = os.environ.get('OPENAI_MODEL', 'gpt-5-mini')  
    max_tokens = int(os.environ.get('OPENAI_MAX_TOKENS', '16000'))

    prompt = PROMPT_ES if language == 'es' else PROMPT_EN

    try:
        text = generate_long_commentary(
            client=client,
            model_name=model_name,
            base_prompt=prompt,
            max_tokens=max_tokens,
            parts=3  # reduced from 6 to 3
        )
    except Exception as e:
        traceback.print_exc()
        return JsonResponse({'error': f'OpenAI failed {e}'}, status=500)

    # ------------------------
    # SPLIT FOR TTS
    # ------------------------
    chunks = split_text_into_chunks(text, words_per_chunk=600)

    tts_model = TTS_MODELS.get(language, TTS_MODELS['en'])
    local_models_root = os.environ.get('TTS_MODEL_PATH', '/app/tts_models')
    safe_name = tts_model.replace('/', '_')
    local_model_path = os.path.join(local_models_root, safe_name)

    audio_urls = []
    session_ts = int(time.time())

    try:
        if os.path.isdir(local_model_path):
            tts = TTS(model_name=local_model_path, progress_bar=False, gpu=False)
        else:
            tts = TTS(model_name=tts_model, progress_bar=False, gpu=False)

        for idx, chunk in enumerate(chunks):
            filename = f"live_{session_ts}_{idx}.wav"
            filepath = os.path.join(settings.MEDIA_ROOT, filename)

            if language == 'es':
                tts.tts_to_file(
                    text=chunk,
                    file_path=filepath,
                    length_scale=1.25,
                    noise_scale=0.65,
                    noise_scale_w=0.8
                )
            else:
                tts.tts_to_file(
                    text=chunk,
                    file_path=filepath
                )

            audio_urls.append(settings.MEDIA_URL + filename)

    except Exception as e:
        traceback.print_exc()
        return JsonResponse({'error': f'TTS failed {e}'}, status=500)

    # ------------------------
    # RESPONSE
    # ------------------------
    return JsonResponse({
        'language': language,
        'total_chunks': len(audio_urls),
        'audio_urls': audio_urls,
        'text': text
    })
