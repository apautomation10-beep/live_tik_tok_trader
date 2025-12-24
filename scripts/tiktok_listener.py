import os
import asyncio
import logging
import signal
import aiohttp

from TikTokLive import TikTokLiveClient
from TikTokLive.types.events import CommentEvent

# ------------------------
# Logging
# ------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("tiktok_listener")

# ------------------------
# Env config
# ------------------------
TIKTOK_USERNAME = os.environ.get("TIKTOK_USERNAME", "historiasdelfol")
POST_URL = os.environ.get("TIKTOK_POST_URL", "http://127.0.0.1:8000/tiktok/comment/")
SECRET = os.environ.get("TIKTOK_SECRET", "")

# ------------------------
# TikTok client
# ------------------------
client = TikTokLiveClient(unique_id=TIKTOK_USERNAME)

# ------------------------
# HTTP session (shared)
# ------------------------
http_session: aiohttp.ClientSession | None = None


async def send_comment_to_server(payload: dict):
    """Send comment payload to Django backend"""
    global http_session

    if http_session is None:
        logger.warning("HTTP session not ready, dropping comment")
        return

    headers = {"Content-Type": "application/json"}
    if SECRET:
        headers["X-TIKTOK-SECRET"] = SECRET

    try:
        async with http_session.post(
            POST_URL,
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            if resp.status == 200:
                logger.info("Comment sent: %s", payload["comment"])
            else:
                logger.warning(
                    "Server error %s: %s",
                    resp.status,
                    await resp.text(),
                )
    except asyncio.TimeoutError:
        logger.warning("POST timeout for comment")
    except Exception:
        logger.exception("Failed sending comment")


@client.on("comment")
async def on_comment(event: CommentEvent):
    """Handle TikTok comment event"""
    try:
        payload = {
            "comment": event.comment.comment,
            "user": event.user.uniqueId,
            "comment_id": getattr(event.comment, "cid", None),
        }

        await send_comment_to_server(payload)

    except Exception:
        logger.exception("Comment handler failed")


async def main():
    """Main async entrypoint"""
    global http_session

    logger.info("Starting TikTok listener for: %s", TIKTOK_USERNAME)

    http_session = aiohttp.ClientSession()

    try:
        await client.start()
    finally:
        if http_session:
            await http_session.close()
            logger.info("HTTP session closed")


def shutdown():
    """Graceful shutdown"""
    logger.info("Shutdown signal received, stopping TikTok client...")
    try:
        client.stop()
    except Exception:
        pass


if __name__ == "__main__":
    loop = asyncio.get_event_loop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown)

    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        shutdown()
    finally:
        loop.stop()
        logger.info("TikTok listener exited cleanly")