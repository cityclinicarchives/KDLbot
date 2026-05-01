
import os
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
LAB_CHAT_ID = int(os.getenv("LAB_CHAT_ID", "0"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-4.1-nano")
OPENAI_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4.1-nano")
OPENAI_AUDIO_MODEL = os.getenv("OPENAI_AUDIO_MODEL", "gpt-4o-mini-transcribe")

if not BOT_TOKEN:
    raise RuntimeError("Не найден BOT_TOKEN. Добавьте его в .env локально или в Variables на Railway.")
