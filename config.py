import os
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
LAB_CHAT_ID = int(os.getenv("LAB_CHAT_ID", "0"))

if not BOT_TOKEN:
    raise RuntimeError("Не найден BOT_TOKEN. Добавьте его в .env локально или в Variables на Railway.")
