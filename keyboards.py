
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton

def get_main_keyboard():
    return ReplyKeyboardMarkup(
        keyboard=[
            [
                KeyboardButton(text="🎤 Продиктовать"),
                KeyboardButton(text="📷 Сфотографировать"),
                KeyboardButton(text="🖼 Загрузить"),
            ]
        ],
        resize_keyboard=True
    )
