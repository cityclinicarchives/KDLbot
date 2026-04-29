from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, ReplyKeyboardMarkup, KeyboardButton


def confirm_list_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Все верно", callback_data="list_ok")],
            [InlineKeyboardButton(text="Начать заново", callback_data="restart")],
        ]
    )


def discount_keyboard() -> InlineKeyboardMarkup:
    rows = []
    for d in [0, 5, 10, 15, 20, 25]:
        text = "Нет скидки" if d == 0 else f"{d}%"
        rows.append([InlineKeyboardButton(text=text, callback_data=f"discount:{d}")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def send_order_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Отправить", callback_data="send_order")],
            [InlineKeyboardButton(text="Начать заново", callback_data="restart")],
        ]
    )


def contact_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text="Поделиться номером телефона", request_contact=True)]],
        resize_keyboard=True,
        one_time_keyboard=True,
    )
