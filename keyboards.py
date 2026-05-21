from aiogram.types import (
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    ReplyKeyboardMarkup,
    KeyboardButton,
)

DEFAULT_VISIBLE_CANDIDATES = 4
MAX_VISIBLE_CANDIDATES = 8


def review_list_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Изменить часть", callback_data="edit_part")],
            [InlineKeyboardButton(text="Удалить часть", callback_data="delete_part")],
            [InlineKeyboardButton(text="Все верно", callback_data="list_ok")],
            [InlineKeyboardButton(text="Начать заново", callback_data="restart")],
        ]
    )


def candidate_keyboard(item_index: int, item, visible_limit: int = DEFAULT_VISIBLE_CANDIDATES) -> InlineKeyboardMarkup:
    rows = []

    visible_limit = max(DEFAULT_VISIBLE_CANDIDATES, min(visible_limit, MAX_VISIBLE_CANDIDATES))
    visible_candidates = item.candidates[:visible_limit]

    for candidate_index, candidate in enumerate(visible_candidates):
        label = candidate.button_text
        text = f"{candidate_index + 1}. {label} — {candidate.price} ₽"
        rows.append([
            InlineKeyboardButton(
                text=text,
                callback_data=f"choose_edit:{item_index}:{candidate_index}",
            )
        ])

    if len(item.candidates) > visible_limit:
        rows.append([
            InlineKeyboardButton(
                text=f"Показать еще ({len(item.candidates) - visible_limit})",
                callback_data=f"show_more_edit:{item_index}",
            )
        ])

    rows.append([
        InlineKeyboardButton(
            text="Уточнить вручную",
            callback_data=f"manual_refine:{item_index}",
        )
    ])
    rows.append([InlineKeyboardButton(text="Вернуться к списку", callback_data="back_to_review")])
    rows.append([InlineKeyboardButton(text="Начать заново", callback_data="restart")])

    return InlineKeyboardMarkup(inline_keyboard=rows)


def discount_keyboard() -> InlineKeyboardMarkup:
    discounts = [0, 5, 10, 15, 20, 25]
    rows = []
    for d in discounts:
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
        keyboard=[
            [KeyboardButton(text="Поделиться номером телефона", request_contact=True)]
        ],
        resize_keyboard=True,
        one_time_keyboard=True,
    )



def main_reply_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="🔍 Новый поиск")],
        ],
        resize_keyboard=True,
        persistent=True,
    )
