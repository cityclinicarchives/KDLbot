from aiogram.types import (
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    ReplyKeyboardMarkup,
    KeyboardButton,
)


def confirm_list_keyboard(has_ambiguous: bool = False) -> InlineKeyboardMarkup:
    rows = []
    if not has_ambiguous:
        rows.append([InlineKeyboardButton(text="Все верно", callback_data="list_ok")])
    rows.append([InlineKeyboardButton(text="Начать заново", callback_data="restart")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def candidate_selection_keyboard(matched_items) -> InlineKeyboardMarkup:
    rows = []

    for item_index, item in enumerate(matched_items):
        if item.status != "ambiguous":
            continue

        rows.append([
            InlineKeyboardButton(
                text=f"Выберите вариант для пункта {item_index + 1}",
                callback_data="noop",
            )
        ])

        for candidate_index, candidate in enumerate(item.candidates):
            text = f"{candidate_index + 1}. {candidate.matched_name[:45]} — {candidate.price} ₽"
            rows.append([
                InlineKeyboardButton(
                    text=text,
                    callback_data=f"choose:{item_index}:{candidate_index}",
                )
            ])

        # Показываем варианты только для первого неоднозначного пункта.
        break

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
