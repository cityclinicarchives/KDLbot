from aiogram.types import (
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    ReplyKeyboardMarkup,
    KeyboardButton,
)

DEFAULT_VISIBLE_CANDIDATES = 4
MAX_VISIBLE_CANDIDATES = 8


def confirm_list_keyboard(has_ambiguous: bool = False) -> InlineKeyboardMarkup:
    rows = []
    if not has_ambiguous:
        rows.append([InlineKeyboardButton(text="Все верно", callback_data="list_ok")])
    rows.append([InlineKeyboardButton(text="Начать заново", callback_data="restart")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def candidate_selection_keyboard(matched_items, candidate_limits: dict | None = None) -> InlineKeyboardMarkup:
    """
    Для первого неоднозначного пункта показывает кнопки выбора.

    Если у анализа есть short_name, на кнопке показывается short_name.
    Если short_name нет, на кнопке показывается полное название анализа.
    """
    rows = []
    candidate_limits = candidate_limits or {}

    for item_index, item in enumerate(matched_items):
        if item.status != "ambiguous":
            continue

        rows.append([
            InlineKeyboardButton(
                text=f"Выберите вариант для пункта {item_index + 1}",
                callback_data="noop",
            )
        ])

        visible_limit = int(candidate_limits.get(str(item_index), DEFAULT_VISIBLE_CANDIDATES))
        visible_limit = max(DEFAULT_VISIBLE_CANDIDATES, min(visible_limit, MAX_VISIBLE_CANDIDATES))
        visible_candidates = item.candidates[:visible_limit]

        for candidate_index, candidate in enumerate(visible_candidates):
            # Если в price_list.csv заполнена колонка short_name, используем ее.
            # Если short_name пустой, выводим полное название анализа.
            label = candidate.button_text
            text = f"{candidate_index + 1}. {label} — {candidate.price} ₽"
            rows.append([
                InlineKeyboardButton(
                    text=text,
                    callback_data=f"choose:{item_index}:{candidate_index}",
                )
            ])

        if len(item.candidates) > visible_limit:
            rows.append([
                InlineKeyboardButton(
                    text=f"Показать еще ({len(item.candidates) - visible_limit})",
                    callback_data=f"show_more:{item_index}",
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
