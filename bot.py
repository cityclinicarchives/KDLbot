import asyncio
import logging
import re
from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import Message, CallbackQuery, ReplyKeyboardRemove

from config import BOT_TOKEN, LAB_CHAT_ID
from pricing import (
    load_price_list,
    split_user_tests,
    match_tests,
    match_one_test,
    calculate_item_price,
    candidate_to_matched,
    learn_group_priority,
)
from order_service import (
    generate_order_id,
    format_preview_message,
    format_candidate_message,
    format_final_calculation,
    format_order_for_lab,
)
from keyboards import (
    review_list_keyboard,
    candidate_keyboard,
    discount_keyboard,
    send_order_keyboard,
    contact_keyboard,
    DEFAULT_VISIBLE_CANDIDATES,
    MAX_VISIBLE_CANDIDATES,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()
price_list = load_price_list()
logger.info("Прайс-лист загружен: %s позиций", len(price_list))


class OrderFlow(StatesGroup):
    waiting_for_tests = State()
    reviewing_list = State()
    waiting_for_edit_numbers = State()
    choosing_replacement = State()
    waiting_for_manual_refine = State()
    waiting_for_delete_numbers = State()
    waiting_for_discount = State()
    waiting_for_send_confirmation = State()
    waiting_for_name = State()
    waiting_for_contact = State()


def parse_numbers(text: str, max_number: int) -> list[int]:
    """
    Превращает строку вида "1, 2, 5" в индексы [0, 1, 4].
    Некорректные номера игнорируются.
    """
    numbers = []
    for part in re.split(r"[,\s;]+", str(text or "").strip()):
        if not part:
            continue
        if not part.isdigit():
            continue
        number = int(part)
        if 1 <= number <= max_number:
            index = number - 1
            if index not in numbers:
                numbers.append(index)
    return numbers


async def show_review(message_or_callback_message, state: FSMContext):
    data = await state.get_data()
    matched_items = data.get("matched_items", [])
    await state.set_state(OrderFlow.reviewing_list)
    await message_or_callback_message.answer(
        format_preview_message(matched_items),
        reply_markup=review_list_keyboard(),
    )


async def show_current_edit_candidate(message_or_callback_message, state: FSMContext):
    data = await state.get_data()
    matched_items = data.get("matched_items", [])
    edit_indices = data.get("edit_indices", [])
    edit_position = data.get("edit_position", 0)
    candidate_limits = data.get("candidate_limits", {})

    if edit_position >= len(edit_indices):
        await state.update_data(edit_indices=[], edit_position=0, candidate_limits={})
        await show_review(message_or_callback_message, state)
        return

    item_index = edit_indices[edit_position]
    if item_index < 0 or item_index >= len(matched_items):
        await state.update_data(edit_position=edit_position + 1)
        await show_current_edit_candidate(message_or_callback_message, state)
        return

    item = matched_items[item_index]

    # Если по пункту нет вариантов, пробуем поискать заново по исходному названию.
    if not item.candidates:
        refreshed = match_one_test(item.input_name, price_list)
        matched_items[item_index] = refreshed
        item = refreshed
        await state.update_data(matched_items=matched_items)

    if not item.candidates:
        await message_or_callback_message.answer(
            f"Для пункта {item_index + 1} не удалось подобрать варианты. "
            f"Нажмите «Уточнить вручную» и напишите анализ другими словами.",
            reply_markup=candidate_keyboard(item_index, item, visible_limit=DEFAULT_VISIBLE_CANDIDATES),
        )
        return

    visible_limit = int(candidate_limits.get(str(item_index), DEFAULT_VISIBLE_CANDIDATES))
    visible_limit = max(DEFAULT_VISIBLE_CANDIDATES, min(visible_limit, MAX_VISIBLE_CANDIDATES))

    await state.set_state(OrderFlow.choosing_replacement)
    await message_or_callback_message.answer(
        format_candidate_message(item_index, item, visible_limit=visible_limit),
        reply_markup=candidate_keyboard(item_index, item, visible_limit=visible_limit),
    )


@dp.message(Command("start"))
async def cmd_start(message: Message, state: FSMContext):
    await state.clear()
    await state.set_state(OrderFlow.waiting_for_tests)
    await message.answer(
        "Здравствуйте! Отправьте мне список анализов текстом.\n\n"
        "Например:\n"
        "ОАК\n"
        "ТТГ\n"
        "витамин Д\n\n"
        "Можно также указать код анализа из прайс-листа."
    )


@dp.message(Command("chatid"))
async def cmd_chatid(message: Message):
    await message.answer(f"chat_id этого чата: {message.chat.id}")


@dp.message(Command("reset"))
async def cmd_reset(message: Message, state: FSMContext):
    await state.clear()
    await state.set_state(OrderFlow.waiting_for_tests)
    await message.answer("Диалог сброшен. Отправьте список анализов текстом.")


@dp.message(StateFilter(None), F.text)
async def any_text_starts_order(message: Message, state: FSMContext):
    user_tests = split_user_tests(message.text)
    if not user_tests:
        await state.set_state(OrderFlow.waiting_for_tests)
        await message.answer("Отправьте список анализов текстом, каждый анализ с новой строки.")
        return

    matched_items = match_tests(user_tests, price_list)
    await state.update_data(matched_items=matched_items, candidate_limits={})
    await show_review(message, state)


@dp.callback_query(F.data == "restart")
async def restart(callback: CallbackQuery, state: FSMContext):
    await state.clear()
    await state.set_state(OrderFlow.waiting_for_tests)
    await callback.message.answer("Начнем заново. Отправьте список анализов текстом.")
    await callback.answer()


@dp.message(OrderFlow.waiting_for_tests)
async def receive_tests(message: Message, state: FSMContext):
    if not message.text:
        await message.answer("Пока на этом этапе я принимаю список анализов только текстом.")
        return

    user_tests = split_user_tests(message.text)
    if not user_tests:
        await message.answer("Не смог найти список анализов. Отправьте каждый анализ с новой строки.")
        return

    matched_items = match_tests(user_tests, price_list)
    await state.update_data(matched_items=matched_items, candidate_limits={})
    await show_review(message, state)


@dp.callback_query(F.data == "back_to_review")
async def back_to_review(callback: CallbackQuery, state: FSMContext):
    await state.update_data(edit_indices=[], edit_position=0, candidate_limits={})
    await show_review(callback.message, state)
    await callback.answer()


@dp.callback_query(F.data == "edit_part")
async def ask_edit_numbers(callback: CallbackQuery, state: FSMContext):
    await state.set_state(OrderFlow.waiting_for_edit_numbers)
    await callback.message.answer(
        "Какие пункты нужно изменить? Напишите номера через запятую.\n\n"
        "Например: 1, 2, 5"
    )
    await callback.answer()


@dp.message(OrderFlow.waiting_for_edit_numbers)
async def receive_edit_numbers(message: Message, state: FSMContext):
    data = await state.get_data()
    matched_items = data.get("matched_items", [])

    indices = parse_numbers(message.text, max_number=len(matched_items))
    if not indices:
        await message.answer(
            "Я не понял номера пунктов. Напишите, например:\n\n"
            "1, 2, 5"
        )
        return

    await state.update_data(edit_indices=indices, edit_position=0, candidate_limits={})
    await show_current_edit_candidate(message, state)


@dp.callback_query(F.data.startswith("show_more_edit:"))
async def show_more_edit(callback: CallbackQuery, state: FSMContext):
    data_parts = callback.data.split(":")
    if len(data_parts) != 2:
        await callback.answer("Не удалось обработать запрос", show_alert=True)
        return

    item_index = int(data_parts[1])
    data = await state.get_data()
    candidate_limits = data.get("candidate_limits", {})
    candidate_limits[str(item_index)] = MAX_VISIBLE_CANDIDATES
    await state.update_data(candidate_limits=candidate_limits)

    await callback.answer("Показываю еще варианты")
    await show_current_edit_candidate(callback.message, state)


@dp.callback_query(F.data.startswith("choose_edit:"))
async def choose_edit_candidate(callback: CallbackQuery, state: FSMContext):
    data_parts = callback.data.split(":")
    if len(data_parts) != 3:
        await callback.answer("Не удалось обработать выбор", show_alert=True)
        return

    item_index = int(data_parts[1])
    candidate_index = int(data_parts[2])

    data = await state.get_data()
    matched_items = data.get("matched_items", [])
    edit_indices = data.get("edit_indices", [])
    edit_position = data.get("edit_position", 0)
    candidate_limits = data.get("candidate_limits", {})

    if item_index < 0 or item_index >= len(matched_items):
        await callback.answer("Пункт не найден", show_alert=True)
        return

    item = matched_items[item_index]
    if candidate_index < 0 or candidate_index >= len(item.candidates):
        await callback.answer("Вариант не найден", show_alert=True)
        return

    selected_candidate = item.candidates[candidate_index]
    matched_items[item_index] = candidate_to_matched(
        item.input_name,
        selected_candidate,
        candidates=item.candidates,
    )

    candidate_limits.pop(str(item_index), None)

    await state.update_data(
        matched_items=matched_items,
        edit_position=edit_position + 1,
        candidate_limits=candidate_limits,
    )

    if getattr(item, "active_group", "") and selected_candidate.code:
        learned = learn_group_priority(item.active_group, selected_candidate.code)
        if learned:
            logger.info("Updated learned priorities for group %s after selecting %s", item.active_group, selected_candidate.code)

    await callback.message.answer(
        f"Пункт {item_index + 1} обновлен: {selected_candidate.matched_name}"
    )

    if edit_position + 1 >= len(edit_indices):
        await state.update_data(edit_indices=[], edit_position=0, candidate_limits={})
        await show_review(callback.message, state)
    else:
        await show_current_edit_candidate(callback.message, state)

    await callback.answer()


@dp.callback_query(F.data.startswith("manual_refine:"))
async def manual_refine(callback: CallbackQuery, state: FSMContext):
    data_parts = callback.data.split(":")
    if len(data_parts) != 2:
        await callback.answer("Не удалось обработать запрос", show_alert=True)
        return

    item_index = int(data_parts[1])
    await state.update_data(manual_refine_index=item_index)
    await state.set_state(OrderFlow.waiting_for_manual_refine)

    await callback.message.answer(
        f"Напишите новую формулировку для пункта {item_index + 1}.\n\n"
        "Например: клинический анализ крови из пальца с лейкоформулой"
    )
    await callback.answer()


@dp.message(OrderFlow.waiting_for_manual_refine)
async def receive_manual_refine(message: Message, state: FSMContext):
    if not message.text or len(message.text.strip()) < 2:
        await message.answer("Пожалуйста, напишите новую формулировку анализа текстом.")
        return

    data = await state.get_data()
    matched_items = data.get("matched_items", [])
    item_index = data.get("manual_refine_index")

    if item_index is None or item_index < 0 or item_index >= len(matched_items):
        await message.answer("Не удалось определить пункт для уточнения. Вернемся к списку.")
        await show_review(message, state)
        return

    new_name = message.text.strip()
    new_match = match_one_test(new_name, price_list)
    matched_items[item_index] = new_match

    # Оставляем пользователя в процессе выбора этого же пункта,
    # чтобы он увидел варианты по новой формулировке.
    data_edit_indices = data.get("edit_indices", [])
    edit_position = data.get("edit_position", 0)
    if not data_edit_indices:
        data_edit_indices = [item_index]
        edit_position = 0

    await state.update_data(
        matched_items=matched_items,
        edit_indices=data_edit_indices,
        edit_position=edit_position,
        manual_refine_index=None,
        candidate_limits={},
    )
    await show_current_edit_candidate(message, state)


@dp.callback_query(F.data == "delete_part")
async def ask_delete_numbers(callback: CallbackQuery, state: FSMContext):
    await state.set_state(OrderFlow.waiting_for_delete_numbers)
    await callback.message.answer(
        "Какие пункты нужно удалить? Напишите номера через запятую.\n\n"
        "Например: 3 или 2, 4"
    )
    await callback.answer()


@dp.message(OrderFlow.waiting_for_delete_numbers)
async def receive_delete_numbers(message: Message, state: FSMContext):
    data = await state.get_data()
    matched_items = data.get("matched_items", [])

    indices = parse_numbers(message.text, max_number=len(matched_items))
    if not indices:
        await message.answer(
            "Я не понял номера пунктов. Напишите, например:\n\n"
            "3"
        )
        return

    for index in sorted(indices, reverse=True):
        del matched_items[index]

    await state.update_data(matched_items=matched_items, candidate_limits={})

    if not matched_items:
        await state.clear()
        await state.set_state(OrderFlow.waiting_for_tests)
        await message.answer("Список стал пустым. Отправьте новый список анализов текстом.")
        return

    await show_review(message, state)


@dp.message(OrderFlow.reviewing_list)
async def review_text_fallback(message: Message):
    await message.answer(
        "Используйте кнопки под списком: «Изменить часть», «Удалить часть», «Все верно» или «Начать заново»."
    )


@dp.callback_query(F.data == "list_ok")
async def list_ok(callback: CallbackQuery, state: FSMContext):
    await state.set_state(OrderFlow.waiting_for_discount)
    await callback.message.answer("Выберите скидку по карте лояльности:", reply_markup=discount_keyboard())
    await callback.answer()


@dp.callback_query(F.data.startswith("discount:"))
async def choose_discount(callback: CallbackQuery, state: FSMContext):
    patient_discount = int(callback.data.split(":")[1])
    data = await state.get_data()
    matched_items = data.get("matched_items", [])

    calculated_items = [calculate_item_price(item, patient_discount) for item in matched_items]
    await state.update_data(patient_discount=patient_discount, calculated_items=calculated_items)
    await state.set_state(OrderFlow.waiting_for_send_confirmation)

    await callback.message.answer(
        format_final_calculation(calculated_items, patient_discount),
        reply_markup=send_order_keyboard(),
    )
    await callback.answer()


@dp.callback_query(F.data == "send_order")
async def ask_patient_name(callback: CallbackQuery, state: FSMContext):
    await state.set_state(OrderFlow.waiting_for_name)
    await callback.message.answer("Введите ваше имя:")
    await callback.answer()


@dp.message(OrderFlow.waiting_for_name)
async def receive_name(message: Message, state: FSMContext):
    if not message.text or len(message.text.strip()) < 2:
        await message.answer("Пожалуйста, введите имя текстом.")
        return

    await state.update_data(patient_name=message.text.strip())
    await state.set_state(OrderFlow.waiting_for_contact)
    await message.answer("Теперь поделитесь номером телефона, нажав кнопку ниже.", reply_markup=contact_keyboard())


@dp.message(OrderFlow.waiting_for_contact, F.contact)
async def receive_contact(message: Message, state: FSMContext):
    phone = message.contact.phone_number
    data = await state.get_data()

    order_id = generate_order_id()
    patient_name = data.get("patient_name", "Не указано")
    patient_discount = data.get("patient_discount", 0)
    calculated_items = data.get("calculated_items", [])

    order_text = format_order_for_lab(order_id, patient_name, phone, patient_discount, calculated_items)

    await message.answer("Ваш заказ принят.\n\n" + order_text, reply_markup=ReplyKeyboardRemove())

    if LAB_CHAT_ID != 0:
        await bot.send_message(LAB_CHAT_ID, order_text)
    else:
        await message.answer(
            "Внимание: LAB_CHAT_ID пока равен 0, поэтому заказ не отправлен в чат лаборатории. "
            "Настройте LAB_CHAT_ID в .env или Railway Variables."
        )

    await state.clear()


@dp.message(OrderFlow.waiting_for_contact)
async def contact_error(message: Message):
    await message.answer(
        "Пожалуйста, нажмите кнопку «Поделиться номером телефона». Это нужно для оформления заказа.",
        reply_markup=contact_keyboard(),
    )


@dp.message()
async def fallback(message: Message, state: FSMContext):
    await state.set_state(OrderFlow.waiting_for_tests)
    await message.answer(
        "Я готов принять список анализов. Отправьте его текстом, каждый анализ с новой строки."
    )


async def main():
    logger.info("Бот запускается в режиме polling")
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
