import asyncio
import logging
import re
from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import Message, CallbackQuery, ReplyKeyboardRemove

from config import BOT_TOKEN, LAB_CHAT_ID
from pricing import load_price_list, split_user_tests, match_tests, match_one_test, calculate_item_price
from order_service import generate_order_id, format_preview_message, format_final_calculation, format_order_for_lab
from keyboards import confirm_list_keyboard, discount_keyboard, send_order_keyboard, contact_keyboard

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()
price_list = load_price_list()
logger.info("Прайс-лист загружен: %s позиций", len(price_list))


class OrderFlow(StatesGroup):
    waiting_for_tests = State()
    editing_list = State()
    waiting_for_discount = State()
    waiting_for_send_confirmation = State()
    waiting_for_name = State()
    waiting_for_contact = State()


@dp.message(Command("start"))
async def cmd_start(message: Message, state: FSMContext):
    await state.clear()
    await state.set_state(OrderFlow.waiting_for_tests)
    await message.answer(
        "Здравствуйте! Отправьте мне список анализов текстом.\n\n"
        "Например:\n"
        "ОАК\n"
        "коагулограмма\n"
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
    """Если пользователь написал список анализов без /start, запускаем сценарий."""
    user_tests = split_user_tests(message.text)
    if not user_tests:
        await state.set_state(OrderFlow.waiting_for_tests)
        await message.answer("Отправьте список анализов текстом, каждый анализ с новой строки.")
        return

    matched_items = match_tests(user_tests, price_list)
    await state.update_data(matched_items=matched_items)
    await state.set_state(OrderFlow.editing_list)
    await message.answer(format_preview_message(matched_items), reply_markup=confirm_list_keyboard())


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
    await state.update_data(matched_items=matched_items)
    await state.set_state(OrderFlow.editing_list)

    await message.answer(format_preview_message(matched_items), reply_markup=confirm_list_keyboard())


@dp.message(OrderFlow.editing_list)
async def edit_list_by_text(message: Message, state: FSMContext):
    if not message.text:
        await message.answer("Напишите исправление текстом, например: 2: ферритин")
        return

    match = re.match(r"^\s*(\d+)\s*[:.)\-]\s*(.+)$", message.text.strip())
    if not match:
        await message.answer(
            "Я не понял, какой пункт нужно исправить. Напишите так:\n\n"
            "2: коагулограмма расширенная"
        )
        return

    item_number = int(match.group(1))
    new_name = match.group(2).strip()

    data = await state.get_data()
    matched_items = data.get("matched_items", [])

    if item_number < 1 or item_number > len(matched_items):
        await message.answer(f"В списке нет пункта №{item_number}.")
        return

    matched_items[item_number - 1] = match_one_test(new_name, price_list)
    await state.update_data(matched_items=matched_items)

    await message.answer(format_preview_message(matched_items), reply_markup=confirm_list_keyboard())


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
