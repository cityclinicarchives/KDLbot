from datetime import datetime
import random


def generate_order_id() -> str:
    date_part = datetime.now().strftime("%Y%m%d")
    random_part = random.randint(100000, 999999)
    return f"LAB-{date_part}-{random_part}"


def format_preview_message(items) -> str:
    lines = ["Я нашел следующие анализы:\n"]

    has_ambiguous = False

    for index, item in enumerate(items, start=1):
        if item.status == "found":
            lines.append(
                f"{index}. {item.matched_name} — {item.price} ₽\n"
                f"   Код: {item.code}\n"
                f"   Тип результата: {item.result_type or 'не указано'}\n"
                f"   Срок выполнения: {item.term_days or 'не указано'} дн.\n"
                f"   Уверенность поиска: {item.confidence}%"
            )
        elif item.status == "ambiguous":
            has_ambiguous = True
            lines.append(f"{index}. По запросу «{item.input_name}» найдено несколько вариантов:")
            for candidate_index, candidate in enumerate(item.candidates, start=1):
                lines.append(
                    f"   {candidate_index}) {candidate.matched_name} — {candidate.price} ₽\n"
                    f"      Код: {candidate.code}; тип: {candidate.result_type or 'не указано'}; "
                    f"срок: {candidate.term_days or 'не указано'} дн.; уверенность: {candidate.confidence}%"
                )
            lines.append("   Выберите нужный вариант кнопкой ниже.")
        else:
            lines.append(
                f"{index}. {item.input_name} — требует уточнения у администратора"
            )

    if has_ambiguous:
        lines.append("\nПеред продолжением выберите варианты для неоднозначных пунктов.")
    else:
        lines.append("\nПроверьте список. Если все верно, нажмите кнопку «Все верно».")

    lines.append("Если есть ошибка, напишите, например:")
    lines.append("2: ферритин")

    return "\n".join(lines)


def format_final_calculation(calculated_items, patient_discount: int) -> str:
    total = 0
    economy = 0

    lines = [f"Расчет заказа со скидкой {patient_discount}%:\n"]

    for index, item in enumerate(calculated_items, start=1):
        if item["status"] == "found":
            total += item["final_price"]
            economy += item["base_price"] - item["final_price"]

            lines.append(
                f"{index}. {item['name']} — {item['final_price']} ₽\n"
                f"   Код: {item['code']}\n"
                f"   Тип результата: {item['result_type']}\n"
                f"   Срок выполнения: {item['term_days']} дн.\n"
                f"   Цена без скидки: {item['base_price']} ₽\n"
                f"   Максимальная скидка: {item['max_discount']}%\n"
                f"   Примененная скидка: {item['discount_used']}%"
            )
        else:
            lines.append(
                f"{index}. {item['input_name']} — наличие и цену нужно уточнить у администратора"
            )

    lines.append(f"\nИтого: {total} ₽")
    lines.append(f"Экономия: {economy} ₽")

    return "\n".join(lines)


def format_order_for_lab(order_id: str, patient_name: str, phone: str, patient_discount: int, calculated_items) -> str:
    total = sum(item["final_price"] for item in calculated_items if item["status"] == "found")

    lines = [
        "Новый заказ из Telegram-бота",
        "",
        f"Код заказа: {order_id}",
        f"Пациент: {patient_name}",
        f"Телефон: {phone}",
        f"Скидка пациента: {patient_discount}%",
        "",
        "Анализы:",
    ]

    for index, item in enumerate(calculated_items, start=1):
        if item["status"] == "found":
            lines.append(
                f"{index}. {item['code']} — {item['name']} — {item['final_price']} ₽ "
                f"(скидка {item['discount_used']}%, тип: {item['result_type']}, срок: {item['term_days']} дн.)"
            )
        else:
            lines.append(
                f"{index}. НЕ НАЙДЕНО — {item['input_name']} — требует уточнения"
            )

    lines.append("")
    lines.append(f"Итого: {total} ₽")

    return "\n".join(lines)
