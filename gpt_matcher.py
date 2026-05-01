import re
from pathlib import Path
from typing import Iterable, Any

from openai import OpenAI
from openpyxl import load_workbook

from config import OPENAI_API_KEY, OPENAI_TEXT_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)
MODEL = OPENAI_TEXT_MODEL or "gpt-4.1-nano"


SYSTEM_PROMPT = """
Ты — опытный медицинский ассистент лаборатории.

Твоя задача: из текста пациента составить корректный список лабораторных анализов и комплексов СТРОГО по переданному каталогу.

Каталог состоит из:
1) ОДИНОЧНЫХ АНАЛИЗОВ.
2) КОМПЛЕКСОВ, где после названия комплекса указан его состав.

ВАЖНЫЕ ПРАВИЛА:
1. Удаляй мусор и не-лабораторные исследования: консультации, приемы врачей, УЗИ, ЭКГ, рентген, МРТ, КТ, ФГДС, колоноскопия, процедуры, вводные слова.
2. Если это похоже на лабораторный анализ, но его нет в каталоге: верни "<исходный запрос> — требует уточнения у администратора".
3. Не дублируй.
4. Если пациент перечислил анализы, входящие в один комплекс, выбери комплекс. Например: ПТ + МНО + АЧТВ + ТВ + ФГ => Коагулограмма, скрининг.
5. Если пациент запросил комплекс и отдельно анализ, который входит в него, верни отдельный анализ как "<анализ> — входит в комплекс "<название комплекса>"".
6. Дефолты:
- ОАК, общий анализ крови, клинический анализ крови => Клинический анализ крови с лейкоцитарной формулой (5DIFF) и СОЭ (венозная кровь)
- ребенок / ребёнок / из пальца / капиллярная кровь => капиллярный ОАК с лейкоформулой и СОЭ
- Коагулограмма без слова "расширенная" => Коагулограмма, скрининг
- Не выбирай расширенную коагулограмму без прямого запроса.
7. Не придумывай, не меняй названия из каталога, не пиши объяснения.
8. Ответ: только список, каждая позиция с новой строки, без нумерации.
"""


def _usage_total(usage: Any) -> int:
    if usage is None:
        return 0
    value = getattr(usage, "total_tokens", None)
    if value is not None:
        return int(value)
    try:
        return int(usage.get("total_tokens", 0))
    except Exception:
        return 0


def normalize_text(text: str) -> str:
    text = str(text or "").lower().replace("ё", "е")
    text = re.sub(r"[.,;:()\[\]{}\"'«»]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_complex_catalog(path: str = "complex_price_list.xlsx") -> list[dict]:
    file_path = Path(path)
    if not file_path.exists():
        return []

    wb = load_workbook(file_path, read_only=True, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        return []

    headers = [str(c or "").strip() for c in rows[0]]

    def idx(name: str):
        return headers.index(name) if name in headers else None

    name_i = idx("name")
    comp_i = idx("components")
    price_i = idx("price")
    term_i = idx("term_days")

    if name_i is None:
        return []

    result = []
    for row in rows[1:]:
        name = row[name_i] if name_i < len(row) else ""
        if not name:
            continue
        result.append({
            "name": str(name).strip(),
            "components": str(row[comp_i] if comp_i is not None and comp_i < len(row) and row[comp_i] else "").strip(),
            "price": str(row[price_i] if price_i is not None and price_i < len(row) and row[price_i] else "").strip(),
            "term_days": str(row[term_i] if term_i is not None and term_i < len(row) and row[term_i] else "").strip(),
        })
    return result


def build_catalog_for_gpt(price_list, max_single: int = 900, max_complex: int = 260) -> str:
    lines: list[str] = []

    lines.append("ОДИНОЧНЫЕ АНАЛИЗЫ:")
    for item in list(price_list)[:max_single]:
        name = getattr(item, "name", "")
        price = getattr(item, "price", "")
        term = getattr(item, "term_days", "")
        if name:
            lines.append(f"- {name} | цена: {price} | срок: {term}")

    complexes = load_complex_catalog()
    if complexes:
        lines.append("")
        lines.append("КОМПЛЕКСЫ:")
        for item in complexes[:max_complex]:
            line = f"- {item['name']} | цена: {item['price']} | срок: {item['term_days']}"
            if item["components"]:
                line += f" | состав: {item['components']}"
            lines.append(line)

    return "\n".join(lines)


def call_gpt(user_text: str, catalog: str):
    response = client.responses.create(
        model=MODEL,
        temperature=0,
        max_output_tokens=500,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Запрос пациента:\n{user_text}\n\nКаталог:\n{catalog}"},
        ],
    )
    return response.output_text, response.usage


def clean_gpt_output(text: str) -> list[str]:
    lines = [str(line).strip() for line in str(text or "").splitlines()]
    cleaned = []
    seen = set()

    for line in lines:
        if not line:
            continue
        line = re.sub(r"^\s*\d+[\).\-\s]+", "", line).strip()
        if not line:
            continue

        if "входит в комплекс" in line.lower():
            left = normalize_text(line.split("—")[0] if "—" in line else line.split("-")[0])
            right = normalize_text(line.split("комплекс", 1)[-1])
            if left and left in right:
                continue

        key = normalize_text(line)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(line)

    return cleaned


def enforce_business_rules(lines: list[str]) -> list[str]:
    result = []
    has_coag_screening = any("коагулограмма, скрининг" in line.lower() for line in lines)

    for line in lines:
        low = line.lower()

        if "коагулограмма" in low and "расширенная" in low and "расширенная" not in low.replace("коагулограмма, расширенная", ""):
            line = re.sub("расширенная", "скрининг", line, flags=re.IGNORECASE)

        if has_coag_screening and any(x in low for x in ["мно", "ачтв", "фибриноген", "тромбиновое", "протромбин"]):
            if "входит в комплекс" not in low and "коагулограмма" not in low:
                line = f'{line} — входит в комплекс "Коагулограмма, скрининг"'

        result.append(line)

    deduped = []
    seen = set()
    for line in result:
        key = normalize_text(line)
        if key not in seen:
            seen.add(key)
            deduped.append(line)
    return deduped


def validate_against_catalog(lines: list[str], catalog_names: Iterable[str]) -> list[str]:
    catalog_names = [str(x) for x in catalog_names if str(x).strip()]
    catalog_norm = {normalize_text(x): x for x in catalog_names}

    complex_names = [x["name"] for x in load_complex_catalog()]
    complex_norm = {normalize_text(x): x for x in complex_names}

    validated = []

    for line in lines:
        low = line.lower()

        if "входит в комплекс" in low or "требует уточнения" in low:
            validated.append(line)
            continue

        norm = normalize_text(line)

        if norm in catalog_norm:
            validated.append(catalog_norm[norm])
            continue

        if norm in complex_norm:
            validated.append(complex_norm[norm])
            continue

        found = None
        for n_norm, original in {**catalog_norm, **complex_norm}.items():
            if n_norm and n_norm in norm:
                found = original
                break

        if found:
            validated.append(found)
        else:
            validated.append(f"{line} — требует уточнения у администратора")

    return validated


def gpt_match(user_text: str, price_list, catalog_names: list[str]):
    catalog = build_catalog_for_gpt(price_list)
    raw, usage = call_gpt(user_text, catalog)

    cleaned = clean_gpt_output(raw)
    business_fixed = enforce_business_rules(cleaned)

    all_catalog_names = list(catalog_names) + [x["name"] for x in load_complex_catalog()]
    validated = validate_against_catalog(business_fixed, all_catalog_names)

    return validated, usage, _usage_total(usage)
