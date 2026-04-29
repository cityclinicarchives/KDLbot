import csv
import re
from dataclasses import dataclass
from pathlib import Path
from rapidfuzz import fuzz


@dataclass
class LabTest:
    code: str
    name: str
    result_type: str
    max_discount: int
    term_days: str
    price: int
    synonyms: list[str]


@dataclass
class MatchedItem:
    input_name: str
    status: str
    matched_name: str | None = None
    code: str | None = None
    result_type: str | None = None
    term_days: str | None = None
    price: int | None = None
    max_discount: int | None = None
    confidence: int = 0


def normalize_text(text: str) -> str:
    text = str(text or "").lower().replace("ё", "е")
    text = re.sub(r"[\s\u00a0]+", " ", text)
    text = re.sub(r"[.,;:()\[\]{}\"'«»]", " ", text)
    text = text.replace("-", " ").replace("—", " ")
    return re.sub(r"\s+", " ", text).strip()


def parse_int(value, default: int = 0) -> int:
    if value is None:
        return default
    text = str(value).strip().replace(" ", "").replace(",", ".")
    if not text:
        return default
    try:
        number = float(text)
    except ValueError:
        digits = re.sub(r"[^0-9.]", "", text)
        if not digits:
            return default
        number = float(digits)
    return int(round(number))


def parse_discount(value) -> int:
    """
    Поддерживает форматы: 25, 25%, 0.25.
    Если значение меньше 1, считаем его долей и переводим в проценты.
    """
    if value is None:
        return 0
    text = str(value).strip().replace("%", "").replace(",", ".")
    if not text:
        return 0
    try:
        number = float(text)
    except ValueError:
        return 0
    if 0 < number < 1:
        number *= 100
    return int(round(number))


def load_price_list(path: str = "price_list.csv") -> list[LabTest]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Не найден файл прайса: {path}")

    items: list[LabTest] = []
    with file_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"code", "name", "result_type", "max_discount", "term_days", "price", "synonyms"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"В price_list.csv не хватает колонок: {', '.join(sorted(missing))}")

        for row in reader:
            code = str(row.get("code", "")).strip()
            name = str(row.get("name", "")).strip()
            if not code and not name:
                continue

            synonyms = [s.strip() for s in str(row.get("synonyms", "")).split(";") if s.strip()]
            items.append(
                LabTest(
                    code=code,
                    name=name,
                    result_type=str(row.get("result_type", "")).strip(),
                    max_discount=parse_discount(row.get("max_discount")),
                    term_days=str(row.get("term_days", "")).strip(),
                    price=parse_int(row.get("price")),
                    synonyms=synonyms,
                )
            )
    return items


def split_user_tests(text: str) -> list[str]:
    raw_parts: list[str] = []
    for line in str(text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        line = line.replace(";", ",")
        raw_parts.extend(line.split(","))

    cleaned: list[str] = []
    for part in raw_parts:
        value = re.sub(r"^\s*[0-9]+[.)\-:]*\s*", "", part).strip(" -—\t")
        if value:
            cleaned.append(value)
    return cleaned


def match_one_test(input_name: str, price_list: list[LabTest]) -> MatchedItem:
    query = normalize_text(input_name)
    raw_query = str(input_name or "").strip().lower()

    best_item: LabTest | None = None
    best_score = 0

    for item in price_list:
        # 1. Точный поиск по коду анализа.
        if raw_query and raw_query == item.code.lower():
            return MatchedItem(
                input_name=input_name,
                status="found",
                matched_name=item.name,
                code=item.code,
                result_type=item.result_type,
                term_days=item.term_days,
                price=item.price,
                max_discount=item.max_discount,
                confidence=100,
            )

        # 2. Поиск по названию и синонимам.
        variants = [item.name] + item.synonyms + [item.code]
        for variant in variants:
            if not variant:
                continue
            score = fuzz.WRatio(query, normalize_text(variant))
            if score > best_score:
                best_score = score
                best_item = item

    if best_item and best_score >= 78:
        return MatchedItem(
            input_name=input_name,
            status="found",
            matched_name=best_item.name,
            code=best_item.code,
            result_type=best_item.result_type,
            term_days=best_item.term_days,
            price=best_item.price,
            max_discount=best_item.max_discount,
            confidence=best_score,
        )

    return MatchedItem(input_name=input_name, status="not_found", confidence=best_score)


def match_tests(user_tests: list[str], price_list: list[LabTest]) -> list[MatchedItem]:
    return [match_one_test(test, price_list) for test in user_tests]


def calculate_item_price(item: MatchedItem, patient_discount: int) -> dict:
    if item.status != "found" or item.price is None or item.max_discount is None:
        return {
            "input_name": item.input_name,
            "status": "not_found",
            "final_price": None,
            "discount_used": None,
        }

    discount_used = min(patient_discount, item.max_discount)
    final_price = round(item.price * (100 - discount_used) / 100)

    return {
        "input_name": item.input_name,
        "status": "found",
        "code": item.code,
        "name": item.matched_name,
        "result_type": item.result_type or "не указано",
        "term_days": item.term_days or "не указано",
        "base_price": item.price,
        "max_discount": item.max_discount,
        "discount_used": discount_used,
        "final_price": final_price,
    }
