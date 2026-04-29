import csv
import re
from dataclasses import dataclass, field
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
class CandidateItem:
    matched_name: str
    code: str
    result_type: str
    term_days: str
    price: int
    max_discount: int
    confidence: int


@dataclass
class MatchedItem:
    input_name: str
    status: str  # found | ambiguous | not_found
    matched_name: str | None = None
    code: str | None = None
    result_type: str | None = None
    term_days: str | None = None
    price: int | None = None
    max_discount: int | None = None
    confidence: int = 0
    candidates: list[CandidateItem] = field(default_factory=list)


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

            synonyms_raw = str(row.get("synonyms", ""))
            synonyms = [s.strip() for s in re.split(r"[;,]", synonyms_raw) if s.strip()]
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


def labtest_to_candidate(item: LabTest, confidence: int) -> CandidateItem:
    return CandidateItem(
        matched_name=item.name,
        code=item.code,
        result_type=item.result_type,
        term_days=item.term_days,
        price=item.price,
        max_discount=item.max_discount,
        confidence=confidence,
    )


def candidate_to_matched(input_name: str, candidate: CandidateItem) -> MatchedItem:
    return MatchedItem(
        input_name=input_name,
        status="found",
        matched_name=candidate.matched_name,
        code=candidate.code,
        result_type=candidate.result_type,
        term_days=candidate.term_days,
        price=candidate.price,
        max_discount=candidate.max_discount,
        confidence=candidate.confidence,
    )


def get_score_for_item(query: str, raw_query: str, item: LabTest) -> int:
    """Возвращает лучший score для одного анализа по коду, названию и синонимам."""
    if raw_query and (raw_query == item.code.lower() or query == normalize_text(item.code)):
        return 100

    best_score = 0
    variants = [item.name] + item.synonyms + [item.code]
    for variant in variants:
        if not variant:
            continue
        normalized_variant = normalize_text(variant)
        if query == normalized_variant:
            score = 100
        elif query and query in normalized_variant:
            score = max(fuzz.WRatio(query, normalized_variant), 92)
        else:
            score = fuzz.WRatio(query, normalized_variant)
        best_score = max(best_score, int(round(score)))
    return best_score


def find_candidate_tests(
    input_name: str,
    price_list: list[LabTest],
    min_score: int = 78,
    close_gap: int = 8,
    max_candidates: int = 5,
) -> list[CandidateItem]:
    """
    Ищет несколько близких вариантов.

    Логика:
    1. Считаем score для каждого анализа.
    2. Берем лучший score.
    3. Возвращаем все варианты, которые не хуже лучшего больше чем на close_gap.
    4. Ограничиваем список max_candidates, чтобы не перегружать пациента.
    """
    query = normalize_text(input_name)
    raw_query = str(input_name or "").strip().lower()

    if not query and not raw_query:
        return []

    scored: list[tuple[int, LabTest]] = []
    for item in price_list:
        score = get_score_for_item(query, raw_query, item)
        if score >= min_score:
            scored.append((score, item))

    if not scored:
        return []

    scored.sort(key=lambda pair: (-pair[0], pair[1].name))
    best_score = scored[0][0]

    close_items = [
        (score, item)
        for score, item in scored
        if best_score - score <= close_gap
    ][:max_candidates]

    return [labtest_to_candidate(item, score) for score, item in close_items]


def match_one_test(input_name: str, price_list: list[LabTest]) -> MatchedItem:
    candidates = find_candidate_tests(input_name, price_list)

    if not candidates:
        # Для диагностики сохраняем лучший score, даже если ниже порога.
        query = normalize_text(input_name)
        raw_query = str(input_name or "").strip().lower()
        best_score = 0
        for item in price_list:
            best_score = max(best_score, get_score_for_item(query, raw_query, item))
        return MatchedItem(input_name=input_name, status="not_found", confidence=best_score)

    if len(candidates) == 1:
        return candidate_to_matched(input_name, candidates[0])

    return MatchedItem(
        input_name=input_name,
        status="ambiguous",
        confidence=candidates[0].confidence,
        candidates=candidates,
    )


def match_tests(user_tests: list[str], price_list: list[LabTest]) -> list[MatchedItem]:
    return [match_one_test(test, price_list) for test in user_tests]


def has_ambiguous_items(items: list[MatchedItem]) -> bool:
    return any(item.status == "ambiguous" for item in items)


def calculate_item_price(item: MatchedItem, patient_discount: int) -> dict:
    if item.status != "found" or item.price is None or item.max_discount is None:
        return {
            "input_name": item.input_name,
            "status": item.status,
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
