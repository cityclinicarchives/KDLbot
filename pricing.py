import csv
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from rapidfuzz import fuzz


# Старый словарь приоритетной фильтрации оставлен как резервный механизм.
# Основная новая логика работает через alias_groups.csv + колонки group/priority в price_list.csv.
PRIORITY_SYNONYMS = {
    "оак": ["клинический анализ крови", "общий анализ крови"],
    "оам": ["общий анализ мочи"],
    "д димер": ["д димер", "д-димер"],
}


@dataclass
class LabTest:
    code: str
    name: str
    result_type: str
    max_discount: int
    term_days: str
    price: int
    synonyms: list[str]
    short_name: str = ""
    group: str = ""
    priority: int = 0


@dataclass
class CandidateItem:
    matched_name: str
    code: str
    result_type: str
    term_days: str
    price: int
    max_discount: int
    confidence: int
    short_name: str = ""
    group: str = ""
    priority: int = 0

    @property
    def button_text(self) -> str:
        return self.short_name or self.matched_name


@dataclass
class MatchedItem:
    input_name: str
    status: str  # found | not_found
    matched_name: str | None = None
    code: str | None = None
    result_type: str | None = None
    term_days: str | None = None
    price: int | None = None
    max_discount: int | None = None
    confidence: int = 0
    short_name: str | None = None
    group: str | None = None
    priority: int = 0
    candidates: list[CandidateItem] = field(default_factory=list)

    @property
    def button_text(self) -> str:
        return self.short_name or self.matched_name or self.input_name


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


@lru_cache(maxsize=1)
def load_alias_groups(path: str = "alias_groups.csv") -> dict[str, str]:
    """
    Читает файл alias_groups.csv.

    Формат:
    alias,group
    оак,cbc
    общий анализ крови,cbc
    """
    file_path = Path(path)
    if not file_path.exists():
        return {}

    aliases: dict[str, str] = {}
    with file_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "alias" not in reader.fieldnames or "group" not in reader.fieldnames:
            return {}

        for row in reader:
            alias = normalize_text(row.get("alias", ""))
            group = str(row.get("group", "")).strip()
            if alias and group:
                aliases[alias] = group

    return aliases


def load_price_list(path: str = "price_list.csv") -> list[LabTest]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Не найден файл прайса: {path}")

    items: list[LabTest] = []
    with file_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"code", "name", "result_type", "max_discount", "term_days", "price", "synonyms"}
        # short_name, group и priority — необязательные колонки.
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
                    short_name=str(row.get("short_name", "")).strip(),
                    group=str(row.get("group", "")).strip(),
                    priority=parse_int(row.get("priority"), default=0),
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
        short_name=item.short_name,
        group=item.group,
        priority=item.priority,
    )


def candidate_to_matched(input_name: str, candidate: CandidateItem, candidates: list[CandidateItem] | None = None) -> MatchedItem:
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
        short_name=candidate.short_name,
        group=candidate.group,
        priority=candidate.priority,
        candidates=candidates or [],
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


def _sort_group_items(items: list[LabTest]) -> list[LabTest]:
    return sorted(items, key=lambda item: (-item.priority, item.name))


def _find_by_alias_group(query: str, price_list: list[LabTest], max_candidates: int) -> list[CandidateItem]:
    alias_groups = load_alias_groups()
    group = alias_groups.get(query)
    if not group:
        return []

    group_items = [item for item in price_list if item.group == group]

    # Внутри ручной группы ранжируем не просто по имени, а по близости к запросу.
    # Это важно для аллергенов: запрос «мука пшеничная» должен поставить
    # «Мука пшеничная IgE» выше остальных видов муки, а «молоко» —
    # «Молоко коровье IgE» выше белковых компонентов молока.
    def group_sort_key(item: LabTest):
        item_name = normalize_text(item.name)
        item_button = normalize_text(item.short_name)
        score = get_score_for_item(query, query, item)
        starts_with_query = 1 if (item_name.startswith(query) or item_button.startswith(query)) else 0
        exact_source = 1 if query == item_name or query == item_button else 0
        component_penalty = 1 if "аллергокомпонент" in item_name else 0
        return (-exact_source, -starts_with_query, -score, component_penalty, -item.priority, item.name)

    group_items = sorted(group_items, key=group_sort_key)[:max_candidates]

    # Для group-match confidence считаем по запросу, но не ниже 100 для точной ручной связки.
    # В интерфейсе confidence сейчас не показывается, но значение полезно для отладки.
    return [labtest_to_candidate(item, max(100, get_score_for_item(query, query, item))) for item in group_items]


def find_candidate_tests(
    input_name: str,
    price_list: list[LabTest],
    min_score: int = 80,
    close_gap: int = 8,
    max_candidates: int = 8,
) -> list[CandidateItem]:
    """
    Ищет несколько близких вариантов.

    Новая логика:
    1. Если запрос есть в alias_groups.csv, НЕ делаем fuzzy-поиск по всему прайсу.
       Сразу возвращаем анализы из соответствующей группы, отсортированные по priority.
    2. Если группы нет, используем прежний fuzzy-поиск.
    """
    query = normalize_text(input_name)
    raw_query = str(input_name or "").strip().lower()

    if not query and not raw_query:
        return []

    group_candidates = _find_by_alias_group(query, price_list, max_candidates=max_candidates)
    if group_candidates:
        return group_candidates

    priority_keywords = PRIORITY_SYNONYMS.get(query)

    if priority_keywords:
        filtered_price_list: list[LabTest] = []

        for item in price_list:
            combined_text = normalize_text(item.name + " " + " ".join(item.synonyms))
            if any(keyword in combined_text for keyword in priority_keywords):
                filtered_price_list.append(item)

        if filtered_price_list:
            price_list = filtered_price_list

    scored: list[tuple[int, LabTest]] = []
    for item in price_list:
        score = get_score_for_item(query, raw_query, item)
        if score >= min_score:
            scored.append((score, item))

    if not scored:
        return []

    # Сначала score, затем priority, затем название.
    scored.sort(key=lambda pair: (-pair[0], -pair[1].priority, pair[1].name))
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

    # В новой логике бот сразу выбирает лучший вариант.
    # Остальные кандидаты сохраняются внутри пункта и используются при кнопке "Изменить часть".
    return candidate_to_matched(input_name, candidates[0], candidates=candidates)


def match_tests(user_tests: list[str], price_list: list[LabTest]) -> list[MatchedItem]:
    return [match_one_test(test, price_list) for test in user_tests]


def has_ambiguous_items(items: list[MatchedItem]) -> bool:
    # Оставлено для обратной совместимости со старым кодом.
    return False


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
        "short_name": item.short_name or "",
        "result_type": item.result_type or "не указано",
        "term_days": item.term_days or "не указано",
        "base_price": item.price,
        "max_discount": item.max_discount,
        "discount_used": discount_used,
        "final_price": final_price,
    }
