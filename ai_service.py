
import base64
import json
import mimetypes
import os
import re
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from openai import OpenAI
from rapidfuzz import fuzz

from config import (
    OPENAI_API_KEY,
    OPENAI_TEXT_MODEL,
    OPENAI_VISION_MODEL,
    OPENAI_AUDIO_MODEL,
)
from pricing import (
    LabTest,
    MatchedItem,
    normalize_text,
    split_user_tests,
    match_tests,
    find_candidate_tests,
    labtest_to_candidate,
    make_matched_from_code,
    make_included_item,
)


MEMORY_PATH = Path("cleaner_memory.json")


@dataclass
class AIUsage:
    used: bool = False
    source: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add(self, other: "AIUsage") -> None:
        if not other:
            return
        self.used = self.used or other.used
        sources = [s for s in [self.source, other.source] if s]
        self.source = "+".join(dict.fromkeys(sources))
        self.prompt_tokens += int(other.prompt_tokens or 0)
        self.completion_tokens += int(other.completion_tokens or 0)
        self.total_tokens += int(other.total_tokens or 0)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExtractedTextResult:
    raw_text: str
    usage: AIUsage


@dataclass
class AIMatchResult:
    matched_items: list[MatchedItem]
    cleaned_items: list[str]
    raw_text: str
    usage: AIUsage
    used_gpt_matching: bool = False


def is_ai_enabled() -> bool:
    return bool(OPENAI_API_KEY)


def _client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY не задан")
    return OpenAI(api_key=OPENAI_API_KEY)


def _usage_from_chat_response(response, source: str) -> AIUsage:
    usage = getattr(response, "usage", None)
    if not usage:
        return AIUsage(used=True, source=source)
    prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion = int(getattr(usage, "completion_tokens", 0) or 0)
    total = int(getattr(usage, "total_tokens", 0) or (prompt + completion))
    return AIUsage(
        used=True,
        source=source,
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
    )


def _usage_from_audio_response(response, source: str) -> AIUsage:
    usage = getattr(response, "usage", None)
    if not usage:
        return AIUsage(used=True, source=source)
    # В разных версиях SDK usage может быть объектом с разными именами полей.
    prompt = int(getattr(usage, "input_tokens", 0) or getattr(usage, "prompt_tokens", 0) or 0)
    completion = int(getattr(usage, "output_tokens", 0) or getattr(usage, "completion_tokens", 0) or 0)
    total = int(getattr(usage, "total_tokens", 0) or (prompt + completion))
    return AIUsage(True, source, prompt, completion, total)


def _load_memory() -> dict:
    if not MEMORY_PATH.exists():
        return {"cleaned": {}, "choices": {}}
    try:
        data = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data.setdefault("cleaned", {})
            data.setdefault("choices", {})
            return data
    except Exception:
        pass
    return {"cleaned": {}, "choices": {}}


def _save_memory(data: dict) -> None:
    try:
        MEMORY_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        # На некоторых хостингах файловая система может быть временной или read-only.
        pass


def remember_manual_choice(input_text: str, selected_code: str, selected_name: str) -> None:
    key = normalize_text(input_text)
    if not key or not selected_code:
        return
    data = _load_memory()
    data["choices"][key] = {"code": selected_code, "name": selected_name}
    _save_memory(data)


FILLER_PHRASES = [
    "ну", "так", "значит", "пожалуйста", "пожалуй", "наверное", "скорее всего",
    "я хочу", "хочу", "мне нужно", "мне надо", "надо", "нужно", "сдать",
    "сделать", "проверить", "посмотреть", "анализы", "анализ", "еще", "ещё",
    "и еще", "а еще", "также", "плюс", "можно", "давайте",
]


SEARCH_EQUIVALENTS = {
    "орешник": ["лещина", "лещина обыкновенная", "лесной орех", "фундук", "corylus", "corylus avellana"],
    "лещина": ["орешник", "лещина обыкновенная", "лесной орех", "фундук", "corylus", "corylus avellana"],
    # Важно: Candida НЕ является бытовым синонимом "плесень" в запросах на аллергены.
    "плесень": ["плесневый гриб", "плесневые грибы", "alternaria", "alternaria alternata", "cladosporium", "aspergillus", "penicillium"],
    "береза": ["береза", "береза бородавчатая", "betula", "betula alba"],
    "ольха": ["ольха", "ольха серая", "alnus", "alnus incana"],
    "капрограмма": ["копрограмма", "общий анализ кала"],
    "копрограмма": ["копрограмма", "общий анализ кала"],
    "общий белок": ["общий белок"],
    "глюкоза": ["глюкоза"],
    "пт": ["протромбин", "протромбиновое"],
    "тв": ["тромбиновое время"],
    "фг": ["фибриноген"],
}

PHRASE_TO_CANONICAL = {
    "общий анализ крови": "ОАК",
    "клинический анализ крови": "ОАК",
    "анализ крови": "ОАК",
    "оак": "ОАК",
    "общий анализ мочи": "ОАМ",
    "анализ мочи": "ОАМ",
    "оам": "ОАМ",
    "моча": "ОАМ",
    "капрограмма": "копрограмма",
    "копрограмма": "копрограмма",
    "общий анализ кала": "копрограмма",
    "алт": "АЛТ",
    "аст": "АСТ",
    "ттг": "ТТГ",
    "мно": "МНО",
    "ачтв": "АЧТВ",
    "тв": "ТВ",
    "фг": "ФГ",
    "фибриноген": "Фибриноген",
    "д димер": "Д-димер",
    "д-димер": "Д-димер",
    "коагулограмма": "Коагулограмма",
    "глюкоза": "Глюкоза",
    "общий белок": "Общий белок",
    "береза": "аллерген береза",
    "берёза": "аллерген береза",
    "ольха": "аллерген ольха",
    "орешник": "аллерген лещина",
    "лещина": "аллерген лещина",
    "плесень": "аллерген плесневый гриб",
}

GENERIC_SHORTLIST_TOKENS = {
    "общий", "общая", "анализ", "анализы", "крови", "кровь", "мочи", "моча", "аллерген", "аллергены",
    "сдать", "посчитать", "пожалуйста", "мне", "нужно", "надо", "хочу", "еще", "ещё"
}

ALLERGEN_QUERY_WORDS = {
    "береза", "берёза", "ольха", "орешник", "лещина", "плесень", "плесневый", "плесневые"
}

ALLERGEN_PANEL_WORDS = ["панель", "аллергокомплекс", "комплекс"]


def clean_recognized_text_locally(text: str) -> list[str]:
    raw = str(text or "").strip()
    if not raw:
        return []

    data = _load_memory()
    memory_key = normalize_text(raw)
    if memory_key in data.get("cleaned", {}):
        stored = data["cleaned"][memory_key]
        if isinstance(stored, list) and stored:
            return [str(x).strip() for x in stored if str(x).strip()]

    norm = normalize_text(raw)

    extracted: list[str] = []
    # Сначала вытаскиваем устойчивые медицинские фразы.
    for phrase, canonical in sorted(PHRASE_TO_CANONICAL.items(), key=lambda pair: len(pair[0]), reverse=True):
        if re.search(rf"(^|\s){re.escape(normalize_text(phrase))}($|\s)", norm):
            if canonical not in extracted:
                extracted.append(canonical)

    # Потом делим остаток на части и удаляем слова-паразиты.
    separators = r"\b(?:и еще|и ещё|а еще|а ещё|также|плюс|,|;|\n|\.|\+)\b"
    parts = re.split(separators, raw, flags=re.IGNORECASE)

    for part in parts:
        value = normalize_text(part)
        for filler in sorted(FILLER_PHRASES, key=len, reverse=True):
            value = re.sub(rf"(^|\s){re.escape(normalize_text(filler))}($|\s)", " ", value)
        value = re.sub(r"\s+", " ", value).strip(" -—,.;:")

        if not value or len(value) < 2:
            continue

        canonical = PHRASE_TO_CANONICAL.get(value, value)
        if canonical not in extracted:
            extracted.append(canonical)

    # Не возвращаем слишком длинные разговорные куски как отдельный анализ.
    cleaned = []
    for item in extracted:
        tokens = normalize_text(item).split()
        if len(tokens) > 8:
            continue
        if item not in cleaned:
            cleaned.append(item)

    return cleaned


def remember_cleaned_text(raw_text: str, cleaned_items: list[str]) -> None:
    key = normalize_text(raw_text)
    if not key or not cleaned_items:
        return
    data = _load_memory()
    data["cleaned"][key] = cleaned_items[:20]
    _save_memory(data)


def _resize_image_for_vision(path: str) -> str:
    try:
        from PIL import Image
    except Exception:
        return path

    try:
        image = Image.open(path)
        image = image.convert("RGB")
        max_side = 1280
        w, h = image.size
        scale = min(max_side / max(w, h), 1.0)
        if scale < 1.0:
            image = image.resize((int(w * scale), int(h * scale)))

        fd, out_path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        image.save(out_path, "JPEG", quality=75, optimize=True)
        return out_path
    except Exception:
        return path


def _encode_image_as_data_url(path: str) -> str:
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


async def extract_tests_from_audio_file(path: str) -> ExtractedTextResult:
    client = _client()
    with open(path, "rb") as f:
        response = client.audio.transcriptions.create(
            model=OPENAI_AUDIO_MODEL,
            file=f,
        )
    raw_text = str(getattr(response, "text", "") or "").strip()
    return ExtractedTextResult(
        raw_text=raw_text,
        usage=_usage_from_audio_response(response, "audio_transcription"),
    )


async def extract_tests_from_image_file(path: str) -> ExtractedTextResult:
    client = _client()
    resized_path = _resize_image_for_vision(path)
    data_url = _encode_image_as_data_url(resized_path)

    prompt = (
        "Извлеки с изображения только текст со списком медицинских анализов. "
        "Не сопоставляй с прайсом. Не добавляй пояснения. "
        "Верни простой текст, каждый анализ или строку назначения с новой строки."
    )

    response = client.chat.completions.create(
        model=OPENAI_VISION_MODEL,
        temperature=0,
        max_tokens=400,
        messages=[
            {"role": "system", "content": "Ты OCR-модуль для медицинских назначений."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
    )
    raw_text = response.choices[0].message.content.strip()
    return ExtractedTextResult(
        raw_text=raw_text,
        usage=_usage_from_chat_response(response, "vision_ocr"),
    )


def _catalog_item(item: LabTest) -> dict[str, Any]:
    return {
        "code": item.code,
        "name": item.name,
        "price": item.price,
        "term_days": item.term_days,
        "result_type": item.result_type,
        "priority": item.priority,
        "components": item.components if item.result_type == "комплекс" else "",
    }


def _candidate_score(query: str, item: LabTest, *, complex_mode: bool = False) -> int:
    q = normalize_text(query)
    name = normalize_text(item.name)
    score = fuzz.WRatio(q, name)
    if q and q in name:
        score = max(score, 92)

    # Для комплексов состав используем только для попадания в shortlist и дедубликации.
    if complex_mode and item.components:
        comp = normalize_text(item.components)
        comp_score = fuzz.token_set_ratio(q, comp)
        shared = len(set(q.split()) & set(comp.split()))
        score = max(score, int(comp_score), min(100, shared * 12))
    return int(score)


def _is_likely_allergen_query(text: str) -> bool:
    qn = normalize_text(text)
    return any(word in qn for word in ALLERGEN_QUERY_WORDS)


def _is_panel_or_complex_allergen(item: LabTest) -> bool:
    name = normalize_text(item.name)
    return any(word in name for word in ALLERGEN_PANEL_WORDS)


def _has_meaningful_token_overlap(query_text: str, item_name: str) -> bool:
    q_tokens = {t for t in normalize_text(query_text).split() if len(t) >= 4 and t not in GENERIC_SHORTLIST_TOKENS}
    if not q_tokens:
        return False
    item_norm = normalize_text(item_name)
    return any(tok in item_norm for tok in q_tokens)


def _force_item_by_name_contains(price_list: list[LabTest], scored: dict[str, tuple[int, LabTest]], words: list[str], *, score: int = 100, exclude_panels: bool = False, prefer_cheaper: bool = True) -> None:
    matches = []
    for item in price_list:
        name_norm = normalize_text(item.name)
        if all(w in name_norm for w in words):
            if exclude_panels and _is_panel_or_complex_allergen(item):
                continue
            matches.append(item)
    if not matches:
        return
    # Для аллергенов обычно в прайсе есть 2 версии: обычная ИФА дешевле и ImmunoCAP дороже.
    # По умолчанию пациент ожидает базовый индивидуальный аллерген, поэтому ставим более дешевый выше.
    matches.sort(key=lambda x: (x.price if prefer_cheaper else -x.price, x.name))
    for item in matches[:5]:
        scored[item.code] = (score, item)


def _build_candidate_catalog(raw_text: str, cleaned_items: list[str], price_list: list[LabTest]) -> dict[str, list[dict[str, Any]]]:
    queries = [raw_text] + cleaned_items
    expanded_queries = list(queries)
    for q in queries:
        qn = normalize_text(q)
        for key, values in SEARCH_EQUIVALENTS.items():
            if key in qn:
                expanded_queries.extend(values)
    queries = [q for q in expanded_queries if q]
    query_text = " ".join([str(q) for q in queries if q])
    qn_full = normalize_text(query_text)

    individual_items = [item for item in price_list if item.result_type != "комплекс"]
    complex_items = [item for item in price_list if item.result_type == "комплекс"]

    scored_individuals: dict[str, tuple[int, LabTest]] = {}
    scored_complexes: dict[str, tuple[int, LabTest]] = {}

    # --- Жесткие медицинские якоря для типичных запросов ---
    if "оак" in qn_full or "общий анализ крови" in qn_full or "клинический анализ крови" in qn_full:
        # Правильный дефолт: ОАК с лейкоформулой и СОЭ из вены.
        for item in individual_items:
            n = normalize_text(item.name)
            if "клинический анализ крови" in n and "соэ" in n and "веноз" in n:
                scored_individuals[item.code] = (120, item)

    if "оам" in qn_full or "общий анализ мочи" in qn_full:
        _force_item_by_name_contains(individual_items, scored_individuals, ["общий", "анализ", "мочи"], score=115)

    if "копрограмм" in qn_full or "капрограмм" in qn_full or "общий анализ кала" in qn_full:
        _force_item_by_name_contains(individual_items, scored_individuals, ["копрограмма"], score=115)

    if "глюкоза" in qn_full:
        _force_item_by_name_contains(individual_items, scored_individuals, ["глюкоза"], score=110)

    if "общий белок" in qn_full:
        _force_item_by_name_contains(individual_items, scored_individuals, ["общий", "белок"], score=110)

    # Индивидуальные аллергены. Не подсовываем панели/аллергокомплексы, если пациент не просил панель.
    allergen_mode = _is_likely_allergen_query(query_text)
    if "берез" in qn_full:
        _force_item_by_name_contains(individual_items, scored_individuals, ["береза"], score=115, exclude_panels=True)
    if "ольх" in qn_full:
        _force_item_by_name_contains(individual_items, scored_individuals, ["ольха"], score=115, exclude_panels=True)
    if "ореш" in qn_full or "лещин" in qn_full:
        _force_item_by_name_contains(individual_items, scored_individuals, ["лещина"], score=115, exclude_panels=True)
    if "плес" in qn_full:
        # По бытовому запросу "плесень" самый безопасный дефолт — Alternaria alternata.
        preferred = []
        for item in individual_items:
            n = normalize_text(item.name)
            if "плеснев" in n and "alternaria" in n and not _is_panel_or_complex_allergen(item):
                preferred.append(item)
        preferred.sort(key=lambda x: (0 if "alternaria alternata" in normalize_text(x.name) else 1, x.price, x.name))
        for item in preferred[:3]:
            scored_individuals[item.code] = (120, item)
        # Добавим и другие индивидуальные плесневые грибы как варианты для ручной замены.
        _force_item_by_name_contains(individual_items, scored_individuals, ["плеснев"], score=100, exclude_panels=True)

    # Коагулограмма: если просили коагулограмму или ее компоненты, обязательно включаем скрининг.
    coag_markers = ["коагул", "мно", "ачтв", "протромбин", "пт", "тв", "тромбиновое", "фг", "фибриноген"]
    if any(m in qn_full for m in coag_markers):
        for item in complex_items:
            n = normalize_text(item.name)
            if "коагулограмма" in n and "скрининг" in n:
                scored_complexes[item.code] = (120, item)
        # Компоненты тоже добавим, чтобы GPT мог пометить их как included, если они продублированы.
        for item in individual_items:
            n = normalize_text(item.name)
            if any(m in n for m in ["мно", "ачтв", "протромбин", "тромбиновое", "фибриноген"]):
                scored_individuals[item.code] = (90, item)

    # --- Обычный fuzzy shortlist ---
    for item in individual_items:
        # Если это allergen-mode, исключаем панели и аллергокомплексы из индивидуального shortlist.
        if allergen_mode and _is_panel_or_complex_allergen(item):
            continue
        best = max((_candidate_score(q, item) for q in queries if q), default=0)
        if item.priority >= 100:
            best = max(best, 95)
        if _has_meaningful_token_overlap(query_text, item.name):
            best = max(best, 72)
        if best >= 58:
            existing = scored_individuals.get(item.code)
            if existing is None or best > existing[0]:
                scored_individuals[item.code] = (best, item)

    for item in complex_items:
        best = max((_candidate_score(q, item, complex_mode=True) for q in queries if q), default=0)
        name_norm = normalize_text(item.name)
        if _has_meaningful_token_overlap(query_text, item.name):
            best = max(best, 70)
        if any(m in qn_full for m in coag_markers) and "коагул" in name_norm:
            best = max(best, 99)
        # Аллергокомплексы не добавляем по запросу отдельных аллергенов, если прямо не просили комплекс/панель.
        if allergen_mode and ("аллергокомплекс" in name_norm or "панель" in name_norm) and not any(x in qn_full for x in ["панель", "комплекс"]):
            continue
        if best >= 55:
            existing = scored_complexes.get(item.code)
            if existing is None or best > existing[0]:
                scored_complexes[item.code] = (best, item)

    individuals = sorted(scored_individuals.values(), key=lambda pair: (-pair[0], -pair[1].priority, pair[1].price, pair[1].name))[:100]
    complexes = sorted(scored_complexes.values(), key=lambda pair: (-pair[0], -pair[1].priority, pair[1].price, pair[1].name))[:50]

    return {
        "individual_analyses": [_catalog_item(item) for _, item in individuals],
        "complexes": [_catalog_item(item) for _, item in complexes],
    }



# === NEW ARCHITECTURE: per-request candidate search ===
# Python ищет кандидатов по всему прайсу, GPT только выбирает лучший вариант.

def _full_price_list(price_list: list[LabTest]) -> list[LabTest]:
    existing_codes = {item.code for item in price_list if item.code}
    result = list(price_list)
    for item in _read_complex_rows_from_xlsx():
        if item.code and item.code not in existing_codes:
            result.append(item)
            existing_codes.add(item.code)
    return result


def _expanded_query_variants(query: str) -> list[str]:
    variants = [query]
    qn = normalize_text(query)

    for mapping in (SEARCH_EQUIVALENTS, UNIVERSAL_SEARCH_EQUIVALENTS):
        for key, values in mapping.items():
            if normalize_text(key) in qn:
                variants.extend(values)

    pcr_num = re.search(r"\bпцр\b.*?\b(\d{1,2})\b", qn)
    if pcr_num:
        n = pcr_num.group(1)
        variants.extend([f"ПЦР-{n}", f"ПЦР {n}", f"{n} инфекций"])

    if any(x in qn for x in ["ковид", "covid", "коронавирус", "sars cov", "sars"]):
        variants.extend([
            "covid", "covid-19", "ковид", "коронавирус", "sars-cov-2", "sars cov 2",
            "РНК SARS-CoV-2", "ДНК SARS-CoV-2", "ПЦР COVID", "мазок на ковид",
        ])

    if "биохим" in qn and ("расшир" in qn or "разверн" in qn):
        variants.extend(["Биохимический анализ крови", "биохимия расширенная"])

    if "посев" in qn or "микрофлор" in qn or "бакпосев" in qn:
        variants.extend(["посев", "бакпосев", "бактериологическое исследование", "микрофлора", "аэробная флора", "анаэробная флора"])

    deduped = []
    seen = set()
    for value in variants:
        key = normalize_text(value)
        if key and key not in seen:
            seen.add(key)
            deduped.append(value)
    return deduped


def _candidate_blob(item: LabTest, include_components: bool = True) -> str:
    parts = [
        item.code,
        item.name,
        getattr(item, "short_name", ""),
        getattr(item, "auto_select_rule", ""),
        " ".join(getattr(item, "synonyms", []) or []),
    ]
    if include_components:
        parts.append(getattr(item, "components", ""))
    return " ".join(str(x or "") for x in parts)


def _is_covid_query(query: str) -> bool:
    q = normalize_text(query)
    return any(x in q for x in ["ковид", "covid", "коронавирус", "sars cov", "sars"])


def _score_candidate_new(request: str, item: LabTest, raw_context: str = "") -> tuple[int, str]:
    variants = _expanded_query_variants(request)
    req_norm = normalize_text(request)
    name_norm = normalize_text(item.name)
    blob_norm = normalize_text(_candidate_blob(item, include_components=True))
    best = 0
    reasons = []

    for variant in variants:
        q = normalize_text(variant)
        if not q:
            continue
        score_name = max(fuzz.WRatio(q, name_norm), fuzz.token_set_ratio(q, name_norm))
        score_blob = max(fuzz.WRatio(q, blob_norm), fuzz.token_set_ratio(q, blob_norm))
        score = int(max(score_name, score_blob * 0.85))
        if q in name_norm:
            score = max(score, 96); reasons.append("query_in_name")
        if q in blob_norm:
            score = max(score, 90); reasons.append("query_in_catalog")
        best = max(best, score)

    if req_norm and normalize_text(item.code) == req_norm:
        best = max(best, 150); reasons.append("code_exact")

    if any(x in req_norm for x in ["оак", "общий анализ крови", "клинический анализ крови", "анализ крови"]):
        if "клинический анализ крови" in name_norm and "соэ" in name_norm and "веноз" in name_norm:
            best = max(best, 145); reasons.append("cbc_default")
        elif "клинический анализ крови" in name_norm and "соэ" in name_norm:
            best = max(best, 130); reasons.append("cbc_with_esr")

    if any(x in req_norm for x in ["оам", "общий анализ мочи", "анализ мочи"]):
        if "общий анализ мочи" in name_norm:
            best = max(best, 140); reasons.append("urine_default")

    if any(x in req_norm for x in ["копрограмм", "капрограмм", "общий анализ кала"]):
        if "копрограмма" in name_norm:
            best = max(best, 140); reasons.append("coprogram")

    if "биохим" in req_norm:
        if "биохимический анализ крови" in name_norm:
            if "базов" in req_norm and "базов" in name_norm:
                best = max(best, 145); reasons.append("basic_biochemistry")
            elif ("расшир" in req_norm or "разверн" in req_norm) and "базов" not in name_norm:
                best = max(best, 142); reasons.append("extended_biochemistry")
            elif "базов" not in req_norm and "расшир" not in req_norm:
                best = max(best, 125); reasons.append("biochemistry")
        if "базов" in req_norm and "биохим" in name_norm and "базов" not in name_norm:
            best -= 35
        if ("расшир" in req_norm or "разверн" in req_norm) and "биохим" in name_norm and "базов" in name_norm:
            best -= 45

    pcr_num = re.search(r"\bпцр\b.*?\b(\d{1,2})\b", req_norm)
    if pcr_num and "пцр" in name_norm:
        n = pcr_num.group(1)
        if re.search(rf"\bпцр\s*{re.escape(n)}\b", name_norm):
            best = max(best, 150); reasons.append("pcr_number_exact")
            if any(x in name_norm for x in ["эякулят", "моча"]):
                best -= 20; reasons.append("specific_biomaterial_penalty")

    coag_markers = ["коагул", "мно", "ачтв", "протромбин", "пт", "тв", "тромбиновое", "фг", "фибриноген"]
    if any(m in req_norm for m in coag_markers):
        if "коагулограмма" in name_norm and "скрининг" in name_norm:
            best = max(best, 145); reasons.append("coag_screening")
        elif "коагулограмма" in name_norm:
            best = max(best, 105); reasons.append("coag")
        if any(x in name_norm for x in ["мно", "ачтв", "протромбин", "тромбиновое", "фибриноген"]):
            best = max(best, 95); reasons.append("coag_component")

    if "впч" in req_norm and ("впч" in name_norm or "папилломавирус" in name_norm):
        best = max(best, 115)
        if "онког" in req_norm and ("онког" in name_norm or "канцер" in name_norm):
            best = max(best, 145); reasons.append("hpv_high_risk")

    if _is_covid_query(req_norm):
        if any(x in blob_norm for x in ["ковид", "covid", "коронавирус", "sars cov", "sars"]):
            best = max(best, 130); reasons.append("covid")
            if any(x in blob_norm for x in ["пцр", "мазок", "рнк", "днк"]):
                best = max(best, 145); reasons.append("covid_pcr")

    if any(x in req_norm for x in ["посев", "микрофлор", "бакпосев", "бактериолог"]):
        if any(x in name_norm for x in ["посев", "микрофлор", "бактериолог", "бакпосев"]):
            best = max(best, 130); reasons.append("culture")

    allergen_needles = {
        "берез": ["берез", "betula"],
        "ольх": ["ольх", "alnus"],
        "ореш": ["лещин", "corylus"],
        "лещин": ["лещин", "corylus"],
        "плес": ["плеснев", "alternaria"],
    }
    for trigger, needles in allergen_needles.items():
        if trigger in req_norm:
            if any(n in name_norm for n in needles):
                best = max(best, 135); reasons.append("individual_allergen")
                if trigger == "плес" and "alternaria" in name_norm:
                    best = max(best, 145)
            if _is_panel_or_complex_allergen(item) and not any(x in req_norm for x in ["панель", "комплекс"]):
                best -= 60; reasons.append("allergen_panel_penalty")

    if item.result_type == "комплекс" and item.components:
        comp = normalize_text(item.components)
        req_tokens = {t for t in req_norm.split() if len(t) >= 3 and t not in GENERIC_SHORTLIST_TOKENS}
        shared = req_tokens & set(comp.split())
        if shared:
            best = max(best, min(110, 55 + len(shared) * 12)); reasons.append("component_match")

    if item.priority >= 100 and best >= 60:
        best += 8; reasons.append("priority")

    return max(0, int(best)), ",".join(dict.fromkeys(reasons))


def _catalog_item_new(item: LabTest, score: int, reason: str = "") -> dict[str, Any]:
    return {
        "code": item.code,
        "name": item.name,
        "price": item.price,
        "term_days": item.term_days,
        "result_type": item.result_type,
        "priority": item.priority,
        "components": item.components if item.result_type == "комплекс" else "",
        "score": score,
        "reason": reason,
    }


def _dedupe_scored_new(items: list[tuple[int, LabTest, str]], limit: int) -> list[tuple[int, LabTest, str]]:
    seen = set(); result = []
    for score, item, reason in sorted(items, key=lambda x: (-x[0], -x[1].priority, x[1].price, x[1].name)):
        key = item.code or normalize_text(item.name)
        if key in seen:
            continue
        seen.add(key); result.append((score, item, reason))
        if len(result) >= limit:
            break
    return result


def build_candidates_for_request(request: str, price_list: list[LabTest], *, raw_context: str = "", max_candidates: int = 20) -> list[dict[str, Any]]:
    full_list = _full_price_list(price_list)
    scored = []
    for item in full_list:
        score, reason = _score_candidate_new(request, item, raw_context=raw_context)
        if score >= 58:
            if _is_likely_allergen_query(request) and _is_panel_or_complex_allergen(item) and not any(x in normalize_text(request) for x in ["панель", "комплекс"]):
                continue
            scored.append((score, item, reason))
    top = _dedupe_scored_new(scored, max_candidates)
    return [_catalog_item_new(item, score, reason) for score, item, reason in top]


def build_per_request_candidate_payload(raw_text: str, cleaned_items: list[str], price_list: list[LabTest]) -> dict[str, Any]:
    source_requests = []
    source_requests.extend(cleaned_items or [])
    source_requests.extend(split_user_tests(raw_text))
    seen = set(); requests = []
    for request in source_requests:
        request = str(request or "").strip()
        key = normalize_text(request)
        if not key or key in seen:
            continue
        seen.add(key)
        requests.append({
            "request": request,
            "candidates": build_candidates_for_request(request, price_list, raw_context=raw_text, max_candidates=20),
        })
    if not requests:
        requests.append({
            "request": raw_text,
            "candidates": build_candidates_for_request(raw_text, price_list, raw_context=raw_text, max_candidates=20),
        })
    return {"raw_text": raw_text, "requests": requests}


def _norm_for_self_include(text: str) -> str:
    value = normalize_text(text)
    value = re.sub(r"\b(анализ|комплекс|исследование)\b", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _is_self_included(requested: str, included_in_complex: str, code: str, by_code: dict[str, LabTest]) -> bool:
    req = _norm_for_self_include(requested); inc = _norm_for_self_include(included_in_complex)
    if req and inc and (req == inc or req in inc or inc in req):
        return True
    item = by_code.get(code)
    if item:
        item_name = _norm_for_self_include(item.name)
        if item_name and req and (req == item_name or req in item_name or item_name in req):
            return True
        if item.result_type == "комплекс" and item_name and inc and (inc == item_name or inc in item_name or item_name in inc):
            return True
    return False


def _postprocess_gpt_items(data: dict, by_code: dict[str, LabTest]) -> list[dict[str, Any]]:
    items = data.get("items", [])
    if not isinstance(items, list):
        return []
    found_complexes = []
    for entry in items:
        code = str(entry.get("code") or "").strip(); status = str(entry.get("status") or "").strip()
        item = by_code.get(code)
        if status == "found" and item and item.result_type == "комплекс":
            found_complexes.append(item)
    processed = []; seen = set()
    for entry in items:
        if not isinstance(entry, dict):
            continue
        requested = str(entry.get("request") or entry.get("requested") or "").strip() or "анализ"
        status = str(entry.get("status") or "").strip()
        code = str(entry.get("code") or "").strip()
        included_in_complex = str(entry.get("included_in_complex") or "").strip()
        if status == "included" and code in by_code and _is_self_included(requested, included_in_complex, code, by_code):
            status = "found"; included_in_complex = ""
        if status == "found" and code in by_code:
            item = by_code[code]; req_norm = normalize_text(requested); item_norm = normalize_text(item.name)
            for complex_item in found_complexes:
                comp = normalize_text(complex_item.components)
                if item.result_type != "комплекс" and (req_norm in comp or item_norm in comp):
                    status = "included"; included_in_complex = complex_item.name; break
        key = (status, code, normalize_text(requested), normalize_text(included_in_complex))
        if key in seen:
            continue
        seen.add(key)
        processed.append({"requested": requested, "status": status, "code": code, "included_in_complex": included_in_complex})
    return processed
GPT_MATCHING_SYSTEM_PROMPT = """
Ты — медицинский эксперт лабораторного прайс-листа.

Тебе передан JSON:
- raw_text: исходный запрос пациента;
- requests: отдельные пункты запроса;
- candidates: кандидаты из прайса для каждого пункта.

Твоя задача — НЕ искать по прайсу самостоятельно, а выбрать лучший вариант из candidates для каждого request.

Строгие правила:
1. Используй только переданные candidates. Не придумывай анализы.
2. Выбирай по медицинскому смыслу, названию name, типу result_type и составу components.
3. Если нет хорошего кандидата для request, верни status=not_found.
4. Если пользователь просит набор показателей, который очевидно входит в комплекс, выбери комплекс.
5. Если пользователь одновременно просит комплекс и отдельный анализ, который входит в этот комплекс, НЕ удаляй отдельный анализ. Верни его отдельной строкой со status=included и included_in_complex.
6. ОАК/общий анализ крови без уточнений = Клинический анализ крови с лейкоцитарной формулой (5DIFF) и СОЭ (венозная кровь).
7. Коагулограмма без слова "расширенная" = Коагулограмма, скрининг.
8. Базовая биохимия = Биохимический анализ крови, базовый.
9. Расширенная биохимия = более широкий комплекс "Биохимический анализ крови", а не базовый.
10. ПЦР на 12 инфекций = ПЦР-12, если такой кандидат есть.
11. Ковид/COVID/мазок на ковид = анализ на SARS-CoV-2/COVID, предпочтительно ПЦР/РНК/ДНК/мазок.
12. Посев на микрофлору = бактериологический посев/исследование микрофлоры, если такой кандидат есть.
13. Аллергены береза/ольха/орешник/лещина/плесень ищи как индивидуальные аллергены. Не выбирай аллергоПАНЕЛЬ или аллергоКОМПЛЕКС, если пациент прямо не просил панель/комплекс.
14. Орешник = лещина. Капрограмма = копрограмма. Плесень = плесневый гриб, чаще всего Alternaria alternata, а не Candida.
15. Не добавляй анализы, которых пациент не просил.
16. Запрещена самоссылка: если request совпадает с названием комплекса, status должен быть found, а не included.

Верни строго JSON:
{
  "items": [
    {
      "request": "исходный пункт",
      "status": "found|included|not_found",
      "code": "код выбранного кандидата или пусто",
      "included_in_complex": "название комплекса, если status=included, иначе пусто",
      "comment": "коротко"
    }
  ]
}
"""
async def match_user_request_with_ai(raw_text: str, price_list: list[LabTest], initial_usage: AIUsage | None = None, source: str = "text") -> AIMatchResult:
    usage = AIUsage(used=False, source="")
    if initial_usage:
        usage.add(initial_usage)

    cleaned_items = clean_recognized_text_locally(raw_text)
    full_price_list = _full_price_list(price_list)

    if not is_ai_enabled():
        fallback_items = cleaned_items or split_user_tests(raw_text)
        return AIMatchResult(
            matched_items=match_tests(fallback_items, full_price_list),
            cleaned_items=fallback_items,
            raw_text=raw_text,
            usage=usage,
            used_gpt_matching=False,
        )

    payload = build_per_request_candidate_payload(raw_text, cleaned_items, full_price_list)

    if not any(req.get("candidates") for req in payload.get("requests", [])):
        fallback_items = cleaned_items or split_user_tests(raw_text)
        return AIMatchResult(
            matched_items=match_tests(fallback_items, full_price_list),
            cleaned_items=fallback_items,
            raw_text=raw_text,
            usage=usage,
            used_gpt_matching=False,
        )

    client = _client()
    response = client.chat.completions.create(
        model=OPENAI_TEXT_MODEL,
        temperature=0,
        max_tokens=1600,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": GPT_MATCHING_SYSTEM_PROMPT.strip()},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    )
    usage.add(_usage_from_chat_response(response, f"{source}_gpt_match"))

    content = response.choices[0].message.content or "{}"
    try:
        data = json.loads(content)
    except Exception:
        data = {"items": []}

    by_code = {item.code: item for item in full_price_list if item.code}
    matched_items: list[MatchedItem] = []

    for entry in _postprocess_gpt_items(data, by_code):
        requested = str(entry.get("requested") or "").strip() or "анализ"
        status = str(entry.get("status") or "").strip()
        code = str(entry.get("code") or "").strip()
        included_in_complex = str(entry.get("included_in_complex") or "").strip()

        if status == "included":
            matched_items.append(make_included_item(requested, included_in_complex, full_price_list, code=code))
            continue

        if status == "found" and code in by_code:
            candidates = find_candidate_tests(requested, full_price_list)
            selected = labtest_to_candidate(by_code[code], 100)
            candidates = [selected] + [c for c in candidates if c.code != code]
            matched_items.append(make_matched_from_code(requested, code, full_price_list, candidates=candidates))
            continue

        matched_items.append(MatchedItem(input_name=requested, status="not_found", confidence=0))

    if not matched_items:
        fallback_items = cleaned_items or split_user_tests(raw_text)
        matched_items = match_tests(fallback_items, full_price_list)

    if cleaned_items:
        remember_cleaned_text(raw_text, cleaned_items)

    return AIMatchResult(
        matched_items=matched_items,
        cleaned_items=cleaned_items,
        raw_text=raw_text,
        usage=usage,
        used_gpt_matching=True,
    )
