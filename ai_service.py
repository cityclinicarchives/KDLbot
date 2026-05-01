
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
    "орешник": ["лещина", "лесной орех", "фундук", "corylus"],
    "лещина": ["орешник", "лесной орех", "фундук", "corylus"],
    "плесень": ["плеснев", "alternaria", "cladosporium", "aspergillus", "penicillium", "мукор", "candida"],
    "береза": ["береза", "betula"],
    "ольха": ["ольха", "alnus"],
    "пт": ["протромбин", "протромбиновое"],
    "тв": ["тромбиновое время"],
    "фг": ["фибриноген"],
}

PHRASE_TO_CANONICAL = {
    "анализ крови": "ОАК",
    "общий анализ крови": "ОАК",
    "клинический анализ крови": "ОАК",
    "анализ мочи": "ОАМ",
    "общий анализ мочи": "ОАМ",
    "моча": "ОАМ",
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
}


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

    individual_items = [item for item in price_list if item.result_type != "комплекс"]
    complex_items = [item for item in price_list if item.result_type == "комплекс"]

    scored_individuals: dict[str, tuple[int, LabTest]] = {}
    scored_complexes: dict[str, tuple[int, LabTest]] = {}

    for item in individual_items:
        best = max((_candidate_score(q, item) for q in queries if q), default=0)
        # Важная ручная страховка: единственный оставленный priority=100 должен попадать в shortlist.
        if item.priority >= 100:
            best = max(best, 95)
        # Аллергены и короткие запросы: береза, ольха, лещина, плесень.
        if any(tok in normalize_text(item.name) for tok in normalize_text(query_text).split() if len(tok) >= 4):
            best = max(best, 70)
        if best >= 55:
            scored_individuals[item.code] = (best, item)

    for item in complex_items:
        best = max((_candidate_score(q, item, complex_mode=True) for q in queries if q), default=0)
        name_norm = normalize_text(item.name)
        if any(tok in name_norm for tok in normalize_text(query_text).split() if len(tok) >= 4):
            best = max(best, 70)
        # Коагулограмма по составу: ПТ, МНО, АЧТВ, ТВ, ФГ.
        qn = normalize_text(query_text)
        coag_markers = ["мно", "ачтв", "протромбин", "пт", "тв", "тромбиновое", "фг", "фибриноген"]
        if sum(1 for m in coag_markers if m in qn) >= 3 and "коагул" in name_norm:
            best = max(best, 99)
        if best >= 50:
            scored_complexes[item.code] = (best, item)

    individuals = sorted(scored_individuals.values(), key=lambda pair: (-pair[0], -pair[1].priority, pair[1].name))[:90]
    complexes = sorted(scored_complexes.values(), key=lambda pair: (-pair[0], -pair[1].priority, pair[1].name))[:50]

    return {
        "individual_analyses": [_catalog_item(item) for _, item in individuals],
        "complexes": [_catalog_item(item) for _, item in complexes],
    }


GPT_MATCHING_SYSTEM_PROMPT = """
Ты — медицинский эксперт лабораторного прайс-листа. Твоя задача — сопоставить запрос пациента с анализами и комплексами из переданного каталога.

Строгие правила:
1. Используй только переданные позиции каталога.
2. Для поиска ориентируйся на поле name. Не используй synonyms, их нет в каталоге.
3. Для комплексов можно использовать поле components, чтобы понять, что отдельные показатели входят в комплекс.
4. Если пользователь просит набор показателей, который очевидно входит в комплекс, выбери комплекс.
5. Если пользователь одновременно просит комплекс и отдельный анализ, который входит в этот комплекс, НЕ удаляй отдельный анализ. Верни его отдельной строкой со статусом included и укажи included_in_complex.
6. Если пользователь пишет ОАК/общий анализ крови без уточнений, выбирай полный вариант: Клинический анализ крови с лейкоцитарной формулой (5DIFF) и СОЭ (венозная кровь).
7. Если пользователь пишет ОАК ребенку/из пальца/капиллярная кровь, выбирай соответствующий вариант из пальца.
8. Используй priority только как редкий tie-breaker при одинаковых вариантах.
9. Не добавляй анализы, которых пациент не просил.
10. Если подходящей позиции нет, верни status=not_found.

Верни строго JSON:
{
  "items": [
    {
      "requested": "как понял запрос",
      "status": "found|included|not_found",
      "code": "код из каталога или пусто",
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

    if not is_ai_enabled():
        # Fallback без GPT: старая локальная логика.
        fallback_items = cleaned_items or split_user_tests(raw_text)
        return AIMatchResult(
            matched_items=match_tests(fallback_items, price_list),
            cleaned_items=fallback_items,
            raw_text=raw_text,
            usage=usage,
            used_gpt_matching=False,
        )

    catalog = _build_candidate_catalog(raw_text, cleaned_items, price_list)
    if not catalog["individual_analyses"] and not catalog["complexes"]:
        fallback_items = cleaned_items or split_user_tests(raw_text)
        return AIMatchResult(
            matched_items=match_tests(fallback_items, price_list),
            cleaned_items=fallback_items,
            raw_text=raw_text,
            usage=usage,
            used_gpt_matching=False,
        )

    user_payload = {
        "raw_text": raw_text,
        "cleaned_items_local": cleaned_items,
        "catalog": catalog,
    }

    client = _client()
    response = client.chat.completions.create(
        model=OPENAI_TEXT_MODEL,
        temperature=0,
        max_tokens=1200,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": GPT_MATCHING_SYSTEM_PROMPT.strip()},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
    )
    usage.add(_usage_from_chat_response(response, f"{source}_gpt_match"))

    content = response.choices[0].message.content or "{}"
    try:
        data = json.loads(content)
    except Exception:
        data = {"items": []}

    by_code = {item.code: item for item in price_list if item.code}
    matched_items: list[MatchedItem] = []

    for entry in data.get("items", []):
        requested = str(entry.get("requested") or "").strip() or "анализ"
        status = str(entry.get("status") or "").strip()
        code = str(entry.get("code") or "").strip()
        included_in_complex = str(entry.get("included_in_complex") or "").strip()

        if status == "included":
            matched_items.append(make_included_item(requested, included_in_complex, price_list, code=code))
            continue

        if status == "found" and code in by_code:
            candidates = find_candidate_tests(requested, price_list)
            # выбранный GPT вариант ставим первым в candidates
            selected = labtest_to_candidate(by_code[code], 100)
            candidates = [selected] + [c for c in candidates if c.code != code]
            matched_items.append(make_matched_from_code(requested, code, price_list, candidates=candidates))
            continue

        matched_items.append(MatchedItem(input_name=requested, status="not_found", confidence=0))

    if not matched_items:
        fallback_items = cleaned_items or split_user_tests(raw_text)
        matched_items = match_tests(fallback_items, price_list)

    if cleaned_items:
        remember_cleaned_text(raw_text, cleaned_items)

    return AIMatchResult(
        matched_items=matched_items,
        cleaned_items=cleaned_items,
        raw_text=raw_text,
        usage=usage,
        used_gpt_matching=True,
    )
