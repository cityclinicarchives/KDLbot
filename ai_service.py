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
    CandidateItem,
    MatchedItem,
    normalize_text,
    split_user_tests,
    match_tests,
    find_candidate_tests,
    labtest_to_candidate,
    make_matched_from_code,
    make_included_item,
    load_alias_groups,
    candidate_to_matched,
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
        # На Railway файловая система может быть временной. Ошибка памяти не должна ломать бота.
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
    "и еще", "а еще", "также", "плюс", "можно", "давайте", "посчитай",
    "подбери", "рассчитай", "список",
]


# Локальные подсказки нужны только как fallback, когда OPENAI_API_KEY отсутствует.
# Основная архитектура: ИИ чистит и разделяет текст; Python ищет по прайсу.
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
    "ггт": "ГГТ",
    "гамма гт": "ГГТ",
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
    "базовая биохимия": "Биохимия базовая",
    "биохимия базовая": "Биохимия базовая",
    "расширенная биохимия": "Биохимия расширенная",
    "биохимия расширенная": "Биохимия расширенная",
    "пцр на 12 инфекций": "ПЦР-12",
    "пцр 12 инфекций": "ПЦР-12",
    "пцр на 15 инфекций": "ПЦР-15",
    "пцр 15 инфекций": "ПЦР-15",
    "впч высокого онкогенного риска": "ВПЧ высокого онкогенного риска",
    "мазок на ковид": "COVID-19",
    "пцр на ковид": "COVID-19",
    "ковид": "COVID-19",
    "covid": "COVID-19",
    "covid-19": "COVID-19",
    "коронавирус": "COVID-19",
    "sars-cov-2": "COVID-19",
    "sars cov 2": "COVID-19",
    "посев на микрофлору": "посев на микрофлору",
    "мазок на микрофлору": "посев на микрофлору",
    "орешник": "аллерген лещина",
    "лещина": "аллерген лещина",
    "береза": "аллерген береза",
    "берёза": "аллерген береза",
    "ольха": "аллерген ольха",
    "плесень": "аллерген плесневый гриб",
}


# Смысловые варианты нужны для Python fuzzy search. Это не автоподбор результата,
# а расширение запроса, чтобы "биохимия" находила "биохимический".
SEARCH_EXPANSIONS = {
    "оак": ["клинический анализ крови соэ", "общий анализ крови", "клинический анализ крови"],
    "оам": ["общий анализ мочи"],
    "коагулограмма": ["коагулограмма скрининг", "коагулограмма"],
    "биохимия базовая": ["биохимический анализ крови базовый", "биохимический анализ крови, базовый", "базовая биохимия"],
    "базовая биохимия": ["биохимический анализ крови базовый", "биохимический анализ крови, базовый"],
    "биохимия расширенная": ["биохимический анализ крови", "биохимический анализ крови расширенный", "расширенная биохимия"],
    "расширенная биохимия": ["биохимический анализ крови", "биохимический анализ крови расширенный"],
    "пцр-12": ["пцр 12", "пцр-12", "пцр на 12 инфекций", "12 инфекций"],
    "пцр-15": ["пцр 15", "пцр-15", "пцр на 15 инфекций", "15 инфекций"],
    "covid-19": ["covid", "covid-19", "ковид", "коронавирус", "sars-cov-2", "sars cov 2", "рнк sars-cov-2", "днк sars-cov-2"],
    "посев на микрофлору": ["посев", "бакпосев", "бактериологическое исследование", "микрофлора", "чувствительность к антибиотикам"],
    "аллерген лещина": ["лещина", "орешник", "corylus", "фундук"],
    "аллерген береза": ["береза", "берёза", "betula"],
    "аллерген ольха": ["ольха", "alnus"],
    "аллерген плесневый гриб": ["плесневый гриб", "плесень", "alternaria", "alternaria alternata"],
}


GENERIC_SHORTLIST_TOKENS = {
    "общий", "общая", "анализ", "анализы", "крови", "кровь", "мочи", "моча", "аллерген", "аллергены",
    "сдать", "посчитать", "пожалуйста", "мне", "нужно", "надо", "хочу", "еще", "ещё", "мазок",
}

ALLERGEN_QUERY_WORDS = {"береза", "берёза", "ольха", "орешник", "лещина", "плесень", "плесневый", "плесневые"}
ALLERGEN_PANEL_WORDS = ["панель", "аллергокомплекс", "комплекс"]


def clean_recognized_text_locally(text: str) -> list[str]:
    """
    Fallback без GPT: пытается вытащить медицинские фразы локально.
    Основной рабочий сценарий при OPENAI_API_KEY — _clean_text_with_gpt().
    """
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

    for phrase, canonical in sorted(PHRASE_TO_CANONICAL.items(), key=lambda pair: len(pair[0]), reverse=True):
        if re.search(rf"(^|\s){re.escape(normalize_text(phrase))}($|\s)", norm):
            if canonical not in extracted:
                extracted.append(canonical)

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


def _split_semicolon_list(text: str) -> list[str]:
    if not text:
        return []

    value = str(text).strip()
    # На случай если GPT вернул JSON.
    try:
        data = json.loads(value)
        if isinstance(data, dict) and isinstance(data.get("items"), list):
            return [str(x).strip() for x in data["items"] if str(x).strip()]
        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x).strip()]
    except Exception:
        pass

    value = value.replace("\n", ";")
    value = re.sub(r"^\s*(?:список|анализы)\s*:\s*", "", value, flags=re.IGNORECASE)
    parts = [re.sub(r"^\s*\d+[.)\-:]*\s*", "", p).strip(" -—\t\r\n") for p in value.split(";")]
    result = []
    seen = set()
    for part in parts:
        if not part:
            continue
        # Убираем случайные кавычки/маркеры.
        part = part.strip("\"'«» ")
        key = normalize_text(part)
        if key and key not in seen:
            seen.add(key)
            result.append(part)
    return result


async def _clean_text_with_gpt(raw_text: str) -> tuple[list[str], AIUsage]:
    """
    ИИ используется только как cleaner:
    - убрать мусор;
    - исправить очевидные ошибки распознавания;
    - разделить анализы символом ';'.

    ИИ НЕ ищет анализы в прайсе и НЕ выбирает коды.
    """
    data = _load_memory()
    memory_key = normalize_text(raw_text)
    if memory_key in data.get("cleaned", {}):
        stored = data["cleaned"][memory_key]
        if isinstance(stored, list) and stored:
            return [str(x).strip() for x in stored if str(x).strip()], AIUsage(False, "")

    client = _client()

    system_prompt = """
Ты модуль очистки текста медицинского Telegram-бота.

Твоя задача:
1. Убрать слова-паразиты и лишние фразы.
2. Исправить очевидные ошибки распознавания речи.
3. Разделить текст на отдельные лабораторные анализы.
4. Вернуть ТОЛЬКО одну строку, где анализы разделены символом ';'.

Не ищи анализы в прайсе.
Не добавляй цены, коды, пояснения.
Не удаляй лабораторный анализ только потому, что он тебе кажется неполным.
Удаляй только явно не лабораторные исследования: УЗИ, МРТ, КТ, ЭКГ, ФГДС, консультации, приемы врачей, процедуры.

Примеры:
"Мне нужны ОАК ОАМ кугалаграма глюкозу, расширеную биохимию обычную, общий белок, аст, ПЦР 15 инфекций и еще посев на микрофлору какой-нибудь, где показывает чувствительность к антибиотикам и анализ на сифилис"
=> ОАК; ОАМ; Коагулограмма; Глюкоза; Биохимия расширенная; Общий белок; АСТ; ПЦР-15; Посев на микрофлору с определением чувствительности к антибиотикам; Сифилис

"ОАК, ОАМ, коагулограмма, общий белок, глюкоза, аллерген лещина"
=> ОАК; ОАМ; Коагулограмма; Общий белок; Глюкоза; Аллерген лещина
"""

    response = client.chat.completions.create(
        model=OPENAI_TEXT_MODEL,
        temperature=0,
        max_tokens=250,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": str(raw_text or "").strip()},
        ],
    )

    usage = _usage_from_chat_response(response, "text_cleaning")
    content = response.choices[0].message.content or ""
    items = _split_semicolon_list(content)

    if not items:
        items = clean_recognized_text_locally(raw_text)

    return items, usage


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


def _is_likely_allergen_query(text: str) -> bool:
    qn = normalize_text(text)
    return any(word in qn for word in ALLERGEN_QUERY_WORDS)


def _is_panel_or_complex_allergen(item: LabTest) -> bool:
    name = normalize_text(item.name)
    return any(word in name for word in ALLERGEN_PANEL_WORDS)


def _expanded_queries(query: str) -> list[str]:
    query = str(query or "").strip()
    result = [query]
    qn = normalize_text(query)

    canonical = PHRASE_TO_CANONICAL.get(qn)
    if canonical:
        result.append(canonical)
        qn = normalize_text(canonical)

    for key, values in SEARCH_EXPANSIONS.items():
        if normalize_text(key) in qn or qn in normalize_text(key):
            result.extend(values)

    # ПЦР-число.
    pcr_num = re.search(r"\bпцр\b.*?\b(\d{1,2})\b", qn)
    if pcr_num:
        n = pcr_num.group(1)
        result.extend([f"ПЦР-{n}", f"ПЦР {n}", f"{n} инфекций"])

    # SARS/COVID.
    if any(x in qn for x in ["ковид", "covid", "коронавирус", "sars"]):
        result.extend(["COVID-19", "SARS-CoV-2", "РНК SARS-CoV-2", "ПЦР COVID"])

    deduped = []
    seen = set()
    for value in result:
        key = normalize_text(value)
        if key and key not in seen:
            seen.add(key)
            deduped.append(value)
    return deduped


def _score_price_item(query: str, item: LabTest) -> int:
    best = 0
    variants = _expanded_queries(query)
    item_name = normalize_text(item.name)
    item_code = normalize_text(item.code)
    item_short = normalize_text(item.short_name)
    item_rule = normalize_text(item.auto_select_rule)
    item_components = normalize_text(item.components)
    item_texts = [item_name, item_code, item_short, item_rule]

    # По требованию: основной поиск по столбцу B name и alias_groups.
    # components используем только для логики комплексов.
    for q in variants:
        qn = normalize_text(q)
        if not qn:
            continue

        for text in item_texts:
            if not text:
                continue
            score = max(fuzz.WRatio(qn, text), fuzz.token_set_ratio(qn, text))
            if qn == text:
                score = 100
            elif qn in text:
                score = max(score, 96)
            best = max(best, int(score))

        if item.result_type == "комплекс" and item_components:
            comp_score = fuzz.token_set_ratio(qn, item_components)
            best = max(best, int(comp_score * 0.65))

    qfull = normalize_text(" ".join(variants))

    # Дефолты, которые должны работать даже без alias_groups.
    if qfull in {"оак"} or "общий анализ крови" in qfull or "клинический анализ крови" in qfull:
        if "клинический анализ крови" in item_name and "соэ" in item_name and "веноз" in item_name:
            best = max(best, 150)
        elif "клинический анализ крови" in item_name and "соэ" in item_name:
            best = max(best, 130)

    if qfull in {"оам"} or "общий анализ мочи" in qfull:
        if "общий анализ мочи" in item_name:
            best = max(best, 150)

    if "коагул" in qfull:
        if "коагулограмма" in item_name and "скрининг" in item_name:
            best = max(best, 150)
        elif "коагулограмма" in item_name:
            best = max(best, 110)

    if "биохим" in qfull:
        if "базов" in qfull:
            if "биохимический анализ крови" in item_name and "базов" in item_name:
                best = max(best, 150)
            elif "биохим" in item_name and "базов" in item_name:
                best = max(best, 135)
        elif "расшир" in qfull or "разверн" in qfull:
            if "биохимический анализ крови" in item_name and "базов" not in item_name:
                best = max(best, 145)
            if "базов" in item_name:
                best -= 45
        else:
            if "биохимический анализ крови" in item_name:
                best = max(best, 115)

    # ПЦР-12 / ПЦР-15.
    pcr_num = re.search(r"\bпцр\b.*?\b(\d{1,2})\b", qfull)
    if pcr_num and "пцр" in item_name:
        n = pcr_num.group(1)
        if re.search(rf"\bпцр\s*{re.escape(n)}\b", item_name):
            best = max(best, 150)
            if any(x in item_name for x in ["эякулят", "моча"]):
                best -= 20

    # COVID/SARS.
    if any(x in qfull for x in ["ковид", "covid", "коронавирус", "sars"]):
        if any(x in item_name for x in ["ковид", "covid", "коронавирус", "sars"]):
            best = max(best, 140)
            if any(x in item_name for x in ["пцр", "рнк", "днк", "мазок"]):
                best = max(best, 150)

    # Посевы.
    if "посев" in qfull or "микрофлор" in qfull or "бакпосев" in qfull:
        if any(x in item_name for x in ["посев", "микрофлор", "бактериолог", "бакпосев"]):
            best = max(best, 135)
        if "чувств" in qfull and "чувств" in item_name:
            best = max(best, 145)

    # Аллергены.
    allergen_map = {
        "берез": ["берез", "betula"],
        "ольх": ["ольх", "alnus"],
        "ореш": ["лещин", "corylus"],
        "лещин": ["лещин", "corylus"],
        "плес": ["плеснев", "alternaria"],
    }
    for trigger, needles in allergen_map.items():
        if trigger in qfull:
            if any(n in item_name for n in needles):
                best = max(best, 140)
                if trigger == "плес" and "alternaria" in item_name:
                    best = max(best, 150)
            if _is_panel_or_complex_allergen(item) and not any(x in qfull for x in ["панель", "комплекс"]):
                best -= 60

    if item.priority >= 100 and best >= 80:
        best += 10

    return max(0, min(200, int(best)))


def _find_by_alias_groups_fuzzy(query: str, price_list: list[LabTest], limit: int = 8) -> list[CandidateItem]:
    alias_groups = load_alias_groups()
    if not alias_groups:
        return []

    qn = normalize_text(query)
    best_group = ""
    best_score = 0

    for alias, group in alias_groups.items():
        score = max(fuzz.WRatio(qn, alias), fuzz.token_set_ratio(qn, alias))
        if qn == alias:
            score = 120
        elif qn in alias or alias in qn:
            score = max(score, 100)
        if score > best_score:
            best_score = int(score)
            best_group = group

    if not best_group or best_score < 78:
        return []

    group_items = [item for item in price_list if item.group == best_group]
    if not group_items:
        return []

    scored: list[tuple[int, LabTest]] = []
    for item in group_items:
        score = _score_price_item(query, item)
        score = max(score, best_score)
        scored.append((score, item))

    scored.sort(key=lambda pair: (-pair[1].priority, -pair[0], pair[1].price, pair[1].name))
    return [labtest_to_candidate(item, min(100, score)) for score, item in scored[:limit]]


def _find_candidates_python(query: str, price_list: list[LabTest], max_candidates: int = 8) -> list[CandidateItem]:
    # 1. Сначала штатный поиск pricing.py: он уже ищет по alias_groups и name.
    candidates = find_candidate_tests(query, price_list, min_score=70, close_gap=12, max_candidates=max_candidates)

    # 2. Если alias_groups сработал fuzzy-совпадением, добавляем кандидатов.
    alias_candidates = _find_by_alias_groups_fuzzy(query, price_list, limit=max_candidates)

    combined: list[CandidateItem] = []
    seen = set()

    for candidate in alias_candidates + candidates:
        if candidate.code and candidate.code not in seen:
            seen.add(candidate.code)
            combined.append(candidate)

    # 3. Универсальный поиск по name/short/code/auto_select_rule.
    scored: list[tuple[int, LabTest]] = []
    for item in price_list:
        if _is_likely_allergen_query(query) and _is_panel_or_complex_allergen(item) and not any(x in normalize_text(query) for x in ["панель", "комплекс"]):
            continue
        score = _score_price_item(query, item)
        if score >= 68:
            scored.append((score, item))

    scored.sort(key=lambda pair: (-pair[0], -pair[1].priority, pair[1].price, pair[1].name))
    for score, item in scored:
        if item.code and item.code not in seen:
            seen.add(item.code)
            combined.append(labtest_to_candidate(item, min(100, score)))
        if len(combined) >= max_candidates:
            break

    # Финальная сортировка: лучший score + priority.
    combined.sort(key=lambda c: (-c.confidence, -c.priority, c.price, c.matched_name))
    return combined[:max_candidates]


def _match_one_python(query: str, price_list: list[LabTest]) -> MatchedItem:
    data = _load_memory()
    memory_key = normalize_text(query)
    remembered = data.get("choices", {}).get(memory_key)
    if remembered and remembered.get("code"):
        return make_matched_from_code(query, remembered["code"], price_list)

    candidates = _find_candidates_python(query, price_list, max_candidates=8)
    if not candidates:
        return MatchedItem(input_name=query, status="not_found", confidence=0)

    selected = candidates[0]
    return candidate_to_matched(query, selected, candidates=candidates)


def _component_codes(components: str) -> set[str]:
    text = str(components or "")
    # Коды вида 3.0.A3.203, 50.0.H49.203 и т.п.
    return set(re.findall(r"\b\d+(?:\.[A-Za-zА-Яа-я0-9]+)+\b", text))


def _deduplicate_matched(items: list[MatchedItem]) -> list[MatchedItem]:
    result = []
    seen = set()
    for item in items:
        key = item.code if item.code else normalize_text(item.matched_name or item.input_name)
        if not key:
            continue
        # Included не удаляем, если это пояснение о вхождении.
        if item.status != "included" and key in seen:
            continue
        if item.status != "included":
            seen.add(key)
        result.append(item)
    return result


def _apply_complex_inclusions(items: list[MatchedItem], price_list: list[LabTest]) -> list[MatchedItem]:
    """
    Если в списке уже есть комплекс, одиночные анализы из его components
    переводим в status=included.
    """
    result: list[MatchedItem] = []
    complexes = [item for item in items if item.status == "found" and item.components]

    for item in items:
        if item.status != "found" or item.components:
            result.append(item)
            continue

        included_in = ""
        for complex_item in complexes:
            codes = _component_codes(complex_item.components)
            if item.code and item.code in codes:
                included_in = complex_item.matched_name
                break

            item_name = normalize_text(item.matched_name or item.input_name)
            comp_text = normalize_text(complex_item.components)
            if item_name and len(item_name) >= 4 and item_name in comp_text:
                included_in = complex_item.matched_name
                break

        if included_in:
            result.append(make_included_item(item.input_name, included_in, price_list, code=item.code or ""))
        else:
            result.append(item)

    return result


def _suggest_cheaper_complexes(items: list[MatchedItem], price_list: list[LabTest]) -> list[MatchedItem]:
    """
    Консервативное правило:
    если 5+ одиночных анализов входят в один комплекс и комплекс дешевле суммы,
    заменяем на комплекс, а одиночные помечаем как included.
    Для коагулограммы допускаем 3+ компонента.
    """
    found_codes = {item.code for item in items if item.status == "found" and item.code}
    existing_complex_codes = {item.code for item in items if item.status == "found" and item.components}
    singles_by_code = {
        item.code: item
        for item in items
        if item.status == "found" and item.code and not item.components and item.price is not None
    }

    candidate_complexes: list[tuple[int, int, LabTest, set[str]]] = []
    for complex_item in price_list:
        if complex_item.code in existing_complex_codes:
            continue
        if complex_item.result_type != "комплекс" and not complex_item.components:
            continue

        codes = _component_codes(complex_item.components)
        matched = codes & set(singles_by_code.keys())
        if not matched:
            continue

        min_count = 3 if "коагулограмма" in normalize_text(complex_item.name) else 5
        if len(matched) < min_count:
            continue

        sum_price = sum(int(singles_by_code[code].price or 0) for code in matched)
        if complex_item.price and sum_price > int(complex_item.price):
            economy = sum_price - int(complex_item.price)
            candidate_complexes.append((economy, len(matched), complex_item, matched))

    if not candidate_complexes:
        return items

    candidate_complexes.sort(key=lambda x: (-x[0], -x[1], x[2].price, x[2].name))
    _, _, selected_complex, matched_component_codes = candidate_complexes[0]

    complex_candidate = labtest_to_candidate(selected_complex, 100)
    complex_matched = candidate_to_matched(selected_complex.name, complex_candidate, candidates=[complex_candidate])

    new_items = [complex_matched]
    for item in items:
        if item.code in matched_component_codes:
            new_items.append(make_included_item(item.input_name, selected_complex.name, price_list, code=item.code or ""))
        else:
            new_items.append(item)

    return _deduplicate_matched(_apply_complex_inclusions(new_items, price_list))


def _python_search(cleaned_items: list[str], price_list: list[LabTest]) -> list[MatchedItem]:
    matched = [_match_one_python(item, price_list) for item in cleaned_items if str(item).strip()]
    matched = _deduplicate_matched(matched)
    matched = _suggest_cheaper_complexes(matched, price_list)
    matched = _apply_complex_inclusions(matched, price_list)
    matched = _deduplicate_matched(matched)
    return matched


async def match_user_request_with_ai(raw_text: str, price_list: list[LabTest], initial_usage: AIUsage | None = None, source: str = "text") -> AIMatchResult:
    """
    Новая архитектура:
    1. ИИ распознает речь/фото в текст — это делают extract_* функции.
    2. ИИ очищает текст и разделяет анализы через ';'.
    3. Python получает готовый список и сам ищет в price_list.xlsx + alias_groups.xlsx.
    4. Python проверяет комплексы, дубли и included-компоненты.
    """
    usage = AIUsage(used=False, source="")
    if initial_usage:
        usage.add(initial_usage)

    if is_ai_enabled():
        try:
            cleaned_items, clean_usage = await _clean_text_with_gpt(raw_text)
            usage.add(clean_usage)
        except Exception:
            cleaned_items = clean_recognized_text_locally(raw_text)
    else:
        cleaned_items = clean_recognized_text_locally(raw_text)

    if not cleaned_items:
        cleaned_items = split_user_tests(raw_text)

    matched_items = _python_search(cleaned_items, price_list)

    if cleaned_items:
        remember_cleaned_text(raw_text, cleaned_items)

    return AIMatchResult(
        matched_items=matched_items,
        cleaned_items=cleaned_items,
        raw_text=raw_text,
        usage=usage,
        used_gpt_matching=False,
    )
