import asyncio
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

from config import (
    OPENAI_API_KEY,
    OPENAI_TEXT_MODEL,
    OPENAI_VISION_MODEL,
    OPENAI_AUDIO_MODEL,
)


@dataclass
class AIUsage:
    used: bool = False
    source: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add(self, other: "AIUsage") -> None:
        self.used = self.used or other.used
        if other.source:
            self.source = other.source if not self.source else f"{self.source} + {other.source}"
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AIExtractResult:
    raw_text: str
    tests: list[str]
    usage: AIUsage


def is_ai_enabled() -> bool:
    return bool(OPENAI_API_KEY)


def _client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY не задан. Добавьте ключ в .env или Railway Variables.")
    return OpenAI(api_key=OPENAI_API_KEY)


def _usage_from_response(response: Any, source: str) -> AIUsage:
    usage = getattr(response, "usage", None)
    prompt_tokens = int(getattr(usage, "input_tokens", 0) or getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "output_tokens", 0) or getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", 0) or (prompt_tokens + completion_tokens))
    return AIUsage(
        used=True,
        source=source,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


def _safe_json_loads(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise


def _normalize_tests(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value or "").strip().strip(" -—\t\n\r")
        item = re.sub(r"\s+", " ", item).strip()
        if not item:
            continue
        key = item.lower().replace("ё", "е")
        if key not in seen:
            result.append(item)
            seen.add(key)
    return result


FILLER_WORDS = {
    "ну", "я", "мы", "мне", "нам", "хочу", "хотим", "хотел", "хотела",
    "сдать", "сдавать", "сдайте", "сделать", "сделайте", "надо", "нужно",
    "нужен", "нужна", "нужны", "пожалуйста", "так", "там", "вот", "это",
    "наверное", "может", "быть", "еще", "ещё", "также", "плюс", "и",
    "а", "список", "анализы", "анализ", "исследование", "исследования",
}

# Частые разговорные фразы, которые лучше сразу приводить к медицинским коротким алиасам.
# Дальше точное сопоставление всё равно делает pricing.py.
CANONICAL_PHRASES: list[tuple[str, str]] = [
    (r"\bбиохим(?:ия|ический анализ)?\s+крови\b", "биохимия крови"),
    (r"\bобщ(?:ий|его)?\s+анализ\s+крови\b", "ОАК"),
    (r"\bклиническ(?:ий|ого)?\s+анализ\s+крови\b", "ОАК"),
    (r"\bанализ\s+крови\b", "ОАК"),
    (r"\bобщ(?:ий|его)?\s+анализ\s+мочи\b", "ОАМ"),
    (r"\bанализ\s+мочи\b", "ОАМ"),
    (r"\bд\s*-?\s*димер\b", "Д-димер"),
    (r"\bтиреотропн(?:ый|ого)?\s+гормон\b", "ТТГ"),
    (r"\bт\s*т\s*г\b", "ТТГ"),
    (r"\bа\s*л\s*т\b", "АЛТ"),
    (r"\bа\s*с\s*т\b", "АСТ"),
    (r"\bс\s*р\s*б\b", "СРБ"),
    (r"\bм\s*н\s*о\b", "МНО"),
    (r"\bсоэ\b", "СОЭ"),
    (r"\bферритин\b", "ферритин"),
    (r"\bглюкоз[ауеы]?\b", "глюкоза"),
    (r"\bсахар\s+крови\b", "глюкоза"),
    (r"\bвитамин\s*д\b", "витамин Д"),
    (r"\bвитамин\s*d\b", "витамин D"),
    (r"\bвитамин\s*б\s*12\b", "витамин B12"),
    (r"\bвитамин\s*b\s*12\b", "витамин B12"),
]

SHORT_TEST_PATTERN = re.compile(
    r"\b(оак|оам|алт|аст|ттг|т3|т4|срб|соэ|мно|пса|хгч|вич|hiv|hbsag|anti\s*hcv|anti-hcv|igg|igm|iga)\b",
    flags=re.IGNORECASE,
)


def _normalize_for_cleaning(text: str) -> str:
    value = str(text or "").replace("ё", "е")
    value = re.sub(r"\b(запятая|точка с запятой|новая строка|следующая строка)\b", ",", value, flags=re.IGNORECASE)
    value = re.sub(r"[\n\r;]+", ",", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _strip_filler_edges(text: str) -> str:
    words = text.split()
    while words and words[0].lower().strip(".,!?;:") in FILLER_WORDS:
        words.pop(0)
    while words and words[-1].lower().strip(".,!?;:") in FILLER_WORDS:
        words.pop()
    return " ".join(words).strip(" -—,.;:\t\n\r")


def _looks_like_filler(text: str) -> bool:
    value = str(text or "").strip().lower()
    if not value:
        return True
    tokens = [t.strip(".,!?;:") for t in value.split() if t.strip(".,!?;:")]
    if not tokens:
        return True
    if all(token in FILLER_WORDS for token in tokens):
        return True
    return value in {"кровь", "моча", "цены", "цену", "список"}


def extract_tests_locally(text: str, *, mode: str = "general") -> list[str]:
    """
    Бесплатный локальный очиститель распознанного текста.

    Цель: из разговорной фразы получить список коротких запросов для pricing.py.
    Например:
    "Ну я хочу сдать анализ крови, а еще анализ мочи и наверное АЛТ и АСТ"
    -> ["ОАК", "ОАМ", "АЛТ", "АСТ"]

    Важно: функция не заменяет медицинский поиск. Она только убирает мусорные слова.
    """
    value = _normalize_for_cleaning(text)
    if not value:
        return []

    extracted: list[str] = []

    # 1. Вытаскиваем частые разговорные фразы и заменяем их каноническими алиасами.
    working = value
    for pattern, canonical in CANONICAL_PHRASES:
        if re.search(pattern, working, flags=re.IGNORECASE):
            extracted.append(canonical)
            working = re.sub(pattern, " , ", working, flags=re.IGNORECASE)

    # 2. Вытаскиваем короткие устойчивые маркеры: АЛТ, АСТ, ТТГ и т.д.
    for match in SHORT_TEST_PATTERN.finditer(working):
        marker = re.sub(r"\s+", "", match.group(1)).upper()
        marker = marker.replace("ANTI", "anti") if marker.startswith("ANTI") else marker
        extracted.append(marker)
        working = working.replace(match.group(0), " , ")

    # 3. Остаток режем по естественным разделителям списков.
    working = re.sub(r"\b(и\s+еще|и\s+ещё|а\s+еще|а\s+ещё|также|плюс|добавьте|еще|ещё)\b", ",", working, flags=re.IGNORECASE)
    # Союз "и" часто разделяет список: "АЛТ и АСТ", "ферритин и глюкоза".
    working = re.sub(r"\s+и\s+", ",", working, flags=re.IGNORECASE)
    working = re.sub(r"[.]+", ",", working)

    for part in working.split(","):
        item = re.sub(r"^\s*[0-9]+[.)\-:]*\s*", "", part).strip(" -—\t\n\r")
        item = re.sub(
            r"(?i)\b(мне|нам|пациенту|хочу|хотим|надо|нужно|сдать|сдайте|сделать|сделайте|пожалуйста|наверное|так|ну|вот|это)\b",
            " ",
            item,
        )
        item = re.sub(r"\s+", " ", item).strip(" -—\t\n\r")
        item = _strip_filler_edges(item)
        if len(item) < 2 or _looks_like_filler(item):
            continue
        extracted.append(item)

    # 4. Убираем дубли, сохраняя порядок.
    result: list[str] = []
    seen: set[str] = set()
    for item in extracted:
        cleaned = re.sub(r"\s+", " ", str(item or "")).strip(" -—\t\n\r")
        if not cleaned or _looks_like_filler(cleaned):
            continue
        key = cleaned.lower().replace("ё", "е")
        if key not in seen:
            result.append(cleaned)
            seen.add(key)

    return result


def local_extraction_is_good_enough(raw_text: str, tests: list[str], *, mode: str = "audio") -> bool:
    """
    Проверяет, можно ли доверять локальному очистителю.
    Если нет — включаем GPT fallback.
    """
    if not tests:
        return False

    # Если остались длинные разговорные куски, лучше один раз спросить дешевую текстовую модель.
    suspicious_words = {
        "хочу", "нужно", "надо", "наверное", "пожалуйста", "добавьте",
        "сдать", "сделать", "анализы", "исследования",
    }
    for item in tests:
        tokens = [t.lower().strip(".,!?;:") for t in item.split()]
        if len(item) > 90:
            return False
        if sum(1 for token in tokens if token in suspicious_words) >= 2:
            return False

    # Для голосовых фраз это хороший признак: список распался на 2+ коротких пунктов.
    if mode == "audio" and len(tests) >= 2:
        return True

    # Один короткий понятный пункт тоже принимаем.
    if len(tests) == 1 and len(tests[0]) <= 50:
        return True

    return len(tests) >= 1


def _extract_tests_from_text_sync(text: str, source: str) -> AIExtractResult:
    """
    Дешевый GPT fallback. Вызывается только если локальный очиститель не справился.
    """
    client = _client()
    prompt = (
        "Ты извлекаешь медицинские анализы из распознанной фразы пациента. "
        "Удали слова-паразиты, просьбы, вводные слова и комментарии. "
        "Верни только JSON {\"tests\":[\"...\"]}. "
        "Если пациент сказал 'анализ крови' без уточнений, верни 'ОАК'. "
        "Если сказал 'анализ мочи' без уточнений, верни 'ОАМ'. "
        "Не добавляй анализы, которых нет в тексте."
    )
    response = client.responses.create(
        model=OPENAI_TEXT_MODEL,
        input=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text or ""},
        ],
        temperature=0,
        max_output_tokens=160,
    )
    output_text = getattr(response, "output_text", "") or ""
    usage = _usage_from_response(response, source)
    try:
        data = _safe_json_loads(output_text)
        tests = _normalize_tests(data.get("tests"))
    except Exception:
        tests = extract_tests_locally(output_text)
    return AIExtractResult(raw_text=text or "", tests=tests, usage=usage)


async def extract_tests_from_text(text: str, source: str = "text_cleanup_fallback") -> AIExtractResult:
    return await asyncio.to_thread(_extract_tests_from_text_sync, text, source)


def _transcribe_audio_sync(path: str) -> tuple[str, AIUsage]:
    client = _client()
    with open(path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model=OPENAI_AUDIO_MODEL,
            file=audio_file,
            language="ru",
        )
    text = getattr(response, "text", "") or str(response)
    usage = _usage_from_response(response, "audio_transcription")
    return text, usage


async def extract_tests_from_audio_file(path: str) -> AIExtractResult:
    # Основной путь: транскрибация -> локальная очистка без дополнительного GPT-запроса.
    text, audio_usage = await asyncio.to_thread(_transcribe_audio_sync, path)
    tests = extract_tests_locally(text, mode="audio")

    if local_extraction_is_good_enough(text, tests, mode="audio"):
        return AIExtractResult(raw_text=text, tests=tests, usage=audio_usage)

    # Fallback включается только при неудачной локальной очистке.
    fallback = await extract_tests_from_text(text, source="text_cleanup_fallback")
    total_usage = audio_usage
    total_usage.add(fallback.usage)
    return AIExtractResult(raw_text=text, tests=fallback.tests, usage=total_usage)


def _try_local_ocr(path: str) -> str:
    """
    Необязательный локальный OCR. Если в окружении нет pytesseract/tesseract,
    функция тихо вернет пустую строку, и код перейдет к vision-модели.
    """
    try:
        import pytesseract  # type: ignore
        from PIL import Image, ImageOps
    except Exception:
        return ""

    try:
        image = Image.open(path)
        image = ImageOps.exif_transpose(image)
        image = image.convert("L")
        text = pytesseract.image_to_string(image, lang="rus+eng")
        return str(text or "").strip()
    except Exception:
        return ""


def _ocr_text_is_good_enough(text: str) -> bool:
    tests = extract_tests_locally(text, mode="ocr")
    # Если OCR дал хотя бы 2 осмысленные строки или достаточно длинный текст,
    # не тратим vision-токены.
    return len(tests) >= 2 or len(str(text or "").strip()) >= 30


def _prepare_image_for_vision(path: str, max_side: int = 1280, quality: int = 75) -> str:
    """
    Сжимает изображение перед отправкой в vision-модель.
    Это снижает количество image-токенов и ускоряет ответ.
    """
    try:
        from PIL import Image, ImageOps
    except Exception:
        return path

    source = Path(path)
    fd, out_path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)

    try:
        image = Image.open(source)
        image = ImageOps.exif_transpose(image)
        image.thumbnail((max_side, max_side))
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        image.save(out_path, format="JPEG", quality=quality, optimize=True)
        return out_path
    except Exception:
        try:
            os.remove(out_path)
        except OSError:
            pass
        return path


def _extract_tests_from_image_with_vision_sync(path: str) -> AIExtractResult:
    client = _client()
    prepared_path = _prepare_image_for_vision(path)
    file_path = Path(prepared_path)
    mime_type = mimetypes.guess_type(file_path.name)[0] or "image/jpeg"
    image_b64 = base64.b64encode(file_path.read_bytes()).decode("utf-8")
    data_url = f"data:{mime_type};base64,{image_b64}"

    prompt = (
        "На изображении список медицинских анализов. "
        "Извлеки только названия анализов/комплексов, без цен, дат, ФИО и комментариев. "
        "Убери слова вроде 'сдать', 'нужно', 'анализ', если они не являются частью названия. "
        "Верни JSON {\"tests\":[\"...\"],\"raw_text\":\"...\"}."
    )
    try:
        response = client.responses.create(
            model=OPENAI_VISION_MODEL,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": data_url, "detail": "low"},
                    ],
                }
            ],
            temperature=0,
            max_output_tokens=300,
        )
    finally:
        if prepared_path != path:
            try:
                os.remove(prepared_path)
            except OSError:
                pass

    output_text = getattr(response, "output_text", "") or ""
    usage = _usage_from_response(response, "image_vision")
    try:
        data = _safe_json_loads(output_text)
        tests = _normalize_tests(data.get("tests"))
        raw_text = str(data.get("raw_text") or output_text)
    except Exception:
        tests = extract_tests_locally(output_text, mode="vision")
        raw_text = output_text
    return AIExtractResult(raw_text=raw_text, tests=tests, usage=usage)


async def extract_tests_from_image_file(path: str) -> AIExtractResult:
    # Шаг 1: пробуем бесплатный локальный OCR, если он доступен.
    ocr_text = await asyncio.to_thread(_try_local_ocr, path)
    if _ocr_text_is_good_enough(ocr_text):
        return AIExtractResult(
            raw_text=ocr_text,
            tests=extract_tests_locally(ocr_text, mode="ocr"),
            usage=AIUsage(used=False, source="local_ocr"),
        )

    # Шаг 2: если OCR недоступен или слабый — используем дешевую vision-модель.
    return await asyncio.to_thread(_extract_tests_from_image_with_vision_sync, path)


def format_ai_usage_for_lab(usage: dict[str, Any] | AIUsage | None) -> str:
    if not usage:
        return ""
    data = usage.to_dict() if isinstance(usage, AIUsage) else usage
    if not data.get("used"):
        return ""
    return (
        "\nИИ-слой:\n"
        f"Источник: {data.get('source') or 'не указано'}\n"
        f"Prompt/Input tokens: {int(data.get('prompt_tokens') or 0)}\n"
        f"Completion/Output tokens: {int(data.get('completion_tokens') or 0)}\n"
        f"Total tokens: {int(data.get('total_tokens') or 0)}"
    )
