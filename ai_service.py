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
        if not item:
            continue
        key = item.lower().replace("ё", "е")
        if key not in seen:
            result.append(item)
            seen.add(key)
    return result


def extract_tests_locally(text: str) -> list[str]:
    """
    Дешевый локальный парсер без ИИ.
    Используется для обычного текста и для текста после аудио/OCR.
    Не пытается медицински расшифровывать сокращения — это делает pricing.py.
    """
    value = str(text or "").strip()
    if not value:
        return []

    # Убираем типичные вводные фразы из голосовых сообщений.
    value = re.sub(
        r"(?i)\b(мне|нам|пациенту)?\s*(нужн[аоы]?|сдать|сдайте|надо|нужно|нужны|пожалуйста|анализы|анализ)\b",
        " ",
        value,
    )
    value = re.sub(r"(?i)\b(и\s+еще|ещ[её]|также|плюс|добавьте)\b", ",", value)
    value = value.replace(";", ",")
    value = value.replace("\n", ",")

    parts = []
    for part in value.split(","):
        item = re.sub(r"^\s*[0-9]+[.)\-:]*\s*", "", part).strip(" -—\t\n\r")
        item = re.sub(r"\s+", " ", item).strip()
        if len(item) < 2:
            continue
        # Отсекаем явные служебные куски.
        if item.lower() in {"кровь", "моча", "список", "цены", "цену"}:
            continue
        parts.append(item)

    return _normalize_tests(parts)


def _extract_tests_from_text_sync(text: str, source: str) -> AIExtractResult:
    """
    Дорогой fallback для сложного текста. В обычном сценарии не используется.
    Оставлен на случай будущего ручного включения.
    """
    client = _client()
    prompt = "Извлеки только список медицинских анализов. Верни JSON {\"tests\":[\"...\"]}."
    response = client.responses.create(
        model=OPENAI_TEXT_MODEL,
        input=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text or ""},
        ],
        temperature=0,
        max_output_tokens=200,
    )
    output_text = getattr(response, "output_text", "") or ""
    usage = _usage_from_response(response, source)
    try:
        data = _safe_json_loads(output_text)
        tests = _normalize_tests(data.get("tests"))
    except Exception:
        tests = extract_tests_locally(output_text)
    return AIExtractResult(raw_text=text or "", tests=tests, usage=usage)


async def extract_tests_from_text(text: str, source: str = "text_ai") -> AIExtractResult:
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
    # ВАЖНО: после транскрибации НЕ отправляем текст во второй GPT-запрос.
    # Экономия: минус один текстовый запрос на каждое голосовое сообщение.
    text, audio_usage = await asyncio.to_thread(_transcribe_audio_sync, path)
    tests = extract_tests_locally(text)
    return AIExtractResult(raw_text=text, tests=tests, usage=audio_usage)


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
    tests = extract_tests_locally(text)
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
        "Извлеки только названия анализов/комплексов, без цен, дат и комментариев. "
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
        tests = extract_tests_locally(output_text)
        raw_text = output_text
    return AIExtractResult(raw_text=raw_text, tests=tests, usage=usage)


async def extract_tests_from_image_file(path: str) -> AIExtractResult:
    # Шаг 1: пробуем бесплатный локальный OCR, если он доступен.
    ocr_text = await asyncio.to_thread(_try_local_ocr, path)
    if _ocr_text_is_good_enough(ocr_text):
        return AIExtractResult(
            raw_text=ocr_text,
            tests=extract_tests_locally(ocr_text),
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
