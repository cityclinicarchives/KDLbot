import base64
import io
from typing import Tuple

from openai import OpenAI

from config import OPENAI_API_KEY, OPENAI_AUDIO_MODEL, OPENAI_VISION_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)


def usage_total(usage) -> int:
    if usage is None:
        return 0
    value = getattr(usage, "total_tokens", None)
    if value is not None:
        return int(value)
    try:
        return int(usage.get("total_tokens", 0))
    except Exception:
        return 0


def compress_image(image_bytes: bytes, max_side: int = 1280, quality: int = 75) -> bytes:
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = img.size
        scale = min(1.0, max_side / max(w, h))
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)))
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=quality, optimize=True)
        return out.getvalue()
    except Exception:
        return image_bytes


async def transcribe_voice_bytes(audio_bytes: bytes, filename: str = "voice.ogg") -> Tuple[str, int]:
    response = client.audio.transcriptions.create(
        model=OPENAI_AUDIO_MODEL,
        file=(filename, audio_bytes),
    )
    text = getattr(response, "text", "") or ""
    return text.strip(), usage_total(getattr(response, "usage", None))


async def extract_text_from_image_bytes(image_bytes: bytes) -> Tuple[str, int]:
    image_bytes = compress_image(image_bytes)
    b64 = base64.b64encode(image_bytes).decode("utf-8")

    response = client.responses.create(
        model=OPENAI_VISION_MODEL,
        temperature=0,
        max_output_tokens=300,
        input=[
            {
                "role": "system",
                "content": (
                    "Извлеки из изображения только список медицинских лабораторных анализов. "
                    "Не добавляй комментарии. Каждый анализ — с новой строки. "
                    "Игнорируй назначения, диагнозы, процедуры, УЗИ, ЭКГ, рентген, консультации и лишний текст."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Распознай список лабораторных анализов."},
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"},
                ],
            },
        ],
    )
    return (response.output_text or "").strip(), usage_total(getattr(response, "usage", None))
