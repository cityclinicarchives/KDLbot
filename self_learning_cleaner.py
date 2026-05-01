import json
import os
import re
from typing import Iterable

MEMORY_FILE = "cleaner_memory.json"

STOP_WORDS = {
    "ну", "вот", "это", "так", "еще", "ещё", "надо", "нужно", "хочу", "сдать",
    "сдавать", "анализ", "анализы", "пожалуйста", "наверное", "можно", "мне",
    "назначили", "посмотрите", "сделать", "сделайте"
}

HUMAN_MAP = {
    "кровь": "ОАК",
    "общий кровь": "ОАК",
    "общий анализ крови": "ОАК",
    "моча": "ОАМ",
    "общий мочи": "ОАМ",
    "общий анализ мочи": "ОАМ",
    "печеночные": "АЛТ\nАСТ",
    "печёночные": "АЛТ\nАСТ",
    "свертываемость": "коагулограмма",
    "свёртываемость": "коагулограмма",
}


def _normalize(text: str) -> str:
    text = str(text or "").lower().replace("ё", "е")
    text = re.sub(r"[,.!?;:()\[\]{}\"'«»]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_memory() -> dict[str, str]:
    if not os.path.exists(MEMORY_FILE):
        return {}
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_memory(memory: dict[str, str]) -> None:
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)


def clean_text(text: str) -> list[str]:
    normalized = _normalize(text)
    if not normalized:
        return []

    results: list[str] = []

    for key, value in HUMAN_MAP.items():
        if key in normalized:
            results.extend([v.strip() for v in value.splitlines() if v.strip()])

    memory = load_memory()
    for key, value in memory.items():
        if key and key in normalized:
            results.extend([v.strip() for v in str(value).splitlines() if v.strip()])

    parts = re.split(r"\b(?:и|также|плюс|а еще|а ещё|,)\b", normalized)
    for part in parts:
        words = [w for w in part.split() if w not in STOP_WORDS]
        candidate = " ".join(words).strip()
        if len(candidate) >= 2 and candidate not in results:
            results.append(candidate)

    deduped = []
    seen = set()
    for item in results:
        key = _normalize(item)
        if key and key not in seen:
            seen.add(key)
            deduped.append(item)

    return deduped


def learn(original_text: str, final_items: Iterable[str]) -> None:
    normalized = _normalize(original_text)
    final_list = [str(x).strip() for x in final_items if str(x).strip()]
    if not normalized or not final_list:
        return

    memory = load_memory()
    tokens = [t for t in normalized.split() if len(t) >= 4 and t not in STOP_WORDS]
    final_value = "\n".join(final_list[:5])

    for token in tokens:
        memory.setdefault(token, final_value)

    save_memory(memory)
