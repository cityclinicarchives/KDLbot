
# Telegram-бот лаборатории — GPT-подбор анализов

## Что изменено

В этой версии первичный подбор анализов выполняет `gpt-4.1-nano`.

Архитектура:

```text
Текст / голос / изображение
↓
если голос/изображение — перевод в текст
↓
self-learning cleaner
↓
GPT-подбор по прайс-листам
↓
текущий сценарий бота: проверка списка, изменение части, удаление части, скидка, заказ
```

## Важные правила новой версии

- GPT получает не весь прайс, а предварительно отобранный shortlist, чтобы снизить стоимость.
- Для одиночных анализов GPT ориентируется на колонку `name` из `price_list.xlsx`.
- Для комплексов GPT ориентируется на `name` и `components` из `complex_price_list.xlsx`.
- Синонимы не используются как основной источник подбора.
- Если анализ входит в выбранный комплекс, он остается в списке, но без цены: `входит в комплекс ...`.
- В `price_list.xlsx` сброшены все старые приоритеты, кроме:
  - `Клинический анализ крови с лейкоцитарной формулой (5DIFF) и СОЭ (венозная кровь)` → `priority = 100`.

## Файлы данных

Обязательные файлы:

```text
price_list.xlsx
complex_price_list.xlsx
alias_groups.xlsx
```

## Переменные Railway

```env
BOT_TOKEN=...
LAB_CHAT_ID=...
OPENAI_API_KEY=...
OPENAI_TEXT_MODEL=gpt-4.1-nano
OPENAI_VISION_MODEL=gpt-4.1-nano
OPENAI_AUDIO_MODEL=gpt-4o-mini-transcribe
```

## Зависимости

```bash
pip install -r requirements.txt
```

## Деплой

```text
commit → push → Railway Redeploy → /reset
```
