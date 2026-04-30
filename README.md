
# Telegram-бот для лаборатории анализов — Excel-версия

В этой версии прайс-листы переведены из CSV в Excel:

```text
price_list.xlsx
alias_groups.xlsx
```

## Файлы данных

### price_list.xlsx

Лист `price_list`.

Обязательные колонки:

```text
code
name
result_type
max_discount
term_days
price
synonyms
```

Необязательные колонки, если они есть в вашей версии прайса:

```text
short_name
group
priority
```

### alias_groups.xlsx

Лист `alias_groups`.

Колонки:

```text
alias
group
```

## Важно

Бот теперь по умолчанию читает:

```text
price_list.xlsx
alias_groups.xlsx
```

CSV-файлы больше не нужны для работы. Для страховки в коде оставлен fallback: если Excel-файла нет, бот попробует прочитать старый CSV.

## Установка зависимостей

В `requirements.txt` добавлен пакет:

```text
openpyxl==3.*
```

На Railway после `commit → push` зависимости установятся автоматически.

## Деплой на Railway

1. Замените файлы в GitHub файлами из этого архива.
2. Убедитесь, что в репозитории есть:
   - `price_list.xlsx`
   - `alias_groups.xlsx`
   - `pricing.py`
   - `requirements.txt`
3. Сделайте `commit → push`.
4. В Railway нажмите `Redeploy`.
5. В Telegram напишите:

```text
/reset
```

И проверьте поиск анализов.
