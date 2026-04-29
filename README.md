# Telegram-бот для лаборатории анализов

В проект уже встроен реальный файл `price_list.csv` с колонками:

```csv
code,name,result_type,max_discount,term_days,price,synonyms
```

Бот ищет анализы:

1. по коду анализа `code`;
2. по официальному названию `name`;
3. по синонимам `synonyms`;
4. через нечеткое совпадение RapidFuzz.

В выдаче пациенту и лаборатории показываются:

- код анализа;
- название;
- тип результата;
- срок выполнения;
- цена;
- максимальная скидка;
- примененная скидка;
- итоговая цена.

## Локальный запуск

1. Установите Python 3.11 или 3.12.
2. Откройте папку проекта в VS Code.
3. Создайте виртуальное окружение:

```powershell
python -m venv .venv
```

4. Активируйте его:

```powershell
.\.venv\Scripts\Activate.ps1
```

5. Установите зависимости:

```powershell
pip install -r requirements.txt
```

6. Создайте файл `.env` по примеру `.env.example`:

```env
BOT_TOKEN=токен_бота_от_BotFather
LAB_CHAT_ID=0
```

7. Запустите бота:

```powershell
python bot.py
```

## Railway

1. Загрузите проект в GitHub.
2. На Railway создайте New Project → Deploy from GitHub repo.
3. Выберите репозиторий.
4. В Variables добавьте:

```env
BOT_TOKEN=токен_бота_от_BotFather
LAB_CHAT_ID=0
```

5. Start Command:

```bash
python bot.py
```

6. После запуска добавьте бота в группу лаборатории, сделайте администратором и напишите в группе:

```text
/chatid
```

7. Скопируйте полученный chat_id и замените `LAB_CHAT_ID=0` на реальный ID в Railway Variables.
8. Выполните Redeploy.
