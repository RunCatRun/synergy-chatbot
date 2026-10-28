"""
telegram_bot.py — чат-бот Университета «Синергия» в Telegram.

Использует то же ядро распознавания, что и консольный bot.py:
тон ответов, уровень логики и семантический метод берутся из настроек
в начале bot.py.

Запуск:
  1. Создать бота у @BotFather и получить токен.
  2. Установить зависимости: pip install -r requirements.txt
  3. Задать токен в переменной окружения BOT_TOKEN:
       set BOT_TOKEN=123456:ABC...          (Windows, cmd)
       $env:BOT_TOKEN = "123456:ABC..."     (Windows, PowerShell)
       export BOT_TOKEN=123456:ABC...       (Linux, macOS)
  4. Если Telegram напрямую недоступен, задать прокси в переменной BOT_PROXY
     (SOCKS5 или HTTP), например:
       export BOT_PROXY=socks5://user:pass@host:1080
     Без этой переменной бот подключается напрямую.
  5. python telegram_bot.py

Бот работает, пока запущена программа. Модель ru_core_news_lg занимает
около 1 ГБ оперативной памяти — это нужно учесть при выборе сервера.
"""

import asyncio
import logging
import os
import threading

from aiogram import Bot, Dispatcher, F
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.filters import Command, CommandStart
from aiogram.types import Message

import bot as console
from chatbot.knowledge import load_knowledge

PREPROCESS, MATCHER = console.build(
    console.TONE, console.LEVEL, console.SEMANTIC_METHOD
)
ANSWERS = {
    c["code"]: c["answer"] for c in load_knowledge(console.TONE)["categories"]
}

# Сопоставитель один на всех пользователей, а память диалога у каждого
# чата своя: перед ответом она подставляется, после ответа сохраняется.
# На уровне 3 память хранится во вложенном сопоставителе уровня 2.
_STATE = getattr(MATCHER, "level2", MATCHER)
_MEMORY = {}
_LOCK = threading.Lock()


def reply(chat_id, text):
    """Возвращает ответ бота на фразу пользователя из чата chat_id."""
    with _LOCK:
        if hasattr(_STATE, "last_intent"):
            _STATE.last_intent = _MEMORY.get(chat_id)
        _code, answer, _score = MATCHER.match(PREPROCESS(text), text)
        if hasattr(_STATE, "last_intent"):
            _MEMORY[chat_id] = _STATE.last_intent
    return answer


dp = Dispatcher()


@dp.message(CommandStart())
async def on_start(message: Message):
    await message.answer(ANSWERS["greeting"])


@dp.message(Command("help"))
async def on_help(message: Message):
    await message.answer(ANSWERS["help"])


@dp.message(F.text)
async def on_text(message: Message):
    # Распознавание занимает процессор, поэтому выполняется в отдельном
    # потоке и не задерживает ответы другим пользователям.
    answer = await asyncio.to_thread(reply, message.chat.id, message.text)
    await message.answer(answer)


async def main():
    token = os.environ.get("BOT_TOKEN")
    if not token:
        raise SystemExit("Не задан токен: переменная окружения BOT_TOKEN.")
    # Прокси для выхода к Telegram, например socks5://user:pass@host:1080.
    # Если переменная пустая, бот подключается напрямую.
    proxy = os.environ.get("BOT_PROXY")
    session = AiohttpSession(proxy=proxy) if proxy else None
    await dp.start_polling(Bot(token=token, session=session))


if __name__ == "__main__":
    # Журнал aiogram (подключение, ошибки сети) — в консоль или journalctl.
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    asyncio.run(main())
