import os
import sqlite3
import logging
import asyncio
import threading
from typing import List

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.types import Message
from aiogram.filters import Command
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramAPIError


logging.getLogger("aiogram").setLevel(logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
DB_PATH = os.getenv("DB_PATH", "/app/db/subscribers.db")


def escape_markdown_v2(text: str) -> str:
    """Надежно экранирует текст для Telegram MarkdownV2."""
    escape_chars = r'_*[]()~`>#+-=|{}.!'
    return ''.join(f'\\{char}' if char in escape_chars else char for char in text)

class TelegramBotManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if not cls._instance:
                cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        if not TELEGRAM_BOT_TOKEN:
            raise ValueError("TELEGRAM_BOT_TOKEN is not set!")
            
        self.db_conn = self._init_db()
        
        default_properties = DefaultBotProperties(parse_mode=ParseMode.MARKDOWN_V2)
        self.bot = Bot(token=TELEGRAM_BOT_TOKEN, default=default_properties)
        self.dp = Dispatcher()
        self._register_handlers()
        
        self._thread = None
        self._initialized = True
        logger.info("Telegram bot (aiogram) initialized.")

    def _init_db(self):
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS subscribers (chat_id INTEGER PRIMARY KEY)")
        conn.commit()
        logger.info(f"Database initialized at {DB_PATH}")
        return conn
        
    def _register_handlers(self):
        self.dp.message.register(self.start_command, Command(commands=["start", "subscribe"]))
        self.dp.message.register(self.stop_command, Command(commands=["stop", "unsubscribe"]))
    
    def _add_subscriber(self, chat_id: int):
        cursor = self.db_conn.cursor()
        cursor.execute("INSERT OR IGNORE INTO subscribers (chat_id) VALUES (?)", (chat_id,))
        self.db_conn.commit()
        logger.info(f"New subscriber added: {chat_id}")

    def _remove_subscriber(self, chat_id: int):
        cursor = self.db_conn.cursor()
        cursor.execute("DELETE FROM subscribers WHERE chat_id = ?", (chat_id,))
        self.db_conn.commit()
        logger.info(f"Subscriber removed: {chat_id}")

    def _get_all_subscribers(self) -> List[int]:
        cursor = self.db_conn.cursor()
        cursor.execute("SELECT chat_id FROM subscribers")
        return [row[0] for row in cursor.fetchall()]

    async def start_command(self, message: Message):
        chat_id = message.chat.id
        logger.info(f"Received /start command from chat_id: {chat_id}")
        try:
            self._add_subscriber(chat_id)
            text = (
                "Вы успешно подписались на уведомления о корреляциях логов\\!\n"
                "Чтобы отписаться, отправьте /stop\\."
            )
            await message.answer(text)
        except Exception as e:
            logger.error(f"Error in start_command for chat_id {chat_id}: {e}", exc_info=True)
            await message.answer("Произошла ошибка при подписке\\. Пожалуйста, попробуйте позже\\.")

    async def stop_command(self, message: Message):
        chat_id = message.chat.id
        logger.info(f"Received /stop command from chat_id: {chat_id}")
        try:
            self._remove_subscriber(chat_id)
            await message.answer("Вы отписались от уведомлений\\.")
        except Exception as e:
            logger.error(f"Error in stop_command for chat_id {chat_id}: {e}", exc_info=True)
            await message.answer("Произошла ошибка при отписке\\.")

    async def send_notification_async(self, result: dict):
        subscribers = self._get_all_subscribers()
        if not subscribers:
            logger.info("No subscribers to notify.")
            return

        problem_id = escape_markdown_v2(str(result.get('problem_id', 'N/A')))
        anomaly_id = escape_markdown_v2(str(result.get('anomaly_id', 'N/A')))
        log_line = escape_markdown_v2(str(result.get('log', 'N/A')))

        message_text = (
            f"*Обнаружена связь*\n\n"
            f"*ID проблемы:* `{problem_id}`\n"
            f"*ID аномалии:* `{anomaly_id}`\n\n"
            f"*Строка лога*:\n"
            f"```\n{log_line}\n```"
        )
        
        for chat_id in subscribers:
            try:
                await self.bot.send_message(chat_id=chat_id, text=message_text)
            except TelegramAPIError as e:
                logger.error(f"Failed to send notification to chat_id {chat_id}: {e}")
            except Exception as e:
                logger.error(f"An unexpected error occurred sending to {chat_id}: {e}")
        
        logger.info(f"Attempted to send notifications to {len(subscribers)} subscribers.")

    async def _start_polling(self):
        await self.bot.delete_webhook(drop_pending_updates=True)
        await self.dp.start_polling(self.bot, handle_signals=False)

    def start_bot_in_thread(self):
        if self._thread and self._thread.is_alive():
            logger.warning("Bot thread is already running.")
            return

        def _run():
            try:
                asyncio.run(self._start_polling())
            except Exception as e:
                logger.critical(f"Bot thread crashed: {e}", exc_info=True)
        
        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        logger.info("Telegram bot (aiogram) thread started.")

bot_manager = None
if TELEGRAM_BOT_TOKEN:
    try:
        bot_manager = TelegramBotManager() 
    except Exception as e:
        logger.error(f"Failed to initialize TelegramBotManager: {e}")

def send_notification(result: dict):
    if not bot_manager:
        return
    
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(bot_manager.send_notification_async(result))
    except RuntimeError:
        asyncio.run(bot_manager.send_notification_async(result))