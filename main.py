import os
import logging
import requests
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
TARGET_CHAT_ID = int(os.environ["TARGET_CHAT_ID"])
API_FOOTBALL_KEY = os.environ["API_FOOTBALL_KEY"]
API_FOOTBALL_HOST = os.environ["API_FOOTBALL_HOST"]

BASE_URL = "https://api-football-v1.p.rapidapi.com/v3/fixtures"

headers = {
    "X-RapidAPI-Key": API_FOOTBALL_KEY,
    "X-RapidAPI-Host": API_FOOTBALL_HOST,
}


# ==========================
# TELEGRAM COMMANDS
# ==========================

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Bot activo ✅")

async def cmd_list(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Comando list recibido.")

async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Historial limpiado.")


# ==========================
# FOOTBALL POLLING
# ==========================

async def poll_results(app: Application):
    try:
        params = {"live": "all"}
        response = requests.get(BASE_URL, headers=headers, params=params, timeout=10)
        data = response.json()

        if "response" in data and data["response"]:
            for match in data["response"]:
                home = match["teams"]["home"]["name"]
                away = match["teams"]["away"]["name"]
                goals_home = match["goals"]["home"]
                goals_away = match["goals"]["away"]

                message = f"⚽ {home} {goals_home}-{goals_away} {away}"

                await app.bot.send_message(
                    chat_id=TARGET_CHAT_ID,
                    text=message
                )

    except Exception as e:
        log.error(f"Error en poll_results: {e}")


async def poll_job(context: ContextTypes.DEFAULT_TYPE):
    await poll_results(context.application)


# ==========================
# STARTUP
# ==========================

async def on_startup(app: Application):
    log.info("Bot started. Target chat id: %s", TARGET_CHAT_ID)

    app.job_queue.run_repeating(
        poll_job,
        interval=60,
        first=5,
        name="poll_results",
    )


# ==========================
# MAIN
# ==========================

def main():
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("list", cmd_list))
    application.add_handler(CommandHandler("clear", cmd_clear))

    application.post_init = on_startup

    application.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
