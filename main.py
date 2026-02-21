import os
import logging
import requests
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

logging.basicConfig(level=logging.INFO)

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
API_FOOTBALL_KEY = os.environ["API_FOOTBALL_KEY"]
API_FOOTBALL_HOST = os.environ.get("API_FOOTBALL_HOST", "v3.football.api-sports.io")
TARGET_CHAT_ID = int(os.environ["TARGET_CHAT_ID"])

TRACKED_FIXTURES = set()
LAST_STATE = {}  # fixture_id -> (status, home_goals, away_goals)

def api_football_get(path, params=None):
    url = f"https://{API_FOOTBALL_HOST}{path}"
    headers = {"x-apisports-key": API_FOOTBALL_KEY}
    r = requests.get(url, headers=headers, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "✅ Bot activo.\n"
        "Comandos:\n"
        "/addfixture <fixture_id>\n"
        "/list\n"
        "/clear\n"
        "Publicaré cambios y resultados finales en el grupo."
    )

async def addfixture(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Uso: /addfixture <fixture_id>")
        return
    try:
        fid = int(context.args[0])
    except ValueError:
        await update.message.reply_text("fixture_id inválido.")
        return

    TRACKED_FIXTURES.add(fid)
    await update.message.reply_text(f"➕ Añadido fixture_id: {fid}")

async def listcmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not TRACKED_FIXTURES:
        await update.message.reply_text("No hay partidos en seguimiento.")
        return
    await update.message.reply_text("📌 En seguimiento:\n" + "\n".join(map(str, sorted(TRACKED_FIXTURES))))

async def clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    TRACKED_FIXTURES.clear()
    LAST_STATE.clear()
    await update.message.reply_text("🧹 Limpio. Sin partidos en seguimiento.")

async def poll_results(app: Application):
    if not TRACKED_FIXTURES:
        return

    for fid in list(TRACKED_FIXTURES):
        try:
            data = api_football_get("/fixtures", params={"id": fid})
            resp = data.get("response", [])
            if not resp:
                continue

            fx = resp[0]
            status = fx["fixture"]["status"]["short"]
            elapsed = fx["fixture"]["status"].get("elapsed")
            home = fx["teams"]["home"]["name"]
            away = fx["teams"]["away"]["name"]
            hg = fx["goals"]["home"]
            ag = fx["goals"]["away"]

            key = (status, hg, ag)
            prev = LAST_STATE.get(fid)

            if prev != key:
                LAST_STATE[fid] = key

                if status == "FT":
                    msg = f"🏁 FINAL\n{home} {hg} - {ag} {away}"
                    await app.bot.send_message(chat_id=TARGET_CHAT_ID, text=msg)
                else:
                    minute = f"{elapsed}' " if elapsed else ""
                    msg = f"⚽ {minute}{home} {hg} - {ag} {away} ({status})"
                    await app.bot.send_message(chat_id=TARGET_CHAT_ID, text=msg)

        except Exception as e:
            logging.exception(f"Error polling fixture {fid}: {e}")

def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("addfixture", addfixture))
    app.add_handler(CommandHandler("list", listcmd))
    app.add_handler(CommandHandler("clear", clear))

    scheduler = AsyncIOScheduler()
    scheduler.add_job(poll_results, "interval", seconds=90, args=[app])
    scheduler.start()

    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
