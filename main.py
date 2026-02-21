import os
import json
import logging
import asyncio
from typing import Dict, Any, List, Optional

import httpx
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
)
log = logging.getLogger("telegram-resultados")


def env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing environment variable: {name}")
    return v


TELEGRAM_TOKEN = env("TELEGRAM_TOKEN")
API_FOOTBALL_KEY = env("API_FOOTBALL_KEY")
API_FOOTBALL_HOST = os.getenv("API_FOOTBALL_HOST", "v3.football.api-sports.io")
TARGET_CHAT_ID = int(env("TARGET_CHAT_ID"))

API_BASE = f"https://{API_FOOTBALL_HOST}"

STATE_FILE = "state.json"
DEFAULT_STATE = {"fixtures": [], "last": {}}  # last: fixture_id -> {"status":..., "score":"H-A"}


def load_state() -> Dict[str, Any]:
    if not os.path.exists(STATE_FILE):
        return DEFAULT_STATE.copy()
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "fixtures" not in data or "last" not in data:
            return DEFAULT_STATE.copy()
        return data
    except Exception:
        return DEFAULT_STATE.copy()


def save_state(state: Dict[str, Any]) -> None:
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


async def api_get_fixture(client: httpx.AsyncClient, fixture_id: int) -> Optional[Dict[str, Any]]:
    headers = {"x-apisports-key": API_FOOTBALL_KEY}
    params = {"id": fixture_id}
    r = await client.get(f"{API_BASE}/fixtures", headers=headers, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    resp = data.get("response") or []
    if not resp:
        return None
    return resp[0]


def format_line(fx: Dict[str, Any]) -> str:
    fixture = fx.get("fixture", {})
    teams = fx.get("teams", {})
    goals = fx.get("goals", {})

    home = (teams.get("home") or {}).get("name", "Home")
    away = (teams.get("away") or {}).get("name", "Away")
    gh = goals.get("home")
    ga = goals.get("away")

    status = (fixture.get("status") or {}).get("short", "")
    minute = (fixture.get("status") or {}).get("elapsed", None)
    time_part = f"{minute}'" if isinstance(minute, int) else status

    score = f"{gh}-{ga}" if gh is not None and ga is not None else "—"
    return f"{home} vs {away} | {score} | {time_part}"


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("✅ Bot activo. Usa /addfixture <id>, /list, /clear")


async def cmd_addfixture(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Uso: /addfixture <fixture_id>")
        return
    try:
        fixture_id = int(context.args[0])
    except ValueError:
        await update.message.reply_text("El fixture_id debe ser un número.")
        return

    state = load_state()
    fixtures: List[int] = state["fixtures"]
    if fixture_id in fixtures:
        await update.message.reply_text(f"Ya estaba añadido: {fixture_id}")
        return
    fixtures.append(fixture_id)
    save_state(state)
    await update.message.reply_text(f"✅ Añadido fixture: {fixture_id}")


async def cmd_list(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state = load_state()
    fixtures: List[int] = state["fixtures"]
    if not fixtures:
        await update.message.reply_text("No hay fixtures guardados. Usa /addfixture <id>.")
        return
    await update.message.reply_text("📌 Fixtures:\n" + "\n".join(f"- {x}" for x in fixtures))


async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    save_state(DEFAULT_STATE.copy())
    await update.message.reply_text("🧹 Lista limpiada.")


async def poll_results(app: Application):
    state = load_state()
    fixtures: List[int] = state["fixtures"]
    if not fixtures:
        return

    async with httpx.AsyncClient() as client:
        for fid in fixtures:
            try:
                fx = await api_get_fixture(client, fid)
                if not fx:
                    continue

                fixture = fx.get("fixture", {})
                status = (fixture.get("status") or {}).get("short", "")
                goals = fx.get("goals", {})
                gh = goals.get("home")
                ga = goals.get("away")
                score = f"{gh}-{ga}" if gh is not None and ga is not None else "—"

                last = state["last"].get(str(fid))
                changed = (last is None) or (last.get("status") != status) or (last.get("score") != score)

                if changed:
                    line = format_line(fx)
                    await app.bot.send_message(chat_id=TARGET_CHAT_ID, text=f"⚽️ {line}")
                    state["last"][str(fid)] = {"status": status, "score": score}
                    save_state(state)

            except Exception as e:
                log.warning("Error en fixture %s: %s", fid, e)


async def on_startup(app: Application):
    log.info("Bot started. Target chat id: %s", TARGET_CHAT_ID)

    scheduler = AsyncIOScheduler()
    # cada 60s (ajusta si quieres)
    scheduler.add_job(lambda: asyncio.create_task(poll_results(app)), "interval", seconds=60)
    scheduler.start()


def main():
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("addfixture", cmd_addfixture))
    application.add_handler(CommandHandler("list", cmd_list))
    application.add_handler(CommandHandler("clear", cmd_clear))

    application.post_init = on_startup

    # IMPORTANTE: esto es lo que “escucha” comandos
    application.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
