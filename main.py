import os
import re
import json
import asyncio
import logging
from dataclasses import dataclass, asdict
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta, date

import httpx
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# -----------------------
# Config
# -----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
)
log = logging.getLogger("telegram-resultados")

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
TARGET_CHAT_ID = int(os.environ["TARGET_CHAT_ID"])

SPORTSDB_KEY = os.getenv("SPORTSDB_KEY", "3").strip()  # demo por defecto
POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", "300"))  # 5 min por defecto
SPORT = os.getenv("SPORT", "Soccer")

# Fichero local (best-effort). En Railway puede perderse tras redeploy, pero sirve para reinicios del proceso.
STATE_FILE = os.getenv("STATE_FILE", "/tmp/tracked_matches.json")

# TheSportsDB base
TSDB_BASE = f"https://www.thesportsdb.com/api/v1/json/{SPORTSDB_KEY}"

# -----------------------
# Helpers
# -----------------------
def _norm(s: str) -> str:
    """Normaliza para matching flexible."""
    s = s.lower().strip()
    s = s.replace("&", "and")
    s = re.sub(r"[\.\,\-\_\(\)\[\]\/\\]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

def _looks_like_vs(text: str) -> Optional[Tuple[str, str]]:
    """
    Acepta:
      /seguir Team A vs Team B
      /seguir Team A - Team B
      /seguir Team A v Team B
    """
    text = text.strip()
    # separadores típicos
    for sep in [" vs ", " v ", " - ", "—", "–"]:
        if sep in text.lower():
            parts = re.split(sep, text, flags=re.IGNORECASE)
            if len(parts) >= 2:
                home = parts[0].strip()
                away = " ".join(parts[1:]).strip()
                if home and away:
                    return home, away
    return None

def _parse_date_arg(arg: Optional[str]) -> Optional[str]:
    """Devuelve YYYY-MM-DD o None."""
    if not arg:
        return None
    arg = arg.strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", arg):
        return arg
    return None

@dataclass
class TrackedMatch:
    home: str
    away: str
    match_date: str  # YYYY-MM-DD
    id_event: Optional[str] = None

    # estado
    last_status: Optional[str] = None
    last_home_score: Optional[int] = None
    last_away_score: Optional[int] = None
    finished: bool = False

# En memoria
TRACKED: Dict[str, TrackedMatch] = {}  # key -> TrackedMatch


def _make_key(home: str, away: str, match_date: str) -> str:
    return f"{_norm(home)}__{_norm(away)}__{match_date}"

def load_state() -> None:
    global TRACKED
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                raw = json.load(f)
            tmp: Dict[str, TrackedMatch] = {}
            for k, v in raw.items():
                tmp[k] = TrackedMatch(**v)
            TRACKED = tmp
            log.info("Loaded state: %d tracked matches", len(TRACKED))
    except Exception as e:
        log.warning("Could not load state: %s", e)

def save_state() -> None:
    try:
        raw = {k: asdict(v) for k, v in TRACKED.items()}
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.warning("Could not save state: %s", e)

def is_finished_status(status: Optional[str]) -> bool:
    if not status:
        return False
    s = status.strip().lower()
    # TheSportsDB suele: "FT", "AET", "PEN", "Match Finished"
    return s in {"ft", "aet", "pen", "match finished", "finished"} or "finished" in s

def pretty_status(status: Optional[str]) -> str:
    if not status:
        return "?"
    return status

def safe_int(x) -> Optional[int]:
    try:
        if x is None:
            return None
        if isinstance(x, int):
            return x
        if isinstance(x, str) and x.strip() == "":
            return None
        return int(x)
    except Exception:
        return None

# -----------------------
# TheSportsDB calls
# -----------------------
async def fetch_events_day(client: httpx.AsyncClient, d: str) -> List[dict]:
    """
    GET /eventsday.php?d=YYYY-MM-DD&s=Soccer
    """
    url = f"{TSDB_BASE}/eventsday.php"
    params = {"d": d, "s": SPORT}
    r = await client.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    return data.get("events") or []

def find_event_for_match(events: List[dict], tm: TrackedMatch) -> Optional[dict]:
    """
    Matching flexible: primero por id_event si existe,
    si no, por home/away normalizados.
    """
    if tm.id_event:
        for ev in events:
            if str(ev.get("idEvent")) == str(tm.id_event):
                return ev

    nh = _norm(tm.home)
    na = _norm(tm.away)

    # 1) match directo por home/away
    for ev in events:
        eh = _norm(ev.get("strHomeTeam") or "")
        ea = _norm(ev.get("strAwayTeam") or "")
        if eh == nh and ea == na:
            return ev

    # 2) match "contiene" (por si hay FC, AC, etc)
    def contains(a: str, b: str) -> bool:
        return (a in b) or (b in a)

    for ev in events:
        eh = _norm(ev.get("strHomeTeam") or "")
        ea = _norm(ev.get("strAwayTeam") or "")
        if contains(nh, eh) and contains(na, ea):
            return ev

    return None

def format_scoreline(tm: TrackedMatch, status: Optional[str], hs: Optional[int], as_: Optional[int]) -> str:
    hname = tm.home
    aname = tm.away
    hs_str = "?" if hs is None else str(hs)
    as_str = "?" if as_ is None else str(as_)
    st = pretty_status(status)
    return f"{hname} {hs_str}-{as_str} {aname} ({st})"

# -----------------------
# Bot commands
# -----------------------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat and update.effective_chat.id != TARGET_CHAT_ID:
        return  # ignorar fuera del grupo objetivo

    msg = (
        "Bot activo ✅\n\n"
        "Comandos:\n"
        "• /seguir <equipo1> vs <equipo2> [YYYY-MM-DD]\n"
        "   Ej: /seguir Lecce vs Inter Milan 2026-02-21\n"
        "   Si no pones fecha, usa hoy.\n"
        "• /lista\n"
        "• /borrar <equipo1> vs <equipo2> [YYYY-MM-DD]\n"
        "• /limpiar\n"
    )
    await context.bot.send_message(chat_id=TARGET_CHAT_ID, text=msg)

async def cmd_seguir(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat and update.effective_chat.id != TARGET_CHAT_ID:
        return

    text = " ".join(context.args).strip()
    if not text:
        await context.bot.send_message(chat_id=TARGET_CHAT_ID, text="Uso: /seguir Equipo A vs Equipo B [YYYY-MM-DD]")
        return

    # intenta detectar fecha al final
    parts = text.split()
    maybe_date = _parse_date_arg(parts[-1]) if parts else None
    if maybe_date:
        date_str = maybe_date
        text_teams = " ".join(parts[:-1]).strip()
    else:
        date_str = date.today().isoformat()
        text_teams = text

    vs = _looks_like_vs(text_teams)
    if not vs:
        await context.bot.send_message(chat_id=TARGET_CHAT_ID, text="No entiendo el formato. Ej: /seguir Roma vs US Cremonese 2026-02-22")
        return

    home, away = vs
    key = _make_key(home, away, date_str)
    if key in TRACKED and not TRACKED[key].finished:
        await context.bot.send_message(chat_id=TARGET_CHAT_ID, text=f"Ya lo estoy siguiendo: {home} vs {away} ({date_str})")
        return

    TRACKED[key] = TrackedMatch(home=home, away=away, match_date=date_str)
    save_state()

    await context.bot.send_message(chat_id=TARGET_CHAT_ID, text=f"✅ Añadido: {home} vs {away} ({date_str}). Actualizo cada {POLL_INTERVAL_SECONDS//60} min.")

async def cmd_lista(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat and update.effective_chat.id != TARGET_CHAT_ID:
        return

    active = [tm for tm in TRACKED.values() if not tm.finished]
    if not active:
        await context.bot.send_message(chat_id=TARGET_CHAT_ID, text="No hay partidos en seguimiento. Usa /seguir ...")
        return

    lines = ["📋 Partidos en seguimiento:"]
    for tm in active:
        lines.append(f"• {tm.home} vs {tm.away} ({tm.match_date})")
    await context.bot.send_message(chat_id=TARGET_CHAT_ID, text="\n".join(lines))

async def cmd_borrar(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat and update.effective_chat.id != TARGET_CHAT_ID:
        return

    text = " ".join(context.args).strip()
    if not text:
        await context.bot.send_message(chat_id=TARGET_CHAT_ID, text="Uso: /borrar Equipo A vs Equipo B [YYYY-MM-DD]")
        return

    parts = text.split()
    maybe_date = _parse_date_arg(parts[-1]) if parts else None
    if maybe_date:
        date_str = maybe_date
        text_teams = " ".join(parts[:-1]).strip()
    else:
        date_str = date.today().isoformat()
        text_teams = text

    vs = _looks_like_vs(text_teams)
    if not vs:
        await context.bot.send_message(chat_id=TARGET_CHAT_ID, text="Formato inválido. Ej: /borrar Lecce vs Inter Milan 2026-02-21")
        return

    home, away = vs
    key = _make_key(home, away, date_str)
    if key not in TRACKED:
        await context.bot.send_message(chat_id=TARGET_CHAT_ID, text="No lo encuentro en la lista.")
        return

    TRACKED.pop(key, None)
    save_state()
    await context.bot.send_message(chat_id=TARGET_CHAT_ID, text=f"🗑️ Borrado: {home} vs {away} ({date_str})")

async def cmd_limpiar(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat and update.effective_chat.id != TARGET_CHAT_ID:
        return
    TRACKED.clear()
    save_state()
    await context.bot.send_message(chat_id=TARGET_CHAT_ID, text="🧹 Lista limpiada.")

# -----------------------
# Polling job (cada X min)
# -----------------------
async def poll_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    active_keys = [k for k, tm in TRACKED.items() if not tm.finished]
    if not active_keys:
        return

    # Pedimos eventos SOLO para los días que realmente necesitamos (hoy + los días de los partidos en seguimiento)
    needed_dates = sorted({TRACKED[k].match_date for k in active_keys})
    # Para evitar “quedarnos ciegos” por cambios horarios, también consultamos el día anterior y el siguiente si hay partidos "hoy"
    # (best-effort)
    expanded_dates = set(needed_dates)
    try:
        for d in needed_dates:
            dd = datetime.strptime(d, "%Y-%m-%d").date()
            expanded_dates.add((dd - timedelta(days=1)).isoformat())
            expanded_dates.add((dd + timedelta(days=1)).isoformat())
    except Exception:
        expanded_dates = set(needed_dates)

    # Descargamos eventos por día (1 request por día)
    all_events_by_date: Dict[str, List[dict]] = {}
    async with httpx.AsyncClient(headers={"User-Agent": "telegram-resultados-bot"}) as client:
        for d in sorted(expanded_dates):
            try:
                evs = await fetch_events_day(client, d)
                all_events_by_date[d] = evs
            except Exception as e:
                log.warning("TSDB eventsday failed for %s: %s", d, e)
                all_events_by_date[d] = []

    # Procesamos
    for k in active_keys:
        tm = TRACKED[k]
        events = all_events_by_date.get(tm.match_date) or []

        ev = find_event_for_match(events, tm)
        # si no aparece en su fecha exacta, probamos en +/-1 día (por timezone/fixture moved)
        if not ev:
            dd = datetime.strptime(tm.match_date, "%Y-%m-%d").date()
            for alt in [(dd - timedelta(days=1)).isoformat(), (dd + timedelta(days=1)).isoformat()]:
                ev = find_event_for_match(all_events_by_date.get(alt, []), tm)
                if ev:
                    # actualizamos la fecha para seguir bien
                    tm.match_date = alt
                    break

        if not ev:
            # No spameamos: solo guardamos que no lo vimos aún
            log.info("No event found yet for: %s vs %s (%s)", tm.home, tm.away, tm.match_date)
            continue

        tm.id_event = str(ev.get("idEvent") or tm.id_event)

        status = ev.get("strStatus")
        hs = safe_int(ev.get("intHomeScore"))
        as_ = safe_int(ev.get("intAwayScore"))

        # Detecta cambios
        changed_score = (hs is not None and as_ is not None) and (
            tm.last_home_score != hs or tm.last_away_score != as_
        )
        changed_status = (status is not None) and (tm.last_status != status)

        # Mensajes (solo si hay info útil)
        if changed_score:
            text = "⚽ " + format_scoreline(tm, status, hs, as_)
            await context.bot.send_message(chat_id=TARGET_CHAT_ID, text=text)

        # Si llega FT, avisamos final y dejamos de seguir
        if is_finished_status(status):
            final_text = "✅ FINAL: " + format_scoreline(tm, status, hs, as_)
            await context.bot.send_message(chat_id=TARGET_CHAT_ID, text=final_text)
            tm.finished = True

        # actualiza last
        tm.last_status = status
        tm.last_home_score = hs
        tm.last_away_score = as_

    save_state()

# -----------------------
# Main
# -----------------------
def main() -> None:
    load_state()

    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("seguir", cmd_seguir))
    app.add_handler(CommandHandler("lista", cmd_lista))
    app.add_handler(CommandHandler("borrar", cmd_borrar))
    app.add_handler(CommandHandler("limpiar", cmd_limpiar))

    # JobQueue: polling
    app.job_queue.run_repeating(poll_job, interval=POLL_INTERVAL_SECONDS, first=10)

    log.info("Bot started. Target chat id: %s | interval=%ss | tsdb_key=%s",
             TARGET_CHAT_ID, POLL_INTERVAL_SECONDS, ("***" if SPORTSDB_KEY else ""))
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
