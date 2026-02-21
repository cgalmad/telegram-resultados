import os
import re
import json
import time
import logging
from dataclasses import dataclass, asdict, field
from typing import Dict, Optional, List, Tuple, Set
from datetime import datetime, timedelta, date

import httpx
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("telegram-resultados")

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
TARGET_CHAT_ID = int(os.environ["TARGET_CHAT_ID"])

# TheSportsDB: en free demo se usa 123 (compartida)
SPORTSDB_KEY = os.getenv("SPORTSDB_KEY", "123").strip()
SPORT = os.getenv("SPORT", "Soccer")

# Tick del job (se ejecuta cada 60s)
TICK_SECONDS = int(os.getenv("TICK_SECONDS", "60"))
# Si no hay partidos en juego, hacemos polling "fuerte" cada 300s (5 min)
IDLE_POLL_SECONDS = int(os.getenv("IDLE_POLL_SECONDS", "300"))

STATE_FILE = os.getenv("STATE_FILE", "/tmp/tracked_matches.json")

TSDB_BASE = f"https://www.thesportsdb.com/api/v1/json/{SPORTSDB_KEY}"

# --- Normalización/matching ---
def _norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = s.replace("&", "and")
    s = re.sub(r"[\.\,\-\_\(\)\[\]\/\\]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

def _parse_date_arg(arg: Optional[str]) -> Optional[str]:
    if not arg:
        return None
    arg = arg.strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", arg):
        return arg
    return None

def _looks_like_vs(text: str) -> Optional[Tuple[str, str]]:
    text = text.strip()
    for sep in [" vs ", " v ", " - ", "—", "–"]:
        if sep in text.lower():
            parts = re.split(sep, text, flags=re.IGNORECASE)
            if len(parts) >= 2:
                home = parts[0].strip()
                away = " ".join(parts[1:]).strip()
                if home and away:
                    return home, away
    return None

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

def is_live_status(status: Optional[str]) -> bool:
    if not status:
        return False
    s = status.strip().lower()
    # TheSportsDB suele usar: 1H, 2H, HT, ET, PEN, AET, Live
    return s in {"1h", "2h", "ht", "et", "pen", "aet", "live"} or "half" in s or "time" in s

def is_ht_status(status: Optional[str]) -> bool:
    if not status:
        return False
    s = status.strip().lower()
    return s in {"ht", "half time", "halftime"}

def is_finished_status(status: Optional[str]) -> bool:
    if not status:
        return False
    s = status.strip().lower()
    return s in {"ft", "aet", "pen", "match finished", "finished"} or "finished" in s

def pretty_status(status: Optional[str]) -> str:
    return status or "?"

@dataclass
class TrackedMatch:
    home: str
    away: str
    match_date: str  # YYYY-MM-DD
    id_event: Optional[str] = None

    last_status: Optional[str] = None
    last_home_score: Optional[int] = None
    last_away_score: Optional[int] = None
    finished: bool = False

    # Para no repetir eventos de timeline
    seen_timeline_keys: Set[str] = field(default_factory=set)

TRACKED: Dict[str, TrackedMatch] = {}
LAST_HEAVY_POLL_TS: float = 0.0  # para idle mode

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
                v.setdefault("seen_timeline_keys", [])
                v["seen_timeline_keys"] = set(v["seen_timeline_keys"])
                tmp[k] = TrackedMatch(**v)
            TRACKED = tmp
            log.info("Loaded state: %d matches", len(TRACKED))
    except Exception as e:
        log.warning("Could not load state: %s", e)

def save_state() -> None:
    try:
        raw = {}
        for k, tm in TRACKED.items():
            d = asdict(tm)
            d["seen_timeline_keys"] = sorted(list(tm.seen_timeline_keys))
            raw[k] = d
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.warning("Could not save state: %s", e)

# --- TheSportsDB API ---
async def fetch_events_day(client: httpx.AsyncClient, d: str) -> List[dict]:
    url = f"{TSDB_BASE}/eventsday.php"
    params = {"d": d, "s": SPORT}
    r = await client.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    return data.get("events") or []

async def fetch_timeline(client: httpx.AsyncClient, id_event: str) -> List[dict]:
    # Endpoint: lookuptimeline.php?id=EVENT
    url = f"{TSDB_BASE}/lookuptimeline.php"
    params = {"id": id_event}
    r = await client.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    # suele venir en "timeline"
    return data.get("timeline") or data.get("timelines") or []

def find_event_for_match(events: List[dict], tm: TrackedMatch) -> Optional[dict]:
    if tm.id_event:
        for ev in events:
            if str(ev.get("idEvent")) == str(tm.id_event):
                return ev

    nh, na = _norm(tm.home), _norm(tm.away)

    for ev in events:
        eh = _norm(ev.get("strHomeTeam") or "")
        ea = _norm(ev.get("strAwayTeam") or "")
        if eh == nh and ea == na:
            return ev

    def contains(a: str, b: str) -> bool:
        return (a in b) or (b in a)

    for ev in events:
        eh = _norm(ev.get("strHomeTeam") or "")
        ea = _norm(ev.get("strAwayTeam") or "")
        if contains(nh, eh) and contains(na, ea):
            return ev

    return None

def scoreline(tm: TrackedMatch, status: Optional[str], hs: Optional[int], as_: Optional[int]) -> str:
    hs_s = "?" if hs is None else str(hs)
    as_s = "?" if as_ is None else str(as_)
    return f"{tm.home} {hs_s}-{as_s} {tm.away} ({pretty_status(status)})"

def timeline_key(item: dict) -> str:
    # Creamos una clave estable para no repetir
    # (minuto + tipo + texto + equipo + jugador)
    t = str(item.get("strTimeline") or item.get("strEvent") or item.get("strType") or "")
    m = str(item.get("intTime") or item.get("strTime") or item.get("strTimeLine") or "")
    p = str(item.get("strPlayer") or item.get("strPlayerName") or "")
    te = str(item.get("strTeam") or item.get("strSide") or "")
    ex = str(item.get("strExtra") or item.get("strAssist") or "")
    return _norm(f"{m}|{t}|{p}|{te}|{ex}")

def classify_timeline(item: dict) -> Tuple[Optional[str], str]:
    """
    Devuelve (tipo, detalle). tipo ∈ {"goal","red","disallowed"} o None
    """
    text = (item.get("strTimeline") or item.get("strEvent") or item.get("strType") or "")
    text_n = _norm(text)

    # Intentos de detección genérica
    if "red" in text_n and "card" in text_n:
        return "red", text.strip() or "Red card"

    # Algunos timelines marcan gol como "Goal" o similar
    if "goal" in text_n:
        # si menciona anulaciones
        if "disallow" in text_n or "var" in text_n or "no goal" in text_n or "overturn" in text_n:
            return "disallowed", text.strip() or "Disallowed goal"
        return "goal", text.strip() or "Goal"

    # Si no aparece "goal", pero hay VAR/disallowed explícito
    if "disallow" in text_n or ("var" in text_n and "goal" in text_n):
        return "disallowed", text.strip() or "Disallowed goal"

    return None, text.strip()

# --- Bot Commands ---
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat and update.effective_chat.id != TARGET_CHAT_ID:
        return
    msg = (
        "Bot activo ✅\n\n"
        "Comandos:\n"
        "• /seguir <equipo1> vs <equipo2> [YYYY-MM-DD]\n"
        "  Ej: /seguir Lecce vs Inter Milan 2026-02-21\n"
        "• /lista\n"
        "• /borrar <equipo1> vs <equipo2> [YYYY-MM-DD]\n"
        "• /limpiar\n\n"
        f"Tick: {TICK_SECONDS}s | Idle poll: {IDLE_POLL_SECONDS}s"
    )
    await context.bot.send_message(chat_id=TARGET_CHAT_ID, text=msg)

async def cmd_seguir(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat and update.effective_chat.id != TARGET_CHAT_ID:
        return
    text = " ".join(context.args).strip()
    if not text:
        await context.bot.send_message(chat_id=TARGET_CHAT_ID, text="Uso: /seguir Equipo A vs Equipo B [YYYY-MM-DD]")
        return

    parts = text.split()
    maybe_date = _parse_date_arg(parts[-1]) if parts else None
    if maybe_date:
        d = maybe_date
        teams_text = " ".join(parts[:-1]).strip()
    else:
        d = date.today().isoformat()
        teams_text = text

    vs = _looks_like_vs(teams_text)
    if not vs:
        await context.bot.send_message(chat_id=TARGET_CHAT_ID, text="Formato inválido. Ej: /seguir Juventus vs Como 2026-02-21")
        return

    home, away = vs
    key = _make_key(home, away, d)

    if key in TRACKED and not TRACKED[key].finished:
        await context.bot.send_message(chat_id=TARGET_CHAT_ID, text=f"Ya lo sigo: {home} vs {away} ({d})")
        return

    TRACKED[key] = TrackedMatch(home=home, away=away, match_date=d)
    save_state()
    await context.bot.send_message(chat_id=TARGET_CHAT_ID, text=f"✅ Añadido: {home} vs {away} ({d}).")

async def cmd_lista(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat and update.effective_chat.id != TARGET_CHAT_ID:
        return
    active = [tm for tm in TRACKED.values() if not tm.finished]
    if not active:
        await context.bot.send_message(chat_id=TARGET_CHAT_ID, text="No hay partidos en seguimiento.")
        return
    lines = ["📋 Seguimiento:"]
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
        d = maybe_date
        teams_text = " ".join(parts[:-1]).strip()
    else:
        d = date.today().isoformat()
        teams_text = text

    vs = _looks_like_vs(teams_text)
    if not vs:
        await context.bot.send_message(chat_id=TARGET_CHAT_ID, text="Formato inválido.")
        return
    home, away = vs
    key = _make_key(home, away, d)
    if key not in TRACKED:
        await context.bot.send_message(chat_id=TARGET_CHAT_ID, text="No lo encuentro en la lista.")
        return
    TRACKED.pop(key, None)
    save_state()
    await context.bot.send_message(chat_id=TARGET_CHAT_ID, text=f"🗑️ Borrado: {home} vs {away} ({d})")

async def cmd_limpiar(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat and update.effective_chat.id != TARGET_CHAT_ID:
        return
    TRACKED.clear()
    save_state()
    await context.bot.send_message(chat_id=TARGET_CHAT_ID, text="🧹 Lista limpiada.")

# --- Poll job ---
async def poll_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    global LAST_HEAVY_POLL_TS

    active_keys = [k for k, tm in TRACKED.items() if not tm.finished]
    if not active_keys:
        return

    # ¿Tenemos alguno en juego? Si no, hacemos heavy poll solo cada IDLE_POLL_SECONDS
    any_live_cached = any(is_live_status(TRACKED[k].last_status) for k in active_keys if TRACKED[k].last_status)
    now = time.time()
    if not any_live_cached and (now - LAST_HEAVY_POLL_TS) < IDLE_POLL_SECONDS:
        return

    needed_dates = sorted({TRACKED[k].match_date for k in active_keys})

    async with httpx.AsyncClient(headers={"User-Agent": "telegram-resultados-bot"}) as client:
        # 1) Cargamos eventos por día (1 request por día)
        events_by_date: Dict[str, List[dict]] = {}
        for d in needed_dates:
            try:
                events_by_date[d] = await fetch_events_day(client, d)
            except Exception as e:
                log.warning("eventsday failed for %s: %s", d, e)
                events_by_date[d] = []

        # 2) Procesamos cada partido seguido
        any_live_now = False
        for k in active_keys:
            tm = TRACKED[k]
            ev = find_event_for_match(events_by_date.get(tm.match_date, []), tm)
            if not ev:
                continue

            tm.id_event = str(ev.get("idEvent") or tm.id_event)

            status = ev.get("strStatus")
            hs = safe_int(ev.get("intHomeScore"))
            as_ = safe_int(ev.get("intAwayScore"))

            if is_live_status(status):
                any_live_now = True
                any_live_cached = True

            # DESCANSO
            if is_ht_status(status) and not is_ht_status(tm.last_status):
                await context.bot.send_message(chat_id=TARGET_CHAT_ID, text=f"⏸ DESCANSO: {scoreline(tm, status, hs, as_)}")

            # GOL (cambio de marcador)
            if (hs is not None and as_ is not None) and (tm.last_home_score != hs or tm.last_away_score != as_):
                await context.bot.send_message(chat_id=TARGET_CHAT_ID, text=f"⚽ GOL: {scoreline(tm, status, hs, as_)}")

            # FINAL
            if is_finished_status(status) and not tm.finished:
                await context.bot.send_message(chat_id=TARGET_CHAT_ID, text=f"✅ FINAL: {scoreline(tm, status, hs, as_)}")
                tm.finished = True

            # 3) Timeline (rojas / gol anulado / detalles)
            # Solo si:
            #  - el partido está en juego, o
            #  - hubo cambio relevante (marcador/status)
            relevant_change = (tm.last_status != status) or (
                (hs is not None and as_ is not None) and (tm.last_home_score != hs or tm.last_away_score != as_)
            )
            if tm.id_event and (is_live_status(status) or relevant_change) and not tm.finished:
                try:
                    timeline = await fetch_timeline(client, tm.id_event)
                    for item in timeline:
                        key_tl = timeline_key(item)
                        if not key_tl or key_tl in tm.seen_timeline_keys:
                            continue
                        tm.seen_timeline_keys.add(key_tl)

                        kind, detail = classify_timeline(item)
                        minute = item.get("intTime") or item.get("strTime") or ""
                        minute_txt = f"{minute}' " if str(minute).strip() else ""

                        if kind == "red":
                            await context.bot.send_message(chat_id=TARGET_CHAT_ID, text=f"🟥 ROJA: {minute_txt}{tm.home} vs {tm.away} — {detail}")
                        elif kind == "disallowed":
                            await context.bot.send_message(chat_id=TARGET_CHAT_ID, text=f"❌ GOL ANULADO: {minute_txt}{tm.home} vs {tm.away} — {detail}")
                        elif kind == "goal":
                            # Ojo: ya mandamos gol por cambio de marcador; esto añade el detalle (si quieres, se puede apagar)
                            await context.bot.send_message(chat_id=TARGET_CHAT_ID, text=f"⚽ EVENTO GOL: {minute_txt}{tm.home} vs {tm.away} — {detail}")
                except Exception as e:
                    log.warning("timeline failed for %s: %s", tm.id_event, e)

            tm.last_status = status
            tm.last_home_score = hs
            tm.last_away_score = as_

        save_state()

        # Si no hay ninguno en juego ahora, marcamos heavy poll timestamp (idle throttling)
        if not any_live_now:
            LAST_HEAVY_POLL_TS = time.time()
        else:
            LAST_HEAVY_POLL_TS = 0.0  # mientras haya live, no throttling

def main() -> None:
    load_state()

    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("seguir", cmd_seguir))
    app.add_handler(CommandHandler("lista", cmd_lista))
    app.add_handler(CommandHandler("borrar", cmd_borrar))
    app.add_handler(CommandHandler("limpiar", cmd_limpiar))

    app.job_queue.run_repeating(poll_job, interval=TICK_SECONDS, first=10)

    log.info("Bot started. chat=%s tick=%ss idle=%ss key=%s",
             TARGET_CHAT_ID, TICK_SECONDS, IDLE_POLL_SECONDS, "***")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
