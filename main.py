import asyncio
import json
import logging
import os
import re
import unicodedata
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, date
from typing import Any, Dict, List, Optional, Tuple

import httpx
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)

# =========================
# CONFIG
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
)
log = logging.getLogger("telegram-resultados")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
TARGET_CHAT_ID = int(os.getenv("TARGET_CHAT_ID", "0").strip())  # grupo destino
POLL_SECONDS = int(os.getenv("POLL_SECONDS", "60").strip())
SPORTSDB_KEY = os.getenv("SPORTSDB_KEY", "123").strip()  # TheSportsDB free shared key
STATE_FILE = os.getenv("STATE_FILE", "state.json").strip()

SPORTSDB_BASE = f"https://www.thesportsdb.com/api/v1/json/{SPORTSDB_KEY}"

INPLAY_STATUSES = {"1H", "2H", "HT", "ET", "P", "BT", "PEN"}  # defensivo
FINISHED_STATUSES = {"FT", "AET", "PEN"}  # a veces PEN también es final
# Nota: TheSportsDB usa strings tipo "Not Started", "Match Finished", etc en algunas rutas.
# Aquí tratamos ambas formas.

# =========================
# HELPERS
# =========================
def _now_utc() -> datetime:
    return datetime.utcnow()

def normalize_team(s: str) -> str:
    """
    Normaliza nombres: minúsculas, sin acentos, sin puntuación, y elimina tokens comunes.
    """
    s = s.strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    # tokens muy comunes que meten ruido
    stop = {"fc", "cf", "sc", "ac", "cd", "ud", "afc", "the", "de", "la"}
    parts = [p for p in s.split() if p not in stop]
    return " ".join(parts)

def parse_teams(text: str) -> Tuple[str, str]:
    """
    Espera: "Equipo A vs Equipo B" o "Equipo A - Equipo B"
    """
    t = text.strip()
    t = re.sub(r"\s+", " ", t)
    if " vs " in t.lower():
        a, b = re.split(r"\s+vs\s+", t, flags=re.IGNORECASE, maxsplit=1)
    elif " - " in t:
        a, b = t.split(" - ", 1)
    else:
        raise ValueError("Formato inválido. Usa: /seguir Equipo1 vs Equipo2 [YYYY-MM-DD]")
    return a.strip(), b.strip()

def parse_optional_date(parts: List[str]) -> Optional[date]:
    """
    Si el último token es YYYY-MM-DD lo interpreta como fecha.
    """
    if not parts:
        return None
    last = parts[-1]
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", last):
        try:
            return datetime.strptime(last, "%Y-%m-%d").date()
        except ValueError:
            return None
    return None

def safe_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        return None

def status_bucket(status: str) -> str:
    s = (status or "").strip()
    s_up = s.upper()

    # TheSportsDB a veces da "Not Started", "Match Finished", etc.
    if s_up in INPLAY_STATUSES:
        return "INPLAY"
    if s_up in FINISHED_STATUSES:
        return "FINISHED"
    if "NOT START" in s_up or s_up in {"NS", "TBD"}:
        return "SCHEDULED"
    if "FINISH" in s_up or s_up in {"FT", "AET"}:
        return "FINISHED"
    if "HALF" in s_up or s_up == "HT":
        return "INPLAY"
    return "OTHER"

# =========================
# STATE
# =========================
@dataclass
class TrackedMatch:
    home: str
    away: str
    date_str: str  # YYYY-MM-DD (fecha objetivo)
    event_id: str  # TheSportsDB idEvent
    league: str = ""
    kickoff_local: str = ""  # strTimeLocal si viene

    last_home: Optional[int] = None
    last_away: Optional[int] = None
    last_status: str = ""
    seen_timeline_ids: List[str] = None

    def __post_init__(self):
        if self.seen_timeline_ids is None:
            self.seen_timeline_ids = []

def load_state() -> Dict[str, Any]:
    if not os.path.exists(STATE_FILE):
        return {"tracked": []}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        log.exception("No se pudo leer state.json, empezando limpio.")
        return {"tracked": []}

def save_state(state: Dict[str, Any]) -> None:
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, STATE_FILE)

def get_tracked(state: Dict[str, Any]) -> List[TrackedMatch]:
    items = state.get("tracked", [])
    out: List[TrackedMatch] = []
    for it in items:
        try:
            out.append(TrackedMatch(**it))
        except Exception:
            log.exception("Entrada inválida en state.json, se ignora: %s", it)
    return out

def set_tracked(state: Dict[str, Any], matches: List[TrackedMatch]) -> None:
    state["tracked"] = [asdict(m) for m in matches]

# =========================
# API: TheSportsDB
# =========================
async def http_get_json(url: str, params: Optional[dict] = None) -> dict:
    timeout = httpx.Timeout(20.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        return r.json()

async def fetch_events_for_day(d: date) -> List[dict]:
    # eventsday.php devuelve eventos por día para Soccer
    url = f"{SPORTSDB_BASE}/eventsday.php"
    js = await http_get_json(url, params={"d": d.strftime("%Y-%m-%d"), "s": "Soccer"})
    events = js.get("events") or []
    return events

def score_from_event(ev: dict) -> Tuple[Optional[int], Optional[int]]:
    # campos comunes: intHomeScore / intAwayScore
    return safe_int(ev.get("intHomeScore")), safe_int(ev.get("intAwayScore"))

def status_from_event(ev: dict) -> str:
    return (ev.get("strStatus") or "").strip()

def teams_from_event(ev: dict) -> Tuple[str, str]:
    return (ev.get("strHomeTeam") or "").strip(), (ev.get("strAwayTeam") or "").strip()

def event_match_score(ev: dict, home_q: str, away_q: str) -> int:
    """
    Score heurístico para matching flexible.
    """
    h, a = teams_from_event(ev)
    nh = normalize_team(h)
    na = normalize_team(a)
    qh = normalize_team(home_q)
    qa = normalize_team(away_q)

    score = 0
    # match directo
    if nh == qh:
        score += 50
    if na == qa:
        score += 50

    # contiene
    if qh and qh in nh:
        score += 20
    if qa and qa in na:
        score += 20

    # match cruzado (por si usuario invierte)
    if nh == qa:
        score += 10
    if na == qh:
        score += 10

    # bonus si está en juego ahora
    bucket = status_bucket(status_from_event(ev))
    if bucket == "INPLAY":
        score += 30

    # bonus si ya tiene marcador (está avanzado o final)
    hs, as_ = score_from_event(ev)
    if hs is not None or as_ is not None:
        score += 5

    return score

async def find_best_event(home: str, away: str, target_date: Optional[date]) -> Optional[dict]:
    """
    Busca evento por fecha exacta. Si falla, busca ±1 día.
    Prioriza INPLAY si existe.
    """
    if target_date is None:
        target_date = _now_utc().date()

    candidates: List[dict] = []
    for delta in [0, -1, 1]:
        d = target_date + timedelta(days=delta)
        try:
            events = await fetch_events_for_day(d)
        except Exception:
            log.exception("Error pidiendo eventos del día %s", d)
            continue
        candidates.extend(events)

    if not candidates:
        return None

    scored = [(event_match_score(ev, home, away), ev) for ev in candidates]
    scored.sort(key=lambda x: x[0], reverse=True)

    best_score, best_ev = scored[0]
    if best_score < 40:
        # umbral: evita emparejar cualquier cosa
        return None
    return best_ev

async def fetch_event_by_id(event_id: str) -> Optional[dict]:
    url = f"{SPORTSDB_BASE}/lookupevent.php"
    js = await http_get_json(url, params={"id": event_id})
    events = js.get("events") or []
    return events[0] if events else None

async def fetch_timeline(event_id: str) -> List[dict]:
    """
    TheSportsDB tiene lookuptimeline.php?id=EVENT
    """
    url = f"{SPORTSDB_BASE}/lookuptimeline.php"
    js = await http_get_json(url, params={"id": event_id})
    tl = js.get("timeline") or []
    return tl

def timeline_key(item: dict) -> str:
    # idTimeline existe normalmente; si no, fallback a combinación
    return (item.get("idTimeline") or "") or f"{item.get('strTime','')}-{item.get('strEvent','')}-{item.get('strTimeline','')}"

def describe_timeline(item: dict, home: str, away: str, hs: Optional[int], aas: Optional[int]) -> Optional[str]:
    """
    Mensaje para rojas y goles anulados si la API lo trae.
    Campos típicos: strTimeline, strEvent, strTeam, strPlayer, intTime.
    Como puede variar, vamos defensivos.
    """
    ttype = (item.get("strTimeline") or item.get("strType") or item.get("strEvent") or "").strip()
    minute = item.get("intTime") or item.get("strTime") or ""
    team = (item.get("strTeam") or "").strip()
    player = (item.get("strPlayer") or "").strip()

    ttype_up = ttype.lower()
    if "red" in ttype_up:
        who = f"{team} — {player}".strip(" —")
        return f"🟥 ROJA {minute}': {who}"
    if "disallow" in ttype_up or "annul" in ttype_up:
        who = f"{team} — {player}".strip(" —")
        return f"🚫 GOL ANULADO {minute}': {who}"
    # si viene un "Goal" aquí, normalmente ya lo cubrimos por delta de marcador
    return None

# =========================
# TELEGRAM OUTPUT
# =========================
async def send_msg(app: Application, text: str) -> None:
    if TARGET_CHAT_ID == 0:
        log.warning("TARGET_CHAT_ID no configurado, no envío: %s", text)
        return
    await app.bot.send_message(chat_id=TARGET_CHAT_ID, text=text)

def fmt_score(home: str, away: str, hs: Optional[int], aas: Optional[int]) -> str:
    if hs is None or aas is None:
        return f"{home} vs {away}"
    return f"{home} {hs}–{aas} {away}"

# =========================
# COMMANDS
# =========================
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    help_text = (
        "Bot activo ✅\n\n"
        "Comandos:\n"
        "• /seguir <equipo1> vs <equipo2> [YYYY-MM-DD]\n"
        "  (La fecha es opcional. Si falla por un día, el bot intenta ±1 día automáticamente)\n"
        "• /lista\n"
        "• /borrar <equipo1> vs <equipo2> [YYYY-MM-DD]\n"
        "• /limpiar\n"
        f"\nTick: {POLL_SECONDS}s"
    )
    await update.message.reply_text(help_text)

async def cmd_list(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    state = load_state()
    tracked = get_tracked(state)
    if not tracked:
        await update.message.reply_text("No hay partidos seguidos.")
        return
    lines = ["Partidos seguidos:"]
    for m in tracked:
        lines.append(f"• {m.home} vs {m.away} ({m.date_str}) [id={m.event_id}]")
    await update.message.reply_text("\n".join(lines))

async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    state = {"tracked": []}
    save_state(state)
    await update.message.reply_text("✅ Lista limpiada.")

async def cmd_delete(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = " ".join(context.args).strip()
    if not text:
        await update.message.reply_text("Uso: /borrar Equipo1 vs Equipo2 [YYYY-MM-DD]")
        return

    parts = text.split()
    d = parse_optional_date(parts)
    if d:
        text_teams = " ".join(parts[:-1]).strip()
        date_str = d.strftime("%Y-%m-%d")
    else:
        text_teams = text
        date_str = _now_utc().date().strftime("%Y-%m-%d")

    try:
        home, away = parse_teams(text_teams)
    except Exception as e:
        await update.message.reply_text(str(e))
        return

    state = load_state()
    tracked = get_tracked(state)
    before = len(tracked)
    tracked = [
        m for m in tracked
        if not (normalize_team(m.home) == normalize_team(home)
                and normalize_team(m.away) == normalize_team(away)
                and m.date_str == date_str)
    ]
    set_tracked(state, tracked)
    save_state(state)

    if len(tracked) == before:
        await update.message.reply_text("No encontré ese partido en la lista.")
    else:
        await update.message.reply_text("✅ Partido eliminado.")

async def cmd_follow(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = " ".join(context.args).strip()
    if not text:
        await update.message.reply_text("Uso: /seguir Equipo1 vs Equipo2 [YYYY-MM-DD]")
        return

    parts = text.split()
    d = parse_optional_date(parts)
    if d:
        text_teams = " ".join(parts[:-1]).strip()
        target_date = d
    else:
        text_teams = text
        target_date = None  # hoy por defecto

    try:
        home, away = parse_teams(text_teams)
    except Exception as e:
        await update.message.reply_text(str(e))
        return

    ev = await find_best_event(home, away, target_date)
    if not ev:
        await update.message.reply_text("❌ No encontré ese partido (ni en ±1 día). Prueba con nombres más exactos o sin fecha.")
        return

    h_ev, a_ev = teams_from_event(ev)
    ev_id = str(ev.get("idEvent") or "").strip()
    ev_date = (ev.get("dateEvent") or (target_date.strftime("%Y-%m-%d") if target_date else _now_utc().date().strftime("%Y-%m-%d"))).strip()
    league = (ev.get("strLeague") or "").strip()
    kickoff_local = (ev.get("strTimeLocal") or ev.get("strTime") or "").strip()

    state = load_state()
    tracked = get_tracked(state)

    # evita duplicados por idEvent
    for m in tracked:
        if m.event_id == ev_id:
            await update.message.reply_text(f"ℹ️ Ya estabas siguiendo: {m.home} vs {m.away} ({m.date_str})")
            return

    tracked.append(
        TrackedMatch(
            home=h_ev or home,
            away=a_ev or away,
            date_str=ev_date,
            event_id=ev_id,
            league=league,
            kickoff_local=kickoff_local,
        )
    )
    set_tracked(state, tracked)
    save_state(state)

    await update.message.reply_text(f"✅ Añadido: {h_ev} vs {a_ev} ({ev_date}). Actualizo cada {POLL_SECONDS//60 if POLL_SECONDS>=60 else POLL_SECONDS}s.")

# =========================
# POLLING JOB
# =========================
async def poll_once(app: Application) -> None:
    state = load_state()
    tracked = get_tracked(state)
    if not tracked:
        return

    changed = False

    for m in tracked:
        try:
            ev = await fetch_event_by_id(m.event_id)
        except Exception:
            log.exception("Error consultando evento id=%s", m.event_id)
            continue

        if not ev:
            continue

        home, away = teams_from_event(ev)
        hs, aas = score_from_event(ev)
        status = status_from_event(ev)
        bucket = status_bucket(status)

        # 1) Avisos de cambio de estado (inicio/descanso/final)
        prev_bucket = status_bucket(m.last_status) if m.last_status else ""
        if m.last_status and status != m.last_status:
            # inicio (pasa a INPLAY)
            if prev_bucket != "INPLAY" and bucket == "INPLAY":
                await send_msg(app, f"🔔 EMPIEZA: {home} vs {away} ({m.league})")
            # descanso
            if status.upper() == "HT" or "HALF" in status.upper():
                await send_msg(app, f"⏸ DESCANSO: {fmt_score(home, away, hs, aas)}")
            # final
            if bucket == "FINISHED" and prev_bucket != "FINISHED":
                await send_msg(app, f"🏁 FINAL: {fmt_score(home, away, hs, aas)}")

        # 2) Detectar goles por delta de marcador (equipo anotador + marcador)
        if m.last_home is not None and m.last_away is not None and hs is not None and aas is not None:
            if hs > m.last_home or aas > m.last_away:
                scorer = home if hs > m.last_home else away
                await send_msg(app, f"⚽ GOL: {scorer} — {fmt_score(home, away, hs, aas)}")

        # 3) Timeline (rojas / anulados) si está en juego (para no gastar a lo loco)
        #    OJO: si la API no trae timeline, no pasa nada.
        if bucket == "INPLAY":
            try:
                tl = await fetch_timeline(m.event_id)
                for item in tl:
                    key = timeline_key(item)
                    if key in m.seen_timeline_ids:
                        continue
                    msg = describe_timeline(item, home, away, hs, aas)
                    if msg:
                        await send_msg(app, msg + f" — {home} vs {away}")
                    m.seen_timeline_ids.append(key)
                    changed = True
            except Exception:
                # no bloqueamos el bot por timeline
                log.info("Timeline no disponible o error para %s", m.event_id)

        # 4) Guardar nuevo estado
        m.last_home = hs if hs is not None else m.last_home
        m.last_away = aas if aas is not None else m.last_away
        m.last_status = status
        changed = True

    if changed:
        set_tracked(state, tracked)
        save_state(state)

async def job_poll(context: ContextTypes.DEFAULT_TYPE) -> None:
    await poll_once(context.application)

# =========================
# MAIN
# =========================
def build_app() -> Application:
    if not TELEGRAM_TOKEN:
        raise RuntimeError("TELEGRAM_TOKEN no configurado")
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("seguir", cmd_follow))
    app.add_handler(CommandHandler("lista", cmd_list))
    app.add_handler(CommandHandler("borrar", cmd_delete))
    app.add_handler(CommandHandler("limpiar", cmd_clear))

    # JobQueue: poll cada POLL_SECONDS
    app.job_queue.run_repeating(job_poll, interval=POLL_SECONDS, first=5)

    return app

def main() -> None:
    app = build_app()
    log.info("Bot started. Target chat id: %s", TARGET_CHAT_ID)
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
