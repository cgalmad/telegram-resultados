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

# =========================
# HELPERS
# =========================
def _now_utc() -> datetime:
    return datetime.utcnow()

def normalize_team(s: str) -> str:
    """
    Normaliza nombres: minúsculas, sin acentos, sin puntuación, y elimina tokens comunes.
    """
    s = (s or "").strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

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
        raise ValueError("Formato inválido. Usa: /seguir Equipo1 vs Equipo2 [YYYY-MM-DD] | pick=...")
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

def fmt_minutes() -> str:
    return f"{POLL_SECONDS}s" if POLL_SECONDS < 60 else f"{POLL_SECONDS//60} min"

# =========================
# PICK parsing
# =========================
def normalize_pick(p: str) -> str:
    p = (p or "").strip().upper()
    p = p.replace(" ", "")
    p = p.replace("Ó", "O")
    return p

def parse_pick_from_tail(tail: str) -> str:
    """
    tail: parte derecha tras '|', ejemplo: 'pick=O2.5'
    """
    if not tail:
        return ""
    m = re.search(r"(?:^|[\s,])pick\s*=\s*([A-Za-z0-9\.\+\-]+)", tail, flags=re.IGNORECASE)
    if not m:
        return ""
    return normalize_pick(m.group(1))

# =========================
# STATE
# =========================
@dataclass
class TrackedMatch:
    home: str
    away: str
    date_str: str  # YYYY-MM-DD (fecha objetivo)
    event_id: str  # TheSportsDB idEvent
    pick: str = ""  # PICK dentro de /seguir

    league: str = ""
    kickoff_local: str = ""

    last_home: Optional[int] = None
    last_away: Optional[int] = None
    last_status: str = ""
    started_notified: bool = False
    ht_notified: bool = False
    ft_notified: bool = False

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
    url = f"{SPORTSDB_BASE}/eventsday.php"
    js = await http_get_json(url, params={"d": d.strftime("%Y-%m-%d"), "s": "Soccer"})
    return js.get("events") or []

async def search_team_id(team_name: str) -> Optional[Tuple[str, str]]:
    url = f"{SPORTSDB_BASE}/searchteams.php"
    js = await http_get_json(url, params={"t": team_name})
    teams = js.get("teams") or []
    if not teams:
        return None

    q = normalize_team(team_name)
    best = None
    best_score = -1
    for t in teams:
        name = (t.get("strTeam") or "").strip()
        nt = normalize_team(name)
        sc = 0
        if nt == q:
            sc += 100
        if q and q in nt:
            sc += 40
        if nt and nt in q:
            sc += 20
        if sc > best_score:
            best_score = sc
            best = t
    if not best:
        best = teams[0]
    return str(best.get("idTeam") or "").strip(), (best.get("strTeam") or team_name).strip()

async def fetch_team_events_window(team_id: str) -> List[dict]:
    out: List[dict] = []
    try:
        js1 = await http_get_json(f"{SPORTSDB_BASE}/eventsnext.php", params={"id": team_id})
        out.extend(js1.get("events") or [])
    except Exception:
        pass
    try:
        js2 = await http_get_json(f"{SPORTSDB_BASE}/eventslast.php", params={"id": team_id})
        out.extend(js2.get("results") or js2.get("events") or [])
    except Exception:
        pass
    return out

def score_from_event(ev: dict) -> Tuple[Optional[int], Optional[int]]:
    return safe_int(ev.get("intHomeScore")), safe_int(ev.get("intAwayScore"))

def status_from_event(ev: dict) -> str:
    return (ev.get("strStatus") or "").strip()

def teams_from_event(ev: dict) -> Tuple[str, str]:
    return (ev.get("strHomeTeam") or "").strip(), (ev.get("strAwayTeam") or "").strip()

def event_match_score(ev: dict, home_q: str, away_q: str) -> int:
    h, a = teams_from_event(ev)
    nh = normalize_team(h)
    na = normalize_team(a)
    qh = normalize_team(home_q)
    qa = normalize_team(away_q)

    score = 0
    if nh == qh:
        score += 60
    if na == qa:
        score += 60

    if qh and qh in nh:
        score += 25
    if qa and qa in na:
        score += 25

    if nh == qa:
        score += 12
    if na == qh:
        score += 12

    bucket = status_bucket(status_from_event(ev))
    if bucket == "INPLAY":
        score += 30

    hs, as_ = score_from_event(ev)
    if hs is not None or as_ is not None:
        score += 5

    return score

async def find_best_event(home: str, away: str, target_date: Optional[date]) -> Optional[dict]:
    if target_date is None:
        target_date = _now_utc().date()

    # 1) DAY search
    candidates: List[dict] = []
    for delta in [0, -1, 1]:
        d = target_date + timedelta(days=delta)
        try:
            candidates.extend(await fetch_events_for_day(d))
        except Exception:
            log.exception("Error pidiendo eventos del día %s", d)

    if candidates:
        scored = [(event_match_score(ev, home, away), ev) for ev in candidates]
        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_ev = scored[0]
        if best_score >= 45:
            return best_ev

    # 2) FALLBACK por equipo (cuando eventsday no trae esa liga)
    try:
        home_team = await search_team_id(home)
        away_norm = normalize_team(away)
        if home_team:
            home_id, _ = home_team
            evs = await fetch_team_events_window(home_id)

            dd = {
                target_date.strftime("%Y-%m-%d"),
                (target_date - timedelta(days=1)).strftime("%Y-%m-%d"),
                (target_date + timedelta(days=1)).strftime("%Y-%m-%d"),
            }
            evs2 = []
            for ev in evs:
                ev_date = (ev.get("dateEvent") or "").strip()
                if ev_date and ev_date in dd:
                    evs2.append(ev)
            if not evs2:
                evs2 = evs

            best = None
            best_sc = -1
            for ev in evs2:
                h, a = teams_from_event(ev)
                if away_norm and away_norm not in (normalize_team(h) + " " + normalize_team(a)):
                    continue
                sc = event_match_score(ev, home, away)
                if sc > best_sc:
                    best_sc = sc
                    best = ev
            if best and best_sc >= 40:
                return best
    except Exception:
        log.exception("Fallback por equipo falló")

    return None

async def fetch_event_by_id(event_id: str) -> Optional[dict]:
    url = f"{SPORTSDB_BASE}/lookupevent.php"
    js = await http_get_json(url, params={"id": event_id})
    events = js.get("events") or []
    return events[0] if events else None

async def fetch_timeline(event_id: str) -> List[dict]:
    url = f"{SPORTSDB_BASE}/lookuptimeline.php"
    js = await http_get_json(url, params={"id": event_id})
    return js.get("timeline") or []

def timeline_key(item: dict) -> str:
    return (item.get("idTimeline") or "") or f"{item.get('strTime','')}-{item.get('strEvent','')}-{item.get('strTimeline','')}"

def describe_timeline(item: dict) -> Optional[Tuple[str, str, str]]:
    ttype = (item.get("strTimeline") or item.get("strType") or item.get("strEvent") or "").strip()
    minute = str(item.get("intTime") or item.get("strTime") or "").strip()
    team = (item.get("strTeam") or "").strip()
    ttype_low = ttype.lower()

    if "red" in ttype_low:
        return ("RED", minute, team)
    if "disallow" in ttype_low or "annul" in ttype_low or "cancel" in ttype_low:
        return ("DISALLOWED_GOAL", minute, team)
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

def fmt_pick(pick: str) -> str:
    pick = (pick or "").strip()
    return f"\nPICK: {pick}" if pick else ""

# =========================
# COMMANDS
# =========================
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    help_text = (
        "Bot activo ✅\n\n"
        "Comandos:\n"
        "• /seguir <equipo1> vs <equipo2> [YYYY-MM-DD] | pick=<PICK>\n"
        "  Ej: /seguir Betis vs Rayo Vallecano 2026-02-21 | pick=O2.5\n"
        "  (La fecha es opcional. Si falla por un día, el bot intenta ±1 día automáticamente)\n"
        "• /lista\n"
        "• /borrar <equipo1> vs <equipo2> [YYYY-MM-DD]\n"
        "• /limpiar\n"
        f"\nTick: {fmt_minutes()}"
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
        p = f" | pick={m.pick}" if m.pick else ""
        lines.append(f"• {m.home} vs {m.away} ({m.date_str}){p}")
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

    left = text.split("|", 1)[0].strip()
    parts = left.split()
    d = parse_optional_date(parts)
    if d:
        text_teams = " ".join(parts[:-1]).strip()
        date_str = d.strftime("%Y-%m-%d")
    else:
        text_teams = left
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
        if not (
            normalize_team(m.home) == normalize_team(home)
            and normalize_team(m.away) == normalize_team(away)
            and m.date_str == date_str
        )
    ]
    set_tracked(state, tracked)
    save_state(state)

    if len(tracked) == before:
        await update.message.reply_text("No encontré ese partido en la lista.")
    else:
        await update.message.reply_text("✅ Partido eliminado.")

async def cmd_follow(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    raw = " ".join(context.args).strip()
    if not raw:
        await update.message.reply_text("Uso: /seguir Equipo1 vs Equipo2 [YYYY-MM-DD] | pick=...")
        return

    left, tail = (raw.split("|", 1) + [""])[:2]
    left = left.strip()
    tail = tail.strip()

    pick = parse_pick_from_tail(tail)

    parts = left.split()
    d = parse_optional_date(parts)
    if d:
        text_teams = " ".join(parts[:-1]).strip()
        target_date = d
    else:
        text_teams = left
        target_date = None

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

    for m in tracked:
        if m.event_id == ev_id:
            changed = False
            if pick and pick != (m.pick or ""):
                m.pick = pick
                changed = True
            if changed:
                set_tracked(state, tracked)
                save_state(state)
                await update.message.reply_text(f"✅ Actualizado PICK: {m.home} vs {m.away} ({m.date_str}) | pick={m.pick}")
            else:
                await update.message.reply_text(f"ℹ️ Ya estabas siguiendo: {m.home} vs {m.away} ({m.date_str})")
            return

    tracked.append(
        TrackedMatch(
            home=h_ev or home,
            away=a_ev or away,
            date_str=ev_date,
            event_id=ev_id,
            pick=pick,
            league=league,
            kickoff_local=kickoff_local,
        )
    )
    set_tracked(state, tracked)
    save_state(state)

    extra = f" | pick={pick}" if pick else ""
    await update.message.reply_text(f"✅ Añadido: {h_ev} vs {a_ev} ({ev_date}){extra}. Actualizo cada {fmt_minutes()}.")

# =========================
# POLLING JOB
# =========================
async def poll_once(app: Application) -> None:
    state = load_state()
    tracked = get_tracked(state)
    if not tracked:
        return

    changed_any = False

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

        # 1) EMPIEZA
        if bucket == "INPLAY" and not m.started_notified:
            await send_msg(app, f"🔔 EMPIEZA: {home} vs {away} ({m.league}){fmt_pick(m.pick)}")
            m.started_notified = True
            changed_any = True

        # 2) DESCANSO
        if (status.upper() == "HT" or "HALF" in status.upper()) and not m.ht_notified:
            await send_msg(app, f"⏸ DESCANSO: {fmt_score(home, away, hs, aas)}{fmt_pick(m.pick)}")
            m.ht_notified = True
            changed_any = True

        # 3) FINAL
        if bucket == "FINISHED" and not m.ft_notified:
            await send_msg(app, f"🏁 FINAL: {fmt_score(home, away, hs, aas)}{fmt_pick(m.pick)}")
            m.ft_notified = True
            changed_any = True

        # 4) ANTI-VAR: si el marcador "retrocede" respecto al último guardado, avisar corrección
        #    (típico: gol anulado / corrección data provider)
        if (
            m.last_home is not None and m.last_away is not None
            and hs is not None and aas is not None
            and (hs < m.last_home or aas < m.last_away)
        ):
            before = fmt_score(home, away, m.last_home, m.last_away)
            now = fmt_score(home, away, hs, aas)
            await send_msg(
                app,
                f"🔄 CORRECCIÓN (posible VAR):\nAntes: {before}\nAhora: {now}{fmt_pick(m.pick)}"
            )
            changed_any = True
            # Nota: seguimos, pero NO disparamos "GOL" en este tick.
        else:
            # 5) GOL por delta de marcador (solo cuando sube)
            if m.last_home is not None and m.last_away is not None and hs is not None and aas is not None:
                if hs > m.last_home or aas > m.last_away:
                    scorer = home if hs > m.last_home else away
                    await send_msg(
                        app,
                        f"⚽ GOL: {scorer}\n{fmt_score(home, away, hs, aas)}{fmt_pick(m.pick)}"
                    )
                    changed_any = True

        # 6) Timeline: rojas / anulados (solo INPLAY)
        if bucket == "INPLAY":
            try:
                tl = await fetch_timeline(m.event_id)
                for item in tl:
                    key = timeline_key(item)
                    if key in m.seen_timeline_ids:
                        continue
                    desc = describe_timeline(item)
                    if desc:
                        kind, minute, team = desc
                        if kind == "RED":
                            msg = f"🟥 ROJA {minute}': {team}\n{home} vs {away}{fmt_pick(m.pick)}"
                        else:
                            msg = f"🚫 GOL ANULADO {minute}': {team}\n{home} vs {away}{fmt_pick(m.pick)}"
                        await send_msg(app, msg)
                    m.seen_timeline_ids.append(key)
                    changed_any = True
            except Exception:
                log.info("Timeline no disponible o error para %s", m.event_id)

        # 7) Guardar nuevo estado (siempre)
        m.last_home = hs if hs is not None else m.last_home
        m.last_away = aas if aas is not None else m.last_away
        m.last_status = status
        changed_any = True

    if changed_any:
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

    app.job_queue.run_repeating(job_poll, interval=POLL_SECONDS, first=5)
    return app

def main() -> None:
    app = build_app()
    log.info("Bot started. Target chat id: %s", TARGET_CHAT_ID)
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
