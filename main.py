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
from telegram.ext import Application, CommandHandler, ContextTypes

# =========================
# CONFIG
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
)
log = logging.getLogger("telegram-resultados")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
TARGET_CHAT_ID = int(os.getenv("TARGET_CHAT_ID", "0").strip())
POLL_SECONDS = int(os.getenv("POLL_SECONDS", "60").strip())
STATE_FILE = os.getenv("STATE_FILE", "state.json").strip()

# Primary: TheSportsDB
# IMPORTANTE: si SPORTSDB_KEY está vacío, NO construimos una URL con doble "//"
SPORTSDB_KEY = os.getenv("SPORTSDB_KEY", "1").strip()  # "1" suele ser la shared key
SPORTSDB_ENABLED = bool(SPORTSDB_KEY)
SPORTSDB_BASE = f"https://www.thesportsdb.com/api/v1/json/{SPORTSDB_KEY}" if SPORTSDB_ENABLED else ""

# Secondary: API-SPORTS (optional)
APISPORTS_KEY = os.getenv("APISPORTS_KEY", "").strip()
APISPORTS_BASE = os.getenv("APISPORTS_BASE", "https://v3.football.api-sports.io").strip()

INPLAY_STATUSES = {"1H", "2H", "HT", "ET", "P", "BT", "PEN"}
FINISHED_STATUSES = {"FT", "AET", "PEN"}


# =========================
# HELPERS
# =========================
def _now_utc() -> datetime:
    return datetime.utcnow()

def normalize_team(s: str) -> str:
    s = (s or "").strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    stop = {"fc", "cf", "sc", "ac", "cd", "ud", "afc", "the", "de", "la"}
    parts = [p for p in s.split() if p not in stop]
    return " ".join(parts)

def parse_teams(text: str) -> Tuple[str, str]:
    t = re.sub(r"\s+", " ", text.strip())
    if " vs " in t.lower():
        a, b = re.split(r"\s+vs\s+", t, flags=re.IGNORECASE, maxsplit=1)
    elif " - " in t:
        a, b = t.split(" - ", 1)
    else:
        raise ValueError("Formato inválido. Usa: /seguir Equipo1 vs Equipo2 [YYYY-MM-DD] | pick=...")
    return a.strip(), b.strip()

def parse_optional_date(parts: List[str]) -> Optional[date]:
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

def parse_date_yyyy_mm_dd(s: str) -> Optional[date]:
    s = (s or "").strip()
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        return None

# =========================
# PICK parsing
# =========================
def normalize_pick(p: str) -> str:
    p = (p or "").strip().upper()
    p = p.replace(" ", "")
    p = p.replace("Ó", "O")
    p = p.replace("OVER", "O")
    return p

def parse_pick_from_tail(tail: str) -> str:
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
    date_str: str
    match_id: str                 # idEvent (SportsDB) o fixture.id (API-SPORTS)
    provider: str = "sportsdb"    # "sportsdb" | "apisports"
    pick: str = ""

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
        return {"tracked": [], "team_cache": {}}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            js = json.load(f)
        if "tracked" not in js:
            js["tracked"] = []
        if "team_cache" not in js:
            js["team_cache"] = {}
        return js
    except Exception:
        log.exception("No se pudo leer state.json, empezando limpio.")
        return {"tracked": [], "team_cache": {}}

def save_state(state: Dict[str, Any]) -> None:
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, STATE_FILE)

def get_tracked(state: Dict[str, Any]) -> List[TrackedMatch]:
    out: List[TrackedMatch] = []
    for it in state.get("tracked", []):
        try:
            if "match_id" not in it and "event_id" in it:
                it = dict(it)
                it["match_id"] = it.pop("event_id")
                it["provider"] = it.get("provider", "sportsdb")
            out.append(TrackedMatch(**it))
        except Exception:
            log.exception("Entrada inválida en state.json, se ignora: %s", it)
    return out

def set_tracked(state: Dict[str, Any], matches: List[TrackedMatch]) -> None:
    state["tracked"] = [asdict(m) for m in matches]

# =========================
# HTTP
# =========================
class UpstreamBlocked(Exception):
    """Usado para marcar que el proveedor está 'bloqueado' / no responde útil."""

async def http_get_json(url: str, params: Optional[dict] = None, headers: Optional[dict] = None) -> dict:
    timeout = httpx.Timeout(20.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.get(url, params=params, headers=headers)
        r.raise_for_status()
        return r.json()

def is_sportsdb_unusable_error(exc: Exception) -> bool:
    # Si la key es mala / endpoint devuelve HTML / 404, etc.
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return code in {401, 403, 404, 429, 500, 502, 503}
    if isinstance(exc, httpx.RequestError):
        return True
    return False

# =========================
# API: TheSportsDB (PRIMARY)
# =========================
async def fetch_events_for_day(d: date) -> List[dict]:
    if not SPORTSDB_ENABLED:
        raise UpstreamBlocked("SPORTSDB_KEY vacío -> SportsDB deshabilitado")
    url = f"{SPORTSDB_BASE}/eventsday.php"
    js = await http_get_json(url, params={"d": d.strftime("%Y-%m-%d"), "s": "Soccer"})
    return js.get("events") or []

async def search_team_id_sportsdb(team_name: str) -> Optional[Tuple[str, str]]:
    if not SPORTSDB_ENABLED:
        return None
    url = f"{SPORTSDB_BASE}/searchteams.php"
    js = await http_get_json(url, params={"t": team_name})
    teams = js.get("teams") or []
    if not teams:
        return None
    q = normalize_team(team_name)
    best, best_score = None, -1
    for t in teams:
        name = (t.get("strTeam") or "").strip()
        nt = normalize_team(name)
        sc = 0
        if nt == q: sc += 100
        if q and q in nt: sc += 40
        if nt and nt in q: sc += 20
        if sc > best_score:
            best_score = sc
            best = t
    if not best:
        best = teams[0]
    return str(best.get("idTeam") or "").strip(), (best.get("strTeam") or team_name).strip()

async def fetch_team_events_window(team_id: str) -> List[dict]:
    if not SPORTSDB_ENABLED:
        return []
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

def score_from_sportsdb(ev: dict) -> Tuple[Optional[int], Optional[int]]:
    return safe_int(ev.get("intHomeScore")), safe_int(ev.get("intAwayScore"))

def status_from_sportsdb(ev: dict) -> str:
    return (ev.get("strStatus") or "").strip()

def teams_from_sportsdb(ev: dict) -> Tuple[str, str]:
    return (ev.get("strHomeTeam") or "").strip(), (ev.get("strAwayTeam") or "").strip()

def minute_from_sportsdb(ev: dict) -> str:
    m = (ev.get("strProgress") or "").strip()
    if m:
        return m
    mt = ev.get("intTime") or ev.get("strTime") or ""
    mt = str(mt).strip()
    if re.fullmatch(r"\d{2}:\d{2}(:\d{2})?", mt):
        return ""
    return mt

def event_match_score(ev: dict, home_q: str, away_q: str) -> int:
    h, a = teams_from_sportsdb(ev)
    nh = normalize_team(h); na = normalize_team(a)
    qh = normalize_team(home_q); qa = normalize_team(away_q)

    score = 0
    if nh == qh: score += 60
    if na == qa: score += 60
    if qh and qh in nh: score += 25
    if qa and qa in na: score += 25
    if nh == qa: score += 12
    if na == qh: score += 12

    if status_bucket(status_from_sportsdb(ev)) == "INPLAY":
        score += 30

    hs, as_ = score_from_sportsdb(ev)
    if hs is not None or as_ is not None:
        score += 5
    return score

async def find_best_event_sportsdb(home: str, away: str, target_date: Optional[date]) -> Optional[dict]:
    if not SPORTSDB_ENABLED:
        return None

    if target_date is None:
        target_date = _now_utc().date()

    candidates: List[dict] = []
    for delta in [0, -1, 1]:
        d = target_date + timedelta(days=delta)
        try:
            candidates.extend(await fetch_events_for_day(d))
        except Exception as e:
            if is_sportsdb_unusable_error(e):
                # No marcamos como fatal aquí; solo significa que el feed diario no sirve.
                log.info("SportsDB eventsday no usable (%s): %s", d, repr(e))
            else:
                log.exception("Error pidiendo eventsday %s", d)

    if candidates:
        scored = [(event_match_score(ev, home, away), ev) for ev in candidates]
        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_ev = scored[0]
        if best_score >= 45:
            return best_ev

    # fallback por equipo (SportsDB)
    try:
        home_team = await search_team_id_sportsdb(home)
        away_norm = normalize_team(away)
        if home_team:
            home_id, _ = home_team
            evs = await fetch_team_events_window(home_id)

            dd = {
                target_date.strftime("%Y-%m-%d"),
                (target_date - timedelta(days=1)).strftime("%Y-%m-%d"),
                (target_date + timedelta(days=1)).strftime("%Y-%m-%d"),
            }
            evs2 = [ev for ev in evs if (ev.get("dateEvent") or "").strip() in dd] or evs

            best, best_sc = None, -1
            for ev in evs2:
                h, a = teams_from_sportsdb(ev)
                if away_norm and away_norm not in (normalize_team(h) + " " + normalize_team(a)):
                    continue
                sc = event_match_score(ev, home, away)
                if sc > best_sc:
                    best_sc, best = sc, ev
            if best and best_sc >= 40:
                return best
    except Exception as e:
        log.info("Fallback por equipo (SportsDB) falló: %s", repr(e))

    return None

async def fetch_event_by_id_sportsdb(event_id: str) -> Optional[dict]:
    if not SPORTSDB_ENABLED:
        return None
    url = f"{SPORTSDB_BASE}/lookupevent.php"
    js = await http_get_json(url, params={"id": event_id})
    events = js.get("events") or []
    return events[0] if events else None

async def verify_event_usable_sportsdb(event_id: str) -> bool:
    """
    Verificación crítica:
    aunque encontremos un partido en SportsDB (por search/feeds),
    si lookupevent.php no devuelve evento, para nosotros está "bloqueado" => usamos API-SPORTS.
    """
    if not SPORTSDB_ENABLED:
        return False
    if not event_id:
        return False
    try:
        ev = await fetch_event_by_id_sportsdb(event_id)
        return ev is not None
    except Exception as e:
        if is_sportsdb_unusable_error(e):
            return False
        return False

async def fetch_timeline_sportsdb(event_id: str) -> List[dict]:
    if not SPORTSDB_ENABLED:
        return []
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
# API: API-SPORTS (SECONDARY)
# =========================
def apisports_headers() -> dict:
    return {"x-apisports-key": APISPORTS_KEY} if APISPORTS_KEY else {}

def score_from_apisports(fx: dict) -> Tuple[Optional[int], Optional[int]]:
    g = (fx.get("goals") or {})
    return safe_int(g.get("home")), safe_int(g.get("away"))

def status_from_apisports(fx: dict) -> str:
    st = (fx.get("fixture") or {}).get("status") or {}
    return (st.get("short") or st.get("long") or "").strip()

def minute_from_apisports(fx: dict) -> str:
    st = (fx.get("fixture") or {}).get("status") or {}
    el = st.get("elapsed")
    if el is None:
        return ""
    try:
        return f"{int(el)}'"
    except Exception:
        return ""

def teams_from_apisports(fx: dict) -> Tuple[str, str]:
    t = fx.get("teams") or {}
    return (t.get("home") or {}).get("name", "").strip(), (t.get("away") or {}).get("name", "").strip()

def league_from_apisports(fx: dict) -> str:
    l = fx.get("league") or {}
    return (l.get("name") or "").strip()

async def apisports_team_id(state: Dict[str, Any], team_name: str) -> Optional[int]:
    cache = state.get("team_cache", {}) or {}
    key = normalize_team(team_name)
    if key in cache:
        try:
            return int(cache[key])
        except Exception:
            pass

    if not APISPORTS_KEY:
        return None

    url = f"{APISPORTS_BASE}/teams"
    js = await http_get_json(url, params={"search": team_name}, headers=apisports_headers())
    resp = js.get("response") or []
    if not resp:
        return None

    best_id = None
    best_sc = -1
    for item in resp:
        name = ((item.get("team") or {}).get("name") or "").strip()
        tid = (item.get("team") or {}).get("id")
        if not name or tid is None:
            continue
        nt = normalize_team(name)
        q = key
        sc = 0
        if nt == q: sc += 100
        if q and q in nt: sc += 40
        if nt and nt in q: sc += 20
        if sc > best_sc:
            best_sc = sc
            best_id = int(tid)

    if best_id is None:
        tid = (resp[0].get("team") or {}).get("id")
        if tid is None:
            return None
        best_id = int(tid)

    cache[key] = best_id
    state["team_cache"] = cache
    return best_id

def match_score_apisports(fx: dict, home_q: str, away_q: str) -> int:
    h, a = teams_from_apisports(fx)
    nh = normalize_team(h); na = normalize_team(a)
    qh = normalize_team(home_q); qa = normalize_team(away_q)
    sc = 0
    if nh == qh: sc += 60
    if na == qa: sc += 60
    if qh and qh in nh: sc += 25
    if qa and qa in na: sc += 25
    if nh == qa: sc += 12
    if na == qh: sc += 12
    if status_bucket(status_from_apisports(fx)) == "INPLAY":
        sc += 30
    hs, as_ = score_from_apisports(fx)
    if hs is not None or as_ is not None:
        sc += 5
    return sc

async def find_best_event_apisports(state: Dict[str, Any], home: str, away: str, target_date: Optional[date]) -> Optional[dict]:
    if not APISPORTS_KEY:
        return None

    if target_date is None:
        target_date = _now_utc().date()

    home_id = await apisports_team_id(state, home)
    away_id = await apisports_team_id(state, away)
    if home_id is None:
        return None

    best = None
    best_sc = -1

    for delta in [0, -1, 1]:
        d = target_date + timedelta(days=delta)
        url = f"{APISPORTS_BASE}/fixtures"
        js = await http_get_json(url, params={"date": d.strftime("%Y-%m-%d"), "team": home_id}, headers=apisports_headers())
        resp = js.get("response") or []
        for fx in resp:
            if away_id is not None:
                th = ((fx.get("teams") or {}).get("home") or {}).get("id")
                ta = ((fx.get("teams") or {}).get("away") or {}).get("id")
                if away_id not in {th, ta}:
                    continue
            sc = match_score_apisports(fx, home, away)
            if sc > best_sc:
                best_sc = sc
                best = fx

    if best and best_sc >= 45:
        return best
    return None

async def fetch_fixture_by_id_apisports(fixture_id: str) -> Optional[dict]:
    if not APISPORTS_KEY:
        return None
    url = f"{APISPORTS_BASE}/fixtures"
    js = await http_get_json(url, params={"id": fixture_id}, headers=apisports_headers())
    resp = js.get("response") or []
    return resp[0] if resp else None

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
# MIGRATION: SportsDB -> API-SPORTS
# =========================
async def migrate_match_to_apisports(state: Dict[str, Any], m: TrackedMatch) -> bool:
    """
    Si SportsDB no permite consultar el evento (bloqueado),
    migramos el seguimiento a API-SPORTS usando home/away/date_str.
    """
    if not APISPORTS_KEY:
        return False

    td = parse_date_yyyy_mm_dd(m.date_str) or _now_utc().date()
    fx = await find_best_event_apisports(state, m.home, m.away, td)
    if not fx:
        return False

    fixture_id = str(((fx.get("fixture") or {}).get("id")) or "").strip()
    if not fixture_id:
        return False

    # actualizar datos principales
    h_ev, a_ev = teams_from_apisports(fx)
    m.home = h_ev or m.home
    m.away = a_ev or m.away
    m.provider = "apisports"
    m.match_id = fixture_id
    m.league = league_from_apisports(fx) or m.league
    m.kickoff_local = ((fx.get("fixture") or {}).get("date") or m.kickoff_local)

    # timeline ya no aplica
    m.seen_timeline_ids = []

    # sincroniza marcador/estado actuales
    hs, aas = score_from_apisports(fx)
    st = status_from_apisports(fx)
    m.last_home = hs if hs is not None else m.last_home
    m.last_away = aas if aas is not None else m.last_away
    m.last_status = st or m.last_status

    # si ya empezó, no queremos mandar "EMPIEZA" retroactivo
    bucket = status_bucket(st)
    if bucket == "INPLAY":
        m.started_notified = True

    return True

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
        f"\nTick: {fmt_minutes()}\n"
        f"Primary SportsDB: {'ON' if SPORTSDB_ENABLED else 'OFF'}\n"
        f"Fallback API-SPORTS: {'ON' if APISPORTS_KEY else 'OFF (APISPORTS_KEY vacío)'}"
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
        prov = f" [{m.provider}]"
        lines.append(f"• {m.home} vs {m.away} ({m.date_str}){p}{prov}")
    await update.message.reply_text("\n".join(lines))

async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    state = load_state()
    state["tracked"] = []
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

    state = load_state()
    tracked = get_tracked(state)

    # 1) intenta SportsDB (pero con verificación fuerte)
    provider = None
    match_id = ""
    ev = None
    fx = None

    if SPORTSDB_ENABLED:
        try:
            ev = await find_best_event_sportsdb(home, away, target_date)
            if ev:
                cand_id = str(ev.get("idEvent") or "").strip()
                if cand_id and await verify_event_usable_sportsdb(cand_id):
                    provider = "sportsdb"
                    match_id = cand_id
                else:
                    # encontrado "en teoría" pero NO usable => lo tratamos como bloqueado y pasamos a API-SPORTS
                    ev = None
        except Exception as e:
            log.info("SportsDB no usable en /seguir: %s", repr(e))
            ev = None

    # 2) fallback real a API-SPORTS si SportsDB falló/bloqueado
    if provider is None:
        fx = await find_best_event_apisports(state, home, away, target_date)
        if fx:
            provider = "apisports"
            match_id = str(((fx.get("fixture") or {}).get("id")) or "").strip()

    if provider is None or not match_id:
        extra = ""
        if not APISPORTS_KEY:
            extra = "\n(Nota: APISPORTS_KEY no está configurado, así que no pude usar el fallback.)"
        await update.message.reply_text("❌ No encontré ese partido (ni en ±1 día). Prueba con nombres más exactos o sin fecha." + extra)
        return

    # Normaliza datos según provider
    if provider == "sportsdb":
        h_ev, a_ev = teams_from_sportsdb(ev)
        ev_date = (ev.get("dateEvent") or (target_date.strftime("%Y-%m-%d") if target_date else _now_utc().date().strftime("%Y-%m-%d"))).strip()
        league = (ev.get("strLeague") or "").strip()
        kickoff_local = (ev.get("strTimeLocal") or ev.get("strTime") or "").strip()
        hs, aas = score_from_sportsdb(ev)
        status = status_from_sportsdb(ev)
        minute = minute_from_sportsdb(ev)
        bucket = status_bucket(status)
    else:
        h_ev, a_ev = teams_from_apisports(fx)
        dt_iso = ((fx.get("fixture") or {}).get("date") or "").strip()
        ev_date = (dt_iso[:10] if dt_iso else (target_date.strftime("%Y-%m-%d") if target_date else _now_utc().date().strftime("%Y-%m-%d")))
        league = league_from_apisports(fx)
        kickoff_local = dt_iso
        hs, aas = score_from_apisports(fx)
        status = status_from_apisports(fx)
        minute = minute_from_apisports(fx)
        bucket = status_bucket(status)

    # evita duplicados por (provider, match_id)
    for m in tracked:
        if m.provider == provider and m.match_id == match_id:
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

    started_notified = (bucket == "INPLAY")
    last_home = hs if hs is not None else None
    last_away = aas if aas is not None else None

    tracked.append(
        TrackedMatch(
            home=h_ev or home,
            away=a_ev or away,
            date_str=ev_date,
            match_id=match_id,
            provider=provider,
            pick=pick,
            league=league,
            kickoff_local=kickoff_local,
            last_home=last_home,
            last_away=last_away,
            last_status=status,
            started_notified=started_notified,
            ht_notified=False,
            ft_notified=False,
        )
    )
    set_tracked(state, tracked)
    save_state(state)

    extra_pick = f" | pick={pick}" if pick else ""
    extra_provider = f" [{provider}]"

    if bucket == "INPLAY":
        min_txt = f" {minute}" if minute else ""
        await update.message.reply_text(
            f"✅ Añadido: {h_ev} vs {a_ev} ({ev_date}){extra_pick}{extra_provider}\n"
            f"🟢 YA INICIADO{min_txt}: {fmt_score(h_ev, a_ev, hs, aas)}"
        )
    else:
        await update.message.reply_text(
            f"✅ Añadido: {h_ev} vs {a_ev} ({ev_date}){extra_pick}{extra_provider}. "
            f"Actualizo cada {fmt_minutes()}."
        )

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
        changed_this = False
        ev = None
        fx = None

        # 0) consulta según provider
        try:
            if m.provider == "sportsdb":
                ev = await fetch_event_by_id_sportsdb(m.match_id)
                # Si SportsDB "se cae/bloquea": intentamos migrar a API-SPORTS
                if not ev:
                    migrated = await migrate_match_to_apisports(state, m)
                    if migrated:
                        changed_any = True
                        changed_this = True
                        await send_msg(app, f"🔁 Migrado a API-SPORTS: {m.home} vs {m.away} ({m.league}){fmt_pick(m.pick)}")
                        # ya migrado, seguimos en este mismo tick con API-SPORTS
                        fx = await fetch_fixture_by_id_apisports(m.match_id)
                        ev = None
            else:
                fx = await fetch_fixture_by_id_apisports(m.match_id)
        except Exception:
            log.exception("Error consultando match provider=%s id=%s", m.provider, m.match_id)
            continue

        # 1) parseo normalizado
        if m.provider == "sportsdb":
            if not ev:
                continue
            home, away = teams_from_sportsdb(ev)
            hs, aas = score_from_sportsdb(ev)
            status = status_from_sportsdb(ev)
            bucket = status_bucket(status)
            timeline_fetch = True
        else:
            if not fx:
                continue
            home, away = teams_from_apisports(fx)
            hs, aas = score_from_apisports(fx)
            status = status_from_apisports(fx)
            bucket = status_bucket(status)
            timeline_fetch = False

        # 2) EMPIEZA
        if bucket == "INPLAY" and not m.started_notified:
            await send_msg(app, f"🔔 EMPIEZA: {home} vs {away} ({m.league}){fmt_pick(m.pick)}")
            m.started_notified = True
            changed_any = True
            changed_this = True

        # 3) DESCANSO
        if (status.upper() == "HT" or "HALF" in status.upper()) and not m.ht_notified:
            await send_msg(app, f"⏸ DESCANSO: {fmt_score(home, away, hs, aas)}{fmt_pick(m.pick)}")
            m.ht_notified = True
            changed_any = True
            changed_this = True

        # 4) FINAL
        if bucket == "FINISHED" and not m.ft_notified:
            await send_msg(app, f"🏁 FINAL: {fmt_score(home, away, hs, aas)}{fmt_pick(m.pick)}")
            m.ft_notified = True
            changed_any = True
            changed_this = True

        # 5) ANTI-VAR (retroceso marcador)
        if (
            m.last_home is not None and m.last_away is not None
            and hs is not None and aas is not None
            and (hs < m.last_home or aas < m.last_away)
        ):
            before = fmt_score(home, away, m.last_home, m.last_away)
            now = fmt_score(home, away, hs, aas)
            await send_msg(app, f"🔄 CORRECCIÓN (posible VAR):\nAntes: {before}\nAhora: {now}{fmt_pick(m.pick)}")
            changed_any = True
            changed_this = True
        else:
            # 6) GOL (subida marcador)
            if m.last_home is not None and m.last_away is not None and hs is not None and aas is not None:
                if hs > m.last_home or aas > m.last_away:
                    scorer = home if hs > m.last_home else away
                    await send_msg(app, f"⚽ GOL: {scorer}\n{fmt_score(home, away, hs, aas)}{fmt_pick(m.pick)}")
                    changed_any = True
                    changed_this = True

        # 7) Timeline (solo SportsDB)
        if timeline_fetch and bucket == "INPLAY":
            try:
                tl = await fetch_timeline_sportsdb(m.match_id)
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
                    changed_this = True
            except Exception:
                log.info("Timeline no disponible o error para %s", m.match_id)

        # 8) Guardar estado (solo si cambió algo real)
        if hs is not None and hs != m.last_home:
            m.last_home = hs
            changed_any = True
            changed_this = True
        if aas is not None and aas != m.last_away:
            m.last_away = aas
            changed_any = True
            changed_this = True
        if status and status != m.last_status:
            m.last_status = status
            changed_any = True
            changed_this = True

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
    log.info("SPORTSDB_ENABLED=%s | APISPORTS_KEY=%s", SPORTSDB_ENABLED, ("present" if APISPORTS_KEY else "missing"))
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
