import asyncio
import json
import logging
import os
import re
import unicodedata
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta, date
from typing import Any, Dict, List, Optional, Tuple
from difflib import SequenceMatcher

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

# TheSportsDB
SPORTSDB_KEY = os.getenv("SPORTSDB_KEY", "1").strip()
SPORTSDB_ENABLED = bool(SPORTSDB_KEY)
SPORTSDB_BASE = f"https://www.thesportsdb.com/api/v1/json/{SPORTSDB_KEY}" if SPORTSDB_ENABLED else ""

# API-SPORTS
APISPORTS_KEY = os.getenv("APISPORTS_KEY", "").strip()
APISPORTS_BASE = os.getenv("APISPORTS_BASE", "https://v3.football.api-sports.io").strip()

# Preferencia
PREFER_APISPORTS = os.getenv("PREFER_APISPORTS", "1").strip() not in {"0", "false", "False", "NO", "no"}

# Auto remove finished
AUTO_REMOVE_FINISHED = os.getenv("AUTO_REMOVE_FINISHED", "0").strip() not in {"0", "false", "False", "NO", "no"}

# Anti-spam
MIN_SECONDS_BETWEEN_ALERTS = int(os.getenv("MIN_SECONDS_BETWEEN_ALERTS", "15").strip())

INPLAY_STATUSES = {"1H", "2H", "HT", "ET", "P", "BT", "PEN"}
FINISHED_STATUSES = {"FT", "AET", "PEN"}

# Ventana de fechas
DATE_WINDOW_DAYS = 4

# Retries/timeout
MAX_HTTP_RETRIES = 2
REQUEST_TIMEOUT = 20.0

# housekeeping
QUERY_CACHE_TTL_HOURS = int(os.getenv("QUERY_CACHE_TTL_HOURS", "48").strip())
MAX_CONSECUTIVE_FAILURES_BEFORE_MIGRATE = int(os.getenv("MAX_CONSECUTIVE_FAILURES_BEFORE_MIGRATE", "2").strip())
MAX_CONSECUTIVE_FAILURES_BEFORE_DROP = int(os.getenv("MAX_CONSECUTIVE_FAILURES_BEFORE_DROP", "12").strip())

# ===== SportsDB Team Index (por ligas) =====
SPORTSDB_LEAGUE_IDS = [
    4337,  # Dutch Eredivisie
    4328,  # English Premier League
    4331,  # German Bundesliga
    4332,  # Italian Serie A
    4335,  # Spanish La Liga
    4621,  # Austrian Bundesliga
    4334,  # French Ligue 1
    4344,  # Portuguese Primeira Liga
]
TEAM_INDEX_TTL_HOURS = int(os.getenv("TEAM_INDEX_TTL_HOURS", "168").strip())  # 7 días

# live-first (se mantiene, pero sin sesgos: solo entra si pasa hard-gate fuerte)
USE_SPORTSDB_LIVE_FIRST = os.getenv("USE_SPORTSDB_LIVE_FIRST", "1").strip() not in {"0", "false", "False", "NO", "no"}

# =========================
# MATCHING POLICY (NUEVO)
# =========================
# Score por equipo 0..100
MIN_SIDE_SCORE = int(os.getenv("MIN_SIDE_SCORE", "58").strip())     # cada lado debe pasar esto
MIN_PAIR_SCORE = int(os.getenv("MIN_PAIR_SCORE", "132").strip())   # suma de los 2 lados (aprox)
# Nota: con nombres “cortos/raros” puedes bajar MIN_SIDE_SCORE a 55

# =========================
# HELPERS
# =========================
def _now_utc() -> datetime:
    return datetime.utcnow()

def parse_date_yyyy_mm_dd(s: str) -> Optional[date]:
    s = (s or "").strip()
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        return None

def _ascii_lower(s: str) -> str:
    s = (s or "").strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s

_STOPWORDS = {
    "fc","cf","sc","ac","cd","ud","afc","cfc",
    "club","de","la","el","the","and","y",
    "sporting","sports","football","futbol","fútbol",
    "team","clubdeportivo","clubde",
    "sv","bv","vv","v.v.","v.v","v","fk","sk",
    "ss","as","ks","nk","rc","real",
}

_ALIAS = {
    "barca": "barcelona",
    "fcb": "barcelona",
    "athleti": "atletico",
    "atleti": "atletico",
    "psg": "paris saint germain",
    "inter": "internazionale",
    "man utd": "manchester united",
    "manutd": "manchester united",
    "man city": "manchester city",
    "spurs": "tottenham",
}

def normalize_team(s: str) -> str:
    s = _ascii_lower(s)
    s = s.replace("&", " ")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if s in _ALIAS:
        s = _ALIAS[s]
    parts = []
    for p in s.split():
        if p in _STOPWORDS:
            continue
        parts.append(p)
    return " ".join(parts).strip()

def team_tokens(s: str) -> set:
    return set(normalize_team(s).split())

def token_similarity(a: str, b: str) -> float:
    ta = team_tokens(a)
    tb = team_tokens(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0

def string_similarity(a: str, b: str) -> float:
    a2 = normalize_team(a)
    b2 = normalize_team(b)
    if not a2 or not b2:
        return 0.0
    return SequenceMatcher(None, a2, b2).ratio()

def parse_teams(text: str) -> Tuple[str, str]:
    t = re.sub(r"\s+", " ", text.strip())
    if re.search(r"\s+vs\s+", t, flags=re.IGNORECASE):
        a, b = re.split(r"\s+vs\s+", t, flags=re.IGNORECASE, maxsplit=1)
    elif re.search(r"\s+v\s+", t, flags=re.IGNORECASE):
        a, b = re.split(r"\s+v\s+", t, flags=re.IGNORECASE, maxsplit=1)
    elif " - " in t:
        a, b = t.split(" - ", 1)
    elif " @ " in t:
        a, b = t.split(" @ ", 1)
    else:
        raise ValueError("Formato inválido. Usa: /seguir Equipo1 vs Equipo2 [YYYY-MM-DD] | pick=...")
    return a.strip(), b.strip()

def parse_optional_date(parts: List[str]) -> Optional[date]:
    if not parts:
        return None
    return parse_date_yyyy_mm_dd(parts[-1])

def safe_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        return None

def status_bucket(status: str) -> str:
    s_up = (status or "").strip().upper()
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

def make_query_key(home: str, away: str, target_date: Optional[date]) -> str:
    td = target_date or _now_utc().date()
    a = normalize_team(home)
    b = normalize_team(away)
    x, y = sorted([a, b])
    return f"{x}__{y}__{td.strftime('%Y-%m-%d')}"

def parse_iso_dt(s: str) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        return None

def should_send(now: datetime, last_sent_iso: str) -> bool:
    if not MIN_SECONDS_BETWEEN_ALERTS:
        return True
    dt = parse_iso_dt(last_sent_iso)
    if not dt:
        return True
    return (now - dt).total_seconds() >= MIN_SECONDS_BETWEEN_ALERTS

def now_iso() -> str:
    return _now_utc().isoformat()

def cleanup_query_cache(state: Dict[str, Any]) -> None:
    qc = state.get("query_cache", {}) or {}
    if not qc:
        return
    ttl = timedelta(hours=QUERY_CACHE_TTL_HOURS)
    now = _now_utc()
    to_del = []
    for k, v in qc.items():
        saved = parse_iso_dt((v or {}).get("saved_utc", ""))
        if not saved:
            continue
        if now - saved > ttl:
            to_del.append(k)
    for k in to_del:
        qc.pop(k, None)
    state["query_cache"] = qc

def unfollow_cmd_line(m: "TrackedMatch") -> str:
    return f"/borrar {m.home} vs {m.away} {m.date_str}"

def minute_prefix(minute: str) -> str:
    minute = (minute or "").strip()
    if not minute:
        return ""
    if minute.endswith("'"):
        return f"⏱ {minute} "
    m2 = re.sub(r"[^\d]", "", minute)
    if m2.isdigit():
        return f"⏱ {m2}' "
    return f"⏱ {minute} "

def split_alternates(s: str) -> List[str]:
    if not s:
        return []
    parts = re.split(r"[;|/]", s)
    out = []
    for p in parts:
        p = (p or "").strip()
        if p:
            out.append(p)
    return out

def parse_state_dt(s: str) -> Optional[datetime]:
    return parse_iso_dt(s)

def is_state_fresh(saved_iso: str, ttl_hours: int) -> bool:
    dt = parse_state_dt(saved_iso or "")
    if not dt:
        return False
    return (_now_utc() - dt) <= timedelta(hours=ttl_hours)

# =========================
# MATCHING (NUEVO: hard gate + scores explicables)
# =========================
def team_similarity_score(event_name: str, query_name: str) -> int:
    """
    Devuelve 0..100 (aprox). Combina tokens + string + exactitud normalizada.
    """
    evn = normalize_team(event_name)
    qn = normalize_team(query_name)
    if not evn or not qn:
        return 0

    ts = token_similarity(event_name, query_name)          # 0..1
    ss = string_similarity(event_name, query_name)         # 0..1
    exact = 1.0 if evn == qn else 0.0

    # Ponderación robusta
    raw = 0.55 * ts + 0.35 * ss + 0.10 * exact
    return int(round(100 * raw))

def best_assignment_scores(ev_home: str, ev_away: str, q_home: str, q_away: str) -> Tuple[int, int, str]:
    """
    Devuelve (home_side_score, away_side_score, mode)
    mode: "STRAIGHT" si (ev_home~q_home y ev_away~q_away), "SWAP" si al revés.
    """
    s_hh = team_similarity_score(ev_home, q_home)
    s_aa = team_similarity_score(ev_away, q_away)
    s_ha = team_similarity_score(ev_home, q_away)
    s_ah = team_similarity_score(ev_away, q_home)

    straight_min = min(s_hh, s_aa)
    swap_min = min(s_ha, s_ah)

    if swap_min > straight_min:
        return s_ha, s_ah, "SWAP"
    return s_hh, s_aa, "STRAIGHT"

def passes_hard_gate(ev_home: str, ev_away: str, q_home: str, q_away: str) -> Tuple[bool, int, int, int, str, str]:
    """
    (ok, pair_score, s1, s2, mode, reason_if_fail)
    """
    s1, s2, mode = best_assignment_scores(ev_home, ev_away, q_home, q_away)
    pair = s1 + s2

    if s1 < MIN_SIDE_SCORE or s2 < MIN_SIDE_SCORE:
        return (False, pair, s1, s2, mode, f"rechazado: lado débil (s1={s1}, s2={s2}, min={MIN_SIDE_SCORE})")
    if pair < MIN_PAIR_SCORE:
        return (False, pair, s1, s2, mode, f"rechazado: suma baja (pair={pair}, min={MIN_PAIR_SCORE})")
    return (True, pair, s1, s2, mode, "aceptado")

# =========================
# PICK parsing (multi-pick)
# =========================
def normalize_pick_one(p: str) -> str:
    p = (p or "").strip().upper()
    p = p.replace(" ", "")
    p = p.replace("Ó", "O")
    p = p.replace("OVER", "O")
    return p

def normalize_pick(p: str) -> str:
    raw = (p or "").strip()
    if not raw:
        return ""
    parts = [x.strip() for x in raw.split(",") if x.strip()]
    norm = [normalize_pick_one(x) for x in parts]
    return ",".join([x for x in norm if x])

def parse_pick_from_tail(tail: str) -> str:
    if not tail:
        return ""
    m = re.search(r"(?:^|[\s,])pick\s*=\s*([A-Za-z0-9\.\+\-,]+)", tail, flags=re.IGNORECASE)
    if not m:
        return ""
    return normalize_pick(m.group(1))

def fmt_pick_inline(pick: str) -> str:
    pick = (pick or "").strip()
    return f" (pick={pick})" if pick else ""

def fmt_pick(pick: str) -> str:
    pick = (pick or "").strip()
    if not pick:
        return ""
    if "," in pick:
        return f"\nPICKS: {pick}"
    return f"\nPICK: {pick}"

# =========================
# STATE
# =========================
@dataclass
class TrackedMatch:
    home: str
    away: str
    date_str: str
    match_id: str
    provider: str = "sportsdb"   # "sportsdb" | "apisports"
    pick: str = ""

    league: str = ""
    kickoff_local: str = ""

    last_home: Optional[int] = None
    last_away: Optional[int] = None
    last_status: str = ""
    started_notified: bool = False
    ht_notified: bool = False
    ft_notified: bool = False

    seen_timeline_ids: List[str] = field(default_factory=list)

    query_key: str = ""
    created_utc: str = ""
    consecutive_failures: int = 0

    last_alert_utc: str = ""

def load_state() -> Dict[str, Any]:
    if not os.path.exists(STATE_FILE):
        return {
            "tracked": [],
            "team_cache": {},
            "query_cache": {},
            "stats": {},
            "team_index": {},
            "team_index_meta": {},
        }
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            js = json.load(f)
        js.setdefault("tracked", [])
        js.setdefault("team_cache", {})
        js.setdefault("query_cache", {})
        js.setdefault("stats", {})
        js.setdefault("team_index", {})
        js.setdefault("team_index_meta", {})
        return js
    except Exception:
        log.exception("No se pudo leer state.json, empezando limpio.")
        return {
            "tracked": [],
            "team_cache": {},
            "query_cache": {},
            "stats": {},
            "team_index": {},
            "team_index_meta": {},
        }

def save_state(state: Dict[str, Any]) -> None:
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, STATE_FILE)

def get_tracked(state: Dict[str, Any]) -> List[TrackedMatch]:
    out: List[TrackedMatch] = []
    for it in state.get("tracked", []):
        try:
            it = dict(it)
            if "match_id" not in it and "event_id" in it:
                it["match_id"] = it.pop("event_id")
            it.setdefault("provider", "sportsdb")
            it.setdefault("seen_timeline_ids", [])
            it.setdefault("query_key", "")
            it.setdefault("created_utc", "")
            it.setdefault("consecutive_failures", 0)
            it.setdefault("last_alert_utc", "")
            it.setdefault("pick", "")
            out.append(TrackedMatch(**it))
        except Exception:
            log.exception("Entrada inválida en state.json, se ignora: %s", it)
    return out

def set_tracked(state: Dict[str, Any], matches: List[TrackedMatch]) -> None:
    state["tracked"] = [asdict(m) for m in matches]

def find_tracked_by_query(tracked: List[TrackedMatch], qkey: str) -> Optional[TrackedMatch]:
    for m in tracked:
        if m.query_key and m.query_key == qkey:
            return m
    return None

def find_tracked_by_names(tracked: List[TrackedMatch], home: str, away: str, target_date: Optional[date]) -> Optional[TrackedMatch]:
    qh = normalize_team(home)
    qa = normalize_team(away)
    td = target_date or _now_utc().date()
    valid_dates = set()
    for dlt in range(-DATE_WINDOW_DAYS, DATE_WINDOW_DAYS + 1):
        valid_dates.add((td + timedelta(days=dlt)).strftime("%Y-%m-%d"))
    for m in tracked:
        if m.date_str not in valid_dates:
            continue
        mh = normalize_team(m.home)
        ma = normalize_team(m.away)
        if {mh, ma} == {qh, qa}:
            return m
    return None

def stats_inc(state: Dict[str, Any], key: str, n: int = 1) -> None:
    st = state.get("stats", {}) or {}
    st[key] = int(st.get(key, 0)) + n
    state["stats"] = st

# =========================
# HTTP (cliente global + retries)
# =========================
class UpstreamBlocked(Exception):
    pass

_client: Optional[httpx.AsyncClient] = None

def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        timeout = httpx.Timeout(REQUEST_TIMEOUT, connect=10.0)
        _client = httpx.AsyncClient(timeout=timeout, headers={"User-Agent": "telegram-resultados/1.0"})
    return _client

async def http_get_json(url: str, params: Optional[dict] = None, headers: Optional[dict] = None) -> dict:
    last_exc: Optional[Exception] = None
    for attempt in range(MAX_HTTP_RETRIES + 1):
        try:
            r = await _get_client().get(url, params=params, headers=headers)
            if r.status_code >= 400:
                # Log compacto pero útil
                body_snip = (r.text or "")[:180].replace("\n", " ")
                log.info("HTTP %s -> %s body=%s", url, r.status_code, body_snip)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_exc = e
            if attempt < MAX_HTTP_RETRIES:
                await asyncio.sleep(0.25)
    raise last_exc  # type: ignore

def is_sportsdb_unusable_error(exc: Exception) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return code in {401, 403, 404, 429, 500, 502, 503}
    if isinstance(exc, httpx.RequestError):
        return True
    return False

# =========================
# API: TheSportsDB
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
        if nt == q:
            sc += 140
        sc += int(70 * token_similarity(name, team_name))
        sc += int(60 * string_similarity(name, team_name))

        if sc > best_score:
            best_score = sc
            best = t

    best = best or teams[0]
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

def sportsdb_candidate_score(ev: dict, home_q: str, away_q: str) -> Tuple[bool, int, int, int, str, str]:
    h, a = teams_from_sportsdb(ev)
    return passes_hard_gate(h, a, home_q, away_q)

# ---- Team index ----
async def fetch_all_teams_in_league_sportsdb(league_id: int) -> List[dict]:
    if not SPORTSDB_ENABLED:
        return []
    url = f"{SPORTSDB_BASE}/lookup_all_teams.php"
    js = await http_get_json(url, params={"id": str(league_id)})
    return js.get("teams") or []

async def ensure_team_index_sportsdb(state: Dict[str, Any]) -> None:
    idx_meta = (state.get("team_index_meta") or {})
    if is_state_fresh(idx_meta.get("saved_utc", ""), TEAM_INDEX_TTL_HOURS) and state.get("team_index"):
        return

    team_index: Dict[str, Dict[str, str]] = {}
    for lid in SPORTSDB_LEAGUE_IDS:
        try:
            teams = await fetch_all_teams_in_league_sportsdb(lid)
        except Exception as e:
            log.info("Team index: fallo liga %s: %s", lid, repr(e))
            continue

        for t in teams:
            tid = str(t.get("idTeam") or "").strip()
            name = (t.get("strTeam") or "").strip()
            alt = (t.get("strAlternate") or "").strip()
            if not tid or not name:
                continue

            candidates = [name] + split_alternates(alt)
            for cand in candidates:
                k = normalize_team(cand)
                if not k:
                    continue
                if k not in team_index:
                    team_index[k] = {"idTeam": tid, "strTeam": name}

    state["team_index"] = team_index
    state["team_index_meta"] = {"saved_utc": now_iso(), "league_ids": SPORTSDB_LEAGUE_IDS}
    log.info("Team index rebuilt: keys=%s", len(team_index))

def best_team_from_index(state: Dict[str, Any], query_name: str) -> Optional[Tuple[str, str]]:
    idx = state.get("team_index") or {}
    if not idx:
        return None

    qn = (query_name or "").strip()
    nq = normalize_team(qn)
    if not nq:
        return None

    if nq in idx:
        it = idx[nq]
        return it["idTeam"], it["strTeam"]

    best_k = None
    best_sc = -1.0
    for k in idx.keys():
        ts = token_similarity(k, nq)
        ss = SequenceMatcher(None, k, nq).ratio()
        sc = 0.65 * ts + 0.35 * ss
        if sc > best_sc:
            best_sc = sc
            best_k = k

    if best_k and best_sc >= 0.45:
        it = idx[best_k]
        return it["idTeam"], it["strTeam"]

    return None

async def fetch_livescore_sportsdb() -> List[dict]:
    if not SPORTSDB_ENABLED:
        return []
    url = f"{SPORTSDB_BASE}/livescore.php"
    js = await http_get_json(url, params={"s": "Soccer"})
    return js.get("events") or []

async def find_best_live_event_sportsdb(state: Dict[str, Any], home: str, away: str) -> Optional[dict]:
    """
    live-first sin sesgo: solo acepta si pasa hard-gate fuerte.
    Si tenemos IDs de ambos equipos y coincide el set de IDs -> preferencia máxima.
    """
    await ensure_team_index_sportsdb(state)
    home_hit = best_team_from_index(state, home)
    away_hit = best_team_from_index(state, away)
    home_id = home_hit[0] if home_hit else None
    away_id = away_hit[0] if away_hit else None

    events = await fetch_livescore_sportsdb()

    scored: List[Tuple[int, dict, str]] = []
    for ev in events:
        ev_hid = str(ev.get("idHomeTeam") or "").strip()
        ev_aid = str(ev.get("idAwayTeam") or "").strip()
        ev_home, ev_away = teams_from_sportsdb(ev)

        ok, pair, s1, s2, mode, reason = passes_hard_gate(ev_home, ev_away, home, away)

        # si tenemos ambos IDs, exigimos match real por IDs (para evitar “en vivo gana”)
        if home_id and away_id:
            if {ev_hid, ev_aid} != {home_id, away_id}:
                continue
            if ok:
                # boost determinista solo por IDs correctos (no por estado)
                pair += 40
        else:
            if not ok:
                continue

        scored.append((pair, ev, f"{reason} mode={mode} s1={s1} s2={s2}"))

    scored.sort(key=lambda x: x[0], reverse=True)
    if scored:
        best_pair, best_ev, why = scored[0]
        top3 = scored[:3]
        try:
            log.info(
                "SportsDB live-first candidates for '%s vs %s': %s",
                home, away,
                "; ".join([f"{teams_from_sportsdb(e)[0]} vs {teams_from_sportsdb(e)[1]} pair={p}" for p, e, _ in top3])
            )
            log.info("SportsDB live-first selected pair=%s why=%s", best_pair, why)
        except Exception:
            pass
        return best_ev
    return None

async def find_best_event_sportsdb(state: Dict[str, Any], home: str, away: str, target_date: Optional[date]) -> Optional[dict]:
    """
    NUEVO orden:
      1) Resolver por team IDs (index -> searchteams) y usar eventsnext/eventslast (espacio pequeño)
      2) Solo si falla, usar eventsday como último recurso
    """
    if not SPORTSDB_ENABLED:
        return None

    td = target_date or _now_utc().date()
    valid_dates = set((td + timedelta(days=dlt)).strftime("%Y-%m-%d") for dlt in range(-DATE_WINDOW_DAYS, DATE_WINDOW_DAYS + 1))

    await ensure_team_index_sportsdb(state)

    # 1) IDs por índice o searchteams
    home_hit = best_team_from_index(state, home)
    away_hit = best_team_from_index(state, away)

    home_id = home_hit[0] if home_hit else None
    away_id = away_hit[0] if away_hit else None

    if not home_id:
        try:
            h = await search_team_id_sportsdb(home)
            if h:
                home_id = h[0]
        except Exception:
            pass
    if not away_id:
        try:
            a = await search_team_id_sportsdb(away)
            if a:
                away_id = a[0]
        except Exception:
            pass

    # 1A) Eventos del equipo (ventana pequeña)
    candidates: List[dict] = []
    if home_id:
        candidates.extend(await fetch_team_events_window(home_id))
    if away_id:
        candidates.extend(await fetch_team_events_window(away_id))

    # dedupe por idEvent
    seen = set()
    cand2 = []
    for ev in candidates:
        eid = str(ev.get("idEvent") or "").strip()
        if not eid or eid in seen:
            continue
        seen.add(eid)
        cand2.append(ev)
    candidates = cand2

    # filtro por fecha (si existe dateEvent)
    c_by_date = [ev for ev in candidates if (ev.get("dateEvent") or "").strip() in valid_dates]
    candidates = c_by_date or candidates

    scored: List[Tuple[int, dict, str]] = []
    for ev in candidates:
        ev_home, ev_away = teams_from_sportsdb(ev)
        ok, pair, s1, s2, mode, reason = passes_hard_gate(ev_home, ev_away, home, away)
        if not ok:
            continue
        scored.append((pair, ev, f"{reason} mode={mode} s1={s1} s2={s2}"))

    scored.sort(key=lambda x: x[0], reverse=True)
    if scored:
        top3 = scored[:3]
        log.info(
            "SportsDB team-events candidates for '%s vs %s': %s",
            home, away,
            "; ".join([f"{teams_from_sportsdb(e)[0]} vs {teams_from_sportsdb(e)[1]} pair={p}" for p, e, _ in top3])
        )
        best_pair, best_ev, why = scored[0]
        log.info("SportsDB team-events selected pair=%s why=%s", best_pair, why)
        return best_ev

    # 2) Último recurso: eventsday (pero con hard gate)
    day_candidates: List[dict] = []
    for dlt in range(-DATE_WINDOW_DAYS, DATE_WINDOW_DAYS + 1):
        d = td + timedelta(days=dlt)
        try:
            day_candidates.extend(await fetch_events_for_day(d))
        except Exception as e:
            if is_sportsdb_unusable_error(e):
                log.info("SportsDB eventsday no usable (%s): %s", d, repr(e))
            else:
                log.exception("Error pidiendo eventsday %s", d)

    scored2: List[Tuple[int, dict, str]] = []
    for ev in day_candidates:
        ev_home, ev_away = teams_from_sportsdb(ev)
        ok, pair, s1, s2, mode, reason = passes_hard_gate(ev_home, ev_away, home, away)
        if not ok:
            continue
        scored2.append((pair, ev, f"{reason} mode={mode} s1={s1} s2={s2}"))

    scored2.sort(key=lambda x: x[0], reverse=True)
    if scored2:
        top3 = scored2[:3]
        log.info(
            "SportsDB eventsday candidates for '%s vs %s': %s",
            home, away,
            "; ".join([f"{teams_from_sportsdb(e)[0]} vs {teams_from_sportsdb(e)[1]} pair={p}" for p, e, _ in top3])
        )
        best_pair, best_ev, why = scored2[0]
        log.info("SportsDB eventsday selected pair=%s why=%s", best_pair, why)
        return best_ev

    log.info("SportsDB: no match after hard gate for '%s vs %s'", home, away)
    return None

async def fetch_event_by_id_sportsdb(event_id: str) -> Optional[dict]:
    if not SPORTSDB_ENABLED:
        return None
    url = f"{SPORTSDB_BASE}/lookupevent.php"
    js = await http_get_json(url, params={"id": event_id})
    events = js.get("events") or []
    return events[0] if events else None

async def verify_event_usable_sportsdb(event_id: str) -> bool:
    if not SPORTSDB_ENABLED or not event_id:
        return False
    try:
        ev = await fetch_event_by_id_sportsdb(event_id)
        return ev is not None
    except Exception:
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
# API: API-SPORTS
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
        team_name2 = normalize_team(team_name)
        if team_name2 and team_name2 != team_name:
            js = await http_get_json(url, params={"search": team_name2}, headers=apisports_headers())
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

        sc = int(110 * token_similarity(name, team_name))
        sc += int(90 * string_similarity(name, team_name))
        if normalize_team(name) == normalize_team(team_name):
            sc += 70

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

def apisports_candidate_score(fx: dict, home_q: str, away_q: str) -> Tuple[bool, int, int, int, str, str]:
    h, a = teams_from_apisports(fx)
    return passes_hard_gate(h, a, home_q, away_q)

async def find_best_event_apisports(state: Dict[str, Any], home: str, away: str, target_date: Optional[date]) -> Optional[dict]:
    if not APISPORTS_KEY:
        return None

    td = target_date or _now_utc().date()
    home_id = await apisports_team_id(state, home)
    away_id = await apisports_team_id(state, away)

    if home_id is None:
        home2 = normalize_team(home)
        if home2 and home2 != home:
            home_id = await apisports_team_id(state, home2)
    if home_id is None:
        return None

    scored: List[Tuple[int, dict, str]] = []

    for dlt in range(-DATE_WINDOW_DAYS, DATE_WINDOW_DAYS + 1):
        d = td + timedelta(days=dlt)
        url = f"{APISPORTS_BASE}/fixtures"
        js = await http_get_json(
            url,
            params={"date": d.strftime("%Y-%m-%d"), "team": home_id},
            headers=apisports_headers(),
        )
        resp = js.get("response") or []
        for fx in resp:
            if away_id is not None:
                th = ((fx.get("teams") or {}).get("home") or {}).get("id")
                ta = ((fx.get("teams") or {}).get("away") or {}).get("id")
                if away_id not in {th, ta}:
                    continue

            ok, pair, s1, s2, mode, reason = apisports_candidate_score(fx, home, away)
            if not ok:
                continue
            scored.append((pair, fx, f"{reason} mode={mode} s1={s1} s2={s2}"))

    scored.sort(key=lambda x: x[0], reverse=True)
    if scored:
        top3 = scored[:3]
        log.info(
            "API-SPORTS candidates for '%s vs %s': %s",
            home, away,
            "; ".join([f"{teams_from_apisports(f)[0]} vs {teams_from_apisports(f)[1]} pair={p}" for p, f, _ in top3])
        )
        best_pair, best_fx, why = scored[0]
        log.info("API-SPORTS selected pair=%s why=%s", best_pair, why)
        return best_fx

    log.info("API-SPORTS: no match after hard gate for '%s vs %s'", home, away)
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

# =========================
# MIGRATION: SportsDB -> API-SPORTS
# =========================
async def migrate_match_to_apisports(state: Dict[str, Any], m: TrackedMatch) -> bool:
    if not APISPORTS_KEY:
        return False
    td = parse_date_yyyy_mm_dd(m.date_str) or _now_utc().date()
    fx = await find_best_event_apisports(state, m.home, m.away, td)
    if not fx:
        return False

    fixture_id = str(((fx.get("fixture") or {}).get("id")) or "").strip()
    if not fixture_id:
        return False

    h_ev, a_ev = teams_from_apisports(fx)
    m.home = h_ev or m.home
    m.away = a_ev or m.away
    m.provider = "apisports"
    m.match_id = fixture_id
    m.league = league_from_apisports(fx) or m.league
    m.kickoff_local = ((fx.get("fixture") or {}).get("date") or m.kickoff_local)

    m.seen_timeline_ids = []

    hs, aas = score_from_apisports(fx)
    st = status_from_apisports(fx)
    m.last_home = hs if hs is not None else m.last_home
    m.last_away = aas if aas is not None else m.last_away
    m.last_status = st or m.last_status

    bucket = status_bucket(st)
    if bucket in {"INPLAY", "FINISHED"}:
        m.started_notified = True
    if bucket == "FINISHED":
        m.ft_notified = True

    return True

# =========================
# RESOLVER
# =========================
async def resolve_match(
    state: Dict[str, Any],
    home: str,
    away: str,
    target_date: Optional[date],
) -> Tuple[Optional[str], Optional[str], Optional[dict], Optional[dict]]:
    """
    Devuelve: (provider, match_id, ev_sportsdb, fx_apisports)
    Reutiliza cache por query_key.
    """
    cleanup_query_cache(state)

    qkey = make_query_key(home, away, target_date)
    qcache = state.get("query_cache", {}) or {}

    if qkey in qcache:
        try:
            prov = qcache[qkey]["provider"]
            mid = qcache[qkey]["match_id"]
            return prov, mid, None, None
        except Exception:
            pass

    # 0) Live-first SportsDB (1 request) si está ON, pero SOLO si pasa hard gate
    if SPORTSDB_ENABLED and USE_SPORTSDB_LIVE_FIRST:
        try:
            live_ev = await find_best_live_event_sportsdb(state, home, away)
            if live_ev:
                mid = str(live_ev.get("idEvent") or "").strip()
                if mid and await verify_event_usable_sportsdb(mid):
                    qcache[qkey] = {"provider": "sportsdb", "match_id": mid, "saved_utc": now_iso()}
                    state["query_cache"] = qcache
                    return "sportsdb", mid, live_ev, None
        except Exception as e:
            log.info("Live-first SportsDB falló: %s", repr(e))

    if PREFER_APISPORTS and APISPORTS_KEY:
        order = ["apisports", "sportsdb"]
    else:
        order = ["sportsdb", "apisports"]

    ev = None
    fx = None
    provider = None
    match_id = None

    for prov in order:
        if prov == "apisports":
            if not APISPORTS_KEY:
                continue
            fx = await find_best_event_apisports(state, home, away, target_date)
            if fx:
                mid = str(((fx.get("fixture") or {}).get("id")) or "").strip()
                if mid:
                    provider, match_id = "apisports", mid
                    break
        else:
            if not SPORTSDB_ENABLED:
                continue
            try:
                ev = await find_best_event_sportsdb(state, home, away, target_date)
                if ev:
                    mid = str(ev.get("idEvent") or "").strip()
                    if mid and await verify_event_usable_sportsdb(mid):
                        provider, match_id = "sportsdb", mid
                        break
            except Exception as e:
                log.info("SportsDB resolver error: %s", repr(e))

    if provider and match_id:
        qcache[qkey] = {"provider": provider, "match_id": match_id, "saved_utc": now_iso()}
        state["query_cache"] = qcache

    log.info("Resolver result for '%s vs %s' -> provider=%s id=%s", home, away, provider, match_id)
    return provider, match_id, ev, fx

# =========================
# COMMANDS
# =========================
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    help_text = (
        "Bot activo ✅\n\n"
        "Comandos:\n"
        "• /seguir <equipo1> vs <equipo2> [YYYY-MM-DD] | pick=<PICK>\n"
        "   (multi-pick: pick=O1.5,O2.5)\n"
        "• /seguirvarios (1 por línea) | pick=<PICK_GLOBAL>\n"
        "• /lista\n"
        "• /borrar <equipo1> vs <equipo2> [YYYY-MM-DD]\n"
        "• /limpiar\n"
        "• /estado\n"
        f"\nTick: {fmt_minutes()}\n"
        f"SportsDB: {'ON' if SPORTSDB_ENABLED else 'OFF'} | API-SPORTS: {'ON' if APISPORTS_KEY else 'OFF'}\n"
        f"Prefer API-SPORTS: {'YES' if (PREFER_APISPORTS and APISPORTS_KEY) else 'NO'}\n"
        f"SportsDB live-first: {'YES' if (SPORTSDB_ENABLED and USE_SPORTSDB_LIVE_FIRST) else 'NO'}\n"
        f"Team index TTL: {TEAM_INDEX_TTL_HOURS}h\n"
        f"Hard gate: MIN_SIDE_SCORE={MIN_SIDE_SCORE}, MIN_PAIR_SCORE={MIN_PAIR_SCORE}\n"
        f"Auto remove finished: {'YES' if AUTO_REMOVE_FINISHED else 'NO'}"
    )
    await update.message.reply_text(help_text)

async def cmd_estado(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    state = load_state()
    tracked = get_tracked(state)
    st = state.get("stats", {}) or {}

    # NUEVO: intentar construir índice aquí (para que no salga 0 eternamente)
    if SPORTSDB_ENABLED:
        try:
            before_n = len(state.get("team_index") or {})
            await ensure_team_index_sportsdb(state)
            after_n = len(state.get("team_index") or {})
            if after_n != before_n:
                save_state(state)
        except Exception as e:
            log.info("ensure_team_index_sportsdb en /estado falló: %s", repr(e))

    cleanup_query_cache(state)
    save_state(state)

    ti = state.get("team_index") or {}
    lines = [
        "Estado del bot:",
        f"- Tick: {fmt_minutes()}",
        f"- SportsDB: {'ON' if SPORTSDB_ENABLED else 'OFF'}",
        f"- API-SPORTS: {'ON' if APISPORTS_KEY else 'OFF'}",
        f"- Prefer API-SPORTS: {'YES' if (PREFER_APISPORTS and APISPORTS_KEY) else 'NO'}",
        f"- SportsDB live-first: {'YES' if (SPORTSDB_ENABLED and USE_SPORTSDB_LIVE_FIRST) else 'NO'}",
        f"- Team index keys: {len(ti)} (TTL {TEAM_INDEX_TTL_HOURS}h)",
        f"- Hard gate: MIN_SIDE_SCORE={MIN_SIDE_SCORE}, MIN_PAIR_SCORE={MIN_PAIR_SCORE}",
        f"- Auto remove finished: {'YES' if AUTO_REMOVE_FINISHED else 'NO'}",
        f"- Seguidos: {len(tracked)}",
        f"- Query cache: {len((state.get('query_cache') or {}))} (TTL {QUERY_CACHE_TTL_HOURS}h)",
        f"- Errores request: {st.get('http_errors', 0)}",
        f"- Migraciones: {st.get('migrations', 0)}",
        f"- Drops (no data): {st.get('drops', 0)}",
    ]
    await update.message.reply_text("\n".join(lines))

async def cmd_list(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    state = load_state()
    tracked = get_tracked(state)
    if not tracked:
        await update.message.reply_text("No hay partidos seguidos.")
        return
    lines = ["Partidos seguidos:"]
    for m in tracked:
        prov = f" [{m.provider}]"
        lines.append(f"• {m.home} vs {m.away}{fmt_pick_inline(m.pick)} ({m.date_str}){prov}")
    await update.message.reply_text("\n".join(lines))

async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    state = load_state()
    state["tracked"] = []
    state["query_cache"] = {}
    save_state(state)
    await update.message.reply_text("✅ Lista limpiada (y cache reseteada).")

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
        td = d
    else:
        text_teams = left
        td = _now_utc().date()
        date_str = td.strftime("%Y-%m-%d")

    try:
        home, away = parse_teams(text_teams)
    except Exception as e:
        await update.message.reply_text(str(e))
        return

    state = load_state()
    tracked = get_tracked(state)
    before = len(tracked)

    qkey = make_query_key(home, away, td)

    tracked = [
        m for m in tracked
        if not (
            {normalize_team(m.home), normalize_team(m.away)} == {normalize_team(home), normalize_team(away)}
            and m.date_str == date_str
        )
    ]

    qc = state.get("query_cache", {}) or {}
    qc.pop(qkey, None)
    state["query_cache"] = qc

    set_tracked(state, tracked)
    save_state(state)

    if len(tracked) == before:
        await update.message.reply_text("No encontré ese partido en la lista.")
    else:
        await update.message.reply_text("✅ Partido eliminado.")

def _parse_follow_payload(raw: str) -> Tuple[str, str, Optional[date], str]:
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

    home, away = parse_teams(text_teams)
    return home, away, target_date, pick

async def cmd_follow(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    raw = " ".join(context.args).strip()
    if not raw:
        await update.message.reply_text("Uso: /seguir Equipo1 vs Equipo2 [YYYY-MM-DD] | pick=...")
        return
    await _follow_one(update, raw)

async def cmd_follow_many(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = (update.message.text or "").strip()
    if not msg:
        return

    first_line, *rest = msg.splitlines()
    global_pick = ""
    if "|" in first_line:
        _, tail = first_line.split("|", 1)
        global_pick = parse_pick_from_tail(tail.strip())

    lines = [ln.strip() for ln in rest if ln.strip()]
    if not lines:
        await update.message.reply_text(
            "Uso:\n/seguirvarios | pick=O1.5\nVillarreal vs Valencia\nBarcelona vs Levante | pick=O2.5\n(1 partido por línea)"
        )
        return

    ok = 0
    exist = 0
    fail = 0
    msgs = []

    for ln in lines:
        if "pick=" not in ln.lower() and global_pick:
            raw = f"{ln} | pick={global_pick}"
        else:
            raw = ln

        res = await _follow_one(update, raw, silent=True)

        if res[0] == "ok":
            ok += 1
        elif res[0] == "exist":
            exist += 1
        else:
            fail += 1
        msgs.append(res[1])

    header = f"Resumen: ✅ {ok} añadidos | ♻️ {exist} ya existían | ❌ {fail} no encontrados"
    await update.message.reply_text(header + "\n\n" + "\n".join(msgs))

async def _follow_one(update: Update, raw: str, silent: bool = False) -> Tuple[str, str]:
    try:
        home, away, target_date, pick = _parse_follow_payload(raw)
    except Exception as e:
        if not silent:
            await update.message.reply_text(str(e))
        return ("fail", f"❌ {raw} -> {e}")

    state = load_state()

    if SPORTSDB_ENABLED:
        try:
            before_n = len(state.get("team_index") or {})
            await ensure_team_index_sportsdb(state)
            after_n = len(state.get("team_index") or {})
            if after_n != before_n:
                save_state(state)
        except Exception as e:
            log.info("ensure_team_index_sportsdb falló (se continúa): %s", repr(e))

    cleanup_query_cache(state)
    tracked = get_tracked(state)
    qkey = make_query_key(home, away, target_date)

    already = find_tracked_by_query(tracked, qkey) or find_tracked_by_names(tracked, home, away, target_date)
    if already:
        changed = False
        if pick and pick != (already.pick or ""):
            already.pick = pick
            changed = True
        if not already.query_key:
            already.query_key = qkey
            changed = True
        if changed:
            set_tracked(state, tracked)
            save_state(state)
            msg = f"♻️ {already.home} vs {already.away}{fmt_pick_inline(already.pick)} ({already.date_str}) [{already.provider}] (actualizado)"
            if not silent:
                await update.message.reply_text(msg)
            return ("exist", msg)

        msg = f"♻️ {already.home} vs {already.away}{fmt_pick_inline(already.pick)} ({already.date_str}) [{already.provider}]"
        if not silent:
            await update.message.reply_text(msg)
        return ("exist", msg)

    provider, match_id, ev, fx = await resolve_match(state, home, away, target_date)
    if not provider or not match_id:
        save_state(state)
        msg = f"❌ {home} vs {away}{fmt_pick_inline(pick)} -> no encontrado (hard gate ON, ±{DATE_WINDOW_DAYS}d)"
        if not silent:
            await update.message.reply_text(msg)
        return ("fail", msg)

    if provider == "sportsdb" and ev is None:
        try:
            ev = await fetch_event_by_id_sportsdb(match_id)
        except Exception as e:
            stats_inc(state, "http_errors", 1)
            log.info("SportsDB lookup error: %s", repr(e))
            ev = None
    if provider == "apisports" and fx is None:
        try:
            fx = await fetch_fixture_by_id_apisports(match_id)
        except Exception as e:
            stats_inc(state, "http_errors", 1)
            log.info("API-SPORTS lookup error: %s", repr(e))
            fx = None

    # Si el lookup falla, invalidamos cache y reintentamos fallback
    if provider == "sportsdb" and not ev:
        qc = state.get("query_cache", {}) or {}
        qc.pop(qkey, None)
        state["query_cache"] = qc
        if APISPORTS_KEY:
            fx2 = await find_best_event_apisports(state, home, away, target_date)
            if fx2:
                provider = "apisports"
                fx = fx2
                match_id = str(((fx.get("fixture") or {}).get("id")) or "").strip()
                state["query_cache"][qkey] = {"provider": provider, "match_id": match_id, "saved_utc": now_iso()}
            else:
                save_state(state)
                msg = f"❌ {home} vs {away}{fmt_pick_inline(pick)} -> ID cache roto (SportsDB) y no hay fallback"
                if not silent:
                    await update.message.reply_text(msg)
                return ("fail", msg)

    if provider == "apisports" and not fx:
        qc = state.get("query_cache", {}) or {}
        qc.pop(qkey, None)
        state["query_cache"] = qc
        if SPORTSDB_ENABLED:
            ev2 = await find_best_event_sportsdb(state, home, away, target_date)
            if ev2:
                mid2 = str(ev2.get("idEvent") or "").strip()
                if mid2 and await verify_event_usable_sportsdb(mid2):
                    provider = "sportsdb"
                    match_id = mid2
                    ev = ev2
                    state["query_cache"][qkey] = {"provider": provider, "match_id": match_id, "saved_utc": now_iso()}
                else:
                    save_state(state)
                    msg = f"❌ {home} vs {away}{fmt_pick_inline(pick)} -> ID cache roto (API-SPORTS) y fallback no usable"
                    if not silent:
                        await update.message.reply_text(msg)
                    return ("fail", msg)
            else:
                save_state(state)
                msg = f"❌ {home} vs {away}{fmt_pick_inline(pick)} -> ID cache roto (API-SPORTS) y fallback no encontrado"
                if not silent:
                    await update.message.reply_text(msg)
                return ("fail", msg)

    if provider == "sportsdb":
        h_ev, a_ev = teams_from_sportsdb(ev)
        ev_date = (ev.get("dateEvent") or ((target_date or _now_utc().date()).strftime("%Y-%m-%d"))).strip()
        league = (ev.get("strLeague") or "").strip()
        kickoff_local = (ev.get("strTimeLocal") or ev.get("strTime") or "").strip()
        hs, aas = score_from_sportsdb(ev)
        status = status_from_sportsdb(ev)
        minute = minute_from_sportsdb(ev)
        bucket = status_bucket(status)
    else:
        h_ev, a_ev = teams_from_apisports(fx)
        dt_iso = ((fx.get("fixture") or {}).get("date") or "").strip()
        ev_date = (dt_iso[:10] if dt_iso else ((target_date or _now_utc().date()).strftime("%Y-%m-%d")))
        league = league_from_apisports(fx)
        kickoff_local = dt_iso
        hs, aas = score_from_apisports(fx)
        status = status_from_apisports(fx)
        minute = minute_from_apisports(fx)
        bucket = status_bucket(status)

    if bucket == "FINISHED" and AUTO_REMOVE_FINISHED:
        save_state(state)
        msg = f"🏁 {h_ev} {hs}–{aas} {a_ev} (FINAL){fmt_pick_inline(pick)}"
        if not silent:
            await update.message.reply_text(msg)
        return ("ok", msg)

    m = TrackedMatch(
        home=h_ev or home,
        away=a_ev or away,
        date_str=ev_date,
        match_id=match_id,
        provider=provider,
        pick=pick,
        league=league,
        kickoff_local=kickoff_local,
        last_home=hs if hs is not None else None,
        last_away=aas if aas is not None else None,
        last_status=status,
        started_notified=(bucket in {"INPLAY", "FINISHED"}),
        ht_notified=False,
        ft_notified=(bucket == "FINISHED"),
        query_key=qkey,
        created_utc=now_iso(),
        consecutive_failures=0,
        last_alert_utc="",
    )

    tracked.append(m)
    set_tracked(state, tracked)
    save_state(state)

    extra_provider = f" [{provider}]"
    title = f"{m.home} vs {m.away}{fmt_pick_inline(pick)} ({m.date_str}){extra_provider}"

    if bucket == "FINISHED":
        msg = f"✅ {title} -> 🏁 FINAL {fmt_score(m.home, m.away, hs, aas)}"
        if not silent:
            await update.message.reply_text(msg)
        return ("ok", msg)

    if bucket == "INPLAY":
        min_txt = f" {minute_prefix(minute)}" if minute else " "
        msg = f"✅ {title} -> 🟢 YA INICIADO{min_txt}{fmt_score(m.home, m.away, hs, aas)}"
        if not silent:
            await update.message.reply_text(msg)
        return ("ok", msg)

    msg = f"✅ {title} -> agregado"
    if not silent:
        await update.message.reply_text(msg)
    return ("ok", msg)

# =========================
# POLLING JOB
# =========================
async def poll_once(app: Application) -> None:
    state = load_state()
    cleanup_query_cache(state)
    tracked = get_tracked(state)
    if not tracked:
        return

    changed_any = False
    kept: List[TrackedMatch] = []

    for m in tracked:
        ev = None
        fx = None
        now = _now_utc()

        try:
            if m.provider == "sportsdb":
                ev = await fetch_event_by_id_sportsdb(m.match_id)
                if not ev:
                    m.consecutive_failures += 1
                    stats_inc(state, "http_errors", 1)
                    changed_any = True

                    if m.consecutive_failures >= MAX_CONSECUTIVE_FAILURES_BEFORE_MIGRATE:
                        migrated = await migrate_match_to_apisports(state, m)
                        if migrated:
                            stats_inc(state, "migrations", 1)
                            if should_send(now, m.last_alert_utc):
                                await send_msg(app, f"🔁 Migrado a API-SPORTS: {m.home} vs {m.away} ({m.league}){fmt_pick(m.pick)}")
                                m.last_alert_utc = now_iso()
                            fx = await fetch_fixture_by_id_apisports(m.match_id)
                            ev = None
                            m.consecutive_failures = 0
                            changed_any = True

                    if not ev and not fx:
                        if m.consecutive_failures >= MAX_CONSECUTIVE_FAILURES_BEFORE_DROP:
                            stats_inc(state, "drops", 1)
                            if should_send(now, m.last_alert_utc):
                                await send_msg(app, f"🧹 Eliminado por fallos repetidos: {m.home} vs {m.away}{fmt_pick_inline(m.pick)} ({m.date_str}) [{m.provider}]")
                                m.last_alert_utc = now_iso()
                            changed_any = True
                            continue
                        kept.append(m)
                        continue
                else:
                    if m.consecutive_failures != 0:
                        m.consecutive_failures = 0
                        changed_any = True
            else:
                fx = await fetch_fixture_by_id_apisports(m.match_id)
                if not fx:
                    m.consecutive_failures += 1
                    stats_inc(state, "http_errors", 1)
                    changed_any = True
                    if m.consecutive_failures >= MAX_CONSECUTIVE_FAILURES_BEFORE_DROP:
                        stats_inc(state, "drops", 1)
                        if should_send(now, m.last_alert_utc):
                            await send_msg(app, f"🧹 Eliminado por fallos repetidos: {m.home} vs {m.away}{fmt_pick_inline(m.pick)} ({m.date_str}) [{m.provider}]")
                            m.last_alert_utc = now_iso()
                        changed_any = True
                        continue
                    kept.append(m)
                    continue
                else:
                    if m.consecutive_failures != 0:
                        m.consecutive_failures = 0
                        changed_any = True
        except Exception:
            log.exception("Error consultando match provider=%s id=%s", m.provider, m.match_id)
            m.consecutive_failures += 1
            stats_inc(state, "http_errors", 1)
            changed_any = True
            kept.append(m)
            continue

        if m.provider == "sportsdb":
            home, away = teams_from_sportsdb(ev)
            hs, aas = score_from_sportsdb(ev)
            status = status_from_sportsdb(ev)
            minute = minute_from_sportsdb(ev)
            bucket = status_bucket(status)
            timeline_fetch = True
        else:
            home, away = teams_from_apisports(fx)
            hs, aas = score_from_apisports(fx)
            status = status_from_apisports(fx)
            minute = minute_from_apisports(fx)
            bucket = status_bucket(status)
            timeline_fetch = False

        if bucket == "INPLAY" and not m.started_notified:
            if should_send(now, m.last_alert_utc):
                await send_msg(app, f"🔔 EMPIEZA: {home} vs {away} ({m.league}){fmt_pick(m.pick)}")
                m.last_alert_utc = now_iso()
            m.started_notified = True
            changed_any = True

        if (status.upper() == "HT" or "HALF" in status.upper()) and not m.ht_notified:
            if should_send(now, m.last_alert_utc):
                await send_msg(app, f"⏸ DESCANSO: {fmt_score(home, away, hs, aas)}{fmt_pick(m.pick)}")
                m.last_alert_utc = now_iso()
            m.ht_notified = True
            changed_any = True

        if bucket == "FINISHED" and not m.ft_notified:
            if should_send(now, m.last_alert_utc):
                await send_msg(app, f"🏁 FINAL: {fmt_score(home, away, hs, aas)}{fmt_pick(m.pick)}")
                m.last_alert_utc = now_iso()
            m.ft_notified = True
            m.started_notified = True
            changed_any = True

        if (
            m.last_home is not None and m.last_away is not None
            and hs is not None and aas is not None
            and (hs < m.last_home or aas < m.last_away)
        ):
            if should_send(now, m.last_alert_utc):
                before = fmt_score(home, away, m.last_home, m.last_away)
                now_s = fmt_score(home, away, hs, aas)
                await send_msg(app, f"🔄 CORRECCIÓN (posible VAR):\nAntes: {before}\nAhora: {now_s}{fmt_pick(m.pick)}")
                m.last_alert_utc = now_iso()
            changed_any = True
        else:
            if m.last_home is not None and m.last_away is not None and hs is not None and aas is not None:
                if hs > m.last_home or aas > m.last_away:
                    if should_send(now, m.last_alert_utc):
                        scorer = home if hs > m.last_home else away
                        cmd = unfollow_cmd_line(m)
                        mp = minute_prefix(minute)
                        await send_msg(
                            app,
                            f"⚽ GOL: {mp}{scorer}\n{fmt_score(home, away, hs, aas)}{fmt_pick(m.pick)}"
                            f"\n\nDejar de seguir partido:\n{cmd}"
                        )
                        m.last_alert_utc = now_iso()
                    changed_any = True

        if timeline_fetch and bucket == "INPLAY":
            try:
                tl = await fetch_timeline_sportsdb(m.match_id)
                for item in tl:
                    key = timeline_key(item)
                    if key in m.seen_timeline_ids:
                        continue
                    desc = describe_timeline(item)
                    if desc and should_send(now, m.last_alert_utc):
                        kind, minute_tl, team = desc
                        mp = minute_prefix(minute_tl)
                        if kind == "RED":
                            msg = f"🟥 ROJA: {mp}{team}\n{home} vs {away}{fmt_pick(m.pick)}"
                        else:
                            msg = f"🚫 GOL ANULADO: {mp}{team}\n{home} vs {away}{fmt_pick(m.pick)}"
                        await send_msg(app, msg)
                        m.last_alert_utc = now_iso()
                    m.seen_timeline_ids.append(key)
                    changed_any = True
            except Exception:
                log.info("Timeline no disponible o error para %s", m.match_id)

        if hs is not None and hs != m.last_home:
            m.last_home = hs
            changed_any = True
        if aas is not None and aas != m.last_away:
            m.last_away = aas
            changed_any = True
        if status and status != m.last_status:
            m.last_status = status
            changed_any = True

        if bucket == "FINISHED" and m.ft_notified and AUTO_REMOVE_FINISHED:
            changed_any = True
            continue

        kept.append(m)

    if changed_any:
        set_tracked(state, kept)
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
    app.add_handler(CommandHandler("estado", cmd_estado))
    app.add_handler(CommandHandler("seguir", cmd_follow))
    app.add_handler(CommandHandler("seguirvarios", cmd_follow_many))
    app.add_handler(CommandHandler("lista", cmd_list))
    app.add_handler(CommandHandler("borrar", cmd_delete))
    app.add_handler(CommandHandler("limpiar", cmd_clear))

    app.job_queue.run_repeating(job_poll, interval=POLL_SECONDS, first=5)
    return app

def main() -> None:
    app = build_app()
    log.info("Bot started. Target chat id: %s", TARGET_CHAT_ID)
    log.info(
        "SPORTSDB=%s | APISPORTS=%s | PREFER_APISPORTS=%s | LIVE_FIRST=%s | AUTO_REMOVE_FINISHED=%s | HARD_GATE(min_side=%s min_pair=%s)",
        "ON" if SPORTSDB_ENABLED else "OFF",
        "ON" if APISPORTS_KEY else "OFF",
        "YES" if (PREFER_APISPORTS and APISPORTS_KEY) else "NO",
        "YES" if (SPORTSDB_ENABLED and USE_SPORTSDB_LIVE_FIRST) else "NO",
        "YES" if AUTO_REMOVE_FINISHED else "NO",
        MIN_SIDE_SCORE,
        MIN_PAIR_SCORE,
    )
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
