from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Any
import requests
import time
import numpy as np
import os
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────────────
BDL_KEY   = os.getenv("BDL_API_KEY", "0b07f4c7-7110-4cab-a4b0-2b22d0f31a83")
ODDS_KEY  = os.getenv("ODDS_API_KEY", "")
BDL_BASE  = "https://api.balldontlie.io/v1"
ODDS_BASE = "https://api.the-odds-api.com/v4"

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="NbaProp-ML API", version="5.0.0")

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── BallDontLie helpers ───────────────────────────────────────────────────────
BDL_HEADERS = {"Authorization": BDL_KEY}

def bdl_get(path: str, params: dict = {}) -> Any:
    r = requests.get(f"{BDL_BASE}{path}", headers=BDL_HEADERS, params=params, timeout=20)
    if not r.ok:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    return r.json()

def bdl_get_all(path: str, params: dict = {}) -> list:
    """Fetch all pages from a paginated BDL endpoint."""
    results = []
    cursor = None
    while True:
        p = {**params, "per_page": 100}
        if cursor:
            p["cursor"] = cursor
        data = bdl_get(path, p)
        results.extend(data.get("data", []))
        cursor = data.get("meta", {}).get("next_cursor")
        if not cursor:
            break
    return results

def odds_get(path: str, params: dict = {}) -> Any:
    params = {"apiKey": ODDS_KEY, **params}
    r = requests.get(f"{ODDS_BASE}{path}", params=params, timeout=15)
    if not r.ok:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    return r.json()

# ── Season helpers ────────────────────────────────────────────────────────────
def current_season_int() -> int:
    """BDL uses integer seasons: 2025 = 2025-26 season."""
    now = datetime.now()
    return now.year if now.month >= 10 else now.year - 1

def current_season_str() -> str:
    y = current_season_int()
    return f"{y}-{str(y+1)[2:]}"

def season_str_to_int(s: str) -> int:
    """Convert '2025-26' → 2025"""
    return int(s.split("-")[0])

# ── Player search ─────────────────────────────────────────────────────────────
def find_player(name: str) -> dict:
    """Search active players first, then all players. Tries full name then last name."""
    name = name.strip()
    parts = name.split()
    last = parts[-1] if parts else name
    first = parts[0] if len(parts) > 1 else ""

    print(f"find_player called with: '{name}', last='{last}', first='{first}'")

    # Try active players with last name search (more reliable than full name)
    data = bdl_get("/players/active", {"search": last, "per_page": 25})
    players = data.get("data", [])
    print(f"Active search for '{last}' returned {len(players)} players: {[p['first_name']+' '+p['last_name'] for p in players[:5]]}")

    if players:
        exact = [p for p in players if f"{p['first_name']} {p['last_name']}".lower() == name.lower()]
        if exact:
            return exact[0]
        if first:
            first_match = [p for p in players if p['last_name'].lower() == last.lower() and p['first_name'].lower().startswith(first.lower())]
            if first_match:
                return first_match[0]
        last_match = [p for p in players if p['last_name'].lower() == last.lower()]
        if last_match:
            return last_match[0]
        return players[0]

    # Fall back to all players
    data = bdl_get("/players", {"search": last, "per_page": 25})
    players = data.get("data", [])
    print(f"All players search for '{last}' returned {len(players)} players")
    if players:
        exact = [p for p in players if f"{p['first_name']} {p['last_name']}".lower() == name.lower()]
        if exact:
            return exact[0]
        if first:
            first_match = [p for p in players if p['last_name'].lower() == last.lower() and p['first_name'].lower().startswith(first.lower())]
            if first_match:
                return first_match[0]
        last_match = [p for p in players if p['last_name'].lower() == last.lower()]
        if last_match:
            return last_match[0]
        return players[0]

    raise HTTPException(status_code=404, detail=f"Player '{name}' not found.")

def format_player(p: dict) -> dict:
    team = p.get("team") or {}
    return {
        "id": p["id"],
        "full_name": f"{p['first_name']} {p['last_name']}",
        "first_name": p["first_name"],
        "last_name": p["last_name"],
        "position": p.get("position", ""),
        "team": team.get("full_name", ""),
        "team_abbr": team.get("abbreviation", ""),
        "team_id": team.get("id"),
        "is_active": True,
        "height": p.get("height", ""),
        "jersey_number": p.get("jersey_number", ""),
    }

# ── Game log helpers ───────────────────────────────────────────────────────────
def get_player_stats(player_id: int, seasons: list[int], postseason: bool = None) -> list[dict]:
    """Fetch all game stats for a player across given seasons."""
    params = {
        "player_ids[]": player_id,
        "per_page": 100,
    }
    for s in seasons:
        params[f"seasons[]"] = s
    # BDL needs seasons as repeated params — use list
    all_stats = []
    cursor = None
    season_params = "&".join([f"seasons[]={s}" for s in seasons])
    base_params = f"player_ids[]={player_id}&per_page=100"
    if postseason is not None:
        base_params += f"&postseason={'true' if postseason else 'false'}"

    while True:
        url = f"{BDL_BASE}/stats?{base_params}&{season_params}"
        if cursor:
            url += f"&cursor={cursor}"
        r = requests.get(url, headers=BDL_HEADERS, timeout=20)
        if not r.ok:
            break
        data = r.json()
        all_stats.extend(data.get("data", []))
        cursor = data.get("meta", {}).get("next_cursor")
        if not cursor:
            break

    return all_stats

def normalize_stat_row(s: dict) -> dict:
    """Convert BDL stat row to our internal format matching old nba_api format."""
    game = s.get("game", {})
    team = s.get("team", {})
    player = s.get("player", {})
    home_id = game.get("home_team_id")
    team_id = team.get("id")
    is_home = home_id == team_id

    # Build matchup string like old nba_api: "OKC vs. SAS" or "OKC @ SAS"
    # We'll approximate since BDL doesn't give opponent abbr directly
    matchup = "vs." if is_home else "@"

    # Parse minutes
    min_str = s.get("min", "0") or "0"
    try:
        if ":" in str(min_str):
            parts = str(min_str).split(":")
            minutes = float(parts[0]) + float(parts[1]) / 60
        else:
            minutes = float(min_str)
    except:
        minutes = 0.0

    pts = s.get("pts") or 0
    reb = s.get("reb") or 0
    ast = s.get("ast") or 0
    stl = s.get("stl") or 0
    blk = s.get("blk") or 0
    tov = s.get("turnover") or 0

    return {
        "GAME_ID":    str(game.get("id", "")),
        "GAME_DATE":  game.get("date", ""),
        "MATCHUP":    matchup,
        "WL":         "W" if (is_home and game.get("home_team_score", 0) > game.get("visitor_team_score", 0))
                       or (not is_home and game.get("visitor_team_score", 0) > game.get("home_team_score", 0))
                       else "L",
        "MIN":        round(minutes, 1),
        "PTS":        pts,
        "REB":        reb,
        "AST":        ast,
        "STL":        stl,
        "BLK":        blk,
        "TOV":        tov,
        "FGM":        s.get("fgm") or 0,
        "FGA":        s.get("fga") or 0,
        "FG_PCT":     s.get("fg_pct") or 0,
        "FG3M":       s.get("fg3m") or 0,
        "FG3A":       s.get("fg3a") or 0,
        "FG3_PCT":    s.get("fg3_pct") or 0,
        "FTM":        s.get("ftm") or 0,
        "FTA":        s.get("fta") or 0,
        "FT_PCT":     s.get("ft_pct") or 0,
        "OREB":       s.get("oreb") or 0,
        "DREB":       s.get("dreb") or 0,
        "PLUS_MINUS": s.get("plus_minus") or 0,
        "SEASON_ID":  str(game.get("season", "")),
        "POSTSEASON": game.get("postseason", False),
        "IS_HOME":    is_home,
        "TEAM_ABBR":  team.get("abbreviation", ""),
        "HOME_TEAM_ID": home_id,
        "VISITOR_TEAM_ID": game.get("visitor_team_id"),
        "P_R":        round(pts + reb, 1),
        "P_A":        round(pts + ast, 1),
        "P_R_A":      round(pts + reb + ast, 1),
        "A_R":        round(ast + reb, 1),
        "S_B":        round(stl + blk, 1),
    }

def get_gamelog(player_id: int, season_int: int, postseason: bool = False) -> list[dict]:
    """Get normalized game log for a player/season."""
    raw = get_player_stats(player_id, [season_int], postseason=postseason)
    rows = [normalize_stat_row(s) for s in raw]
    return sorted(rows, key=lambda x: x.get("GAME_DATE", ""))

# ── Injury report ──────────────────────────────────────────────────────────────
def get_injuries() -> list[dict]:
    try:
        data = bdl_get_all("/player_injuries")
        return [
            {
                "player_name": f"{i['player']['first_name']} {i['player']['last_name']}",
                "team": "",
                "status": i.get("status", ""),
                "reason": i.get("description", ""),
                "return_date": i.get("return_date", ""),
            }
            for i in data
        ]
    except Exception:
        return []

# ── Opponent defense ──────────────────────────────────────────────────────────
def get_opponent_defense(season_int: int) -> dict:
    """
    Calculate opponent defensive stats from team game stats.
    Returns {team_abbr: {opp_pts, opp_reb, opp_ast, ...}}
    """
    try:
        # Get all games this season
        url = f"{BDL_BASE}/games?seasons[]={season_int}&per_page=100"
        all_games = []
        cursor = None
        while True:
            r = requests.get(url + (f"&cursor={cursor}" if cursor else ""), headers=BDL_HEADERS, timeout=20)
            if not r.ok: break
            d = r.json()
            all_games.extend(d.get("data", []))
            cursor = d.get("meta", {}).get("next_cursor")
            if not cursor: break

        # Build points allowed per team
        defense = {}
        for g in all_games:
            if g.get("status") != "Final": continue
            h_abbr = g.get("home_team", {}).get("abbreviation", "")
            v_abbr = g.get("visitor_team", {}).get("abbreviation", "")
            h_pts = g.get("home_team_score") or 0
            v_pts = g.get("visitor_team_score") or 0

            if h_abbr not in defense:
                defense[h_abbr] = {"pts_allowed": [], "games": 0}
            if v_abbr not in defense:
                defense[v_abbr] = {"pts_allowed": [], "games": 0}

            defense[h_abbr]["pts_allowed"].append(v_pts)
            defense[h_abbr]["games"] += 1
            defense[v_abbr]["pts_allowed"].append(h_pts)
            defense[v_abbr]["games"] += 1

        result = {}
        for abbr, d in defense.items():
            if d["games"] > 0:
                result[abbr] = {
                    "opp_pts":  round(np.mean(d["pts_allowed"]), 1),
                    "opp_reb":  44,  # league avg fallback
                    "opp_ast":  25,
                    "opp_fg3m": 12,
                    "opp_stl":  7,
                    "opp_blk":  5,
                    "opp_pace": 100,
                }
        return result
    except Exception:
        return {}

# ── H2H stats ─────────────────────────────────────────────────────────────────
def get_h2h_stats(games: list[dict], opp_abbr: str) -> dict:
    h2h = [g for g in games if opp_abbr and (
        g.get("TEAM_ABBR") != opp_abbr and opp_abbr in str(g.get("HOME_TEAM_ID", "")) + str(g.get("VISITOR_TEAM_ID", ""))
    )]
    if not h2h:
        return {"games": 0}
    result = {"games": len(h2h)}
    for k in ["PTS", "REB", "AST", "STL", "BLK"]:
        vals = [g.get(k) or 0 for g in h2h]
        result[f"h2h_{k.lower()}"] = round(np.mean(vals), 1)
    return result

# ── Days rest ─────────────────────────────────────────────────────────────────
def days_rest(games: list[dict], idx: int) -> int:
    if idx == 0:
        return 3
    try:
        d1 = datetime.strptime(games[idx - 1]["GAME_DATE"], "%Y-%m-%d")
        d2 = datetime.strptime(games[idx]["GAME_DATE"], "%Y-%m-%d")
        return max((d2 - d1).days - 1, 0)
    except Exception:
        return 1

# ── Travel fatigue ────────────────────────────────────────────────────────────
def get_travel_fatigue(player_team_abbr: str, opp_abbr: str) -> float:
    TIMEZONES = {
        "BOS":1,"NYK":1,"BKN":1,"PHI":1,"TOR":1,"MIA":1,"ORL":1,"ATL":1,
        "CHA":1,"WAS":1,"CLE":1,"DET":1,"IND":1,"CHI":2,"MIL":2,"MIN":2,
        "MEM":2,"NOP":2,"SAS":2,"HOU":2,"DAL":2,"OKC":2,"DEN":3,"UTA":3,
        "POR":4,"GSW":4,"LAL":4,"LAC":4,"SAC":4,"PHX":3,
    }
    HIGH_ALTITUDE = {"DEN"}
    tz_diff = abs(TIMEZONES.get(player_team_abbr, 2) - TIMEZONES.get(opp_abbr, 2))
    fatigue = 2.0 if tz_diff >= 3 else 1.0 if tz_diff == 2 else 0.0
    if opp_abbr in HIGH_ALTITUDE:
        fatigue += 1.5
    return fatigue

# ── Playoff series ────────────────────────────────────────────────────────────
def get_playoff_series_games(playoff_games: list[dict], opponent_abbr: str) -> dict:
    if not playoff_games or not opponent_abbr:
        return {}
    series_games = [g for g in playoff_games if opponent_abbr in str(g.get("HOME_TEAM_ID","")) + str(g.get("VISITOR_TEAM_ID",""))]
    series_games = sorted(series_games, key=lambda x: x.get("GAME_DATE", ""))
    if not series_games:
        return {}
    wins = sum(1 for g in series_games if g.get("WL") == "W")
    losses = sum(1 for g in series_games if g.get("WL") == "L")
    keys = ["PTS","REB","AST","STL","BLK","FGM","FGA","FTM","FTA","FG3M","MIN"]
    series_avgs = {}
    for k in keys:
        vals = [g.get(k) or 0 for g in series_games]
        series_avgs[k] = round(np.mean(vals), 1) if vals else 0
    recent_series = series_games[-2:] if len(series_games) >= 2 else series_games
    series_trend = {}
    for k in ["PTS","REB","AST"]:
        series_trend[k] = round(np.mean([g.get(k) or 0 for g in recent_series]) - series_avgs.get(k, 0), 1)
    return {
        "games_played":   len(series_games),
        "game_number":    len(series_games) + 1,
        "wins":           wins,
        "losses":         losses,
        "is_must_win":    losses == 3 or wins == 3,
        "is_close_out":   wins == 3,
        "is_elimination": losses == 3,
        "series_avgs":    series_avgs,
        "series_trend":   series_trend,
        "series_games":   series_games,
    }

# ── Teammate usage bump ───────────────────────────────────────────────────────
def get_teammate_bump(player_name: str, team_id: int, injuries: list[dict], season_int: int) -> dict:
    try:
        # Get team roster stats — approximate from recent game stats
        return {"out_players": [], "estimated_pts_bump": 0, "estimated_ast_bump": 0, "estimated_reb_bump": 0}
    except Exception:
        return {"out_players": [], "estimated_pts_bump": 0, "estimated_ast_bump": 0, "estimated_reb_bump": 0}

# ── Opponent recent defense ───────────────────────────────────────────────────
def get_opponent_recent_defense(opp_team_id: int, season_int: int, last_n: int = 10) -> dict:
    try:
        url = f"{BDL_BASE}/games?team_ids[]={opp_team_id}&seasons[]={season_int}&per_page={last_n}"
        r = requests.get(url, headers=BDL_HEADERS, timeout=15)
        if not r.ok: return {}
        rows = r.json().get("data", [])
        final = [g for g in rows if g.get("status") == "Final"][:last_n]
        if not final: return {}
        pts_allowed = []
        plus_minus = []
        wins = 0
        for g in final:
            is_home = g.get("home_team", {}).get("id") == opp_team_id
            if is_home:
                pts_allowed.append(g.get("visitor_team_score") or 0)
                pm = (g.get("home_team_score") or 0) - (g.get("visitor_team_score") or 0)
                wins += 1 if g.get("home_team_score", 0) > g.get("visitor_team_score", 0) else 0
            else:
                pts_allowed.append(g.get("home_team_score") or 0)
                pm = (g.get("visitor_team_score") or 0) - (g.get("home_team_score") or 0)
                wins += 1 if g.get("visitor_team_score", 0) > g.get("home_team_score", 0) else 0
            plus_minus.append(pm)
        return {
            "recent_pts_allowed_avg": round(np.mean(pts_allowed), 1),
            "recent_wins": wins,
            "recent_losses": len(final) - wins,
            "recent_form_score": round(np.mean(plus_minus), 1),
        }
    except Exception:
        return {}

def get_opponent_rest(opp_team_id: int, season_int: int) -> int:
    try:
        url = f"{BDL_BASE}/games?team_ids[]={opp_team_id}&seasons[]={season_int}&per_page=5"
        r = requests.get(url, headers=BDL_HEADERS, timeout=10)
        if not r.ok: return 2
        rows = [g for g in r.json().get("data", []) if g.get("status") == "Final"]
        if len(rows) < 2: return 2
        d1 = datetime.strptime(rows[1]["date"], "%Y-%m-%d")
        d2 = datetime.strptime(rows[0]["date"], "%Y-%m-%d")
        return max((d2 - d1).days - 1, 0)
    except Exception:
        return 2

# ── Odds team map ─────────────────────────────────────────────────────────────
ODDS_TEAM_MAP = {
    "Atlanta Hawks":"ATL","Boston Celtics":"BOS","Brooklyn Nets":"BKN",
    "Charlotte Hornets":"CHA","Chicago Bulls":"CHI","Cleveland Cavaliers":"CLE",
    "Dallas Mavericks":"DAL","Denver Nuggets":"DEN","Detroit Pistons":"DET",
    "Golden State Warriors":"GSW","Houston Rockets":"HOU","Indiana Pacers":"IND",
    "LA Clippers":"LAC","Los Angeles Clippers":"LAC","Los Angeles Lakers":"LAL",
    "Memphis Grizzlies":"MEM","Miami Heat":"MIA","Milwaukee Bucks":"MIL",
    "Minnesota Timberwolves":"MIN","New Orleans Pelicans":"NOP","New York Knicks":"NYK",
    "Oklahoma City Thunder":"OKC","Orlando Magic":"ORL","Philadelphia 76ers":"PHI",
    "Phoenix Suns":"PHX","Portland Trail Blazers":"POR","Sacramento Kings":"SAC",
    "San Antonio Spurs":"SAS","Toronto Raptors":"TOR","Utah Jazz":"UTA",
    "Washington Wizards":"WAS",
}

# BDL team abbr → team id cache
_team_id_cache: dict = {}

def get_team_id_by_abbr(abbr: str) -> int | None:
    global _team_id_cache
    if not _team_id_cache:
        data = bdl_get_all("/teams")
        _team_id_cache = {t["abbreviation"]: t["id"] for t in data}
    return _team_id_cache.get(abbr)

# ── Build features ────────────────────────────────────────────────────────────
def build_features(
    recent: list[dict],
    all_games: list[dict],
    opp_defense: dict,
    h2h: dict,
    is_home: bool = False,
    rest: int = 1,
    pts_bump: float = 0,
    ast_bump: float = 0,
    reb_bump: float = 0,
    ref_fta_rate: float = 0,
    shot_profile_score: float = 0,
    expected_min: float = 32.0,
    is_playoffs: bool = False,
    usg_trend: float = 0,
    opp_rest: int = 2,
    opp_recent_form: float = 0,
    travel_fatigue: float = 0,
    clutch_usg: float = 0,
    series_games_played: int = 0,
    series_pts_avg: float = 0,
    series_reb_avg: float = 0,
    series_ast_avg: float = 0,
    series_weight: float = 0,
    is_elimination: bool = False,
    is_close_out: bool = False,
) -> list[float]:
    keys = ["PTS","REB","AST","STL","BLK","FGM","FGA","FTM","FTA","FG3M"]
    feat = []

    def parse_min(m) -> float:
        if not m: return 0.0
        try:
            if isinstance(m, str) and ":" in m:
                p = m.split(":")
                return float(p[0]) + float(p[1]) / 60
            return float(m)
        except: return 0.0

    recent_mins = [parse_min(g.get("MIN")) for g in recent]
    all_mins    = [parse_min(g.get("MIN")) for g in all_games]
    avg_recent_min = np.mean(recent_mins) if recent_mins else 32.0

    # 1. L10 raw mean + std (20)
    for k in keys:
        vals = [g.get(k) or 0 for g in recent]
        feat.append(np.mean(vals)); feat.append(np.std(vals))

    # 2. L10 per-36 rates (10)
    for k in keys:
        pr36 = [(g.get(k) or 0) / m * 36 for g, m in zip(recent, recent_mins) if m > 5]
        feat.append(np.mean(pr36) if pr36 else 0)

    # 3. Minutes-weighted L10 (10)
    for k in keys:
        tw = sum(recent_mins)
        feat.append(sum((g.get(k) or 0) * m for g, m in zip(recent, recent_mins)) / tw if tw > 0 else 0)

    # 4. L5 mean (10)
    last5 = recent[-5:] if len(recent) >= 5 else recent
    for k in keys:
        feat.append(np.mean([g.get(k) or 0 for g in last5]))

    # 5. Season mean (10)
    for k in keys:
        vals = [g.get(k) or 0 for g in all_games]
        feat.append(np.mean(vals) if vals else 0)

    # 6. Season per-36 (10)
    for k in keys:
        pr36 = [(g.get(k) or 0) / m * 36 for g, m in zip(all_games, all_mins) if m > 5]
        feat.append(np.mean(pr36) if pr36 else 0)

    # 7. Home/away split (10)
    home_g = [g for g in all_games if g.get("IS_HOME")]
    away_g = [g for g in all_games if not g.get("IS_HOME")]
    split = home_g if is_home else away_g
    for k in keys:
        vals = [g.get(k) or 0 for g in split]
        feat.append(np.mean(vals) if vals else 0)

    # 8. Opponent defense (7)
    feat += [
        opp_defense.get("opp_pts", 110), opp_defense.get("opp_reb", 44),
        opp_defense.get("opp_ast", 25),  opp_defense.get("opp_fg3m", 12),
        opp_defense.get("opp_stl", 7),   opp_defense.get("opp_blk", 5),
        opp_defense.get("opp_pace", 100),
    ]

    # 9. H2H (5)
    for k in ["pts","reb","ast","stl","blk"]:
        feat.append(h2h.get(f"h2h_{k}", 0))

    # 10. Game context (4)
    feat += [float(is_home), float(min(rest, 5)), float(expected_min), float(is_playoffs)]

    # 11. Minutes context (2)
    feat += [float(avg_recent_min), float(expected_min - avg_recent_min)]

    # 12. Teammate bump (3)
    feat += [float(pts_bump), float(ast_bump), float(reb_bump)]

    # 13. Ref foul rate (1)
    feat.append(float(ref_fta_rate))

    # 14. Shot profile (1)
    feat.append(float(shot_profile_score))

    # 15. Usage trend (2)
    feat += [float(usg_trend), float(clutch_usg)]

    # 16. Opponent context (3)
    feat += [float(min(opp_rest, 5)), float(opp_recent_form), float(travel_fatigue)]

    # 17. Playoff series (7)
    feat += [
        float(series_games_played), float(series_pts_avg), float(series_reb_avg),
        float(series_ast_avg), float(series_weight), float(is_elimination), float(is_close_out),
    ]

    return feat

# ── Train + predict ───────────────────────────────────────────────────────────
def train_predict(
    games: list[dict], target: str, opp_defense: dict, h2h: dict,
    is_home: bool = False, rest: int = 1,
    pts_bump: float = 0, ast_bump: float = 0, reb_bump: float = 0,
    ref_fta_rate: float = 0, shot_profile_score: float = 0,
    expected_min: float = 32.0, is_playoffs: bool = False,
    usg_trend: float = 0, opp_rest: int = 2, opp_recent_form: float = 0,
    travel_fatigue: float = 0, clutch_usg: float = 0,
    series_context: dict = None,
) -> dict:
    if series_context is None:
        series_context = {}

    sc = series_context
    series_games_played = sc.get("games_played", 0)
    series_avgs = sc.get("series_avgs", {})
    series_weight = min(series_games_played / 5.0, 0.8) if series_games_played >= 2 else 0

    games = sorted(games, key=lambda x: x.get("GAME_DATE", ""))

    augmented = list(games)
    if series_weight > 0 and sc.get("series_games"):
        repeats = max(1, int(series_games_played * 2))
        for _ in range(repeats):
            augmented.extend(sc["series_games"])
        augmented = sorted(augmented, key=lambda x: x.get("GAME_DATE", ""))

    if len(games) < 10:
        return {"error": "Not enough games to predict (need at least 10)"}

    window = min(10, max(5, len(augmented) // 3))
    X, y = [], []

    for i in range(window, len(augmented)):
        recent = augmented[i - window:i]
        r = days_rest(augmented, i)
        home = augmented[i].get("IS_HOME", False)
        feat = build_features(recent, augmented[:i], {}, {}, home, r,
                              expected_min=32.0, is_playoffs=False)
        X.append(feat); y.append(augmented[i].get(target) or 0)

    if len(X) < 5:
        return {"error": "Not enough games to predict (need at least 10)"}

    X, y = np.array(X), np.array(y)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, subsample=0.8, random_state=42)
    model.fit(Xs, y)
    rf = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
    rf.fit(Xs, y)

    recent = games[-window:]
    feat = build_features(
        recent, games, opp_defense, h2h, is_home, rest,
        pts_bump, ast_bump, reb_bump, ref_fta_rate, shot_profile_score,
        expected_min, is_playoffs, usg_trend, opp_rest, opp_recent_form,
        travel_fatigue, clutch_usg,
        series_games_played, series_avgs.get("PTS", 0), series_avgs.get("REB", 0),
        series_avgs.get("AST", 0), series_weight,
        sc.get("is_elimination", False), sc.get("is_close_out", False),
    )
    feat_scaled = scaler.transform([feat])
    pred = model.predict(feat_scaled)[0]
    std  = np.std([t.predict(feat_scaled)[0] for t in rf.estimators_])

    recent_avg = np.mean([g.get(target) or 0 for g in recent])
    season_avg = np.mean([g.get(target) or 0 for g in games])

    home_g = [g for g in games if g.get("IS_HOME")]
    away_g = [g for g in games if not g.get("IS_HOME")]
    split  = home_g if is_home else away_g
    split_avg = round(np.mean([g.get(target) or 0 for g in split]), 1) if split else None

    return {
        "prediction": round(float(pred), 1),
        "std_dev": round(float(std), 2),
        "recent_avg_10": round(float(recent_avg), 1),
        "season_avg": round(float(season_avg), 1),
        "home_away_avg": split_avg,
        "games_used": len(augmented),
        "series_weight_applied": round(series_weight, 2),
    }

# ── Fast predict (for best picks bulk scan) ───────────────────────────────────
def fast_predict(all_games, stat, opp_defense, h2h, is_home, rest,
                 expected_min, is_playoffs, opp_rest, travel_fatigue, series_context):
    if len(all_games) < 10:
        return None

    def parse_min(m):
        if not m: return 0.0
        try:
            if isinstance(m, str) and ":" in m:
                p = m.split(":"); return float(p[0]) + float(p[1]) / 60
            return float(m)
        except: return 0.0

    games = sorted(all_games, key=lambda x: x.get("GAME_DATE", ""))
    window = min(10, max(5, len(games) // 3))
    keys = ["PTS","REB","AST","STL","BLK","FGM","FGA","FTM","FTA","FG3M"]
    X, y = [], []

    for i in range(window, len(games)):
        recent = games[i - window:i]
        recent_mins = [parse_min(g.get("MIN")) for g in recent]
        feat = []
        for k in keys:
            vals = [g.get(k) or 0 for g in recent]
            feat.append(np.mean(vals)); feat.append(np.std(vals))
        for k in keys:
            pr36 = [(g.get(k) or 0) / m * 36 for g, m in zip(recent, recent_mins) if m > 5]
            feat.append(np.mean(pr36) if pr36 else 0)
        feat += [opp_defense.get("opp_pts", 110), opp_defense.get("opp_reb", 44),
                 opp_defense.get("opp_ast", 25), opp_defense.get("opp_fg3m", 12),
                 opp_defense.get("opp_pace", 100)]
        for k in ["pts","reb","ast","stl","blk"]:
            feat.append(h2h.get(f"h2h_{k}", 0))
        feat += [float(is_home), float(min(rest, 5)), float(expected_min),
                 float(is_playoffs), float(min(opp_rest, 5)), float(travel_fatigue)]
        sc = series_context or {}
        feat += [float(sc.get("games_played", 0)),
                 float(sc.get("series_avgs", {}).get(stat, 0)),
                 float(sc.get("is_elimination", False))]
        X.append(feat); y.append(games[i].get(stat) or 0)

    if len(X) < 5: return None
    try:
        X, y = np.array(X), np.array(y)
        scaler = StandardScaler(); Xs = scaler.fit_transform(X)
        model = RandomForestRegressor(n_estimators=80, random_state=42, n_jobs=-1)
        model.fit(Xs, y)

        recent = games[-window:]
        recent_mins = [parse_min(g.get("MIN")) for g in recent]
        feat = []
        for k in keys:
            vals = [g.get(k) or 0 for g in recent]
            feat.append(np.mean(vals)); feat.append(np.std(vals))
        for k in keys:
            pr36 = [(g.get(k) or 0) / m * 36 for g, m in zip(recent, recent_mins) if m > 5]
            feat.append(np.mean(pr36) if pr36 else 0)
        feat += [opp_defense.get("opp_pts", 110), opp_defense.get("opp_reb", 44),
                 opp_defense.get("opp_ast", 25), opp_defense.get("opp_fg3m", 12),
                 opp_defense.get("opp_pace", 100)]
        for k in ["pts","reb","ast","stl","blk"]:
            feat.append(h2h.get(f"h2h_{k}", 0))
        feat += [float(is_home), float(min(rest, 5)), float(expected_min),
                 float(is_playoffs), float(min(opp_rest, 5)), float(travel_fatigue)]
        sc = series_context or {}
        feat += [float(sc.get("games_played", 0)),
                 float(sc.get("series_avgs", {}).get(stat, 0)),
                 float(sc.get("is_elimination", False))]

        pred = model.predict(scaler.transform([feat]))[0]
        recent_avg = np.mean([g.get(stat) or 0 for g in recent])
        season_avg = np.mean([g.get(stat) or 0 for g in games])
        return {"prediction": round(float(pred), 1),
                "recent_avg": round(float(recent_avg), 1),
                "season_avg": round(float(season_avg), 1)}
    except Exception:
        return None

# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {"message": "NbaProp-ML API 🏀", "version": "5.0.0"}

# ── Player search ─────────────────────────────────────────────────────────────
@app.get("/player/search")
def search_player(name: str = Query(...)):
    # Try active players first
    data = bdl_get("/players/active", {"search": name, "per_page": 10})
    players = data.get("data", [])
    if not players:
        data = bdl_get("/players", {"search": name, "per_page": 10})
        players = data.get("data", [])
    if not players:
        raise HTTPException(status_code=404, detail="No players found.")
    return {"players": [format_player(p) for p in players]}

# ── Player info ───────────────────────────────────────────────────────────────
@app.get("/player/{player_name}/info")
def player_info(player_name: str):
    # URL decode the name
    from urllib.parse import unquote
    player_name = unquote(player_name)
    p = find_player(player_name)
    season_int = current_season_int()
    try:
        games = get_gamelog(p["id"], season_int)
        if not games:
            games = get_gamelog(p["id"], season_int - 1)
        fp = format_player(p)
        if games:
            latest = games[-1]
            avg_pts = round(np.mean([g.get("PTS", 0) for g in games]), 1)
            avg_reb = round(np.mean([g.get("REB", 0) for g in games]), 1)
            avg_ast = round(np.mean([g.get("AST", 0) for g in games]), 1)
            avg_stl = round(np.mean([g.get("STL", 0) for g in games]), 1)
            avg_blk = round(np.mean([g.get("BLK", 0) for g in games]), 1)
            avg_min = round(np.mean([g.get("MIN", 0) for g in games]), 1)
            avg_fgm = round(np.mean([g.get("FGM", 0) for g in games]), 1)
            avg_fga = round(np.mean([g.get("FGA", 0) for g in games]), 1)
            avg_fg3m = round(np.mean([g.get("FG3M", 0) for g in games]), 1)
            avg_fg3a = round(np.mean([g.get("FG3A", 0) for g in games]), 1)
            avg_ftm = round(np.mean([g.get("FTM", 0) for g in games]), 1)
            avg_fta = round(np.mean([g.get("FTA", 0) for g in games]), 1)
            avg_tov = round(np.mean([g.get("TOV", 0) for g in games]), 1)
            fg_pct  = round(avg_fgm / avg_fga, 3) if avg_fga > 0 else 0
            fg3_pct = round(avg_fg3m / avg_fg3a, 3) if avg_fg3a > 0 else 0
            ft_pct  = round(avg_ftm / avg_fta, 3) if avg_fta > 0 else 0
            stats = {
                "GP": len(games), "MIN": avg_min,
                "PTS": avg_pts, "REB": avg_reb, "AST": avg_ast,
                "STL": avg_stl, "BLK": avg_blk, "TOV": avg_tov,
                "FGM": avg_fgm, "FGA": avg_fga, "FG_PCT": fg_pct,
                "FG3M": avg_fg3m, "FG3A": avg_fg3a, "FG3_PCT": fg3_pct,
                "FTM": avg_ftm, "FTA": avg_fta, "FT_PCT": ft_pct,
                "P_R": round(avg_pts + avg_reb, 1),
                "P_A": round(avg_pts + avg_ast, 1),
                "P_R_A": round(avg_pts + avg_reb + avg_ast, 1),
                "TEAM_ID": p.get("team", {}).get("id") if isinstance(p.get("team"), dict) else None,
            }
        else:
            stats = {}
    except Exception:
        stats = {}
        fp = format_player(p)

    return {
        "common_player_info": {
            "DISPLAY_FIRST_LAST": fp["full_name"],
            "TEAM_NAME": fp["team"],
            "POSITION": fp["position"],
            "HEIGHT": fp["height"],
            "JERSEY": fp["jersey_number"],
            "WEIGHT": "", "COUNTRY": "",
        },
        "player_headline_stats": stats,
    }

# ── Career stats ──────────────────────────────────────────────────────────────
@app.get("/player/{player_name}/career")
def player_career(player_name: str):
    from urllib.parse import unquote
    player_name = unquote(player_name)
    p = find_player(player_name)
    season_int = current_season_int()
    seasons_to_fetch = list(range(max(2015, season_int - 9), season_int + 1))
    career_rows = []
    for s in seasons_to_fetch:
        try:
            games = get_gamelog(p["id"], s)
            if not games: continue
            gp = len(games)
            def avg(k): return round(np.mean([g.get(k, 0) for g in games]), 1)
            def pct(m, a): mv = avg(m); av = avg(a); return round(mv/av, 3) if av > 0 else 0
            career_rows.append({
                "SEASON_ID": f"{s}-{str(s+1)[2:]}",
                "PLAYER_ID": p["id"],
                "TEAM_ABBREVIATION": games[-1].get("TEAM_ABBR", ""),
                "GP": gp, "MIN": avg("MIN"),
                "PTS": avg("PTS"), "REB": avg("REB"), "AST": avg("AST"),
                "STL": avg("STL"), "BLK": avg("BLK"), "TOV": avg("TOV"),
                "FGM": avg("FGM"), "FGA": avg("FGA"), "FG_PCT": pct("FGM","FGA"),
                "FG3M": avg("FG3M"), "FG3A": avg("FG3A"), "FG3_PCT": pct("FG3M","FG3A"),
                "FTM": avg("FTM"), "FTA": avg("FTA"), "FT_PCT": pct("FTM","FTA"),
                "P_R": round(avg("PTS")+avg("REB"),1),
                "P_A": round(avg("PTS")+avg("AST"),1),
                "P_R_A": round(avg("PTS")+avg("REB")+avg("AST"),1),
            })
        except Exception:
            continue

    if not career_rows:
        raise HTTPException(status_code=404, detail="No career data found.")

    # Overall career totals
    all_rows = career_rows
    career_total = {
        "PLAYER_ID": p["id"], "GP": sum(r["GP"] for r in all_rows),
        "PTS": round(np.mean([r["PTS"] for r in all_rows]),1),
        "REB": round(np.mean([r["REB"] for r in all_rows]),1),
        "AST": round(np.mean([r["AST"] for r in all_rows]),1),
    }
    return {
        "season_totals_regular_season": career_rows,
        "career_totals_regular_season": [career_total],
    }

# ── Game log ──────────────────────────────────────────────────────────────────
@app.get("/player/{player_name}/gamelog")
def player_gamelog(player_name: str, season: str = Query(None)):
    from urllib.parse import unquote
    player_name = unquote(player_name)
    if not season:
        season = current_season_str()
    season_int = season_str_to_int(season)
    p = find_player(player_name)
    games = get_gamelog(p["id"], season_int)
    if not games:
        raise HTTPException(status_code=404, detail=f"No games found for {season}.")
    return {"game_log": list(reversed(games))}  # most recent first

# ── League leaders (calculated from stats) ────────────────────────────────────
@app.get("/league/leaders")
def league_leaders(
    season: str = Query(None),
    stat_category: str = Query("PTS"),
    top: int = Query(15, ge=1, le=50),
):
    if not season:
        season = current_season_str()
    season_int = season_str_to_int(season)

    STAT_MAP = {
        "PTS": "pts", "REB": "reb", "AST": "ast", "STL": "stl", "BLK": "blk",
        "FG_PCT": "fg_pct", "FG3_PCT": "fg3_pct", "FT_PCT": "ft_pct",
    }

    bdl_stat = STAT_MAP.get(stat_category, "pts")

    try:
        # Use BDL leaders endpoint (GOAT tier) — fallback to calculating
        data = bdl_get("/leaders", {"season": season_int, "stat_type": bdl_stat})
        raw = data.get("data", [])
        leaders = []
        for i, row in enumerate(raw[:top]):
            player = row.get("player", {})
            team = player.get("team") or {}
            leaders.append({
                "PLAYER_ID": player.get("id"),
                "RANK": i + 1,
                "PLAYER": f"{player.get('first_name','')} {player.get('last_name','')}",
                "TEAM": team.get("abbreviation", "") if isinstance(team, dict) else "",
                "GP": row.get("games_played", 0),
                stat_category: row.get("value", 0),
                "PTS": row.get("value", 0) if stat_category == "PTS" else 0,
                "REB": row.get("value", 0) if stat_category == "REB" else 0,
                "AST": row.get("value", 0) if stat_category == "AST" else 0,
            })
        return {"leaders": leaders}
    except Exception:
        # Fallback: calculate from game stats
        raise HTTPException(status_code=503, detail="Leaders endpoint requires GOAT tier. Please upgrade BallDontLie plan.")

# ── Odds status ───────────────────────────────────────────────────────────────
@app.get("/odds/status")
def odds_status():
    try:
        r = requests.get(
            f"{ODDS_BASE}/sports/basketball_nba/events",
            params={"apiKey": ODDS_KEY}, timeout=10,
        )
        return {
            "status": r.status_code,
            "requests_remaining": r.headers.get("x-requests-remaining", "unknown"),
            "requests_used": r.headers.get("x-requests-used", "unknown"),
            "events_found": len(r.json()) if r.ok else 0,
        }
    except Exception as e:
        return {"error": str(e)}

# ── Predict ───────────────────────────────────────────────────────────────────
@app.get("/predict/{player_name}")
def predict_player(player_name: str, season: str = Query(None)):
    from urllib.parse import unquote
    player_name = unquote(player_name)
    if not season:
        season = current_season_str()
    season_int = season_str_to_int(season)

    # 1. Find player
    p = find_player(player_name)
    pid = p["id"]
    fp = format_player(p)
    player_team_abbr = fp.get("team_abbr", "")

    # 2. Fetch game logs (current + 2 previous seasons)
    all_games = []
    for offset in range(3):
        s = season_int - offset
        try:
            games = get_gamelog(pid, s)
            all_games = games + all_games
        except Exception:
            pass

    if not all_games:
        raise HTTPException(status_code=500, detail=f"Could not fetch game log for {player_name}.")

    # 3. Fetch today's props from Odds API
    props_found = {}
    opponent_abbr = None
    opponent_name = None
    today_game = None
    is_home = False

    try:
        player_last  = fp["last_name"].lower()
        player_first = fp["first_name"].lower()
        market_keys  = "player_points,player_rebounds,player_assists,player_steals,player_blocks"
        stat_map = {
            "player_points": "PTS", "player_rebounds": "REB",
            "player_assists": "AST", "player_steals": "STL", "player_blocks": "BLK",
        }
        events_r = requests.get(
            f"{ODDS_BASE}/sports/basketball_nba/events",
            params={"apiKey": ODDS_KEY}, timeout=10,
        )
        events = events_r.json() if events_r.ok else []

        for event in (events if isinstance(events, list) else []):
            event_id = event.get("id")
            home = event.get("home_team", "")
            away = event.get("away_team", "")
            if not event_id: continue
            try:
                odds_r = requests.get(
                    f"{ODDS_BASE}/sports/basketball_nba/events/{event_id}/odds",
                    params={"apiKey": ODDS_KEY, "regions": "us", "markets": market_keys,
                            "oddsFormat": "american", "bookmakers": "draftkings,fanduel,betmgm"},
                    timeout=10,
                )
                if not odds_r.ok: continue
                odds = odds_r.json()
                player_found = False
                for bm in odds.get("bookmakers", []):
                    for mkt in bm.get("markets", []):
                        stat = stat_map.get(mkt.get("key", ""))
                        if not stat: continue
                        for outcome in mkt.get("outcomes", []):
                            desc = outcome.get("description", "").lower()
                            if player_last in desc or player_first in desc:
                                player_found = True
                                if stat not in props_found and outcome.get("name") == "Over":
                                    props_found[stat] = outcome.get("point")
                if player_found and not opponent_abbr:
                    today_game = {"home": home, "away": away}
                    home_abbr = ODDS_TEAM_MAP.get(home, "")
                    away_abbr = ODDS_TEAM_MAP.get(away, "")
                    if player_team_abbr == home_abbr:
                        opponent_abbr = away_abbr; opponent_name = away; is_home = True
                    else:
                        opponent_abbr = home_abbr; opponent_name = home; is_home = False
            except Exception:
                continue
    except Exception:
        pass

    # 4. Opponent defense + H2H
    sorted_games = sorted(all_games, key=lambda x: x.get("GAME_DATE", ""))
    opp_defense = {}
    h2h = {}
    series_context = {}
    opp_team_id = None

    if opponent_abbr:
        try:
            all_defense = get_opponent_defense(season_int)
            opp_defense = all_defense.get(opponent_abbr, {})
        except Exception:
            pass
        h2h = get_h2h_stats(all_games, opponent_abbr)
        opp_team_id = get_team_id_by_abbr(opponent_abbr)

        # Playoff series
        try:
            playoff_games = get_gamelog(pid, season_int, postseason=True)
            if playoff_games:
                series_context = get_playoff_series_games(playoff_games, opponent_abbr)
        except Exception:
            pass

    # 5. Context
    rest = days_rest(sorted_games, len(sorted_games) - 1) if len(sorted_games) > 1 else 1

    def parse_min(m):
        if not m: return 0.0
        try:
            if isinstance(m, str) and ":" in m:
                p2 = m.split(":"); return float(p2[0]) + float(p2[1]) / 60
            return float(m)
        except: return 0.0

    recent_10 = sorted_games[-10:]
    recent_mins = [parse_min(g.get("MIN")) for g in recent_10 if parse_min(g.get("MIN")) > 5]
    expected_min = round(np.mean(recent_mins), 1) if recent_mins else 32.0
    is_playoffs = any(g.get("POSTSEASON") for g in sorted_games[-5:])

    # 6. Injuries
    injuries = get_injuries()
    relevant_injuries = [i for i in injuries if
        player_team_abbr.lower() in i.get("team", "").lower() or
        (opponent_name or "").lower() in i.get("team", "").lower()]

    # 7. Opponent context
    opp_rest = 2
    opp_recent_defense = {}
    opp_recent_form = 0
    travel_fatigue = 0.0
    if opp_team_id:
        try:
            opp_rest = get_opponent_rest(opp_team_id, season_int)
            opp_recent_defense = get_opponent_recent_defense(opp_team_id, season_int)
            opp_recent_form = opp_recent_defense.get("recent_form_score", 0)
        except Exception:
            pass
    if player_team_abbr and opponent_abbr:
        travel_fatigue = get_travel_fatigue(player_team_abbr, opponent_abbr)

    # 8. Train & predict
    targets = ["PTS", "REB", "AST", "STL", "BLK"]
    predictions = {}
    for stat in targets:
        ml = train_predict(
            all_games, stat, opp_defense, h2h, is_home, rest,
            0, 0, 0, 0, 0, expected_min, is_playoffs,
            0, opp_rest, opp_recent_form, travel_fatigue, 0,
            series_context,
        )
        line = props_found.get(stat)
        entry = {
            "stat": stat, "line": line,
            "ml_prediction": ml.get("prediction"),
            "std_dev": ml.get("std_dev"),
            "recent_avg_10": ml.get("recent_avg_10"),
            "season_avg": ml.get("season_avg"),
            "home_away_avg": ml.get("home_away_avg"),
            "games_used": ml.get("games_used"),
            "series_weight_applied": ml.get("series_weight_applied", 0),
            "error": ml.get("error"),
        }
        if line is not None and ml.get("prediction") is not None:
            diff = ml["prediction"] - line
            entry["recommendation"] = "OVER" if diff > 0 else "UNDER"
            entry["edge"] = round(abs(diff), 1)
        else:
            entry["recommendation"] = "NO LINE"
            entry["edge"] = None
        predictions[stat] = entry

    # 9. Combos
    combos = {}
    for combo_name, stats in [("PR",["PTS","REB"]),("PA",["PTS","AST"]),("PRA",["PTS","REB","AST"]),("RA",["REB","AST"])]:
        total_pred = sum(predictions[s]["ml_prediction"] or 0 for s in stats)
        total_line = round(sum(props_found[s] for s in stats), 1) if all(props_found.get(s) for s in stats) else None
        entry = {"stat": combo_name, "ml_prediction": round(total_pred, 1), "line": total_line}
        if total_line:
            diff = total_pred - total_line
            entry["recommendation"] = "OVER" if diff > 0 else "UNDER"
            entry["edge"] = round(abs(diff), 1)
        else:
            entry["recommendation"] = "NO LINE"
        combos[combo_name] = entry

    return {
        "player": fp,
        "season": season,
        "opponent": opponent_name,
        "opponent_abbr": opponent_abbr,
        "today_game": today_game,
        "is_home": is_home,
        "rest_days": rest,
        "opp_rest_days": opp_rest,
        "expected_min": expected_min,
        "is_playoffs": is_playoffs,
        "travel_fatigue": travel_fatigue,
        "usage_data": {"usg_trend": 0, "trending_up": False},
        "clutch_data": {},
        "opponent_defense": opp_defense,
        "opp_recent_defense": opp_recent_defense,
        "h2h": h2h,
        "series_context": {k: v for k, v in series_context.items() if k != "series_games"},
        "shot_profile_edge": {"score": 0, "interpretation": "N/A", "zones": {}},
        "teammate_context": {"out_players": []},
        "relevant_injuries": relevant_injuries[:10],
        "lineup_news": [],
        "predictions": predictions,
        "combos": combos,
        "props_found": len(props_found),
        "note": "GradientBoosting · BallDontLie API · Per-36 rates · Opp defense · H2H · Injuries · Playoff series weighting",
    }

# ── Best picks ────────────────────────────────────────────────────────────────
_picks_cache: dict = {}

@app.get("/picks/today")
def best_picks_today(
    season: str = Query(None),
    top: int = Query(10, ge=1, le=30),
    min_edge: float = Query(0.5),
    force_refresh: bool = Query(False),
):
    if not season:
        season = current_season_str()
    season_int = season_str_to_int(season)

    cache_key = f"picks_{season}_{datetime.now().strftime('%Y%m%d')}"
    if not force_refresh and cache_key in _picks_cache:
        cached = _picks_cache[cache_key]
        filtered = [p for p in cached["all_picks"] if p["edge"] >= min_edge]
        filtered.sort(key=lambda x: x["edge"], reverse=True)
        return {**cached, "picks": filtered[:top], "min_edge": min_edge, "cached": True}

    # Fetch all props
    all_props = {}
    player_event_map = {}
    try:
        events_r = requests.get(
            f"{ODDS_BASE}/sports/basketball_nba/events",
            params={"apiKey": ODDS_KEY}, timeout=10,
        )
        if not events_r.ok:
            raise HTTPException(status_code=500, detail="Could not fetch events")
        events = events_r.json()
        market_keys = "player_points,player_rebounds,player_assists,player_steals,player_blocks"
        stat_map = {"player_points":"PTS","player_rebounds":"REB","player_assists":"AST","player_steals":"STL","player_blocks":"BLK"}

        for event in (events if isinstance(events, list) else []):
            event_id = event.get("id")
            home = event.get("home_team", "")
            away = event.get("away_team", "")
            if not event_id: continue
            try:
                odds_r = requests.get(
                    f"{ODDS_BASE}/sports/basketball_nba/events/{event_id}/odds",
                    params={"apiKey": ODDS_KEY, "regions": "us", "markets": market_keys,
                            "oddsFormat": "american", "bookmakers": "draftkings,fanduel,betmgm"},
                    timeout=10,
                )
                if not odds_r.ok: continue
                for bm in odds_r.json().get("bookmakers", []):
                    for mkt in bm.get("markets", []):
                        stat = stat_map.get(mkt.get("key",""))
                        if not stat: continue
                        for outcome in mkt.get("outcomes", []):
                            if outcome.get("name") != "Over": continue
                            pname = outcome.get("description","")
                            if not pname: continue
                            if pname not in all_props:
                                all_props[pname] = {}
                                player_event_map[pname] = {"home": home, "away": away}
                            if stat not in all_props[pname]:
                                all_props[pname][stat] = outcome.get("point")
            except Exception:
                continue
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not fetch today's props: {e}")

    if not all_props:
        return {"picks": [], "total_players": 0, "message": "No props found for today"}

    # Pre-fetch defense
    all_defense = {}
    try:
        all_defense = get_opponent_defense(season_int)
    except Exception:
        pass

    picks = []
    gamelog_cache = {}

    for player_name, props in all_props.items():
        if not props: continue
        try:
            p = find_player(player_name)
            pid = p["id"]
            fp = format_player(p)
            player_team_abbr = fp.get("team_abbr", "")

            if pid not in gamelog_cache:
                games = []
                for offset in range(2):
                    try:
                        g = get_gamelog(pid, season_int - offset)
                        games = g + games
                    except Exception:
                        pass
                gamelog_cache[pid] = games
            all_games = gamelog_cache[pid]
            if len(all_games) < 10: continue

            sorted_games = sorted(all_games, key=lambda x: x.get("GAME_DATE",""))
            event_info = player_event_map.get(player_name, {})
            home_team = event_info.get("home","")
            away_team = event_info.get("away","")
            home_abbr = ODDS_TEAM_MAP.get(home_team,"")
            away_abbr = ODDS_TEAM_MAP.get(away_team,"")
            is_home = player_team_abbr == home_abbr
            opponent_abbr = away_abbr if is_home else home_abbr

            opp_defense = all_defense.get(opponent_abbr, {})
            h2h = get_h2h_stats(all_games, opponent_abbr) if opponent_abbr else {}

            def parse_min(m):
                if not m: return 0.0
                try:
                    if isinstance(m, str) and ":" in m:
                        pp = m.split(":"); return float(pp[0]) + float(pp[1])/60
                    return float(m)
                except: return 0.0

            recent_mins = [parse_min(g.get("MIN")) for g in sorted_games[-10:] if parse_min(g.get("MIN")) > 5]
            expected_min = round(np.mean(recent_mins), 1) if recent_mins else 32.0
            is_playoffs = any(g.get("POSTSEASON") for g in sorted_games[-5:])
            rest = days_rest(sorted_games, len(sorted_games)-1) if len(sorted_games) > 1 else 1
            travel_fatigue = get_travel_fatigue(player_team_abbr, opponent_abbr) if player_team_abbr and opponent_abbr else 0

            series_context = {}
            if is_playoffs and opponent_abbr:
                try:
                    poff = get_gamelog(pid, season_int, postseason=True)
                    series_context = get_playoff_series_games(poff, opponent_abbr)
                except Exception:
                    pass

            for stat, line in props.items():
                if line is None: continue
                try:
                    result = fast_predict(all_games, stat, opp_defense, h2h,
                                          is_home, rest, expected_min, is_playoffs,
                                          2, travel_fatigue, series_context)
                    if not result: continue
                    pred = result["prediction"]
                    diff = pred - line
                    edge = round(abs(diff), 1)
                    if edge < 0.5: continue
                    picks.append({
                        "player": fp["full_name"],
                        "stat": stat, "line": line, "prediction": pred,
                        "recommendation": "OVER" if diff > 0 else "UNDER",
                        "edge": edge,
                        "recent_avg": result["recent_avg"],
                        "season_avg": result["season_avg"],
                        "matchup": f"{away_team} @ {home_team}",
                        "is_playoffs": is_playoffs,
                        "series_game": series_context.get("game_number"),
                        "is_elimination": series_context.get("is_elimination", False),
                    })
                except Exception:
                    continue
        except Exception:
            continue

    picks.sort(key=lambda x: x["edge"], reverse=True)
    _picks_cache[cache_key] = {
        "all_picks": picks,
        "total_players": len(all_props),
        "total_picks_found": len([p for p in picks if p["edge"] >= min_edge]),
        "season": season, "cached": False,
    }
    filtered = [p for p in picks if p["edge"] >= min_edge]
    return {
        "picks": filtered[:top],
        "total_players": len(all_props),
        "total_picks_found": len(filtered),
        "season": season, "min_edge": min_edge, "cached": False,
    }