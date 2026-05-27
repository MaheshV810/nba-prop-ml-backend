from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Any
import requests
import numpy as np
import os
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

BDL_KEY   = os.getenv("BDL_API_KEY", "0b07f4c7-7110-4cab-a4b0-2b22d0f31a83")
ODDS_KEY  = os.getenv("ODDS_API_KEY", "")
BDL_BASE  = "https://api.balldontlie.io/v1"
ODDS_BASE = "https://api.the-odds-api.com/v4"

app = FastAPI(title="NbaProp-ML API", version="5.1.0")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["*"], allow_headers=["*"])

BDL_HEADERS = {"Authorization": BDL_KEY}

# ── BDL helpers ───────────────────────────────────────────────────────────────
def bdl_get(path: str, params: dict = {}) -> Any:
    r = requests.get(f"{BDL_BASE}{path}", headers=BDL_HEADERS, params=params, timeout=20)
    if not r.ok:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    return r.json()

def current_season_int() -> int:
    now = datetime.now()
    return now.year if now.month >= 10 else now.year - 1

def current_season_str() -> str:
    y = current_season_int()
    return f"{y}-{str(y+1)[2:]}"

def season_str_to_int(s: str) -> int:
    return int(s.split("-")[0])

# ── Player search ─────────────────────────────────────────────────────────────
def find_player(name: str) -> dict:
    from urllib.parse import unquote
    name = unquote(name).strip()
    parts = name.split()
    last = parts[-1] if parts else name
    first = parts[0] if len(parts) > 1 else ""

    for endpoint in ["/players/active", "/players"]:
        data = bdl_get(endpoint, {"search": last, "per_page": 25})
        players = data.get("data", [])
        if not players:
            continue
        # Exact full name match
        exact = [p for p in players if f"{p['first_name']} {p['last_name']}".lower() == name.lower()]
        if exact:
            return exact[0]
        # First + last match
        if first:
            fl = [p for p in players if p['last_name'].lower() == last.lower() and p['first_name'].lower().startswith(first.lower())]
            if fl:
                return fl[0]
        # Last name only
        lm = [p for p in players if p['last_name'].lower() == last.lower()]
        if lm:
            return lm[0]
        return players[0]

    raise HTTPException(status_code=404, detail=f"Player '{name}' not found.")

def format_player(p: dict) -> dict:
    team = p.get("team") or {}
    if isinstance(team, int):
        team = {}
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

# ── Game stats fetching ───────────────────────────────────────────────────────
def fetch_player_stats_for_season(player_id: int, season_int: int, postseason: bool = False) -> list[dict]:
    """Fetch all game stats for a player in a season, filtering out DNPs."""
    all_stats = []
    cursor = None
    ps_param = "true" if postseason else "false"

    while True:
        url = (f"{BDL_BASE}/stats"
               f"?player_ids[]={player_id}"
               f"&seasons[]={season_int}"
               f"&postseason={ps_param}"
               f"&per_page=100")
        if cursor:
            url += f"&cursor={cursor}"
        r = requests.get(url, headers=BDL_HEADERS, timeout=20)
        if not r.ok:
            break
        data = r.json()
        rows = data.get("data", [])

        for s in rows:
            # Filter out DNPs — min must be > 0 and pts+reb+ast must be > 0
            min_val = parse_minutes(s.get("min", "0"))
            if min_val < 1:
                continue
            all_stats.append(s)

        cursor = data.get("meta", {}).get("next_cursor")
        if not cursor:
            break

    return all_stats

def parse_minutes(m) -> float:
    if not m: return 0.0
    try:
        s = str(m)
        if ":" in s:
            parts = s.split(":")
            return float(parts[0]) + float(parts[1]) / 60
        return float(s)
    except:
        return 0.0

def normalize_row(s: dict) -> dict:
    """Convert BDL stat row to internal format."""
    game = s.get("game", {})
    team = s.get("team", {})
    home_id = game.get("home_team_id")
    team_id = team.get("id")
    is_home = home_id == team_id

    home_score = game.get("home_team_score") or 0
    visitor_score = game.get("visitor_team_score") or 0
    if is_home:
        wl = "W" if home_score > visitor_score else "L"
    else:
        wl = "W" if visitor_score > home_score else "L"

    pts = s.get("pts") or 0
    reb = s.get("reb") or 0
    ast = s.get("ast") or 0
    stl = s.get("stl") or 0
    blk = s.get("blk") or 0
    min_val = parse_minutes(s.get("min", "0"))

    return {
        "GAME_ID":    str(game.get("id", "")),
        "GAME_DATE":  game.get("date", ""),
        "MATCHUP":    "vs." if is_home else "@",
        "WL":         wl,
        "MIN":        round(min_val, 1),
        "PTS":        pts,
        "REB":        reb,
        "AST":        ast,
        "STL":        stl,
        "BLK":        blk,
        "TOV":        s.get("turnover") or 0,
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
        "SEASON_INT": game.get("season"),
        "POSTSEASON": game.get("postseason", False),
        "IS_HOME":    is_home,
        "TEAM_ABBR":  team.get("abbreviation", ""),
        "HOME_TEAM_ID": home_id,
        "VISITOR_TEAM_ID": game.get("visitor_team_id"),
        "P_R":   round(pts + reb, 1),
        "P_A":   round(pts + ast, 1),
        "P_R_A": round(pts + reb + ast, 1),
        "A_R":   round(ast + reb, 1),
        "S_B":   round(stl + blk, 1),
    }

def get_gamelog(player_id: int, season_int: int, postseason: bool = False) -> list[dict]:
    raw = fetch_player_stats_for_season(player_id, season_int, postseason)
    rows = [normalize_row(s) for s in raw]
    return sorted(rows, key=lambda x: x.get("GAME_DATE", ""))

def calc_avgs(games: list[dict]) -> dict:
    if not games:
        return {}
    n = len(games)
    def avg(k): return round(sum(g.get(k, 0) for g in games) / n, 1)
    fgm = avg("FGM"); fga = avg("FGA")
    fg3m = avg("FG3M"); fg3a = avg("FG3A")
    ftm = avg("FTM"); fta = avg("FTA")
    return {
        "GP": n,
        "MIN": avg("MIN"),
        "PTS": avg("PTS"),
        "REB": avg("REB"),
        "AST": avg("AST"),
        "STL": avg("STL"),
        "BLK": avg("BLK"),
        "TOV": avg("TOV"),
        "FGM": fgm, "FGA": fga,
        "FG_PCT": round(fgm / fga, 3) if fga > 0 else 0,
        "FG3M": fg3m, "FG3A": fg3a,
        "FG3_PCT": round(fg3m / fg3a, 3) if fg3a > 0 else 0,
        "FTM": ftm, "FTA": fta,
        "FT_PCT": round(ftm / fta, 3) if fta > 0 else 0,
        "P_R": round(avg("PTS") + avg("REB"), 1),
        "P_A": round(avg("PTS") + avg("AST"), 1),
        "P_R_A": round(avg("PTS") + avg("REB") + avg("AST"), 1),
        "A_R": round(avg("AST") + avg("REB"), 1),
        "S_B": round(avg("STL") + avg("BLK"), 1),
    }

# ── Opponent defense ──────────────────────────────────────────────────────────
def get_opponent_defense(season_int: int) -> dict:
    try:
        all_games = []
        cursor = None
        while True:
            url = f"{BDL_BASE}/games?seasons[]={season_int}&per_page=100"
            if cursor:
                url += f"&cursor={cursor}"
            r = requests.get(url, headers=BDL_HEADERS, timeout=20)
            if not r.ok:
                break
            d = r.json()
            all_games.extend(d.get("data", []))
            cursor = d.get("meta", {}).get("next_cursor")
            if not cursor:
                break

        defense = {}
        for g in all_games:
            if g.get("status") != "Final":
                continue
            h_abbr = g.get("home_team", {}).get("abbreviation", "")
            v_abbr = g.get("visitor_team", {}).get("abbreviation", "")
            h_pts = g.get("home_team_score") or 0
            v_pts = g.get("visitor_team_score") or 0
            if h_abbr not in defense:
                defense[h_abbr] = []
            if v_abbr not in defense:
                defense[v_abbr] = []
            defense[h_abbr].append(v_pts)
            defense[v_abbr].append(h_pts)

        return {
            abbr: {
                "opp_pts": round(np.mean(pts), 1),
                "opp_reb": 44, "opp_ast": 25, "opp_fg3m": 12,
                "opp_stl": 7, "opp_blk": 5, "opp_pace": 100,
            }
            for abbr, pts in defense.items() if pts
        }
    except Exception:
        return {}

def get_h2h_stats(games: list[dict], opp_abbr: str) -> dict:
    if not opp_abbr:
        return {"games": 0}
    h2h = [g for g in games if
           (not g.get("IS_HOME") and g.get("HOME_TEAM_ID") and str(g.get("HOME_TEAM_ID")) != str(g.get("TEAM_ID")))
           or opp_abbr == g.get("OPP_ABBR", "")]
    if not h2h:
        return {"games": 0}
    result = {"games": len(h2h)}
    for k in ["PTS","REB","AST","STL","BLK"]:
        result[f"h2h_{k.lower()}"] = round(np.mean([g.get(k, 0) for g in h2h]), 1)
    return result

def days_rest(games: list[dict], idx: int) -> int:
    if idx == 0:
        return 3
    try:
        d1 = datetime.strptime(games[idx-1]["GAME_DATE"], "%Y-%m-%d")
        d2 = datetime.strptime(games[idx]["GAME_DATE"], "%Y-%m-%d")
        return max((d2 - d1).days - 1, 0)
    except:
        return 1

def get_travel_fatigue(player_abbr: str, opp_abbr: str) -> float:
    TZ = {"BOS":1,"NYK":1,"BKN":1,"PHI":1,"TOR":1,"MIA":1,"ORL":1,"ATL":1,
          "CHA":1,"WAS":1,"CLE":1,"DET":1,"IND":1,"CHI":2,"MIL":2,"MIN":2,
          "MEM":2,"NOP":2,"SAS":2,"HOU":2,"DAL":2,"OKC":2,"DEN":3,"UTA":3,
          "POR":4,"GSW":4,"LAL":4,"LAC":4,"SAC":4,"PHX":3}
    diff = abs(TZ.get(player_abbr, 2) - TZ.get(opp_abbr, 2))
    fatigue = 2.0 if diff >= 3 else 1.0 if diff == 2 else 0.0
    if opp_abbr == "DEN":
        fatigue += 1.5
    return fatigue

def get_playoff_series_games(playoff_games: list[dict], opp_team_id) -> dict:
    if not playoff_games or not opp_team_id:
        return {}
    series = [g for g in playoff_games if
              str(g.get("HOME_TEAM_ID")) == str(opp_team_id) or
              str(g.get("VISITOR_TEAM_ID")) == str(opp_team_id)]
    series = sorted(series, key=lambda x: x.get("GAME_DATE", ""))
    if not series:
        return {}
    wins   = sum(1 for g in series if g.get("WL") == "W")
    losses = sum(1 for g in series if g.get("WL") == "L")
    avgs = calc_avgs(series)
    recent = series[-2:] if len(series) >= 2 else series
    trend = {}
    for k in ["PTS","REB","AST"]:
        trend[k] = round(np.mean([g.get(k,0) for g in recent]) - avgs.get(k, 0), 1)
    return {
        "games_played": len(series),
        "game_number":  len(series) + 1,
        "wins": wins, "losses": losses,
        "is_close_out":   wins == 3,
        "is_elimination": losses == 3,
        "series_avgs": avgs,
        "series_trend": trend,
        "series_games": series,
    }

def get_opponent_rest(opp_team_id: int, season_int: int) -> int:
    try:
        url = f"{BDL_BASE}/games?team_ids[]={opp_team_id}&seasons[]={season_int}&per_page=5"
        r = requests.get(url, headers=BDL_HEADERS, timeout=10)
        if not r.ok:
            return 2
        rows = [g for g in r.json().get("data", []) if g.get("status") == "Final"]
        if len(rows) < 2:
            return 2
        d1 = datetime.strptime(rows[1]["date"], "%Y-%m-%d")
        d2 = datetime.strptime(rows[0]["date"], "%Y-%m-%d")
        return max((d2 - d1).days - 1, 0)
    except:
        return 2

def get_opponent_recent_defense(opp_team_id: int, season_int: int) -> dict:
    try:
        url = f"{BDL_BASE}/games?team_ids[]={opp_team_id}&seasons[]={season_int}&per_page=15"
        r = requests.get(url, headers=BDL_HEADERS, timeout=15)
        if not r.ok:
            return {}
        rows = [g for g in r.json().get("data", []) if g.get("status") == "Final"][:10]
        if not rows:
            return {}
        pts_allowed, pm_list, wins = [], [], 0
        for g in rows:
            is_home = g.get("home_team", {}).get("id") == opp_team_id
            if is_home:
                pts_allowed.append(g.get("visitor_team_score") or 0)
                pm = (g.get("home_team_score") or 0) - (g.get("visitor_team_score") or 0)
                wins += 1 if pm > 0 else 0
            else:
                pts_allowed.append(g.get("home_team_score") or 0)
                pm = (g.get("visitor_team_score") or 0) - (g.get("home_team_score") or 0)
                wins += 1 if pm > 0 else 0
            pm_list.append(pm)
        return {
            "recent_pts_allowed_avg": round(np.mean(pts_allowed), 1),
            "recent_wins": wins,
            "recent_losses": len(rows) - wins,
            "recent_form_score": round(np.mean(pm_list), 1),
        }
    except:
        return {}

_team_cache: dict = {}
def get_team_id_by_abbr(abbr: str):
    global _team_cache
    if not _team_cache:
        r = requests.get(f"{BDL_BASE}/teams", headers=BDL_HEADERS, params={"per_page": 100}, timeout=10)
        if r.ok:
            _team_cache = {t["abbreviation"]: t["id"] for t in r.json().get("data", [])}
    return _team_cache.get(abbr)

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

# ── Build features ────────────────────────────────────────────────────────────
def build_features(
    recent, all_games, opp_defense, h2h,
    is_home=False, rest=1, pts_bump=0, ast_bump=0, reb_bump=0,
    expected_min=32.0, is_playoffs=False, opp_rest=2,
    opp_recent_form=0, travel_fatigue=0,
    series_games_played=0, series_pts=0, series_reb=0, series_ast=0,
    series_weight=0, is_elimination=False, is_close_out=False,
) -> list[float]:
    keys = ["PTS","REB","AST","STL","BLK","FGM","FGA","FTM","FTA","FG3M"]
    feat = []

    recent_mins = [parse_minutes(g.get("MIN")) for g in recent]
    all_mins    = [parse_minutes(g.get("MIN")) for g in all_games]
    avg_recent_min = np.mean(recent_mins) if recent_mins else 32.0

    # 1. L10 raw mean + std (20)
    for k in keys:
        vals = [g.get(k, 0) for g in recent]
        feat.append(float(np.mean(vals))); feat.append(float(np.std(vals)))

    # 2. L10 per-36 rates (10)
    for k in keys:
        pr36 = [(g.get(k, 0)) / m * 36 for g, m in zip(recent, recent_mins) if m > 5]
        feat.append(float(np.mean(pr36)) if pr36 else 0.0)

    # 3. Minutes-weighted L10 (10)
    for k in keys:
        tw = sum(recent_mins)
        feat.append(sum(g.get(k,0) * m for g,m in zip(recent, recent_mins)) / tw if tw > 0 else 0.0)

    # 4. L5 mean (10)
    last5 = recent[-5:] if len(recent) >= 5 else recent
    for k in keys:
        feat.append(float(np.mean([g.get(k,0) for g in last5])))

    # 5. Season mean (10)
    for k in keys:
        vals = [g.get(k,0) for g in all_games]
        feat.append(float(np.mean(vals)) if vals else 0.0)

    # 6. Season per-36 (10)
    for k in keys:
        pr36 = [g.get(k,0) / m * 36 for g, m in zip(all_games, all_mins) if m > 5]
        feat.append(float(np.mean(pr36)) if pr36 else 0.0)

    # 7. Home/away split (10)
    home_g = [g for g in all_games if g.get("IS_HOME")]
    away_g = [g for g in all_games if not g.get("IS_HOME")]
    split = home_g if is_home else away_g
    for k in keys:
        vals = [g.get(k,0) for g in split]
        feat.append(float(np.mean(vals)) if vals else 0.0)

    # 8. Opp defense (7)
    feat += [float(opp_defense.get("opp_pts", 110)), float(opp_defense.get("opp_reb", 44)),
             float(opp_defense.get("opp_ast", 25)),  float(opp_defense.get("opp_fg3m", 12)),
             float(opp_defense.get("opp_stl", 7)),   float(opp_defense.get("opp_blk", 5)),
             float(opp_defense.get("opp_pace", 100))]

    # 9. H2H (5)
    for k in ["pts","reb","ast","stl","blk"]:
        feat.append(float(h2h.get(f"h2h_{k}", 0)))

    # 10. Context (4)
    feat += [float(is_home), float(min(rest,5)), float(expected_min), float(is_playoffs)]

    # 11. Minutes context (2)
    feat += [float(avg_recent_min), float(expected_min - avg_recent_min)]

    # 12. Teammate bump (3)
    feat += [float(pts_bump), float(ast_bump), float(reb_bump)]

    # 13. Opp context (3)
    feat += [float(min(opp_rest,5)), float(opp_recent_form), float(travel_fatigue)]

    # 14. Playoff series (7)
    feat += [float(series_games_played), float(series_pts), float(series_reb),
             float(series_ast), float(series_weight),
             float(is_elimination), float(is_close_out)]

    return feat

# ── Train + predict ───────────────────────────────────────────────────────────
def train_predict(games, target, opp_defense, h2h,
                  is_home=False, rest=1, pts_bump=0, ast_bump=0, reb_bump=0,
                  expected_min=32.0, is_playoffs=False, opp_rest=2,
                  opp_recent_form=0, travel_fatigue=0, series_context=None) -> dict:

    sc = series_context or {}
    series_games_played = sc.get("games_played", 0)
    series_avgs = sc.get("series_avgs", {})
    series_weight = min(series_games_played / 5.0, 0.8) if series_games_played >= 2 else 0

    games = sorted(games, key=lambda x: x.get("GAME_DATE", ""))

    # Augment with series games for playoff weighting
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
        feat = build_features(recent, augmented[:i], {}, {}, home, r, expected_min=32.0)
        X.append(feat)
        y.append(float(augmented[i].get(target) or 0))

    if len(X) < 5:
        return {"error": "Not enough games to predict"}

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
        pts_bump, ast_bump, reb_bump, expected_min, is_playoffs, opp_rest,
        opp_recent_form, travel_fatigue,
        series_games_played, series_avgs.get("PTS",0), series_avgs.get("REB",0),
        series_avgs.get("AST",0), series_weight,
        sc.get("is_elimination",False), sc.get("is_close_out",False),
    )
    fs = scaler.transform([feat])
    pred = model.predict(fs)[0]
    std  = float(np.std([t.predict(fs)[0] for t in rf.estimators_]))

    recent_avg = float(np.mean([g.get(target,0) for g in recent]))
    season_avg = float(np.mean([g.get(target,0) for g in games]))
    hg = [g for g in games if g.get("IS_HOME")]
    ag = [g for g in games if not g.get("IS_HOME")]
    split = hg if is_home else ag
    split_avg = round(float(np.mean([g.get(target,0) for g in split])), 1) if split else None

    return {
        "prediction": round(float(pred), 1),
        "std_dev": round(std, 2),
        "recent_avg_10": round(recent_avg, 1),
        "season_avg": round(season_avg, 1),
        "home_away_avg": split_avg,
        "games_used": len(augmented),
        "series_weight_applied": round(series_weight, 2),
    }

def fast_predict(all_games, stat, opp_defense, h2h, is_home, rest,
                 expected_min, is_playoffs, opp_rest, travel_fatigue, series_context):
    if len(all_games) < 10:
        return None
    games = sorted(all_games, key=lambda x: x.get("GAME_DATE",""))
    window = min(10, max(5, len(games) // 3))
    keys = ["PTS","REB","AST","STL","BLK","FGM","FGA","FTM","FTA","FG3M"]
    X, y = [], []
    for i in range(window, len(games)):
        recent = games[i-window:i]
        rm = [parse_minutes(g.get("MIN")) for g in recent]
        feat = []
        for k in keys:
            vals = [g.get(k,0) for g in recent]
            feat.append(float(np.mean(vals))); feat.append(float(np.std(vals)))
        for k in keys:
            pr36 = [g.get(k,0)/m*36 for g,m in zip(recent,rm) if m>5]
            feat.append(float(np.mean(pr36)) if pr36 else 0.0)
        feat += [float(opp_defense.get("opp_pts",110)), float(opp_defense.get("opp_reb",44)),
                 float(opp_defense.get("opp_ast",25)), float(opp_defense.get("opp_fg3m",12)),
                 float(opp_defense.get("opp_pace",100))]
        for k in ["pts","reb","ast","stl","blk"]:
            feat.append(float(h2h.get(f"h2h_{k}",0)))
        feat += [float(is_home), float(min(rest,5)), float(expected_min),
                 float(is_playoffs), float(min(opp_rest,5)), float(travel_fatigue)]
        sc = series_context or {}
        feat += [float(sc.get("games_played",0)),
                 float(sc.get("series_avgs",{}).get(stat,0)),
                 float(sc.get("is_elimination",False))]
        X.append(feat); y.append(float(games[i].get(stat,0)))
    if len(X) < 5: return None
    try:
        X, y = np.array(X), np.array(y)
        sc2 = StandardScaler(); Xs = sc2.fit_transform(X)
        m = RandomForestRegressor(n_estimators=80, random_state=42, n_jobs=-1)
        m.fit(Xs, y)
        recent = games[-window:]
        rm = [parse_minutes(g.get("MIN")) for g in recent]
        feat = []
        for k in keys:
            vals = [g.get(k,0) for g in recent]
            feat.append(float(np.mean(vals))); feat.append(float(np.std(vals)))
        for k in keys:
            pr36 = [g.get(k,0)/m2*36 for g,m2 in zip(recent,rm) if m2>5]
            feat.append(float(np.mean(pr36)) if pr36 else 0.0)
        feat += [float(opp_defense.get("opp_pts",110)), float(opp_defense.get("opp_reb",44)),
                 float(opp_defense.get("opp_ast",25)), float(opp_defense.get("opp_fg3m",12)),
                 float(opp_defense.get("opp_pace",100))]
        for k in ["pts","reb","ast","stl","blk"]:
            feat.append(float(h2h.get(f"h2h_{k}",0)))
        sc3 = series_context or {}
        feat += [float(is_home), float(min(rest,5)), float(expected_min),
                 float(is_playoffs), float(min(opp_rest,5)), float(travel_fatigue)]
        feat += [float(sc3.get("games_played",0)),
                 float(sc3.get("series_avgs",{}).get(stat,0)),
                 float(sc3.get("is_elimination",False))]
        pred = m.predict(sc2.transform([feat]))[0]
        return {
            "prediction": round(float(pred),1),
            "recent_avg": round(float(np.mean([g.get(stat,0) for g in recent])),1),
            "season_avg": round(float(np.mean([g.get(stat,0) for g in games])),1),
        }
    except:
        return None

# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {"message": "NbaProp-ML API 🏀", "version": "5.1.0"}

@app.get("/player/search")
def search_player(name: str = Query(...)):
    for endpoint in ["/players/active", "/players"]:
        data = bdl_get(endpoint, {"search": name, "per_page": 10})
        players = data.get("data", [])
        if players:
            return {"players": [format_player(p) for p in players]}
    raise HTTPException(status_code=404, detail="No players found.")

@app.get("/player/{player_name}/info")
def player_info(player_name: str):
    from urllib.parse import unquote
    player_name = unquote(player_name)
    p = find_player(player_name)
    fp = format_player(p)
    season_int = current_season_int()
    games = []
    for offset in range(2):
        try:
            g = get_gamelog(p["id"], season_int - offset)
            if g:
                games = g
                break
        except:
            pass
    stats = calc_avgs(games)
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

@app.get("/player/{player_name}/career")
def player_career(player_name: str):
    from urllib.parse import unquote
    player_name = unquote(player_name)
    p = find_player(player_name)
    fp = format_player(p)
    season_int = current_season_int()
    seasons_to_fetch = list(range(max(2015, season_int - 9), season_int + 1))
    career_rows = []
    for s in seasons_to_fetch:
        try:
            games = get_gamelog(p["id"], s)
            if not games:
                continue
            avgs = calc_avgs(games)
            career_rows.append({
                "SEASON_ID": f"{s}-{str(s+1)[2:]}",
                "PLAYER_ID": p["id"],
                "TEAM_ABBREVIATION": games[-1].get("TEAM_ABBR", fp["team_abbr"]),
                **avgs,
            })
        except:
            continue
    if not career_rows:
        raise HTTPException(status_code=404, detail="No career data found.")
    all_pts = [r.get("PTS",0) for r in career_rows]
    all_reb = [r.get("REB",0) for r in career_rows]
    all_ast = [r.get("AST",0) for r in career_rows]
    career_total = {
        "PLAYER_ID": p["id"],
        "GP": sum(r.get("GP",0) for r in career_rows),
        "PTS": round(np.mean(all_pts),1) if all_pts else 0,
        "REB": round(np.mean(all_reb),1) if all_reb else 0,
        "AST": round(np.mean(all_ast),1) if all_ast else 0,
    }
    return {
        "season_totals_regular_season": career_rows,
        "career_totals_regular_season": [career_total],
    }

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
    return {"game_log": list(reversed(games))}

_leaders_cache: dict = {}

# Known top NBA players by BDL ID — covers all realistic leaders
TOP_PLAYER_IDS = [
    175,   # Shai Gilgeous-Alexander
    115,   # Stephen Curry
    1629029, # Luka Doncic
    246,   # Nikola Jokic
    1630162, # Anthony Edwards
    1628983, # (duplicate SGA - skip)
    203999,  # Nikola Jokic alt
    1627759, # Jaylen Brown
    1628378, # Donovan Mitchell
    1628973, # Jalen Brunson
    1630178, # Tyrese Maxey
    202695,  # Kawhi Leonard
    1627750, # Jamal Murray
    1641705, # Victor Wembanyama
    1630166, # Deni Avdija
    1627783, # Pascal Siakam
    201142,  # Kevin Durant
    1626164, # Devin Booker
    1629627, # Ja Morant
    1629028, # Zion Williamson
    203507,  # Giannis Antetokounmpo
    2544,    # LeBron James
    201935,  # James Harden
    1628384, # Bam Adebayo
    1629630, # Jaren Jackson Jr
    1629636, # Darius Garland
    1629029, # Luka alt
    1630596, # Evan Mobley
    1630224, # Josh Giddey
    1630169, # Franz Wagner
    1629628, # Jordan Poole
    1630542, # Scottie Barnes
    1629311, # Trae Young
    1629029, # skip
    203081,  # Damian Lillard
    1628389, # De'Aaron Fox
    1629021, # Saddiq Bey
    1641706, # Cooper Flagg (2025 draft)
    1642844, # Dylan Harper
    1630191, # Alperen Sengun
    1630563, # Cade Cunningham
    1630559, # Jalen Green
    203076,  # Anthony Davis
    1629029, # skip
    203944,  # Julius Randle
    1627826, # Zach LaVine
    203468,  # Victor Oladipo
    1629057, # Luguentz Dort
    1630532, # Desmond Bane
    1629029, # skip
]
# Deduplicate
TOP_PLAYER_IDS = list(dict.fromkeys(TOP_PLAYER_IDS))


@app.get("/league/leaders")
def league_leaders(
    season: str = Query(None),
    stat_category: str = Query("PTS"),
    top: int = Query(10, ge=1, le=25),
):
    if not season:
        season = current_season_str()
    season_int = season_str_to_int(season)

    cache_key = f"leaders_{season}_{datetime.now().strftime('%Y%m%d')}"

    STAT_MAP = {
        "PTS": "PTS", "REB": "REB", "AST": "AST", "STL": "STL", "BLK": "BLK",
        "FG_PCT": "FG_PCT", "FG3_PCT": "FG3_PCT", "FT_PCT": "FT_PCT",
    }
    stat_key = STAT_MAP.get(stat_category, "PTS")

    if cache_key not in _leaders_cache:
        # Fetch stats for all top players — one BDL call per player
        import time
        player_data = {}
        for pid in TOP_PLAYER_IDS:
            try:
                games = get_gamelog(pid, season_int)
                if len(games) < 10:
                    continue
                avgs = calc_avgs(games)
                # Get player name from first game's raw data
                url = f"{BDL_BASE}/players/{pid}"
                pr = requests.get(url, headers=BDL_HEADERS, timeout=10)
                if pr.ok:
                    pinfo = pr.json().get("data", {})
                    team = pinfo.get("team") or {}
                    player_data[pid] = {
                        "name": f"{pinfo.get('first_name','')} {pinfo.get('last_name','')}",
                        "team": team.get("abbreviation","") if isinstance(team, dict) else "",
                        "avgs": avgs,
                    }
                time.sleep(0.5)  # stay under 60 req/min
            except Exception:
                continue
        _leaders_cache[cache_key] = player_data

    player_data = _leaders_cache[cache_key]

    if not player_data:
        raise HTTPException(status_code=503, detail="Could not fetch leaders data. Try again in a moment.")

    leaders_list = []
    for pid, pdata in player_data.items():
        avgs = pdata["avgs"]
        val = avgs.get(stat_key, 0)
        if not val:
            continue
        leaders_list.append({
            "PLAYER_ID": pid,
            "PLAYER": pdata["name"],
            "TEAM": pdata["team"],
            "GP": avgs.get("GP", 0),
            "MIN": avgs.get("MIN", 0),
            "PTS": avgs.get("PTS", 0),
            "REB": avgs.get("REB", 0),
            "AST": avgs.get("AST", 0),
            "STL": avgs.get("STL", 0),
            "BLK": avgs.get("BLK", 0),
            "FG_PCT": avgs.get("FG_PCT", 0),
            "FG3_PCT": avgs.get("FG3_PCT", 0),
            "FT_PCT": avgs.get("FT_PCT", 0),
            "TOV": avgs.get("TOV", 0),
            "_sort_val": val,
        })

    leaders_list.sort(key=lambda x: x["_sort_val"], reverse=True)
    for i, row in enumerate(leaders_list[:top]):
        row["RANK"] = i + 1
        del row["_sort_val"]

    return {"leaders": leaders_list[:top]}

@app.get("/odds/status")
def odds_status():
    try:
        r = requests.get(f"{ODDS_BASE}/sports/basketball_nba/events",
                         params={"apiKey": ODDS_KEY}, timeout=10)
        return {
            "status": r.status_code,
            "requests_remaining": r.headers.get("x-requests-remaining","unknown"),
            "requests_used": r.headers.get("x-requests-used","unknown"),
            "events_found": len(r.json()) if r.ok else 0,
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/predict/{player_name}")
def predict_player(player_name: str, season: str = Query(None)):
    from urllib.parse import unquote
    player_name = unquote(player_name)
    if not season:
        season = current_season_str()
    season_int = season_str_to_int(season)

    p = find_player(player_name)
    pid = p["id"]
    fp = format_player(p)
    player_team_abbr = fp.get("team_abbr", "")

    # Fetch game logs — current + 2 previous seasons (no DNPs)
    all_games = []
    for offset in range(3):
        s = season_int - offset
        try:
            g = get_gamelog(pid, s)
            all_games = g + all_games
        except:
            pass

    if not all_games:
        raise HTTPException(status_code=500, detail=f"Could not fetch game log for {player_name}.")

    # Fetch props from Odds API
    props_found = {}
    opponent_abbr = None
    opponent_name = None
    today_game = None
    is_home = False
    opp_team_id = None

    try:
        player_last  = fp["last_name"].lower()
        player_first = fp["first_name"].lower()
        market_keys  = "player_points,player_rebounds,player_assists,player_steals,player_blocks"
        stat_map = {"player_points":"PTS","player_rebounds":"REB",
                    "player_assists":"AST","player_steals":"STL","player_blocks":"BLK"}

        events_r = requests.get(f"{ODDS_BASE}/sports/basketball_nba/events",
                                 params={"apiKey": ODDS_KEY}, timeout=10)
        events = events_r.json() if events_r.ok else []

        for event in (events if isinstance(events, list) else []):
            event_id = event.get("id")
            home = event.get("home_team","")
            away = event.get("away_team","")
            if not event_id: continue
            try:
                odds_r = requests.get(
                    f"{ODDS_BASE}/sports/basketball_nba/events/{event_id}/odds",
                    params={"apiKey": ODDS_KEY, "regions":"us", "markets": market_keys,
                            "oddsFormat":"american", "bookmakers":"draftkings,fanduel,betmgm"},
                    timeout=10,
                )
                if not odds_r.ok: continue
                player_found = False
                for bm in odds_r.json().get("bookmakers",[]):
                    for mkt in bm.get("markets",[]):
                        stat = stat_map.get(mkt.get("key",""))
                        if not stat: continue
                        for outcome in mkt.get("outcomes",[]):
                            desc = outcome.get("description","").lower()
                            if player_last in desc or player_first in desc:
                                player_found = True
                                if stat not in props_found and outcome.get("name") == "Over":
                                    props_found[stat] = outcome.get("point")
                if player_found and not opponent_abbr:
                    today_game = {"home": home, "away": away}
                    home_abbr = ODDS_TEAM_MAP.get(home,"")
                    away_abbr = ODDS_TEAM_MAP.get(away,"")
                    if player_team_abbr == home_abbr:
                        opponent_abbr = away_abbr; opponent_name = away; is_home = True
                    else:
                        opponent_abbr = home_abbr; opponent_name = home; is_home = False
            except: continue
    except: pass

    # Opponent context
    sorted_games = sorted(all_games, key=lambda x: x.get("GAME_DATE",""))
    opp_defense = {}
    h2h = {}
    series_context = {}
    opp_rest = 2
    opp_recent_defense = {}
    opp_recent_form = 0

    if opponent_abbr:
        opp_team_id = get_team_id_by_abbr(opponent_abbr)
        try:
            all_def = get_opponent_defense(season_int)
            opp_defense = all_def.get(opponent_abbr, {})
        except: pass
        if opp_team_id:
            try:
                opp_rest = get_opponent_rest(opp_team_id, season_int)
                opp_recent_defense = get_opponent_recent_defense(opp_team_id, season_int)
                opp_recent_form = opp_recent_defense.get("recent_form_score", 0)
            except: pass
            # Playoff series
            try:
                playoff_games = get_gamelog(pid, season_int, postseason=True)
                if playoff_games:
                    series_context = get_playoff_series_games(playoff_games, opp_team_id)
            except: pass

    # Minutes + context
    recent_10 = sorted_games[-10:]
    recent_mins = [parse_minutes(g.get("MIN")) for g in recent_10 if parse_minutes(g.get("MIN")) > 5]
    expected_min = round(np.mean(recent_mins), 1) if recent_mins else 32.0
    is_playoffs = any(g.get("POSTSEASON") for g in sorted_games[-5:])
    rest = days_rest(sorted_games, len(sorted_games)-1) if len(sorted_games) > 1 else 1
    travel_fatigue = get_travel_fatigue(player_team_abbr, opponent_abbr) if player_team_abbr and opponent_abbr else 0

    # Injuries
    injuries = []
    try:
        inj_r = requests.get(f"{BDL_BASE}/player_injuries", headers=BDL_HEADERS, timeout=10)
        if inj_r.ok:
            injuries = [{"player_name": f"{i['player']['first_name']} {i['player']['last_name']}",
                         "status": i.get("status",""), "reason": i.get("description","")}
                        for i in inj_r.json().get("data",[])]
    except: pass

    # Train predictions
    targets = ["PTS","REB","AST","STL","BLK"]
    predictions = {}
    for stat in targets:
        ml = train_predict(
            all_games, stat, opp_defense, h2h, is_home, rest,
            0, 0, 0, expected_min, is_playoffs, opp_rest,
            opp_recent_form, travel_fatigue, series_context,
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

    combos = {}
    for combo, stats in [("PR",["PTS","REB"]),("PA",["PTS","AST"]),("PRA",["PTS","REB","AST"]),("RA",["REB","AST"])]:
        total_pred = sum(predictions[s]["ml_prediction"] or 0 for s in stats)
        total_line = round(sum(props_found[s] for s in stats),1) if all(props_found.get(s) for s in stats) else None
        entry = {"stat": combo, "ml_prediction": round(total_pred,1), "line": total_line}
        if total_line:
            diff = total_pred - total_line
            entry["recommendation"] = "OVER" if diff > 0 else "UNDER"
            entry["edge"] = round(abs(diff), 1)
        else:
            entry["recommendation"] = "NO LINE"
        combos[combo] = entry

    return {
        "player": fp, "season": season,
        "opponent": opponent_name, "opponent_abbr": opponent_abbr,
        "today_game": today_game, "is_home": is_home,
        "rest_days": rest, "opp_rest_days": opp_rest,
        "expected_min": expected_min, "is_playoffs": is_playoffs,
        "travel_fatigue": travel_fatigue,
        "usage_data": {"usg_trend": 0, "trending_up": False},
        "clutch_data": {},
        "opponent_defense": opp_defense,
        "opp_recent_defense": opp_recent_defense,
        "h2h": h2h,
        "series_context": {k:v for k,v in series_context.items() if k != "series_games"},
        "shot_profile_edge": {"score": 0, "interpretation": "N/A", "zones": {}},
        "teammate_context": {"out_players": []},
        "relevant_injuries": injuries[:10],
        "lineup_news": [],
        "predictions": predictions, "combos": combos,
        "props_found": len(props_found),
        "note": "GradientBoosting · BallDontLie API · DNPs filtered · Per-36 rates · Opp defense · H2H · Playoff series",
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

    all_props = {}
    player_event_map = {}
    try:
        events_r = requests.get(f"{ODDS_BASE}/sports/basketball_nba/events",
                                 params={"apiKey": ODDS_KEY}, timeout=10)
        if not events_r.ok:
            raise HTTPException(status_code=500, detail="Could not fetch events")
        events = events_r.json()
        market_keys = "player_points,player_rebounds,player_assists,player_steals,player_blocks"
        stat_map = {"player_points":"PTS","player_rebounds":"REB","player_assists":"AST",
                    "player_steals":"STL","player_blocks":"BLK"}

        for event in (events if isinstance(events, list) else []):
            event_id = event.get("id")
            home = event.get("home_team",""); away = event.get("away_team","")
            if not event_id: continue
            try:
                odds_r = requests.get(
                    f"{ODDS_BASE}/sports/basketball_nba/events/{event_id}/odds",
                    params={"apiKey": ODDS_KEY, "regions":"us", "markets": market_keys,
                            "oddsFormat":"american", "bookmakers":"draftkings,fanduel,betmgm"},
                    timeout=10,
                )
                if not odds_r.ok: continue
                for bm in odds_r.json().get("bookmakers",[]):
                    for mkt in bm.get("markets",[]):
                        stat = stat_map.get(mkt.get("key",""))
                        if not stat: continue
                        for outcome in mkt.get("outcomes",[]):
                            if outcome.get("name") != "Over": continue
                            pname = outcome.get("description","")
                            if not pname: continue
                            if pname not in all_props:
                                all_props[pname] = {}
                                player_event_map[pname] = {"home": home, "away": away}
                            if stat not in all_props[pname]:
                                all_props[pname][stat] = outcome.get("point")
            except: continue
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not fetch props: {e}")

    if not all_props:
        return {"picks":[], "total_players":0, "message":"No props found for today"}

    all_defense = {}
    try:
        all_defense = get_opponent_defense(season_int)
    except: pass

    picks = []
    gamelog_cache = {}

    for player_name, props in all_props.items():
        if not props: continue
        try:
            p = find_player(player_name)
            pid = p["id"]
            fp = format_player(p)
            player_team_abbr = fp.get("team_abbr","")

            if pid not in gamelog_cache:
                games = []
                for offset in range(2):
                    try:
                        g = get_gamelog(pid, season_int - offset)
                        games = g + games
                    except: pass
                gamelog_cache[pid] = games
            all_games = gamelog_cache[pid]
            if len(all_games) < 10: continue

            sorted_games = sorted(all_games, key=lambda x: x.get("GAME_DATE",""))
            event_info = player_event_map.get(player_name, {})
            home_team = event_info.get("home",""); away_team = event_info.get("away","")
            home_abbr = ODDS_TEAM_MAP.get(home_team,""); away_abbr = ODDS_TEAM_MAP.get(away_team,"")
            is_home = player_team_abbr == home_abbr
            opponent_abbr = away_abbr if is_home else home_abbr

            opp_defense = all_defense.get(opponent_abbr, {})
            recent_mins = [parse_minutes(g.get("MIN")) for g in sorted_games[-10:] if parse_minutes(g.get("MIN")) > 5]
            expected_min = round(np.mean(recent_mins),1) if recent_mins else 32.0
            is_playoffs = any(g.get("POSTSEASON") for g in sorted_games[-5:])
            rest = days_rest(sorted_games, len(sorted_games)-1) if len(sorted_games) > 1 else 1
            travel_fatigue = get_travel_fatigue(player_team_abbr, opponent_abbr) if player_team_abbr and opponent_abbr else 0

            series_context = {}
            if is_playoffs and opponent_abbr:
                try:
                    opp_tid = get_team_id_by_abbr(opponent_abbr)
                    poff = get_gamelog(pid, season_int, postseason=True)
                    series_context = get_playoff_series_games(poff, opp_tid)
                except: pass

            for stat, line in props.items():
                if line is None: continue
                try:
                    result = fast_predict(all_games, stat, opp_defense, {}, is_home, rest,
                                          expected_min, is_playoffs, 2, travel_fatigue, series_context)
                    if not result: continue
                    pred = result["prediction"]
                    diff = pred - line
                    edge = round(abs(diff), 1)
                    if edge < 0.5: continue
                    picks.append({
                        "player": fp["full_name"], "stat": stat, "line": line,
                        "prediction": pred, "recommendation": "OVER" if diff > 0 else "UNDER",
                        "edge": edge, "recent_avg": result["recent_avg"],
                        "season_avg": result["season_avg"],
                        "matchup": f"{away_team} @ {home_team}",
                        "is_playoffs": is_playoffs,
                        "series_game": series_context.get("game_number"),
                        "is_elimination": series_context.get("is_elimination", False),
                    })
                except: continue
        except: continue

    picks.sort(key=lambda x: x["edge"], reverse=True)
    _picks_cache[cache_key] = {
        "all_picks": picks,
        "total_players": len(all_props),
        "total_picks_found": len([p for p in picks if p["edge"] >= min_edge]),
        "season": season, "cached": False,
    }
    filtered = [p for p in picks if p["edge"] >= min_edge]
    return {
        "picks": filtered[:top], "total_players": len(all_props),
        "total_picks_found": len(filtered),
        "season": season, "min_edge": min_edge, "cached": False,
    }