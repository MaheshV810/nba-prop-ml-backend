from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from nba_api.stats.static import players, teams
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

# ── Patch nba_api http.py to use browser headers + longer timeout ─────────────
try:
    import nba_api.library.http as nba_http
    import inspect, re

    BROWSER_HEADERS = {
        "Host": "stats.nba.com",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "x-nba-stats-origin": "stats",
        "x-nba-stats-token": "true",
        "Referer": "https://www.nba.com/",
        "Connection": "keep-alive",
        "Origin": "https://www.nba.com",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }

    # Patch the class-level headers and timeout
    nba_http.NBAStatsHTTP.HEADERS = BROWSER_HEADERS
    nba_http.NBAStatsHTTP.TIMEOUT = 60

    # Also patch the module-level send function if it exists
    if hasattr(nba_http, 'HEADERS'):
        nba_http.HEADERS = BROWSER_HEADERS
    if hasattr(nba_http, 'TIMEOUT'):
        nba_http.TIMEOUT = 60

    # Monkey-patch the actual file on disk so nba_api uses our headers
    import nba_api
    import os
    http_path = os.path.join(os.path.dirname(nba_api.__file__), 'library', 'http.py')
    with open(http_path, 'r') as f:
        src = f.read()

    # Replace timeout value
    src = re.sub(r'timeout\s*=\s*\d+', 'timeout=60', src)

    # Replace headers dict if present
    if 'x-nba-stats-token' not in src:
        src = src.replace(
            "'User-Agent':",
            "'x-nba-stats-origin': 'stats', 'x-nba-stats-token': 'true', 'Referer': 'https://www.nba.com/', 'Origin': 'https://www.nba.com/', 'User-Agent':"
        )

    with open(http_path, 'w') as f:
        f.write(src)
except Exception as e:
    print(f"nba_api patch warning: {e}")

ODDS_KEY  = os.getenv("ODDS_API_KEY", "44adfb9534b54975e4ff98b9bf8f503a")
ODDS_BASE = "https://api.the-odds-api.com/v4"

app = FastAPI(title="NBA Stats + ML Predictions", version="4.0.0")

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEAMS_BY_ID   = {t["id"]: t for t in teams.get_teams()}
TEAMS_BY_ABBR = {t["abbreviation"]: t for t in teams.get_teams()}
TEAMS_BY_NAME = {t["full_name"]: t for t in teams.get_teams()}


# ── Helpers ───────────────────────────────────────────────────────────────────

def search_players_live(name: str) -> list[dict]:
    """
    Search NBA.com live player index — includes rookies not in nba_api static list.
    Returns list of {id, full_name, first_name, last_name, is_active}.
    """
    try:
        from nba_api.stats.endpoints import commonallplayers
        time.sleep(0.6)
        all_p = commonallplayers.CommonAllPlayers(
            is_only_current_season=0,
            league_id="00",
            season="2024-25",
        )
        rows = all_p.get_normalized_dict().get("CommonAllPlayers", [])
        name_lower = name.lower()
        matches = []
        for r in rows:
            full = r.get("DISPLAY_FIRST_LAST", "") or ""
            if name_lower in full.lower():
                matches.append({
                    "id": r.get("PERSON_ID"),
                    "full_name": full,
                    "first_name": r.get("DISPLAY_FIRST_LAST", "").split()[0] if full else "",
                    "last_name": " ".join(r.get("DISPLAY_FIRST_LAST", "").split()[1:]) if full else "",
                    "is_active": r.get("ROSTERSTATUS") == "Active",
                })
        return matches
    except Exception:
        return []


def get_player(name: str) -> dict:
    """Find player by name — checks static list first, falls back to live NBA.com search."""
    # Try static list first (fast, no API call)
    result = players.find_players_by_full_name(name)
    if result:
        return result[0]
    # Fall back to live search (catches rookies, new signings)
    live = search_players_live(name)
    if live:
        return live[0]
    raise HTTPException(status_code=404, detail=f"Player '{name}' not found.")


def enrich(row: dict[str, Any]) -> dict[str, Any]:
    pts = row.get("PTS") or 0
    reb = row.get("REB") or 0
    ast = row.get("AST") or 0
    stl = row.get("STL") or 0
    blk = row.get("BLK") or 0
    row["P_R"]   = round(pts + reb, 1)
    row["A_R"]   = round(ast + reb, 1)
    row["P_A"]   = round(pts + ast, 1)
    row["P_R_A"] = round(pts + reb + ast, 1)
    row["S_B"]   = round(stl + blk, 1)
    return row


def odds_get(path: str, params: dict = {}) -> Any:
    params = {"apiKey": ODDS_KEY, **params}
    r = requests.get(f"{ODDS_BASE}{path}", params=params, timeout=15)
    if not r.ok:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    return r.json()


def get_injury_report(team_id: int, season: str = "2024-25") -> list[dict]:
    """Pull today's injury/availability report for a team via nba_api."""
    try:
        from nba_api.stats.endpoints import commonteamroster
        time.sleep(0.6)
        roster = commonteamroster.CommonTeamRoster(team_id=team_id, season=season)
        data = roster.get_normalized_dict()
        return data.get("CommonTeamRoster", [])
    except Exception:
        return []


def get_league_injury_report() -> list[dict]:
    """
    Fetch NBA injury report from the official NBA stats injury endpoint.
    Returns list of {player_name, team, status, reason}.
    """
    try:
        from nba_api.stats.endpoints import leaguedashplayerstats
        # Use ESPN's public injury feed as fallback — no auth needed
        r = requests.get(
            "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries",
            timeout=10,
        )
        if not r.ok:
            return []
        data = r.json()
        injured = []
        for team_entry in data.get("injuries", []):
            team_name = team_entry.get("team", {}).get("displayName", "")
            for p in team_entry.get("injuries", []):
                injured.append({
                    "player_name": p.get("athlete", {}).get("displayName", ""),
                    "team": team_name,
                    "status": p.get("status", ""),
                    "reason": p.get("injuries", [{}])[0].get("type", "") if p.get("injuries") else "",
                })
        return injured
    except Exception:
        return []


def get_teammate_usage(player_name: str, team_abbr: str, season: str, injury_report: list[dict]) -> dict:
    """
    Estimate usage bump if key teammates are out.
    Looks at injured teammates on same team and estimates how much extra
    usage flows to the target player based on historical usage %.
    """
    try:
        from nba_api.stats.endpoints import leaguedashplayerstats
        time.sleep(0.6)
        stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            per_mode_simple="PerGame",
        )
        data = stats.get_normalized_dict().get("LeagueDashPlayerStats", [])

        # Get all players on same team
        team_players = [p for p in data if p.get("TEAM_ABBREVIATION") == team_abbr]

        # Find injured teammates
        out_players = []
        for inj in injury_report:
            status = inj.get("status", "").lower()
            if status in ("out", "doubtful") and inj.get("player_name", "").lower() != player_name.lower():
                inj_name = inj.get("player_name", "").lower()
                for tp in team_players:
                    full = f"{tp.get('PLAYER_NAME', '')}".lower()
                    if inj_name in full or full in inj_name:
                        out_players.append(tp)
                        break

        # Target player's current usage
        target = next((p for p in team_players if player_name.lower() in p.get("PLAYER_NAME", "").lower()), None)
        target_usg = target.get("USG_PCT", 0) if target else 0
        target_min = target.get("MIN", 0) if target else 0

        # Sum up minutes/usage being vacated
        vacated_min = sum(p.get("MIN", 0) for p in out_players)
        vacated_pts = sum(p.get("PTS", 0) for p in out_players)
        vacated_ast = sum(p.get("AST", 0) for p in out_players)
        vacated_reb = sum(p.get("REB", 0) for p in out_players)

        # Estimate how much of that flows to target (proportional to their usage)
        team_usg_total = sum(p.get("USG_PCT", 0) for p in team_players if p.get("PLAYER_NAME", "").lower() != player_name.lower() and p not in out_players) or 1
        share = target_usg / team_usg_total if team_usg_total > 0 else 0

        return {
            "out_players": [p.get("PLAYER_NAME") for p in out_players],
            "vacated_min": round(vacated_min, 1),
            "vacated_pts": round(vacated_pts, 1),
            "vacated_ast": round(vacated_ast, 1),
            "vacated_reb": round(vacated_reb, 1),
            "estimated_pts_bump": round(vacated_pts * share, 1),
            "estimated_ast_bump": round(vacated_ast * share, 1),
            "estimated_reb_bump": round(vacated_reb * share, 1),
            "target_usg_pct": round(target_usg * 100, 1) if target_usg else None,
        }
    except Exception:
        return {"out_players": [], "vacated_min": 0}


def get_referee_stats(season: str = "2024-25") -> dict:
    """
    Pull referee foul-rate tendencies from nba_api.
    Returns {ref_name: {foul_rate, pace_impact}} — high foul ref = more FTA.
    """
    try:
        from nba_api.stats.endpoints import leaguegamelog
        time.sleep(0.6)
        # nba_api doesn't have a direct ref endpoint but we can use
        # the official NBA stats ref report
        r = requests.get(
            "https://stats.nba.com/stats/refereereport?LeagueID=00",
            timeout=15,
        )
        if not r.ok:
            return {}
        data = r.json()
        refs = {}
        for row in data.get("resultSets", [{}])[0].get("rowSet", []):
            headers = data["resultSets"][0]["headers"]
            ref_dict = dict(zip(headers, row))
            name = ref_dict.get("REFEREE_NAME", "")
            if name:
                refs[name] = {
                    "fta_per_game": ref_dict.get("FOUL_RATE_HOME", 0),
                    "games": ref_dict.get("G", 0),
                }
        return refs
    except Exception:
        return {}


def get_line_movement(event_id: str, stat_markets: list[str]) -> dict:
    """
    Fetch opening vs current line for a player prop to detect sharp movement.
    Returns {stat: {open_line, current_line, movement, sharp_direction}}.
    """
    movement = {}
    try:
        # Get historical odds (opening line)
        hist = odds_get(
            f"/sports/basketball_nba/events/{event_id}/odds",
            {
                "regions": "us",
                "markets": ",".join(stat_markets),
                "oddsFormat": "american",
                "dateFormat": "iso",
            }
        )
        for bm in hist.get("bookmakers", []):
            # Use DraftKings or FanDuel as reference book
            if bm.get("key") not in ("draftkings", "fanduel", "betmgm"):
                continue
            for mkt in bm.get("markets", []):
                key = mkt.get("key", "")
                stat_map = {
                    "player_points": "PTS", "player_rebounds": "REB",
                    "player_assists": "AST", "player_steals": "STL", "player_blocks": "BLK",
                }
                stat = stat_map.get(key)
                if not stat or stat in movement:
                    continue
                outcomes = mkt.get("outcomes", [])
                over = next((o for o in outcomes if o.get("name") == "Over"), None)
                if over:
                    current = over.get("point")
                    # last_update gives us current; we'll flag significant moves
                    movement[stat] = {
                        "current_line": current,
                        "bookmaker": bm.get("title"),
                    }
    except Exception:
        pass
    return movement


def get_player_shot_profile(player_id: int, season: str) -> dict:
    """
    Pull player shot distribution by zone from shotchartdetail.
    Returns freq and FG% per zone: restricted_area, paint_non_ra,
    mid_range, corner_3, above_break_3.
    """
    try:
        from nba_api.stats.endpoints import shotchartdetail
        time.sleep(0.6)
        chart = shotchartdetail.ShotChartDetail(
            player_id=player_id,
            team_id=0,
            season_nullable=season,
            context_measure_simple="FGA",
        )
        data = chart.get_normalized_dict().get("Shot_Chart_Detail", [])

        zone_map = {
            "Restricted Area":       "restricted_area",
            "In The Paint (Non-RA)": "paint_non_ra",
            "Mid-Range":             "mid_range",
            "Left Corner 3":         "corner_3",
            "Right Corner 3":        "corner_3",
            "Above the Break 3":     "above_break_3",
        }

        zone_attempts = {}
        zone_makes = {}
        total = 0

        for shot in data:
            zone = zone_map.get(shot.get("SHOT_ZONE_BASIC"))
            if zone is None:
                continue
            total += 1
            zone_attempts[zone] = zone_attempts.get(zone, 0) + 1
            if shot.get("SHOT_MADE_FLAG") == 1:
                zone_makes[zone] = zone_makes.get(zone, 0) + 1

        if total == 0:
            return {}

        profile = {}
        for zone in ["restricted_area", "paint_non_ra", "mid_range", "corner_3", "above_break_3"]:
            att = zone_attempts.get(zone, 0)
            mak = zone_makes.get(zone, 0)
            profile[zone] = {
                "freq":     round(att / total, 3),
                "fg_pct":   round(mak / att, 3) if att > 0 else 0,
                "attempts": att,
            }
        return profile
    except Exception:
        return {}


def get_opponent_shot_defense(team_abbr: str, season: str) -> dict:
    """
    Pull opponent defensive FG% allowed by shot zone.
    Returns {zone: {opp_fg_pct_allowed, vs_league_avg}}.
    Positive vs_league_avg = defense gives up more than average there (easier).
    """
    LEAGUE_AVGS = {
        "restricted_area": 0.630,
        "paint_non_ra":    0.395,
        "mid_range":       0.400,
        "corner_3":        0.385,
        "above_break_3":   0.360,
    }
    try:
        from nba_api.stats.endpoints import leaguedashteamptshot
        time.sleep(0.6)
        stats = leaguedashteamptshot.LeagueDashTeamPtShot(
            season=season,
            per_mode_simple="PerGame",
            defense_category="Overall",
        )
        rows = stats.get_normalized_dict().get("LeagueDashTeamPtShot", [])
        team_row = next((r for r in rows if r.get("TEAM_ABBREVIATION") == team_abbr), None)
        if not team_row:
            return {}

        col_map = {
            "restricted_area": ("RA_FGM",    "RA_FGA"),
            "paint_non_ra":    ("PAINT_FGM",  "PAINT_FGA"),
            "mid_range":       ("MR_FGM",     "MR_FGA"),
            "corner_3":        ("C3_FGM",     "C3_FGA"),
            "above_break_3":   ("AB3_FGM",    "AB3_FGA"),
        }

        defense = {}
        for zone, (fgm_col, fga_col) in col_map.items():
            fgm = team_row.get(fgm_col) or 0
            fga = team_row.get(fga_col) or 0
            opp_pct = round(fgm / fga, 3) if fga > 0 else LEAGUE_AVGS[zone]
            defense[zone] = {
                "opp_fg_pct_allowed": opp_pct,
                "vs_league_avg": round(opp_pct - LEAGUE_AVGS[zone], 3),
            }
        return defense
    except Exception:
        return {}


def compute_shot_profile_edge(player_profile: dict, opp_shot_defense: dict) -> dict:
    """
    Cross-multiply player shot frequency vs opponent zone weakness.
    Positive score = opponent leaks in zones player frequents (favorable).
    Negative score = opponent is stingy in those zones (unfavorable).
    """
    if not player_profile or not opp_shot_defense:
        return {"score": 0, "interpretation": "UNKNOWN", "zones": {}}

    total_edge = 0
    zones_detail = {}

    for zone, pdata in player_profile.items():
        freq = pdata.get("freq", 0)
        if freq < 0.05:
            continue
        ddata = opp_shot_defense.get(zone, {})
        vs_avg = ddata.get("vs_league_avg", 0)
        zone_edge = round(freq * vs_avg * 100, 2)
        total_edge += zone_edge
        zones_detail[zone] = {
            "player_freq":    f"{round(freq * 100, 1)}%",
            "player_fg_pct":  f"{round(pdata.get('fg_pct', 0) * 100, 1)}%",
            "opp_allows":     f"{round(ddata.get('opp_fg_pct_allowed', 0) * 100, 1)}%",
            "vs_league":      f"{'+' if vs_avg >= 0 else ''}{round(vs_avg * 100, 1)}%",
            "zone_edge":      zone_edge,
        }

    return {
        "score": round(total_edge, 2),
        "interpretation": (
            "FAVORABLE" if total_edge > 0.5
            else "UNFAVORABLE" if total_edge < -0.5
            else "NEUTRAL"
        ),
        "zones": zones_detail,
    }


# ── Standard Stats Routes ─────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "NBA Stats + ML API 🏀"}


@app.get("/player/search")
def search_player(name: str = Query(...)):
    # Try static list first
    result = players.find_players_by_full_name(name)
    if result:
        return {"players": result}
    # Fall back to live search for rookies/new players
    live = search_players_live(name)
    if not live:
        raise HTTPException(status_code=404, detail="No players found.")
    return {"players": live}


@app.get("/player/{player_name}/info")
def player_info(player_name: str):
    from nba_api.stats.endpoints import playercareerstats
    p = get_player(player_name)
    time.sleep(0.6)
    career = playercareerstats.PlayerCareerStats(player_id=p["id"], per_mode36="PerGame")
    data = career.get_normalized_dict()
    seasons = data.get("SeasonTotalsRegularSeason", [])
    latest = enrich(seasons[-1]) if seasons else {}
    return {
        "common_player_info": {
            "DISPLAY_FIRST_LAST": p["full_name"],
            "JERSEY": "",
            "POSITION": "",
            "TEAM_NAME": TEAMS_BY_ID.get(latest.get("TEAM_ID"), {}).get("full_name", ""),
            "HEIGHT": "",
            "WEIGHT": "",
            "COUNTRY": "",
        },
        "player_headline_stats": latest,
    }


@app.get("/player/{player_name}/career")
def player_career(player_name: str):
    from nba_api.stats.endpoints import playercareerstats
    p = get_player(player_name)
    time.sleep(0.6)
    career = playercareerstats.PlayerCareerStats(player_id=p["id"], per_mode36="PerGame")
    data = career.get_normalized_dict()
    return {
        "season_totals_regular_season": [enrich(r) for r in data["SeasonTotalsRegularSeason"]],
        "career_totals_regular_season": [enrich(r) for r in data["CareerTotalsRegularSeason"]],
    }


@app.get("/player/{player_name}/gamelog")
def player_gamelog(player_name: str, season: str = Query(None)):
    from nba_api.stats.endpoints import playergamelog
    if not season:
        now = datetime.now()
        season = f"{now.year}-{str(now.year+1)[2:]}" if now.month >= 10 else f"{now.year-1}-{str(now.year)[2:]}"
    p = get_player(player_name)
    time.sleep(0.6)
    log = playergamelog.PlayerGameLog(player_id=p["id"], season=season)
    data = log.get_normalized_dict()
    return {"game_log": [enrich(r) for r in data["PlayerGameLog"]]}


@app.get("/league/leaders")
def league_leaders(
    season: str = Query(None),
    stat_category: str = Query("PTS"),
    top: int = Query(15, ge=1, le=50),
):
    from nba_api.stats.endpoints import leagueleaders
    if not season:
        now = datetime.now()
        season = f"{now.year}-{str(now.year+1)[2:]}" if now.month >= 10 else f"{now.year-1}-{str(now.year)[2:]}"
    time.sleep(0.6)
    # Percentage categories don't work with PerGame — use Totals for those
    pct_categories = {"FG_PCT", "FG3_PCT", "FT_PCT"}
    per_mode = "Totals" if stat_category in pct_categories else "PerGame"
    leaders = leagueleaders.LeagueLeaders(
        season=season,
        stat_category_abbreviation=stat_category,
        per_mode48=per_mode,
    )
    data = leaders.get_normalized_dict()
    return {"leaders": [enrich(r) for r in data["LeagueLeaders"][:top]]}


# ── ML Helpers ────────────────────────────────────────────────────────────────

ODDS_TEAM_MAP = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "LA Clippers": "LAC", "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM", "Miami Heat": "MIA", "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NOP", "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX", "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS", "Toronto Raptors": "TOR", "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}


def get_gamelog_flat(player_id: int, season: str) -> list[dict]:
    from nba_api.stats.endpoints import playergamelog
    time.sleep(0.6)
    log = playergamelog.PlayerGameLog(player_id=player_id, season=season)
    return log.get_normalized_dict()["PlayerGameLog"]


def get_playoff_gamelog(player_id: int, season: str) -> list[dict]:
    """Fetch playoff game log specifically."""
    from nba_api.stats.endpoints import playergamelog
    try:
        time.sleep(0.6)
        log = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=season,
            season_type_all_star="Playoffs",
        )
        return log.get_normalized_dict()["PlayerGameLog"]
    except Exception:
        return []


def get_playoff_series_games(playoff_games: list[dict], opponent_abbr: str) -> dict:
    """
    From playoff game log, extract games against the current opponent (this series).
    Returns series context: games played, player's stats in series, series game number,
    must-win flag (elimination game).
    """
    if not playoff_games or not opponent_abbr:
        return {}

    # Filter to games against this opponent in playoffs
    series_games = [g for g in playoff_games if opponent_abbr in g.get("MATCHUP", "")]
    series_games = sorted(series_games, key=lambda x: x.get("GAME_DATE", ""))

    if not series_games:
        return {}

    game_num = len(series_games) + 1  # next game number in series

    # Count wins/losses in series
    wins   = sum(1 for g in series_games if g.get("WL") == "W")
    losses = sum(1 for g in series_games if g.get("WL") == "L")

    # Elimination game detection: team is down 3-1 (must win) or 3-2, or opponent is
    is_must_win   = losses == 3 or (wins == 3 and losses <= 2)  # down 3-x or up 3-x (close out)
    is_close_out  = wins == 3  # player's team can close out
    is_elimination = losses == 3  # player's team faces elimination

    # Per-series averages
    keys = ["PTS","REB","AST","STL","BLK","FGM","FGA","FTM","FTA","FG3M","MIN"]
    series_avgs = {}
    for k in keys:
        vals = [g.get(k) or 0 for g in series_games]
        series_avgs[k] = round(np.mean(vals), 1) if vals else 0

    # Trend within series: last 2 games vs series avg (is player heating up?)
    recent_series = series_games[-2:] if len(series_games) >= 2 else series_games
    series_trend = {}
    for k in ["PTS","REB","AST"]:
        recent_avg = np.mean([g.get(k) or 0 for g in recent_series])
        series_avg = series_avgs.get(k, 0)
        series_trend[k] = round(recent_avg - series_avg, 1)

    return {
        "games_played":    len(series_games),
        "game_number":     game_num,
        "wins":            wins,
        "losses":          losses,
        "is_must_win":     is_must_win,
        "is_close_out":    is_close_out,
        "is_elimination":  is_elimination,
        "series_avgs":     series_avgs,
        "series_trend":    series_trend,
        "series_games":    series_games,
    }


def get_opponent_defense(season: str) -> dict:
    from nba_api.stats.endpoints import leaguedashteamstats
    try:
        time.sleep(0.6)
        stats = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            measure_type_detailed_defense="Opponent",
            per_mode_simple="PerGame",
        )
        defense = {}
        for row in stats.get_normalized_dict().get("LeagueDashTeamStats", []):
            abbr = row.get("TEAM_ABBREVIATION")
            if abbr:
                defense[abbr] = {
                    "opp_pts":  row.get("OPP_PTS") or 110,
                    "opp_reb":  row.get("OPP_REB") or 44,
                    "opp_ast":  row.get("OPP_AST") or 25,
                    "opp_fg3m": row.get("OPP_FG3M") or 12,
                    "opp_stl":  row.get("OPP_STL") or 7,
                    "opp_blk":  row.get("OPP_BLK") or 5,
                    "opp_pace": row.get("OPP_PACE") or 100,
                }
        return defense
    except Exception:
        return {}


def get_h2h_stats(games: list[dict], opp_abbr: str) -> dict:
    h2h = [g for g in games if opp_abbr in g.get("MATCHUP", "")]
    if not h2h:
        return {"games": 0}
    result = {"games": len(h2h)}
    for k in ["PTS","REB","AST","STL","BLK"]:
        vals = [g.get(k) or 0 for g in h2h]
        result[f"h2h_{k.lower()}"] = round(np.mean(vals), 1)
    return result


def days_rest(games: list[dict], idx: int) -> int:
    """Compute rest days before game at idx."""
    if idx == 0:
        return 3
    try:
        from datetime import datetime
        d1 = datetime.strptime(games[idx - 1]["GAME_DATE"], "%b %d, %Y")
        d2 = datetime.strptime(games[idx]["GAME_DATE"], "%b %d, %Y")
        return max((d2 - d1).days - 1, 0)
    except Exception:
        return 1


def get_usage_trend(player_id: int, season: str) -> dict:
    """
    Pull player usage % and usage trend over last 10 vs last 30 games.
    Positive trend = coach giving player more plays = bullish signal.
    """
    try:
        from nba_api.stats.endpoints import playerdashboardbylastngames
        time.sleep(0.6)
        dash = playerdashboardbylastngames.PlayerDashboardByLastNGames(
            player_id=player_id,
            season=season,
            per_mode_simple="PerGame",
            measure_type_detailed_defense="Base",
        )
        data = dash.get_normalized_dict()
        rows = data.get("LastNGamesPlayerDashboard", [])
        # rows are grouped: Last5, Last10, Last15, Last20, Last25, SeasonToDate
        usage = {}
        for row in rows:
            group = row.get("GROUP_VALUE", "")
            usg = row.get("USG_PCT") or 0
            usage[group] = {
                "usg_pct": round(usg * 100, 1),
                "pts": row.get("PTS") or 0,
                "reb": row.get("REB") or 0,
                "ast": row.get("AST") or 0,
                "min": row.get("MIN") or 0,
            }
        # Compute trend: Last10 usage vs SeasonToDate usage
        l10_usg  = usage.get("Last 10", {}).get("usg_pct", 0)
        seas_usg = usage.get("Season To Date", {}).get("usg_pct", 0)
        trend = round(l10_usg - seas_usg, 1)  # positive = trending up
        return {
            "last10": usage.get("Last 10", {}),
            "season": usage.get("Season To Date", {}),
            "usg_trend": trend,
            "trending_up": trend > 2.0,
        }
    except Exception:
        return {"usg_trend": 0, "trending_up": False, "last10": {}, "season": {}}


def get_opponent_recent_defense(opp_abbr: str, season: str, last_n: int = 10) -> dict:
    """
    Get opponent's defensive stats over their last N games only (not full season).
    A tired/injured defense shows up here but not in season averages.
    """
    try:
        from nba_api.stats.endpoints import teamgamelog
        team = TEAMS_BY_ABBR.get(opp_abbr)
        if not team:
            return {}
        time.sleep(0.6)
        log = teamgamelog.TeamGameLog(
            team_id=team["id"],
            season=season,
            season_type_all_star="Regular Season",
        )
        rows = log.get_normalized_dict().get("TeamGameLog", [])
        recent = rows[:last_n]  # most recent N games first
        if not recent:
            return {}

        # Points, reb, ast allowed = opponent stats in team game log
        # We need to look at PTS allowed — nba_api TeamGameLog has PTS (scored by team)
        # We approximate defensive form by looking at +/- and point differential
        pts_scored  = [r.get("PTS") or 0 for r in recent]
        plus_minus  = [r.get("PLUS_MINUS") or 0 for r in recent]
        pts_allowed = [s - pm for s, pm in zip(pts_scored, plus_minus)]

        wins = sum(1 for r in recent if r.get("WL") == "W")
        return {
            "recent_pts_allowed_avg": round(np.mean(pts_allowed), 1),
            "recent_wins": wins,
            "recent_losses": last_n - wins,
            "recent_form_score": round(np.mean(plus_minus), 1),  # positive = good defense recently
            "games": last_n,
        }
    except Exception:
        return {}


def get_opponent_rest(opp_abbr: str, season: str) -> int:
    """
    Get rest days for the opponent team — if they're on a back-to-back
    their defense is slower which benefits the target player.
    """
    try:
        from nba_api.stats.endpoints import teamgamelog
        team = TEAMS_BY_ABBR.get(opp_abbr)
        if not team:
            return 2
        time.sleep(0.6)
        log = teamgamelog.TeamGameLog(
            team_id=team["id"],
            season=season,
            season_type_all_star="Regular Season",
        )
        rows = log.get_normalized_dict().get("TeamGameLog", [])
        if len(rows) < 2:
            return 2
        # rows[0] = most recent game
        try:
            d1 = datetime.strptime(rows[1]["GAME_DATE"], "%b %d, %Y")
            d2 = datetime.strptime(rows[0]["GAME_DATE"], "%b %d, %Y")
            return max((d2 - d1).days - 1, 0)
        except Exception:
            return 2
    except Exception:
        return 2


def get_clutch_stats(player_id: int, season: str) -> dict:
    """
    Pull player's clutch stats (last 5 min, within 5 pts).
    High clutch performers elevate in playoffs — important context.
    """
    try:
        from nba_api.stats.endpoints import leaguedashplayerclutch
        time.sleep(0.6)
        clutch = leaguedashplayerclutch.LeagueDashPlayerClutch(
            season=season,
            per_mode_simple="PerGame",
        )
        rows = clutch.get_normalized_dict().get("LeagueDashPlayerClutch", [])
        player_row = next((r for r in rows if r.get("PLAYER_ID") == player_id), None)
        if not player_row:
            return {}
        return {
            "clutch_pts": player_row.get("PTS") or 0,
            "clutch_usg": round((player_row.get("USG_PCT") or 0) * 100, 1),
            "clutch_plus_minus": player_row.get("PLUS_MINUS") or 0,
            "clutch_gp": player_row.get("GP") or 0,
        }
    except Exception:
        return {}


def get_travel_fatigue(player_team_abbr: str, opp_abbr: str) -> float:
    """
    Estimate travel fatigue based on team locations.
    Cross-country travel (3+ time zones) = fatigue penalty.
    Denver altitude = additional fatigue for visiting team.
    Returns a fatigue score: 0 = no fatigue, 1 = mild, 2 = severe.
    """
    TIMEZONES = {
        "BOS":1,"NYK":1,"BKN":1,"PHI":1,"TOR":1,"MIA":1,"ORL":1,"ATL":1,
        "CHA":1,"WAS":1,"CLE":1,"DET":1,"IND":1,"CHI":2,"MIL":2,"MIN":2,
        "MEM":2,"NOP":2,"SAS":2,"HOU":2,"DAL":2,"OKC":2,"DEN":3,"UTA":3,
        "POR":4,"GSW":4,"LAL":4,"LAC":4,"SAC":4,"PHX":3,"OKC":2,
    }
    HIGH_ALTITUDE = {"DEN"}
    player_tz  = TIMEZONES.get(player_team_abbr, 2)
    opp_tz     = TIMEZONES.get(opp_abbr, 2)
    tz_diff    = abs(player_tz - opp_tz)
    fatigue    = 0.0
    if tz_diff >= 3:
        fatigue += 2.0  # cross country
    elif tz_diff == 2:
        fatigue += 1.0
    if opp_abbr in HIGH_ALTITUDE:
        fatigue += 1.5  # Denver altitude
    return fatigue


def get_espn_lineup_news(player_team_abbr: str) -> list[dict]:
    """
    Pull ESPN lineup/injury news for a team — free, no auth needed.
    Returns list of {player, status, description} for relevant players.
    """
    TEAM_ESPN_IDS = {
        "ATL":"1","BOS":"2","BKN":"17","CHA":"30","CHI":"4","CLE":"5",
        "DAL":"6","DEN":"7","DET":"8","GSW":"9","HOU":"10","IND":"11",
        "LAC":"12","LAL":"13","MEM":"29","MIA":"14","MIL":"15","MIN":"16",
        "NOP":"3","NYK":"18","OKC":"25","ORL":"19","PHI":"20","PHX":"21",
        "POR":"22","SAC":"23","SAS":"24","TOR":"28","UTA":"26","WAS":"27",
    }
    try:
        team_id = TEAM_ESPN_IDS.get(player_team_abbr)
        if not team_id:
            return []
        r = requests.get(
            f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{team_id}/injuries",
            timeout=8,
        )
        if not r.ok:
            return []
        data = r.json()
        news = []
        for item in data.get("injuries", []):
            news.append({
                "player": item.get("athlete", {}).get("displayName", ""),
                "status": item.get("status", ""),
                "description": item.get("shortComment", ""),
            })
        return news
    except Exception:
        return []


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
        """Parse MIN which can be int, float, or 'MM:SS' string."""
        if not m:
            return 0.0
        try:
            if isinstance(m, str) and ":" in m:
                parts = m.split(":")
                return float(parts[0]) + float(parts[1]) / 60
            return float(m)
        except Exception:
            return 0.0

    # Minutes for each game
    recent_mins = [parse_min(g.get("MIN")) for g in recent]
    all_mins    = [parse_min(g.get("MIN")) for g in all_games]
    avg_recent_min = np.mean(recent_mins) if recent_mins else 32.0

    # 1. L10 raw mean + std (20)
    for k in keys:
        vals = [g.get(k) or 0 for g in recent]
        feat.append(np.mean(vals))
        feat.append(np.std(vals))

    # 2. L10 per-36 minute rates (10) — neutralizes low-minute games
    for k in keys:
        per36_vals = []
        for g, m in zip(recent, recent_mins):
            if m > 5:  # ignore garbage time (<5 min)
                per36_vals.append((g.get(k) or 0) / m * 36)
        feat.append(np.mean(per36_vals) if per36_vals else 0)

    # 3. Minutes-weighted averages L10 (10) — high-min games count more
    for k in keys:
        total_w = sum(recent_mins)
        if total_w > 0:
            w_avg = sum((g.get(k) or 0) * m for g, m in zip(recent, recent_mins)) / total_w
        else:
            w_avg = 0
        feat.append(w_avg)

    # 4. L5 mean (10)
    last5 = recent[-5:] if len(recent) >= 5 else recent
    for k in keys:
        vals = [g.get(k) or 0 for g in last5]
        feat.append(np.mean(vals))

    # 5. Season mean (10)
    for k in keys:
        vals = [g.get(k) or 0 for g in all_games]
        feat.append(np.mean(vals) if vals else 0)

    # 6. Season per-36 rates (10)
    for k in keys:
        per36_vals = []
        for g, m in zip(all_games, all_mins):
            if m > 5:
                per36_vals.append((g.get(k) or 0) / m * 36)
        feat.append(np.mean(per36_vals) if per36_vals else 0)

    # 7. Home/away split (10)
    home_games = [g for g in all_games if "vs." in g.get("MATCHUP", "")]
    away_games = [g for g in all_games if " @ " in g.get("MATCHUP", "")]
    split = home_games if is_home else away_games
    for k in keys:
        vals = [g.get(k) or 0 for g in split]
        feat.append(np.mean(vals) if vals else 0)

    # 8. Opponent defense (7)
    feat.append(opp_defense.get("opp_pts", 110))
    feat.append(opp_defense.get("opp_reb", 44))
    feat.append(opp_defense.get("opp_ast", 25))
    feat.append(opp_defense.get("opp_fg3m", 12))
    feat.append(opp_defense.get("opp_stl", 7))
    feat.append(opp_defense.get("opp_blk", 5))
    feat.append(opp_defense.get("opp_pace", 100))

    # 9. H2H (5)
    for k in ["pts","reb","ast","stl","blk"]:
        feat.append(h2h.get(f"h2h_{k}", 0))

    # 10. Game context (4)
    feat.append(float(is_home))
    feat.append(float(min(rest, 5)))
    feat.append(float(expected_min))          # expected minutes next game
    feat.append(float(is_playoffs))           # playoff games = higher intensity

    # 11. Minutes context (2)
    feat.append(float(avg_recent_min))        # recent avg minutes
    feat.append(float(expected_min - avg_recent_min))  # minutes bump/drop vs recent

    # 12. Teammate absence bump (3)
    feat.append(float(pts_bump))
    feat.append(float(ast_bump))
    feat.append(float(reb_bump))

    # 13. Referee foul rate (1)
    feat.append(float(ref_fta_rate))

    # 14. Shot profile matchup edge (1)
    feat.append(float(shot_profile_score))

    # 15. Usage trend (2) — is player getting more/less plays recently
    feat.append(float(usg_trend))           # L10 usage % minus season usage %
    feat.append(float(clutch_usg))          # clutch usage % — elevates in playoffs

    # 16. Opponent context (3)
    feat.append(float(min(opp_rest, 5)))    # opponent rest days — 0 = back-to-back (easy defense)
    feat.append(float(opp_recent_form))     # opponent recent +/- (negative = struggling defense)
    feat.append(float(travel_fatigue))      # player travel fatigue (altitude, time zones)

    # 16. Playoff series context (7) — only non-zero during playoffs
    feat.append(float(series_games_played))   # games played in this series (0-6)
    feat.append(float(series_pts_avg))        # player's avg PTS in this series
    feat.append(float(series_reb_avg))        # player's avg REB in this series
    feat.append(float(series_ast_avg))        # player's avg AST in this series
    feat.append(float(series_weight))         # how much to weight series (0-1)
    feat.append(float(is_elimination))        # elimination game = higher intensity
    feat.append(float(is_close_out))          # close-out game = also high stakes

    return feat


def train_predict(
    games: list[dict],
    target: str,
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
    series_context: dict = None,
) -> dict:
    from sklearn.ensemble import GradientBoostingRegressor

    if series_context is None:
        series_context = {}

    # Extract series info
    sc = series_context
    series_games_played = sc.get("games_played", 0)
    series_avgs  = sc.get("series_avgs", {})
    series_pts   = series_avgs.get("PTS", 0)
    series_reb   = series_avgs.get("REB", 0)
    series_ast   = series_avgs.get("AST", 0)
    is_elim      = sc.get("is_elimination", False)
    is_closeout  = sc.get("is_close_out", False)

    # Series weight: grows with games played (0 games=0, 4+ games=0.8 max)
    # Once 3+ series games played, series stats matter a lot more than reg season
    series_weight = min(series_games_played / 5.0, 0.8) if series_games_played >= 2 else 0

    games = sorted(games, key=lambda x: x.get("GAME_DATE", ""))

    # If in playoffs with series data, prepend synthetic "series average" games
    # to boost their weight in training relative to regular season
    augmented_games = list(games)
    if series_weight > 0 and sc.get("series_games"):
        # Add series games with 3x weight by repeating them
        repeats = max(1, int(series_games_played * 2))
        for _ in range(repeats):
            augmented_games.extend(sc["series_games"])
        augmented_games = sorted(augmented_games, key=lambda x: x.get("GAME_DATE", ""))
    games = sorted(games, key=lambda x: x.get("GAME_DATE", ""))
    if len(games) < 10:
        return {"error": "Not enough games to predict (need at least 10)"}

    # Adjust window for rookies with fewer games
    window = min(10, max(5, len(augmented_games) // 3))
    X, y = [], []

    for i in range(window, len(augmented_games)):
        recent = augmented_games[i - window:i]
        r = days_rest(augmented_games, i)
        home = "vs." in augmented_games[i].get("MATCHUP", "")
        feat = build_features(recent, augmented_games[:i], {}, {}, home, r,
                              expected_min=32.0, is_playoffs=False,
                              usg_trend=0, opp_rest=2, opp_recent_form=0,
                              travel_fatigue=0, clutch_usg=0,
                              series_games_played=0, series_pts_avg=0,
                              series_reb_avg=0, series_ast_avg=0,
                              series_weight=0, is_elimination=False, is_close_out=False)
        X.append(feat)
        y.append(augmented_games[i].get(target) or 0)

    if len(X) < 5:
        return {"error": "Not enough games to predict (need at least 10)"}

    X, y = np.array(X), np.array(y)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # Gradient Boosting — more accurate than RandomForest for tabular sports data
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        random_state=42,
    )
    model.fit(Xs, y)

    # Also train RF for uncertainty estimation (std across trees)
    rf = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
    rf.fit(Xs, y)

    # Predict with full context
    recent = games[-window:]
    feat = build_features(recent, games, opp_defense, h2h, is_home, rest,
                          pts_bump, ast_bump, reb_bump, ref_fta_rate,
                          shot_profile_score, expected_min, is_playoffs,
                          usg_trend, opp_rest, opp_recent_form,
                          travel_fatigue, clutch_usg,
                          series_games_played, series_pts, series_reb, series_ast,
                          series_weight, is_elim, is_closeout)
    feat_scaled = scaler.transform([feat])

    pred = model.predict(feat_scaled)[0]
    tree_preds = [t.predict(feat_scaled)[0] for t in rf.estimators_]
    std = np.std(tree_preds)

    recent_avg = np.mean([g.get(target) or 0 for g in recent])
    season_avg = np.mean([g.get(target) or 0 for g in games])

    # Home/away avg
    home_games = [g for g in games if "vs." in g.get("MATCHUP", "")]
    away_games = [g for g in games if " @ " in g.get("MATCHUP", "")]
    split = home_games if is_home else away_games
    split_avg = round(np.mean([g.get(target) or 0 for g in split]), 1) if split else None

    return {
        "prediction": round(float(pred), 1),
        "std_dev": round(float(std), 2),
        "recent_avg_10": round(float(recent_avg), 1),
        "season_avg": round(float(season_avg), 1),
        "home_away_avg": split_avg,
        "games_used": len(augmented_games),
        "series_weight_applied": round(series_weight, 2),
    }


@app.get("/predict/{player_name}")
def predict_player(
    player_name: str,
    season: str = Query(None),
):
    p = get_player(player_name)
    pid = p["id"]

    # Auto-detect current season based on today's date
    # NBA season starts in October — if month >= 10, current season is this year
    if not season:
        now = datetime.now()
        if now.month >= 10:
            season = f"{now.year}-{str(now.year + 1)[2:]}"
        else:
            season = f"{now.year - 1}-{str(now.year)[2:]}"

    # ── 1. Fetch 3 seasons of game logs ──────────────────────────────────────
    try:
        games = get_gamelog_flat(pid, season)
    except Exception as e:
        games = []
    all_games = list(games)
    for offset in [1, 2]:
        y_off = int(season[:4]) - offset
        prev = f"{y_off}-{str(y_off+1)[2:]}"
        try:
            all_games = get_gamelog_flat(pid, prev) + all_games
        except Exception:
            pass

    if not all_games:
        raise HTTPException(
            status_code=500,
            detail=f"Could not fetch game log for {player_name} in {season}. NBA.com may be rate limiting — try again in 30 seconds."
        )

    # ── 1b. Playoff series context ────────────────────────────────────────────
    playoff_games = []
    series_context = {}
    if is_playoffs if False else True:  # always try — we detect playoffs from data
        try:
            playoff_games = get_playoff_gamelog(pid, season)
        except Exception:
            playoff_games = []

    # ── 2. Fetch today's games + props in ONE bulk call ───────────────────────
    props_found = {}
    opponent_abbr = None
    opponent_name = None
    today_game = None
    is_home = False
    today_event_id = None

    try:
        player_last  = p["full_name"].split()[-1].lower()
        player_first = p["full_name"].split()[0].lower()
        market_keys  = "player_points,player_rebounds,player_assists,player_steals,player_blocks"
        stat_map = {
            "player_points": "PTS", "player_rebounds": "REB",
            "player_assists": "AST", "player_steals": "STL", "player_blocks": "BLK",
        }

        # Fetch events list (1 call), then per-event odds (1 call per game)
        events_r = requests.get(
            f"{ODDS_BASE}/sports/basketball_nba/events",
            params={"apiKey": ODDS_KEY},
            timeout=10,
        )
        events = events_r.json() if events_r.ok else []

        for event in (events if isinstance(events, list) else []):
            event_id = event.get("id")
            home = event.get("home_team", "")
            away = event.get("away_team", "")
            if not event_id:
                continue
            player_found_in_event = False
            try:
                odds_r = requests.get(
                    f"{ODDS_BASE}/sports/basketball_nba/events/{event_id}/odds",
                    params={"apiKey": ODDS_KEY, "regions": "us", "markets": market_keys, "oddsFormat": "american", "bookmakers": "draftkings,fanduel,betmgm"},
                    timeout=10,
                )
                if not odds_r.ok:
                    continue
                odds = odds_r.json()
                for bm in odds.get("bookmakers", []):
                    for mkt in bm.get("markets", []):
                        stat = stat_map.get(mkt.get("key", ""))
                        if not stat:
                            continue
                        for outcome in mkt.get("outcomes", []):
                            desc = outcome.get("description", "").lower()
                            if player_last in desc or player_first in desc:
                                player_found_in_event = True
                                if stat not in props_found and outcome.get("name") == "Over":
                                    props_found[stat] = outcome.get("point")
            except Exception:
                continue

            if player_found_in_event and not opponent_abbr:
                today_event_id = event_id
                today_game = {"home": home, "away": away}
                if all_games:
                    recent_matchup = all_games[-1].get("MATCHUP", "")
                    player_team_abbr = recent_matchup.split()[0] if recent_matchup else ""
                    home_abbr = ODDS_TEAM_MAP.get(home, "")
                    away_abbr = ODDS_TEAM_MAP.get(away, "")
                    if player_team_abbr == home_abbr:
                        opponent_abbr = away_abbr
                        opponent_name = away
                        is_home = True
                    else:
                        opponent_abbr = home_abbr
                        opponent_name = home
                        is_home = False
    except Exception:
        pass

    # ── 3. Opponent defense + H2H + Series context ────────────────────────────
    opp_defense = {}
    h2h = {}
    series_context = {}
    if opponent_abbr:
        all_defense = get_opponent_defense(season)
        opp_defense = all_defense.get(opponent_abbr, {})
        h2h = get_h2h_stats(all_games, opponent_abbr)
        # Build playoff series context now that we know opponent
        if playoff_games:
            series_context = get_playoff_series_games(playoff_games, opponent_abbr)

    # ── 4. Rest days ──────────────────────────────────────────────────────────
    sorted_games = sorted(all_games, key=lambda x: x.get("GAME_DATE", ""))
    rest = days_rest(sorted_games, len(sorted_games) - 1) if len(sorted_games) > 1 else 1

    # ── 4b. Expected minutes + playoff detection ───────────────────────────────
    def parse_min(m) -> float:
        if not m: return 0.0
        try:
            if isinstance(m, str) and ":" in m:
                p = m.split(":")
                return float(p[0]) + float(p[1]) / 60
            return float(m)
        except Exception:
            return 0.0

    recent_10 = sorted_games[-10:]
    recent_mins = [parse_min(g.get("MIN")) for g in recent_10 if parse_min(g.get("MIN")) > 5]
    expected_min = round(np.mean(recent_mins), 1) if recent_mins else 32.0

    # Detect playoffs: SEASON_ID starts with "4" for playoffs (e.g. "42025")
    is_playoffs = any(
        str(g.get("SEASON_ID", "")).startswith("4") for g in sorted_games[-5:]
    )

    # ── 5. Injury report + teammate absence bump ──────────────────────────────
    injury_report = get_league_injury_report()
    player_team_abbr = sorted_games[-1].get("MATCHUP", "").split()[0] if sorted_games else ""
    teammate_context = {}
    pts_bump = ast_bump = reb_bump = 0
    if player_team_abbr:
        teammate_context = get_teammate_usage(p["full_name"], player_team_abbr, season, injury_report)
        pts_bump = teammate_context.get("estimated_pts_bump", 0)
        ast_bump = teammate_context.get("estimated_ast_bump", 0)
        reb_bump = teammate_context.get("estimated_reb_bump", 0)

    # Filter injury report to player's team and opponent
    relevant_injuries = [
        i for i in injury_report
        if player_team_abbr.lower() in i.get("team", "").lower()
        or (opponent_name or "").lower() in i.get("team", "").lower()
    ]

    # ── 6. Referee foul rate ──────────────────────────────────────────────────
    ref_stats = get_referee_stats(season)
    ref_fta_rate = np.mean([v.get("fta_per_game", 0) for v in ref_stats.values()]) if ref_stats else 0

    # ── 7. Usage trend ────────────────────────────────────────────────────────
    usage_data = get_usage_trend(pid, season)
    usg_trend = usage_data.get("usg_trend", 0)

    # ── 8. Clutch stats ───────────────────────────────────────────────────────
    clutch_data = get_clutch_stats(pid, season)
    clutch_usg = clutch_data.get("clutch_usg", 0)

    # ── 9. Opponent rest + recent defensive form ──────────────────────────────
    opp_rest = 2
    opp_recent_defense = {}
    opp_recent_form = 0
    if opponent_abbr:
        opp_rest = get_opponent_rest(opponent_abbr, season)
        opp_recent_defense = get_opponent_recent_defense(opponent_abbr, season)
        opp_recent_form = opp_recent_defense.get("recent_form_score", 0)

    # ── 10. Travel fatigue ────────────────────────────────────────────────────
    travel_fatigue = 0.0
    if player_team_abbr and opponent_abbr:
        travel_fatigue = get_travel_fatigue(player_team_abbr, opponent_abbr)

    # ── 11. ESPN lineup news ──────────────────────────────────────────────────
    lineup_news = []
    if player_team_abbr:
        lineup_news = get_espn_lineup_news(player_team_abbr)

    # ── 12. Shot profile matchup ──────────────────────────────────────────────
    player_shot_profile = get_player_shot_profile(pid, season)
    opp_shot_defense = {}
    shot_profile_edge = {"score": 0, "interpretation": "UNKNOWN", "zones": {}}
    if opponent_abbr and player_shot_profile:
        opp_shot_defense = get_opponent_shot_defense(opponent_abbr, season)
        shot_profile_edge = compute_shot_profile_edge(player_shot_profile, opp_shot_defense)
    shot_profile_score = shot_profile_edge.get("score", 0)

    # ── 13. Line movement (disabled to preserve API quota) ───────────────────
    line_movement = {}

    # ── 14. Train & predict ───────────────────────────────────────────────────
    targets = ["PTS", "REB", "AST", "STL", "BLK"]
    predictions = {}
    for stat in targets:
        ml = train_predict(
            all_games, stat, opp_defense, h2h,
            is_home, rest, pts_bump, ast_bump, reb_bump, ref_fta_rate,
            shot_profile_score, expected_min, is_playoffs,
            usg_trend, opp_rest, opp_recent_form, travel_fatigue, clutch_usg,
            series_context,
        )
        line = props_found.get(stat)
        movement = line_movement.get(stat, {})
        entry = {
            "stat": stat,
            "line": line,
            "ml_prediction": ml.get("prediction"),
            "std_dev": ml.get("std_dev"),
            "recent_avg_10": ml.get("recent_avg_10"),
            "season_avg": ml.get("season_avg"),
            "home_away_avg": ml.get("home_away_avg"),
            "games_used": ml.get("games_used"),
            "error": ml.get("error"),
            "line_movement": movement,
        }
        if line is not None and ml.get("prediction") is not None:
            diff = ml["prediction"] - line
            std = ml.get("std_dev") or 0.01
            confidence = round(min(abs(diff) / (std + 0.01) * 33.3, 99), 0)
            entry["recommendation"] = "OVER" if diff > 0 else "UNDER"
            entry["edge"] = round(abs(diff), 1)
            entry["confidence"] = confidence
        else:
            entry["recommendation"] = "NO LINE"
            entry["edge"] = None
            entry["confidence"] = None
        predictions[stat] = entry

    # ── 15. Combos ────────────────────────────────────────────────────────────
    combos = {}
    for combo_name, stats in [("PR", ["PTS","REB"]), ("PA", ["PTS","AST"]), ("PRA", ["PTS","REB","AST"]), ("RA", ["REB","AST"])]:
        total_pred = sum(predictions[s]["ml_prediction"] or 0 for s in stats)
        total_line = None
        if all(props_found.get(s) for s in stats):
            total_line = round(sum(props_found[s] for s in stats), 1)
        entry = {
            "stat": combo_name,
            "ml_prediction": round(total_pred, 1),
            "line": total_line,
        }
        if total_line:
            diff = total_pred - total_line
            entry["recommendation"] = "OVER" if diff > 0 else "UNDER"
            entry["edge"] = round(abs(diff), 1)
        else:
            entry["recommendation"] = "NO LINE"
        combos[combo_name] = entry

    return {
        "player": p,
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
        "usage_data": usage_data,
        "clutch_data": clutch_data,
        "opponent_defense": opp_defense,
        "opp_recent_defense": opp_recent_defense,
        "h2h": h2h,
        "series_context": {k: v for k, v in series_context.items() if k != "series_games"},
        "shot_profile_edge": shot_profile_edge,
        "teammate_context": teammate_context,
        "relevant_injuries": relevant_injuries[:10],
        "lineup_news": lineup_news[:8],
        "predictions": predictions,
        "combos": combos,
        "props_found": len(props_found),
        "note": "GradientBoosting · 100+ features: per-36 rates, usage trend, clutch stats, opp rest, travel fatigue, recent opp form, lineup news, shot profile, H2H, injuries · Lines from The Odds API",
    }

@app.get("/odds/status")
def odds_status():
    """Check Odds API quota remaining."""
    try:
        r = requests.get(
            f"{ODDS_BASE}/sports/basketball_nba/events",
            params={"apiKey": ODDS_KEY},
            timeout=10,
        )
        remaining = r.headers.get("x-requests-remaining", "unknown")
        used = r.headers.get("x-requests-used", "unknown")
        return {
            "status": r.status_code,
            "requests_remaining": remaining,
            "requests_used": used,
            "events_found": len(r.json()) if r.ok else 0,
        }
    except Exception as e:
        return {"error": str(e)}

# In-memory cache so we don't re-fetch defense + game logs on every refresh
_picks_cache: dict = {}

def fast_predict(all_games: list, stat: str, opp_defense: dict, h2h: dict,
                 is_home: bool, rest: int, expected_min: float,
                 is_playoffs: bool, opp_rest: int, travel_fatigue: float,
                 series_context: dict) -> dict | None:
    """
    Lightweight prediction for bulk scanning — uses RandomForest only with
    core features (no NBA.com calls for shot profile, usage trend, clutch etc).
    Much faster than full train_predict.
    """
    if len(all_games) < 10:
        return None

    def parse_min(m) -> float:
        if not m: return 0.0
        try:
            if isinstance(m, str) and ":" in m:
                p = m.split(":")
                return float(p[0]) + float(p[1]) / 60
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
        # L10 mean + std
        for k in keys:
            vals = [g.get(k) or 0 for g in recent]
            feat.append(np.mean(vals))
            feat.append(np.std(vals))
        # Per-36 rates
        for k in keys:
            pr36 = []
            for g, m in zip(recent, recent_mins):
                if m > 5: pr36.append((g.get(k) or 0) / m * 36)
            feat.append(np.mean(pr36) if pr36 else 0)
        # Opp defense
        feat += [
            opp_defense.get("opp_pts", 110), opp_defense.get("opp_reb", 44),
            opp_defense.get("opp_ast", 25),  opp_defense.get("opp_fg3m", 12),
            opp_defense.get("opp_pace", 100),
        ]
        # H2H
        for k in ["pts","reb","ast","stl","blk"]:
            feat.append(h2h.get(f"h2h_{k}", 0))
        # Context
        feat += [float(is_home), float(min(rest, 5)), float(expected_min),
                 float(is_playoffs), float(min(opp_rest, 5)), float(travel_fatigue)]
        # Series
        sc = series_context or {}
        feat += [float(sc.get("games_played", 0)),
                 float(sc.get("series_avgs", {}).get(stat, 0)),
                 float(sc.get("is_elimination", False))]

        X.append(feat)
        y.append(games[i].get(stat) or 0)

    if len(X) < 5:
        return None

    try:
        X, y = np.array(X), np.array(y)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        model = RandomForestRegressor(n_estimators=80, random_state=42, n_jobs=-1)
        model.fit(Xs, y)

        recent = games[-window:]
        recent_mins = [parse_min(g.get("MIN")) for g in recent]
        feat = []
        for k in keys:
            vals = [g.get(k) or 0 for g in recent]
            feat.append(np.mean(vals))
            feat.append(np.std(vals))
        for k in keys:
            pr36 = []
            for g, m in zip(recent, recent_mins):
                if m > 5: pr36.append((g.get(k) or 0) / m * 36)
            feat.append(np.mean(pr36) if pr36 else 0)
        feat += [
            opp_defense.get("opp_pts", 110), opp_defense.get("opp_reb", 44),
            opp_defense.get("opp_ast", 25),  opp_defense.get("opp_fg3m", 12),
            opp_defense.get("opp_pace", 100),
        ]
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
        return {
            "prediction": round(float(pred), 1),
            "recent_avg": round(float(recent_avg), 1),
            "season_avg": round(float(season_avg), 1),
        }
    except Exception:
        return None


@app.get("/picks/today")
def best_picks_today(
    season: str = Query(None),
    top: int = Query(10, ge=1, le=30),
    min_edge: float = Query(1.5),
    force_refresh: bool = Query(False),
):
    if not season:
        now = datetime.now()
        season = f"{now.year}-{str(now.year+1)[2:]}" if now.month >= 10 else f"{now.year-1}-{str(now.year)[2:]}"

    cache_key = f"picks_{season}_{datetime.now().strftime('%Y%m%d')}"
    if not force_refresh and cache_key in _picks_cache:
        cached = _picks_cache[cache_key]
        filtered = [p for p in cached["all_picks"] if p["edge"] >= min_edge]
        filtered.sort(key=lambda x: x["edge"], reverse=True)
        return {**cached, "picks": filtered[:top], "min_edge": min_edge, "cached": True}

    # ── Step 1: Fetch all props in one pass ───────────────────────────────────
    all_props = {}
    player_event_map = {}
    try:
        # Fetch events first (1 call), then odds per event (1 call each)
        events_r = requests.get(
            f"{ODDS_BASE}/sports/basketball_nba/events",
            params={"apiKey": ODDS_KEY},
            timeout=10,
        )
        if not events_r.ok:
            raise HTTPException(status_code=events_r.status_code, detail=events_r.json().get("message", "Odds API error"))
        events = events_r.json()
        stat_map = {
            "player_points": "PTS", "player_rebounds": "REB",
            "player_assists": "AST", "player_steals": "STL", "player_blocks": "BLK",
        }
        market_keys = "player_points,player_rebounds,player_assists,player_steals,player_blocks"
        for event in (events if isinstance(events, list) else []):
            event_id = event.get("id")
            home = event.get("home_team", "")
            away = event.get("away_team", "")
            if not event_id: continue
            try:
                odds_r = requests.get(
                    f"{ODDS_BASE}/sports/basketball_nba/events/{event_id}/odds",
                    params={"apiKey": ODDS_KEY, "regions": "us", "markets": market_keys, "oddsFormat": "american", "bookmakers": "draftkings,fanduel,betmgm"},
                    timeout=10,
                )
                if not odds_r.ok: continue
                odds = odds_r.json()
                for bm in odds.get("bookmakers", []):
                    for mkt in bm.get("markets", []):
                        stat = stat_map.get(mkt.get("key",""))
                        if not stat: continue
                        for outcome in mkt.get("outcomes", []):
                            if outcome.get("name") != "Over": continue
                            pname = outcome.get("description","")
                            if not pname: continue
                            if pname not in all_props:
                                all_props[pname] = {}
                                player_event_map[pname] = {"event_id": event_id, "home": home, "away": away}
                            if stat not in all_props[pname]:
                                all_props[pname][stat] = outcome.get("point")
            except Exception:
                continue
    except Exception:
        raise HTTPException(status_code=500, detail="Could not fetch today's props")

    if not all_props:
        return {"picks": [], "total_players": 0, "message": "No props found for today"}

    # ── Step 2: Pre-fetch defense once for all teams (1 NBA.com call) ─────────
    all_defense = {}
    try:
        all_defense = get_opponent_defense(season)
    except Exception:
        pass

    # ── Step 3: Check for playoffs once ───────────────────────────────────────
    is_playoffs_global = False
    try:
        sample_player = list(all_props.keys())[0]
        sp = get_player(sample_player)
        sample_games = get_gamelog_flat(sp["id"], season)
        is_playoffs_global = any(str(g.get("SEASON_ID","")).startswith("4") for g in sample_games[:5])
    except Exception:
        pass

    # ── Step 4: Fast predictions — no sleep, minimal NBA.com calls ────────────
    picks = []
    gamelog_cache = {}  # cache game logs within this request

    def parse_min_local(m) -> float:
        if not m: return 0.0
        try:
            if isinstance(m, str) and ":" in m:
                parts = m.split(":")
                return float(parts[0]) + float(parts[1]) / 60
            return float(m)
        except: return 0.0

    for player_name, props in all_props.items():
        if not props: continue
        try:
            p = get_player(player_name)
            pid = p["id"]

            # Use cached game log if available
            if pid not in gamelog_cache:
                try:
                    games = get_gamelog_flat(pid, season)
                    all_games = list(games)
                    # Only fetch 1 extra season in fast mode
                    y_off = int(season[:4]) - 1
                    prev = f"{y_off}-{str(y_off+1)[2:]}"
                    try:
                        all_games = get_gamelog_flat(pid, prev) + all_games
                    except Exception:
                        pass
                    gamelog_cache[pid] = all_games
                except Exception:
                    continue
            else:
                all_games = gamelog_cache[pid]

            if len(all_games) < 10: continue

            sorted_games = sorted(all_games, key=lambda x: x.get("GAME_DATE",""))
            player_team_abbr = sorted_games[-1].get("MATCHUP","").split()[0] if sorted_games else ""
            event_info = player_event_map.get(player_name, {})
            home_team = event_info.get("home","")
            away_team = event_info.get("away","")
            home_abbr = ODDS_TEAM_MAP.get(home_team,"")
            away_abbr = ODDS_TEAM_MAP.get(away_team,"")
            is_home = player_team_abbr == home_abbr
            opponent_abbr = away_abbr if is_home else home_abbr

            opp_defense = all_defense.get(opponent_abbr, {})
            h2h = get_h2h_stats(all_games, opponent_abbr) if opponent_abbr else {}

            recent_mins = [parse_min_local(g.get("MIN")) for g in sorted_games[-10:] if parse_min_local(g.get("MIN")) > 5]
            expected_min = round(np.mean(recent_mins), 1) if recent_mins else 32.0
            is_playoffs = is_playoffs_global or any(str(g.get("SEASON_ID","")).startswith("4") for g in sorted_games[-5:])
            rest = days_rest(sorted_games, len(sorted_games)-1) if len(sorted_games) > 1 else 1
            opp_rest = 2  # skip slow opp rest call in fast mode
            travel_fatigue = get_travel_fatigue(player_team_abbr, opponent_abbr) if player_team_abbr and opponent_abbr else 0

            series_context = {}
            if is_playoffs and opponent_abbr:
                try:
                    poff = get_playoff_gamelog(pid, season)
                    series_context = get_playoff_series_games(poff, opponent_abbr)
                except Exception:
                    pass

            for stat, line in props.items():
                if line is None: continue
                try:
                    result = fast_predict(
                        all_games, stat, opp_defense, h2h,
                        is_home, rest, expected_min, is_playoffs,
                        opp_rest, travel_fatigue, series_context,
                    )
                    if not result: continue

                    pred = result["prediction"]
                    diff = pred - line
                    edge = round(abs(diff), 1)
                    if edge < 0.5: continue  # store all, filter by min_edge at end

                    picks.append({
                        "player":         p["full_name"],
                        "stat":           stat,
                        "line":           line,
                        "prediction":     pred,
                        "recommendation": "OVER" if diff > 0 else "UNDER",
                        "edge":           edge,
                        "recent_avg":     result["recent_avg"],
                        "season_avg":     result["season_avg"],
                        "matchup":        f"{away_team} @ {home_team}",
                        "is_playoffs":    is_playoffs,
                        "series_game":    series_context.get("game_number"),
                        "is_elimination": series_context.get("is_elimination", False),
                    })
                except Exception:
                    continue
        except Exception:
            continue

    picks.sort(key=lambda x: x["edge"], reverse=True)

    # Cache results for the day
    _picks_cache[cache_key] = {
        "all_picks": picks,
        "total_players": len(all_props),
        "total_picks_found": len([p for p in picks if p["edge"] >= min_edge]),
        "season": season,
        "cached": False,
    }

    filtered = [p for p in picks if p["edge"] >= min_edge]
    return {
        "picks": filtered[:top],
        "total_players": len(all_props),
        "total_picks_found": len(filtered),
        "season": season,
        "min_edge": min_edge,
        "cached": False,
    }