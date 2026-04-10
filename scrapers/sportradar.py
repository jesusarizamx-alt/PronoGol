"""
scrapers/sportradar.py — Integración con Sportradar Soccer API v4
Rutas confirmadas disponibles en trial:
  ✅ competitions.json
  ✅ schedules/live/schedule.json
  ✅ competitions/{id}/seasons.json
  ✅ seasons/{id}/summaries.json
  ❌ schedules/{fecha}/schedule.json  (no disponible en trial)
"""
import os
import requests
import time
from datetime import datetime, timedelta

ACCESS_LEVEL = os.environ.get('SPORTRADAR_ACCESS', 'trial')
BASE_URL     = f"https://api.sportradar.com/soccer/{ACCESS_LEVEL}/v4/en"

# Competencias principales a escanear para GLAI
TOP_COMPETITIONS = [
    'sr:competition:17',   # Premier League
    'sr:competition:18',   # La Liga
    'sr:competition:23',   # Bundesliga
    'sr:competition:31',   # Serie A
    'sr:competition:34',   # Ligue 1
    'sr:competition:7',    # UEFA Champions League
    'sr:competition:679',  # MLS
    'sr:competition:242',  # Liga MX
    'sr:competition:325',  # Argentina Primera
    'sr:competition:390',  # Brasileirao
]

LEAGUE_MAP = {
    'sr:competition:17':  'eng.1',
    'sr:competition:18':  'esp.1',
    'sr:competition:23':  'ger.1',
    'sr:competition:31':  'ita.1',
    'sr:competition:34':  'fra.1',
    'sr:competition:7':   'uefa.champions',
    'sr:competition:679': 'usa.1',
    'sr:competition:242': 'mex.1',
    'sr:competition:325': 'arg.1',
    'sr:competition:390': 'bra.1',
    'sr:competition:304': 'por.1',
    'sr:competition:37':  'ned.1',
}


class SportradarScraper:
    def __init__(self, api_key=None, db=None):
        self.db      = db
        self.session = requests.Session()
        self.api_key = api_key or os.environ.get('SPORTRADAR_API_KEY') or self._find_key_in_db()
        self._update_session_headers()

    def _update_session_headers(self):
        headers = {'Accept': 'application/json'}
        if self.api_key:
            headers['x-api-key'] = self.api_key
        self.session.headers.update(headers)

    def _find_key_in_db(self):
        if not self.db:
            return None
        try:
            all_keys = self.db.get_all_keys()
            for name, value in all_keys.items():
                if 'sportradar' in name.lower() and value:
                    print(f"[Sportradar] ✅ Key en DB: '{name}'")
                    return value
        except Exception:
            pass
        return None

    def _ok(self):
        return bool(self.api_key)

    def _get(self, endpoint, retries=2):
        if not self._ok():
            return None
        url     = f"{BASE_URL}/{endpoint}"
        headers = {'Accept': 'application/json', 'x-api-key': self.api_key}
        for attempt in range(retries + 1):
            try:
                r = self.session.get(url, headers=headers, timeout=12)
                if r.status_code == 429:
                    print("[Sportradar] ⏳ Rate limit — esperando 6 seg...")
                    time.sleep(6)
                    continue
                if r.status_code in (401, 403):
                    print(f"[Sportradar] ❌ Auth error {r.status_code}")
                    return None
                if r.status_code == 404:
                    print(f"[Sportradar] ⚠️ 404: {endpoint}")
                    return None
                if not r.ok:
                    print(f"[Sportradar] ⚠️ HTTP {r.status_code}: {endpoint}")
                    return None
                return r.json()
            except Exception as e:
                print(f"[Sportradar] Error {endpoint}: {e}")
                if attempt < retries:
                    time.sleep(2)
        return None

    # ── Partidos en vivo ──────────────────────────────────────────
    def get_live_matches(self):
        """Partidos Soccer en vivo — endpoint confirmado en trial."""
        data = self._get("schedules/live/schedule.json")
        if not data:
            return []
        results = []
        for sched in data.get('sport_events', []):
            sport_event = sched.get('sport_event', sched)
            status      = sched.get('sport_event_status', {})
            competitors = sport_event.get('competitors', [])
            if len(competitors) < 2:
                continue
            home    = next((c for c in competitors if c.get('qualifier') == 'home'), competitors[0])
            away    = next((c for c in competitors if c.get('qualifier') == 'away'), competitors[1])
            comp_id = sport_event.get('tournament', {}).get('id', '')
            sr_stat = status.get('status', 'live')
            results.append({
                'id':        sport_event.get('id', ''),
                'name':      f"{home.get('name','')} vs {away.get('name','')}",
                'date':      sport_event.get('start_time', ''),
                'status':    'in',
                'league':    LEAGUE_MAP.get(comp_id, comp_id),
                'sport':     'soccer',
                'homeTeam':  home.get('name', ''),
                'awayTeam':  away.get('name', ''),
                'homeScore': status.get('home_score', ''),
                'awayScore': status.get('away_score', ''),
                'homeAbbr':  home.get('abbreviation', ''),
                'awayAbbr':  away.get('abbreviation', ''),
                'homeLogo':  '', 'awayLogo': '',
                'period':    status.get('period', 0),
                'clock':     str(status.get('clock', '')),
                'detail':    sr_stat,
                'source':    'sportradar',
            })
        return results

    def get_today_schedule(self):
        """Hoy: usa partidos en vivo + los programados del live feed."""
        return self.get_live_matches()

    # ── Competencias y temporadas ─────────────────────────────────
    def get_competitions(self):
        """Lista de competencias disponibles — confirmado 200 en trial."""
        data = self._get("competitions.json")
        return data.get('competitions', []) if data else []

    def get_current_season(self, competition_id):
        """Obtiene la temporada activa de una competencia."""
        data = self._get(f"competitions/{competition_id}/seasons.json")
        if not data:
            return None
        seasons = data.get('seasons', [])
        if not seasons:
            return None
        # La primera suele ser la más reciente
        return seasons[0].get('id')

    def get_season_summaries(self, season_id):
        """
        Resultados de una temporada — incluye marcadores finales.
        Endpoint: seasons/{season_id}/summaries.json
        """
        data = self._get(f"seasons/{season_id}/summaries.json")
        if not data:
            return []
        results = []
        for summary in data.get('summaries', []):
            sport_event = summary.get('sport_event', {})
            status      = summary.get('sport_event_status', {})
            if status.get('status') not in ('closed', 'complete'):
                continue
            competitors = sport_event.get('competitors', [])
            if len(competitors) < 2:
                continue
            home = next((c for c in competitors if c.get('qualifier') == 'home'), competitors[0])
            away = next((c for c in competitors if c.get('qualifier') == 'away'), competitors[1])
            hg = status.get('home_score')
            ag = status.get('away_score')
            if hg is None or ag is None:
                continue
            comp_id = sport_event.get('tournament', {}).get('id', '')
            results.append({
                'event_id':  sport_event.get('id', ''),
                'league':    LEAGUE_MAP.get(comp_id, comp_id),
                'homeTeam':  home.get('name', ''),
                'awayTeam':  away.get('name', ''),
                'homeGoals': int(hg),
                'awayGoals': int(ag),
            })
        return results

    # ── Scan histórico para GLAI ──────────────────────────────────
    def scan(self, days_back=7, glai=None, on_progress=None):
        """
        Escanea las principales ligas via competitions → seasons → summaries
        y alimenta a GLAI con los resultados encontrados.
        """
        if not self._ok():
            print("[Sportradar] scan() omitido — sin API key")
            return 0

        total = 0
        comps = TOP_COMPETITIONS
        print(f"[Sportradar] Escaneando {len(comps)} competencias...")

        for i, comp_id in enumerate(comps):
            try:
                season_id = self.get_current_season(comp_id)
                time.sleep(1.1)
                if not season_id:
                    continue

                results = self.get_season_summaries(season_id)
                time.sleep(1.1)

                league = LEAGUE_MAP.get(comp_id, comp_id)
                for r in results:
                    if glai and r['homeTeam'] and r['awayTeam']:
                        glai.learn('soccer', league,
                                   r['homeTeam'], r['awayTeam'],
                                   r['homeGoals'], r['awayGoals'],
                                   source='sportradar', event_id=f"sr_{r['event_id']}")
                        total += 1

                if on_progress:
                    pct = round((i + 1) / len(comps) * 30)
                    on_progress(pct, f"Sportradar {league} — {total} resultados", total)

                print(f"[Sportradar] {league}: {len(results)} resultados")

            except Exception as e:
                print(f"[Sportradar] Error en {comp_id}: {e}")
                continue

        print(f"[Sportradar] ✅ Scan completo — {total} resultados aprendidos")
        return total

    # ── Estado ────────────────────────────────────────────────────
    def status(self):
        if not self._ok():
            return {'ok': False, 'msg': 'Sin API key configurada', 'key_set': False}
        # Usa competitions.json — confirmado 200 en trial
        try:
            url = f"{BASE_URL}/competitions.json"
            r   = self.session.get(url, headers={
                'Accept': 'application/json',
                'x-api-key': self.api_key,
            }, timeout=10)
            print(f"[Sportradar] status() HTTP {r.status_code}")
            if r.status_code in (401, 403):
                return {'ok': False, 'msg': f'API key inválida (HTTP {r.status_code})', 'key_set': True}
            if not r.ok:
                return {'ok': False, 'msg': f'Error HTTP {r.status_code}', 'key_set': True}
            comps = len(r.json().get('competitions', []))
            return {
                'ok':           True,
                'msg':          f'Sportradar Soccer conectado — {comps} competencias disponibles',
                'key_set':      True,
                'access_level': ACCESS_LEVEL,
                'competitions': comps,
            }
        except Exception as e:
            return {'ok': False, 'msg': f'Error: {e}', 'key_set': True}
