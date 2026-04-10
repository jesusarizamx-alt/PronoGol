"""
scrapers/sportradar.py — Integración con Sportradar (Soccer + NBA + MLB)
Lee la API key desde la variable de entorno SPORTRADAR_API_KEY
o desde la base de datos (key_name que contenga 'sportradar').
"""
import os
import requests
import time
from datetime import datetime, timedelta

ACCESS = os.environ.get('SPORTRADAR_ACCESS', 'trial')

# URLs base por deporte
URLS = {
    'soccer':  f"https://api.sportradar.com/soccer/{ACCESS}/v4/en",
    'nba':     f"https://api.sportradar.com/nba/{ACCESS}/v8/en",
    'mlb':     f"https://api.sportradar.com/mlb/{ACCESS}/v7/en",
}

# Mapa competencias Soccer → league_id del sistema
SOCCER_LEAGUE_MAP = {
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
        self.db = db
        self.session = requests.Session()
        self.session.headers.update({'Accept': 'application/json'})
        self.api_key = api_key or os.environ.get('SPORTRADAR_API_KEY') or self._find_key_in_db()

    def _find_key_in_db(self):
        if not self.db:
            return None
        try:
            for name, value in self.db.get_all_keys().items():
                if 'sportradar' in name.lower() and value:
                    print(f"[Sportradar] ✅ API key encontrada en DB: '{name}'")
                    return value
        except Exception:
            pass
        return None

    def _ok(self):
        return bool(self.api_key)

    def _get(self, sport, endpoint, params=None, retries=2):
        """GET genérico para cualquier deporte."""
        if not self._ok():
            return None
        base = URLS.get(sport, URLS['soccer'])
        url  = f"{base}/{endpoint}"
        p    = {'api_key': self.api_key}
        if params:
            p.update(params)
        for attempt in range(retries + 1):
            try:
                r = self.session.get(url, params=p, timeout=10)
                if r.status_code == 403:
                    print(f"[Sportradar] ❌ Sin acceso a {sport}/{endpoint}")
                    return None
                if r.status_code == 429:
                    print("[Sportradar] ⏳ Rate limit — esperando 6s...")
                    time.sleep(6)
                    continue
                if r.status_code == 404:
                    print(f"[Sportradar] ⚠️ 404 en {sport}/{endpoint}")
                    return None
                if not r.ok:
                    print(f"[Sportradar] ⚠️ HTTP {r.status_code} en {sport}/{endpoint}")
                    return None
                return r.json()
            except Exception as e:
                print(f"[Sportradar] Error {sport}/{endpoint}: {e}")
                if attempt < retries:
                    time.sleep(2)
        return None

    # ═══════════════════════════════════════════════
    # SOCCER
    # ═══════════════════════════════════════════════
    def _parse_soccer_event(self, sport_event, status, sr_status='scheduled'):
        competitors = sport_event.get('competitors', [])
        if len(competitors) < 2:
            return None
        home = next((c for c in competitors if c.get('qualifier') == 'home'), competitors[0])
        away = next((c for c in competitors if c.get('qualifier') == 'away'), competitors[1])
        comp_id = sport_event.get('tournament', {}).get('id', '')
        league  = SOCCER_LEAGUE_MAP.get(comp_id, comp_id)
        return {
            'id':        sport_event.get('id', ''),
            'name':      f"{home.get('name','')} vs {away.get('name','')}",
            'date':      sport_event.get('start_time', ''),
            'status':    'in' if sr_status in ('live','inprogress') else ('post' if sr_status in ('closed','complete') else 'pre'),
            'league':    league,
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
            'detail':    sr_status,
            'source':    'sportradar',
        }

    def get_soccer_live(self):
        data = self._get('soccer', 'schedules/live/schedule.json')
        if not data:
            return []
        results = []
        for sched in data.get('sport_events', []):
            se     = sched.get('sport_event', sched)
            status = sched.get('sport_event_status', {})
            ev = self._parse_soccer_event(se, status, status.get('status', 'live'))
            if ev:
                results.append(ev)
        return results

    def get_soccer_results(self, date_str):
        data = self._get('soccer', f'schedules/{date_str}/results.json')
        if not data:
            return []
        results = []
        for entry in data.get('results', []):
            se     = entry.get('sport_event', {})
            status = entry.get('sport_event_status', {})
            if status.get('status') not in ('closed', 'complete'):
                continue
            competitors = se.get('competitors', [])
            if len(competitors) < 2:
                continue
            home = next((c for c in competitors if c.get('qualifier') == 'home'), competitors[0])
            away = next((c for c in competitors if c.get('qualifier') == 'away'), competitors[1])
            hg = status.get('home_score')
            ag = status.get('away_score')
            if hg is None or ag is None:
                continue
            comp_id = se.get('tournament', {}).get('id', '')
            results.append({
                'sport':     'soccer',
                'event_id':  se.get('id', ''),
                'league':    SOCCER_LEAGUE_MAP.get(comp_id, comp_id),
                'homeTeam':  home.get('name', ''),
                'awayTeam':  away.get('name', ''),
                'homeGoals': int(hg),
                'awayGoals': int(ag),
            })
        return results

    # ═══════════════════════════════════════════════
    # NBA (Basketball)
    # ═══════════════════════════════════════════════
    def _parse_nba_event(self, game, status=None):
        home = game.get('home', {})
        away = game.get('away', {})
        if not home or not away:
            return None
        st   = status or game.get('status', 'scheduled')
        return {
            'id':        game.get('id', ''),
            'name':      f"{home.get('name','')} vs {away.get('name','')}",
            'date':      game.get('scheduled', ''),
            'status':    'in' if st in ('inprogress',) else ('post' if st in ('closed','complete') else 'pre'),
            'league':    'nba',
            'sport':     'nba',
            'homeTeam':  home.get('name', ''),
            'awayTeam':  away.get('name', ''),
            'homeScore': game.get('home_points', ''),
            'awayScore': game.get('away_points', ''),
            'homeAbbr':  home.get('alias', ''),
            'awayAbbr':  away.get('alias', ''),
            'homeLogo':  '', 'awayLogo': '',
            'period':    game.get('quarter', 0),
            'clock':     game.get('clock', ''),
            'detail':    st,
            'source':    'sportradar',
        }

    def get_nba_schedule(self, date_str):
        data = self._get('nba', f'games/{date_str}/schedule.json')
        if not data:
            return []
        return [ev for g in data.get('games', [])
                for ev in [self._parse_nba_event(g)] if ev]

    def get_nba_results(self, date_str):
        data = self._get('nba', f'games/{date_str}/summary.json')
        if not data:
            # fallback a schedule
            data = self._get('nba', f'games/{date_str}/schedule.json')
        if not data:
            return []
        results = []
        for g in data.get('games', []):
            if g.get('status') not in ('closed', 'complete'):
                continue
            hp = g.get('home_points')
            ap = g.get('away_points')
            if hp is None or ap is None:
                continue
            results.append({
                'sport':     'nba',
                'event_id':  g.get('id', ''),
                'league':    'nba',
                'homeTeam':  g.get('home', {}).get('name', ''),
                'awayTeam':  g.get('away', {}).get('name', ''),
                'homeGoals': int(hp),
                'awayGoals': int(ap),
            })
        return results

    # ═══════════════════════════════════════════════
    # MLB (Baseball)
    # ═══════════════════════════════════════════════
    def _parse_mlb_event(self, game):
        home = game.get('home', {})
        away = game.get('away', {})
        if not home or not away:
            return None
        st = game.get('status', 'scheduled')
        return {
            'id':        game.get('id', ''),
            'name':      f"{home.get('name','')} vs {away.get('name','')}",
            'date':      game.get('scheduled', ''),
            'status':    'in' if st == 'inprogress' else ('post' if st in ('closed','complete') else 'pre'),
            'league':    'mlb',
            'sport':     'mlb',
            'homeTeam':  home.get('name', ''),
            'awayTeam':  away.get('name', ''),
            'homeScore': game.get('home_runs', ''),
            'awayScore': game.get('away_runs', ''),
            'homeAbbr':  home.get('abbr', ''),
            'awayAbbr':  away.get('abbr', ''),
            'homeLogo':  '', 'awayLogo': '',
            'period':    game.get('inning', 0),
            'clock':     game.get('inning_half', ''),
            'detail':    st,
            'source':    'sportradar',
        }

    def get_mlb_schedule(self, date_str):
        data = self._get('mlb', f'games/{date_str}/schedule.json')
        if not data:
            return []
        return [ev for g in data.get('games', [])
                for ev in [self._parse_mlb_event(g)] if ev]

    def get_mlb_results(self, date_str):
        data = self._get('mlb', f'games/{date_str}/summary.json')
        if not data:
            data = self._get('mlb', f'games/{date_str}/schedule.json')
        if not data:
            return []
        results = []
        for g in data.get('games', []):
            if g.get('status') not in ('closed', 'complete'):
                continue
            hr = g.get('home_runs')
            ar = g.get('away_runs')
            if hr is None or ar is None:
                continue
            results.append({
                'sport':     'mlb',
                'event_id':  g.get('id', ''),
                'league':    'mlb',
                'homeTeam':  g.get('home', {}).get('name', ''),
                'awayTeam':  g.get('away', {}).get('name', ''),
                'homeGoals': int(hr),
                'awayGoals': int(ar),
            })
        return results

    # ═══════════════════════════════════════════════
    # COMBINADO — todos los deportes
    # ═══════════════════════════════════════════════
    def get_all_live(self):
        """Partidos en vivo de los 3 deportes."""
        if not self._ok():
            return []
        results = []
        results.extend(self.get_soccer_live())
        today = datetime.utcnow().strftime('%Y-%m-%d')
        results.extend(self.get_nba_schedule(today))
        results.extend(self.get_mlb_schedule(today))
        return results

    def scan(self, days_back=7, glai=None, on_progress=None):
        """Descarga resultados de los últimos N días (Soccer + NBA + MLB) y entrena GLAI."""
        if not self._ok():
            print("[Sportradar] scan() omitido — sin API key")
            return 0

        total = 0
        today = datetime.utcnow()
        dates = [(today - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, days_back + 1)]

        for i, date_str in enumerate(dates):
            # Soccer
            for r in self.get_soccer_results(date_str):
                if glai and r['homeTeam'] and r['awayTeam']:
                    glai.learn('soccer', r['league'], r['homeTeam'], r['awayTeam'],
                               r['homeGoals'], r['awayGoals'],
                               source='sportradar', event_id=f"sr_soc_{r['event_id']}")
                    total += 1
            time.sleep(1.1)

            # NBA
            for r in self.get_nba_results(date_str):
                if glai and r['homeTeam'] and r['awayTeam']:
                    glai.learn('nba', 'nba', r['homeTeam'], r['awayTeam'],
                               r['homeGoals'], r['awayGoals'],
                               source='sportradar', event_id=f"sr_nba_{r['event_id']}")
                    total += 1
            time.sleep(1.1)

            # MLB
            for r in self.get_mlb_results(date_str):
                if glai and r['homeTeam'] and r['awayTeam']:
                    glai.learn('mlb', 'mlb', r['homeTeam'], r['awayTeam'],
                               r['homeGoals'], r['awayGoals'],
                               source='sportradar', event_id=f"sr_mlb_{r['event_id']}")
                    total += 1
            time.sleep(1.1)

            if on_progress:
                pct = round((i + 1) / len(dates) * 30)
                on_progress(pct, f"Sportradar {date_str} — {total} resultados", total)

        print(f"[Sportradar] ✅ Scan completo — {total} resultados (Soccer+NBA+MLB)")
        return total

    def status(self):
        if not self._ok():
            return {'ok': False, 'msg': 'Sin API key', 'key_set': False}
        data = self._get('soccer', 'competitions.json')
        if data is None:
            return {'ok': False, 'msg': 'API key inválida o sin conectividad', 'key_set': True}
        return {
            'ok':          True,
            'msg':         f'Sportradar conectado — Soccer + NBA + MLB activos',
            'key_set':     True,
            'access_level': ACCESS,
            'sports':      ['soccer', 'nba', 'mlb'],
        }
