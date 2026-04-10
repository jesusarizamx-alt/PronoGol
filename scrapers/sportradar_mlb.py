"""
scrapers/sportradar_mlb.py — Integración con Sportradar MLB API v8
La llave NO va aquí — vive en Render como variable de entorno SPORTRADAR_API_KEY.
El código la lee automáticamente desde ahí o desde la base de datos.

Base URL trial: https://api.sportradar.com/mlb/trial/v8/en/
"""
import os
import requests
import time
from datetime import datetime, timedelta

MLB_ACCESS_LEVEL = os.environ.get('SPORTRADAR_ACCESS', 'trial')
MLB_BASE_URL = f"https://api.sportradar.com/mlb/{MLB_ACCESS_LEVEL}/v8/en"


class SportradarMLBScraper:
    def __init__(self, api_key=None, db=None):
        self.db = db
        self.session = requests.Session()
        # La key viene de: 1) parámetro, 2) env var Render, 3) base de datos
        self.api_key = api_key or os.environ.get('SPORTRADAR_API_KEY') or self._find_key_in_db()

    def _find_key_in_db(self):
        """Busca en la DB cualquier key cuyo nombre contenga 'sportradar'."""
        if not self.db:
            return None
        try:
            all_keys = self.db.get_all_keys()
            for name, value in all_keys.items():
                if 'sportradar' in name.lower() and value:
                    print(f"[MLB] ✅ API key encontrada en DB: '{name}'")
                    return value
        except Exception:
            pass
        return None

    def _update_session_headers(self):
        headers = {'Accept': 'application/json'}
        if self.api_key:
            headers['x-api-key'] = self.api_key
        self.session.headers.update(headers)

    def _ok(self):
        return bool(self.api_key)

    def _get(self, endpoint, params=None, retries=2):
        if not self._ok():
            print("[MLB] ⚠️ Sin API key — omitiendo llamada")
            return None
        url = f"{MLB_BASE_URL}/{endpoint}"
        # ✅ Key va en HEADER, no en URL
        headers = {
            'Accept':    'application/json',
            'x-api-key': self.api_key,
        }
        for attempt in range(retries + 1):
            try:
                r = self.session.get(url, params=params, headers=headers, timeout=10)
                if r.status_code in (401, 403):
                    print(f"[MLB] ❌ API key inválida (HTTP {r.status_code})")
                    return None
                if r.status_code == 429:
                    print("[MLB] ⏳ Rate limit — esperando 6 seg...")
                    time.sleep(6)
                    continue
                if r.status_code == 404:
                    print(f"[MLB] ⚠️ 404 en {endpoint}")
                    return None
                if not r.ok:
                    print(f"[MLB] ⚠️ HTTP {r.status_code} en {endpoint}")
                    return None
                return r.json()
            except Exception as e:
                print(f"[MLB] Error en {endpoint}: {e}")
                if attempt < retries:
                    time.sleep(2)
        return None

    def _parse_game(self, game):
        """Convierte un juego MLB en formato interno."""
        home = game.get('home', {})
        away = game.get('away', {})
        sr_status = game.get('status', 'scheduled')

        status_map = {
            'inprogress': 'in', 'live': 'in',
            'closed': 'post', 'complete': 'post',
            'scheduled': 'pre', 'created': 'pre',
            'delayed': 'pre', 'postponed': 'pre', 'canceled': 'canceled',
            'unnecessary': 'canceled',
        }
        internal_status = status_map.get(sr_status, 'pre')

        # MLB usa 'runs' para el marcador
        home_score = home.get('runs', home.get('points', ''))
        away_score = away.get('runs', away.get('points', ''))

        # Inning actual
        inning = game.get('inning', 0)
        inning_half = game.get('inning_half', '')
        clock_str = f"Inn {inning} {inning_half}".strip() if inning else ''

        return {
            'id':        game.get('id', ''),
            'name':      f"{home.get('name','')} vs {away.get('name','')}",
            'date':      game.get('scheduled', ''),
            'status':    internal_status,
            'league':    'mlb',
            'sport':     'baseball',
            'homeTeam':  home.get('name', ''),
            'awayTeam':  away.get('name', ''),
            'homeScore': str(home_score) if internal_status != 'pre' else '',
            'awayScore': str(away_score) if internal_status != 'pre' else '',
            'homeAbbr':  home.get('abbr', home.get('alias', '')),
            'awayAbbr':  away.get('abbr', away.get('alias', '')),
            'homeLogo':  '', 'awayLogo': '',
            'period':    inning,
            'clock':     clock_str,
            'detail':    sr_status,
            'source':    'sportradar_mlb',
        }

    def get_today_schedule(self):
        """Partidos MLB de hoy."""
        now = datetime.utcnow()
        return self.get_schedule_by_date(now.year, now.month, now.day)

    def get_schedule_by_date(self, year, month, day):
        """Partidos MLB de una fecha específica."""
        data = self._get(f"games/{year}/{month:02d}/{day:02d}/schedule.json")
        if not data:
            return []
        return [self._parse_game(g) for g in data.get('games', [])]

    def get_live_matches(self):
        """Partidos MLB en curso ahora mismo."""
        today = datetime.utcnow()
        games = self.get_schedule_by_date(today.year, today.month, today.day)
        return [g for g in games if g['status'] == 'in']

    def get_results_by_date(self, year, month, day):
        """Resultados finales para alimentar a GLAI."""
        data = self._get(f"games/{year}/{month:02d}/{day:02d}/summary.json")
        if not data:
            return []
        results = []
        for game in data.get('games', []):
            if game.get('status') not in ('closed', 'complete'):
                continue
            home = game.get('home', {})
            away = game.get('away', {})
            # Runs = carreras en béisbol
            hp = home.get('runs', home.get('points'))
            ap = away.get('runs', away.get('points'))
            if hp is None or ap is None:
                continue
            results.append({
                'event_id':  game.get('id', ''),
                'league':    'mlb',
                'homeTeam':  home.get('name', ''),
                'awayTeam':  away.get('name', ''),
                'homeGoals': int(hp),   # en MLB = carreras (runs)
                'awayGoals': int(ap),
            })
        return results

    def scan(self, days_back=7, glai=None, on_progress=None):
        """Escanea días anteriores y alimenta a GLAI con resultados MLB."""
        if not self._ok():
            print("[MLB] scan() omitido — sin API key")
            return 0
        total = 0
        today = datetime.utcnow()
        dates = [(today - timedelta(days=i)) for i in range(1, days_back + 1)]
        for i, d in enumerate(dates):
            for r in self.get_results_by_date(d.year, d.month, d.day):
                if glai and r['homeTeam'] and r['awayTeam']:
                    glai.learn('baseball', r['league'], r['homeTeam'], r['awayTeam'],
                               r['homeGoals'], r['awayGoals'],
                               source='sportradar_mlb', event_id=f"mlb_{r['event_id']}")
                    total += 1
            if on_progress:
                pct = round((i + 1) / len(dates) * 30)
                on_progress(pct, f"MLB {d.strftime('%Y-%m-%d')} — {total} resultados", total)
            time.sleep(1.1)
        print(f"[MLB] ✅ Scan completo — {total} resultados aprendidos")
        return total

    def status(self):
        """Verifica la conexión con la MLB API."""
        if not self._ok():
            return {'ok': False, 'msg': 'Sin API key MLB configurada', 'key_set': False}
        try:
            now = datetime.utcnow()
            url = f"{MLB_BASE_URL}/games/{now.year}/{now.month:02d}/{now.day:02d}/schedule.json"
            r = self.session.get(url, headers={
                'Accept': 'application/json',
                'x-api-key': self.api_key,
            }, timeout=10)
            print(f"[MLB] status() HTTP {r.status_code}")
            if r.status_code in (401, 403):
                return {'ok': False, 'msg': f'MLB API key inválida (HTTP {r.status_code})', 'key_set': True}
            if not r.ok:
                return {'ok': False, 'msg': f'MLB error HTTP {r.status_code}', 'key_set': True}
            data = r.json()
            count = len(data.get('games', []))
            return {
                'ok':           True,
                'msg':          f'MLB conectado — {count} partidos hoy',
                'key_set':      True,
                'access_level': MLB_ACCESS_LEVEL,
            }
        except Exception as e:
            return {'ok': False, 'msg': f'Error de conexión MLB: {e}', 'key_set': True}
