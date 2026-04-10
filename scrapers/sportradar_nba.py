"""
scrapers/sportradar_nba.py — Integración con Sportradar NBA API v8
La llave NO va aquí — vive en Render como variable de entorno SPORTRADAR_API_KEY.
El código la lee automáticamente desde ahí o desde la base de datos.

Base URL trial: https://api.sportradar.com/nba/trial/v8/en/
"""
import os
import requests
import time
from datetime import datetime, timedelta

NBA_ACCESS_LEVEL = os.environ.get('SPORTRADAR_ACCESS', 'trial')
NBA_BASE_URL = f"https://api.sportradar.com/nba/{NBA_ACCESS_LEVEL}/v8/en"


class SportradarNBAScraper:
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
                    print(f"[NBA] ✅ API key encontrada en DB: '{name}'")
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
            print("[NBA] ⚠️ Sin API key — omitiendo llamada")
            return None
        url = f"{NBA_BASE_URL}/{endpoint}"
        # ✅ Key va en HEADER, no en URL
        headers = {
            'Accept':    'application/json',
            'x-api-key': self.api_key,
        }
        for attempt in range(retries + 1):
            try:
                r = self.session.get(url, params=params, headers=headers, timeout=10)
                if r.status_code in (401, 403):
                    print(f"[NBA] ❌ API key inválida (HTTP {r.status_code})")
                    return None
                if r.status_code == 429:
                    print("[NBA] ⏳ Rate limit — esperando 6 seg...")
                    time.sleep(6)
                    continue
                if r.status_code == 404:
                    print(f"[NBA] ⚠️ 404 en {endpoint}")
                    return None
                if not r.ok:
                    print(f"[NBA] ⚠️ HTTP {r.status_code} en {endpoint}")
                    return None
                return r.json()
            except Exception as e:
                print(f"[NBA] Error en {endpoint}: {e}")
                if attempt < retries:
                    time.sleep(2)
        return None

    def _parse_game(self, game):
        """Convierte un juego NBA en formato interno."""
        home = game.get('home', {})
        away = game.get('away', {})
        sr_status = game.get('status', 'scheduled')

        status_map = {
            'inprogress': 'in', 'halftime': 'in',
            'closed': 'post', 'complete': 'post',
            'scheduled': 'pre', 'created': 'pre',
            'delayed': 'pre', 'postponed': 'pre', 'canceled': 'canceled',
        }
        internal_status = status_map.get(sr_status, 'pre')

        home_pts = home.get('points', '')
        away_pts = away.get('points', '')

        return {
            'id':        game.get('id', ''),
            'name':      f"{home.get('name','')} vs {away.get('name','')}",
            'date':      game.get('scheduled', ''),
            'status':    internal_status,
            'league':    'nba',
            'sport':     'basketball',
            'homeTeam':  home.get('name', ''),
            'awayTeam':  away.get('name', ''),
            'homeScore': str(home_pts) if internal_status != 'pre' else '',
            'awayScore': str(away_pts) if internal_status != 'pre' else '',
            'homeAbbr':  home.get('alias', ''),
            'awayAbbr':  away.get('alias', ''),
            'homeLogo':  '', 'awayLogo': '',
            'period':    game.get('quarter', 0),
            'clock':     str(game.get('clock', '')),
            'detail':    sr_status,
            'source':    'sportradar_nba',
        }

    def get_today_schedule(self):
        """Partidos NBA de hoy."""
        now = datetime.utcnow()
        return self.get_schedule_by_date(now.year, now.month, now.day)

    def get_schedule_by_date(self, year, month, day):
        """Partidos NBA de una fecha específica."""
        data = self._get(f"games/{year}/{month:02d}/{day:02d}/schedule.json")
        if not data:
            return []
        return [self._parse_game(g) for g in data.get('games', [])]

    def get_live_matches(self):
        """Partidos NBA en curso ahora mismo."""
        today = datetime.utcnow()
        games = self.get_schedule_by_date(today.year, today.month, today.day)
        return [g for g in games if g['status'] == 'in']

    def get_results_by_date(self, year, month, day):
        """
        Resultados finales para alimentar a GLAI.
        Trial usa schedule.json y filtramos por status closed/complete.
        El endpoint summary.json puede no estar disponible en trial.
        """
        data = self._get(f"games/{year}/{month:02d}/{day:02d}/schedule.json")
        if not data:
            return []
        results = []
        for game in data.get('games', []):
            if game.get('status') not in ('closed', 'complete'):
                continue
            home = game.get('home', {})
            away = game.get('away', {})
            hp = home.get('points')
            ap = away.get('points')
            if hp is None or ap is None:
                continue
            results.append({
                'event_id':  game.get('id', ''),
                'league':    'nba',
                'homeTeam':  home.get('name', ''),
                'awayTeam':  away.get('name', ''),
                'homeGoals': int(hp),
                'awayGoals': int(ap),
            })
        return results

    def scan(self, days_back=7, glai=None, on_progress=None):
        """Escanea días anteriores y alimenta a GLAI con resultados NBA."""
        if not self._ok():
            print("[NBA] scan() omitido — sin API key")
            return 0
        total = 0
        today = datetime.utcnow()
        dates = [(today - timedelta(days=i)) for i in range(1, days_back + 1)]
        for i, d in enumerate(dates):
            for r in self.get_results_by_date(d.year, d.month, d.day):
                if glai and r['homeTeam'] and r['awayTeam']:
                    glai.learn('basketball', r['league'], r['homeTeam'], r['awayTeam'],
                               r['homeGoals'], r['awayGoals'],
                               source='sportradar_nba', event_id=f"nba_{r['event_id']}")
                    total += 1
            if on_progress:
                pct = round((i + 1) / len(dates) * 30)
                on_progress(pct, f"NBA {d.strftime('%Y-%m-%d')} — {total} resultados", total)
            time.sleep(1.1)
        print(f"[NBA] ✅ Scan completo — {total} resultados aprendidos")
        return total

    def status(self):
        """Verifica la conexión con la NBA API."""
        if not self._ok():
            return {'ok': False, 'msg': 'Sin API key NBA configurada', 'key_set': False}
        try:
            now = datetime.utcnow()
            url = f"{NBA_BASE_URL}/games/{now.year}/{now.month:02d}/{now.day:02d}/schedule.json"
            r = self.session.get(url, headers={
                'Accept': 'application/json',
                'x-api-key': self.api_key,
            }, timeout=10)
            print(f"[NBA] status() HTTP {r.status_code}")
            if r.status_code in (401, 403):
                return {'ok': False, 'msg': f'NBA API key inválida (HTTP {r.status_code})', 'key_set': True}
            if not r.ok:
                return {'ok': False, 'msg': f'NBA error HTTP {r.status_code}', 'key_set': True}
            data = r.json()
            count = len(data.get('games', []))
            return {
                'ok':           True,
                'msg':          f'NBA conectado — {count} partidos hoy',
                'key_set':      True,
                'access_level': NBA_ACCESS_LEVEL,
            }
        except Exception as e:
            return {'ok': False, 'msg': f'Error de conexión NBA: {e}', 'key_set': True}
