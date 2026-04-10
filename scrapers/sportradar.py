"""
scrapers/sportradar.py — Integración con Sportradar Soccer API v4
Lee la API key desde la variable de entorno SPORTRADAR_API_KEY
o desde la base de datos (key_name = 'sportradar').
"""
import os
import requests
import time
from datetime import datetime, timedelta

# ── Configuración ──────────────────────────────────────────────────
# Sportradar ofrece nivel "trial" y "production" — ajusta con SPORTRADAR_ACCESS
# Valor por defecto: trial (plan de prueba)
ACCESS_LEVEL = os.environ.get('SPORTRADAR_ACCESS', 'trial')
BASE_URL = f"https://api.sportradar.com/soccer/{ACCESS_LEVEL}/v4/en"

# Mapa de competencias Sportradar → league_id del sistema
LEAGUE_MAP = {
    'sr:competition:17':  'eng.1',    # Premier League
    'sr:competition:18':  'esp.1',    # La Liga
    'sr:competition:23':  'ger.1',    # Bundesliga
    'sr:competition:31':  'ita.1',    # Serie A
    'sr:competition:34':  'fra.1',    # Ligue 1
    'sr:competition:7':   'uefa.champions',
    'sr:competition:679': 'usa.1',    # MLS
    'sr:competition:242': 'mex.1',    # Liga MX
    'sr:competition:325': 'arg.1',    # Superliga Argentina
    'sr:competition:390': 'bra.1',    # Brasileirão
    'sr:competition:304': 'por.1',    # Primeira Liga
    'sr:competition:37':  'ned.1',    # Eredivisie
}


class SportradarScraper:
    def __init__(self, api_key=None, db=None):
        """
        api_key: llave directa (opcional)
        db:      instancia de Database — se usa como fallback si no hay api_key
        """
        # 1. Argumento directo
        # 2. Variable de entorno
        # 3. Base de datos
        self.api_key = (
            api_key
            or os.environ.get('SPORTRADAR_API_KEY')
            or (db.get_key('sportradar') if db else None)
            or (db.get_key('SPORTRADAR_API_KEY') if db else None)
        )
        self.db = db
        self.session = requests.Session()
        self.session.headers.update({'Accept': 'application/json'})

    def _ok(self):
        """Devuelve True si hay API key configurada."""
        return bool(self.api_key)

    def _get(self, endpoint, params=None, retries=2):
        """Hace GET a la API de Sportradar con reintentos."""
        if not self._ok():
            print("[Sportradar] ⚠️ Sin API key — omitiendo llamada")
            return None
        url = f"{BASE_URL}/{endpoint}"
        p = {'api_key': self.api_key}
        if params:
            p.update(params)
        for attempt in range(retries + 1):
            try:
                r = self.session.get(url, params=p, timeout=10)
                if r.status_code == 403:
                    print(f"[Sportradar] ❌ API key inválida o sin acceso a: {endpoint}")
                    return None
                if r.status_code == 429:
                    print("[Sportradar] ⏳ Rate limit — esperando 6 seg...")
                    time.sleep(6)
                    continue
                if not r.ok:
                    print(f"[Sportradar] ⚠️ HTTP {r.status_code} en {endpoint}")
                    return None
                return r.json()
            except Exception as e:
                print(f"[Sportradar] Error en {endpoint}: {e}")
                if attempt < retries:
                    time.sleep(2)
        return None

    # ─── Partidos en vivo ─────────────────────────────────────────
    def get_live_matches(self):
        """
        Retorna lista de partidos EN VIVO ahora mismo.
        Formato compatible con ESPNScraper._event_dict()
        """
        data = self._get("schedules/live/results.json")
        if not data:
            return []
        results = []
        for sched in data.get('results', []):
            sport_event = sched.get('sport_event', {})
            status      = sched.get('sport_event_status', {})
            competitors = sport_event.get('competitors', [])
            if len(competitors) < 2:
                continue
            home = next((c for c in competitors if c.get('qualifier') == 'home'), competitors[0])
            away = next((c for c in competitors if c.get('qualifier') == 'away'), competitors[1])
            comp_id  = sport_event.get('tournament', {}).get('id', '')
            league   = LEAGUE_MAP.get(comp_id, comp_id)
            sr_status = status.get('status', 'live')
            results.append({
                'id':        sport_event.get('id', ''),
                'name':      f"{home.get('name','')} vs {away.get('name','')}",
                'date':      sport_event.get('start_time', ''),
                'status':    'in' if sr_status == 'live' else sr_status,
                'league':    league,
                'sport':     'soccer',
                'homeTeam':  home.get('name', ''),
                'awayTeam':  away.get('name', ''),
                'homeScore': status.get('home_score', ''),
                'awayScore': status.get('away_score', ''),
                'homeAbbr':  home.get('abbreviation', ''),
                'awayAbbr':  away.get('abbreviation', ''),
                'homeLogo':  '',
                'awayLogo':  '',
                'period':    status.get('period', 0),
                'clock':     str(status.get('clock', '')),
                'detail':    sr_status,
                'source':    'sportradar',
            })
        return results

    # ─── Partidos de hoy ──────────────────────────────────────────
    def get_today_schedule(self):
        """Partidos programados para hoy (incluye pre-juego y en vivo)."""
        today = datetime.utcnow().strftime('%Y-%m-%d')
        return self.get_schedule_by_date(today)

    def get_schedule_by_date(self, date_str):
        """Partidos de una fecha específica (YYYY-MM-DD)."""
        data = self._get(f"schedules/{date_str}/schedule.json")
        if not data:
            return []
        results = []
        for sched in data.get('sport_events', []):
            competitors = sched.get('competitors', [])
            if len(competitors) < 2:
                continue
            home = next((c for c in competitors if c.get('qualifier') == 'home'), competitors[0])
            away = next((c for c in competitors if c.get('qualifier') == 'away'), competitors[1])
            comp_id = sched.get('tournament', {}).get('id', '')
            league  = LEAGUE_MAP.get(comp_id, comp_id)
            results.append({
                'id':        sched.get('id', ''),
                'name':      f"{home.get('name','')} vs {away.get('name','')}",
                'date':      sched.get('start_time', ''),
                'status':    'pre',
                'league':    league,
                'sport':     'soccer',
                'homeTeam':  home.get('name', ''),
                'awayTeam':  away.get('name', ''),
                'homeScore': '',
                'awayScore': '',
                'homeAbbr':  home.get('abbreviation', ''),
                'awayAbbr':  away.get('abbreviation', ''),
                'homeLogo':  '',
                'awayLogo':  '',
                'period':    0,
                'clock':     '',
                'detail':    'Programado',
                'source':    'sportradar',
            })
        return results

    # ─── Resultados históricos (para entrenar GLAI) ───────────────
    def get_results_by_date(self, date_str):
        """Resultados finales de una fecha (YYYY-MM-DD)."""
        data = self._get(f"schedules/{date_str}/results.json")
        if not data:
            return []
        results = []
        for entry in data.get('results', []):
            sport_event = entry.get('sport_event', {})
            status      = entry.get('sport_event_status', {})
            if status.get('status') != 'closed':
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
            league  = LEAGUE_MAP.get(comp_id, comp_id)
            results.append({
                'event_id':  sport_event.get('id', ''),
                'league':    league,
                'homeTeam':  home.get('name', ''),
                'awayTeam':  away.get('name', ''),
                'homeGoals': int(hg),
                'awayGoals': int(ag),
            })
        return results

    # ─── Scan histórico para GLAI ─────────────────────────────────
    def scan(self, days_back=7, glai=None, on_progress=None):
        """
        Descarga resultados de los últimos N días y los enseña a GLAI.
        Compatible con el método scan() de ESPNScraper.
        """
        if not self._ok():
            print("[Sportradar] scan() omitido — sin API key")
            return 0

        total = 0
        today = datetime.utcnow()
        dates = [(today - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, days_back + 1)]

        for i, date_str in enumerate(dates):
            results = self.get_results_by_date(date_str)
            for r in results:
                if glai and r['homeTeam'] and r['awayTeam']:
                    glai.learn(
                        'soccer',
                        r['league'],
                        r['homeTeam'],
                        r['awayTeam'],
                        r['homeGoals'],
                        r['awayGoals'],
                        source='sportradar',
                        event_id=f"sr_{r['event_id']}"
                    )
                    total += 1
            if on_progress:
                pct = round((i + 1) / len(dates) * 30)
                on_progress(pct, f"Sportradar {date_str} — {total} resultados", total)
            # Sportradar permite ~1 req/seg en trial
            time.sleep(1.1)

        print(f"[Sportradar] ✅ Scan completo — {total} resultados aprendidos")
        return total

    def status(self):
        """Verifica si la API key está configurada y funciona."""
        if not self._ok():
            return {'ok': False, 'msg': 'Sin API key configurada', 'key_set': False}
        # Prueba con un endpoint liviano
        data = self._get("competitions.json")
        if data is None:
            return {'ok': False, 'msg': 'API key inválida o sin conectividad', 'key_set': True}
        comp_count = len(data.get('competitions', []))
        return {
            'ok': True,
            'msg': f'Sportradar conectado — {comp_count} competencias disponibles',
            'key_set': True,
            'access_level': ACCESS_LEVEL,
        }
