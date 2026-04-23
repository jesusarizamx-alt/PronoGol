"""
scrapers/sportradar.py — Integración con Sportradar Soccer API v4

Endpoints confirmados disponibles en trial (2026-04-10):
  ✅ competitions.json                          → status / lista de ligas
  ✅ competitions/{id}/seasons.json             → temporadas por liga
  ✅ schedules/{fecha}/summaries.json           → resultados del día (SCAN)
  ❌ schedules/live/schedule.json               → 404 en trial
  ❌ schedules/{fecha}/schedule.json            → 404 en trial
  ❌ schedules/{fecha}/results.json             → 404 en trial

NBA/MLB requieren suscripción separada en developer.sportradar.com (403 actualmente).
"""
import os
import requests
import time
from datetime import datetime, timedelta

ACCESS_LEVEL = os.environ.get('SPORTRADAR_ACCESS', 'trial')
BASE_URL     = f"https://api.sportradar.com/soccer/{ACCESS_LEVEL}/v4/en"

LEAGUE_MAP = {
    'sr:competition:17':  'eng.1',    # Premier League
    'sr:competition:18':  'esp.1',    # La Liga
    'sr:competition:23':  'ger.1',    # Bundesliga
    'sr:competition:31':  'ita.1',    # Serie A
    'sr:competition:34':  'fra.1',    # Ligue 1
    'sr:competition:7':   'uefa.champions',
    'sr:competition:679': 'usa.1',    # MLS
    'sr:competition:242': 'mex.1',    # Liga MX
    'sr:competition:325': 'arg.1',    # Argentina Primera
    'sr:competition:390': 'bra.1',    # Brasileirao
    'sr:competition:304': 'por.1',    # Primeira Liga
    'sr:competition:37':  'ned.1',    # Eredivisie
    'sr:competition:238': 'tur.1',    # Süper Lig
    'sr:competition:8':   'uefa.europa',
    'sr:competition:24':  'sco.1',    # Scottish Prem
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
                    print(f"[Sportradar] ⚠️ 404 (no disponible en trial): {endpoint}")
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

    # ── Parser de summaries ───────────────────────────────────────
    def _parse_summary(self, summary):
        """
        Convierte un item de summaries.json en formato interno.
        Funciona para resultados finales y partidos en curso.
        """
        sport_event = summary.get('sport_event', {})
        status      = summary.get('sport_event_status', {})
        competitors = sport_event.get('competitors', [])
        if len(competitors) < 2:
            return None

        home    = next((c for c in competitors if c.get('qualifier') == 'home'), competitors[0])
        away    = next((c for c in competitors if c.get('qualifier') == 'away'), competitors[1])
        comp_id = sport_event.get('tournament', {}).get('id', '')
        sr_stat = status.get('status', 'scheduled')

        status_map = {
            'closed':      'post', 'ended': 'post', 'complete': 'post',
            'live':        'in',   'inprogress': 'in', '1st_half': 'in',
            '2nd_half':    'in',   'halftime': 'in', 'overtime': 'in',
            'scheduled':   'pre',  'not_started': 'pre', 'created': 'pre',
            'postponed':   'pre',  'cancelled': 'canceled',
        }
        internal_status = status_map.get(sr_stat, 'pre')

        hg = status.get('home_score', '')
        ag = status.get('away_score', '')
        minute = status.get('clock', {}).get('played', '') if isinstance(status.get('clock'), dict) else ''

        return {
            'id':        sport_event.get('id', ''),
            'name':      f"{home.get('name','')} vs {away.get('name','')}",
            'date':      sport_event.get('start_time', ''),
            'status':    internal_status,
            'league':    LEAGUE_MAP.get(comp_id, comp_id),
            'sport':     'soccer',
            'homeTeam':  home.get('name', ''),
            'awayTeam':  away.get('name', ''),
            'homeScore': str(hg) if internal_status != 'pre' else '',
            'awayScore': str(ag) if internal_status != 'pre' else '',
            'homeAbbr':  home.get('abbreviation', ''),
            'awayAbbr':  away.get('abbreviation', ''),
            'homeLogo':  '', 'awayLogo': '',
            'period':    status.get('period', 0),
            'clock':     str(minute) + "'" if minute else '',
            'detail':    sr_stat,
            'source':    'sportradar',
            # Para el scan:
            '_homeGoals': int(hg) if str(hg).isdigit() else None,
            '_awayGoals': int(ag) if str(ag).isdigit() else None,
            '_final':     internal_status == 'post',
        }

    # ── Summaries por fecha ───────────────────────────────────────
    def get_summaries_by_date(self, date_str):
        """
        Obtiene todos los partidos de una fecha.
        date_str: 'YYYY-MM-DD'
        Endpoint confirmado 200 en trial: schedules/{fecha}/summaries.json
        """
        data = self._get(f"schedules/{date_str}/summaries.json")
        if not data:
            return []
        parsed = []
        for s in data.get('summaries', []):
            m = self._parse_summary(s)
            if m:
                parsed.append(m)
        return parsed

    # ── Partidos de hoy y en vivo ─────────────────────────────────
    def get_today_schedule(self):
        """Todos los partidos de hoy."""
        today = datetime.utcnow().strftime('%Y-%m-%d')
        return self.get_summaries_by_date(today)

    def get_live_matches(self):
        """Partidos Soccer en vivo ahora mismo (filtrado de summaries de hoy)."""
        return [m for m in self.get_today_schedule() if m['status'] == 'in']

    def get_upcoming_matches(self, days_ahead=1):
        """
        Partidos programados de hoy + próximos días.
        Incluye La Liga, Premier, Champions, etc.
        Útil para mostrar fixtures futuros en pronósticos.
        """
        all_matches = []
        seen = set()
        for i in range(days_ahead + 1):
            date_str = (datetime.utcnow() + timedelta(days=i)).strftime('%Y-%m-%d')
            matches = self.get_summaries_by_date(date_str)
            for m in matches:
                key = (m['homeTeam'], m['awayTeam'])
                if key not in seen:
                    seen.add(key)
                    all_matches.append(m)
            if i < days_ahead:
                time.sleep(1.2)  # respetar rate limit entre fechas
        return all_matches

    # ── Competencias ──────────────────────────────────────────────
    def get_competitions(self):
        """Lista de competencias — confirmado 200 en trial."""
        data = self._get("competitions.json")
        return data.get('competitions', []) if data else []

    # ── Scan histórico para GLAI ──────────────────────────────────
    def scan(self, days_back=7, glai=None, on_progress=None):
        """
        Escanea los últimos N días usando schedules/{fecha}/summaries.json
        y alimenta a GLAI con los resultados finales encontrados.
        Endpoint confirmado 200 en trial.
        """
        if not self._ok():
            print("[Sportradar] scan() omitido — sin API key")
            return 0

        total = 0
        today = datetime.utcnow()
        dates = [(today - timedelta(days=i)).strftime('%Y-%m-%d')
                 for i in range(0, days_back + 1)]

        print(f"[Sportradar] Escaneando {len(dates)} días via summaries.json...")

        for i, date_str in enumerate(dates):
            try:
                matches = self.get_summaries_by_date(date_str)
                finals  = [m for m in matches if m['_final']
                           and m['_homeGoals'] is not None
                           and m['homeTeam'] and m['awayTeam']]

                for m in finals:
                    if glai:
                        glai.learn('soccer', m['league'],
                                   m['homeTeam'], m['awayTeam'],
                                   m['_homeGoals'], m['_awayGoals'],
                                   source='sportradar', event_id=f"sr_{m['id']}")
                        total += 1

                if finals:
                    print(f"[Sportradar] {date_str}: {len(finals)} resultados")

                if on_progress:
                    pct = round((i + 1) / len(dates) * 30)
                    on_progress(pct, f"Sportradar {date_str} — {total} resultados", total)

            except Exception as e:
                print(f"[Sportradar] Error en {date_str}: {e}")
                continue

            time.sleep(1.2)  # respetar rate limit trial

        print(f"[Sportradar] ✅ Scan completo — {total} resultados aprendidos")
        return total

    # ── Estado ────────────────────────────────────────────────────
    def status(self):
        if not self._ok():
            return {'ok': False, 'msg': 'Sin API key configurada', 'key_set': False}
        try:
            url = f"{BASE_URL}/competitions.json"
            r   = self.session.get(url, headers={
                'Accept': 'application/json', 'x-api-key': self.api_key,
            }, timeout=10)
            if r.status_code in (401, 403):
                return {'ok': False, 'msg': f'API key inválida ({r.status_code})', 'key_set': True}
            if not r.ok:
                return {'ok': False, 'msg': f'Error HTTP {r.status_code}', 'key_set': True}
            comps = len(r.json().get('competitions', []))
            return {
                'ok':           True,
                'msg':          f'Sportradar Soccer conectado — {comps} competencias',
                'key_set':      True,
                'access_level': ACCESS_LEVEL,
                'competitions': comps,
            }
        except Exception as e:
            return {'ok': False, 'msg': f'Error: {e}', 'key_set': True}
