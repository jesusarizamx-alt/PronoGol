"""
scrapers/footballdata.py — football-data.org API (plan gratuito)
Registro gratuito en: https://www.football-data.org/client/register
Cubre: Premier League, La Liga, Bundesliga, Serie A, Ligue 1,
       Champions League, Europa League, Eredivisie, Primeira Liga,
       Championship, Serie B y más.
Sin límite de solicitudes en el plan free (10 req/min).
"""
import requests
import time
from datetime import datetime, timedelta

BASE_URL = "https://api.football-data.org/v4"

# Competiciones disponibles en el plan GRATUITO
FREE_COMPETITIONS = {
    'PL':  'eng.1',   # Premier League
    'PD':  'esp.1',   # La Liga
    'BL1': 'ger.1',   # Bundesliga
    'SA':  'ita.1',   # Serie A
    'FL1': 'fra.1',   # Ligue 1
    'DED': 'ned.1',   # Eredivisie
    'PPL': 'por.1',   # Primeira Liga
    'CL':  'uefa.champions',  # Champions League
    'EL':  'uefa.europa',     # Europa League
    'ECL': 'uefa.conference', # Conference League
    'BSA': 'bra.1',   # Brasileirão
    'MLS': 'usa.1',   # MLS
}

# Mapeo de status de football-data → nuestro status
STATUS_MAP = {
    'FINISHED':   'final',
    'IN_PLAY':    'live',
    'PAUSED':     'live',
    'SCHEDULED':  'pre',
    'TIMED':      'pre',
    'POSTPONED':  'postponed',
    'CANCELLED':  'cancelled',
    'SUSPENDED':  'live',
}


class FootballDataScraper:
    def __init__(self, token: str):
        """
        token: API token gratuito de football-data.org
        Registro en https://www.football-data.org/client/register
        """
        self.token = token
        self.session = requests.Session()
        self.session.headers.update({
            'X-Auth-Token': token,
            'User-Agent': 'SoccerPredictorPro/1.0',
        })

    def _get(self, path, params=None, timeout=10):
        """GET con reintentos y respeto al rate limit (10 req/min)."""
        url = f"{BASE_URL}/{path}"
        for attempt in range(3):
            try:
                r = self.session.get(url, params=params, timeout=timeout)
                if r.status_code == 200:
                    return r.json()
                if r.status_code == 429:
                    # Rate limit — esperar 12 segundos
                    print(f"[FBData] Rate limit, esperando 12s...")
                    time.sleep(12)
                    continue
                if r.status_code == 403:
                    print(f"[FBData] Token inválido o plan no cubre: {path}")
                    return None
                if r.status_code == 404:
                    return None
            except Exception as e:
                print(f"[FBData] Error GET {path}: {e}")
                time.sleep(2)
        return None

    def get_matches(self, competition_code, date_from=None, date_to=None, status=None):
        """
        Obtiene partidos de una competición.
        competition_code: PL, PD, BL1, SA, FL1, CL, etc.
        date_from/date_to: YYYY-MM-DD
        status: FINISHED | SCHEDULED | IN_PLAY
        """
        params = {}
        if date_from:
            params['dateFrom'] = date_from
        if date_to:
            params['dateTo'] = date_to
        if status:
            params['status'] = status

        data = self._get(f"competitions/{competition_code}/matches", params=params)
        if not data:
            return []

        matches = []
        for m in (data.get('matches') or []):
            home = m.get('homeTeam', {}).get('name', '')
            away = m.get('awayTeam', {}).get('name', '')
            if not home or not away:
                continue

            score = m.get('score', {})
            full  = score.get('fullTime', {})
            home_score = full.get('home')
            away_score = full.get('away')

            status_raw = m.get('status', 'SCHEDULED')
            our_status = STATUS_MAP.get(status_raw, 'pre')

            matches.append({
                'homeTeam':  home,
                'awayTeam':  away,
                'homeScore': home_score,
                'awayScore': away_score,
                'homeLogo':  m.get('homeTeam', {}).get('crest', ''),
                'awayLogo':  m.get('awayTeam', {}).get('crest', ''),
                'date':      m.get('utcDate', ''),
                'league':    FREE_COMPETITIONS.get(competition_code, competition_code),
                'status':    our_status,
                'eventId':   f"fd_{m.get('id','')}",
            })
        return matches

    def get_past_results(self, days_back=14):
        """Resultados de los últimos N días en todas las ligas del plan free."""
        date_to   = datetime.utcnow().strftime('%Y-%m-%d')
        date_from = (datetime.utcnow() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        all_matches = []
        for code in FREE_COMPETITIONS:
            try:
                matches = self.get_matches(
                    code,
                    date_from=date_from,
                    date_to=date_to,
                    status='FINISHED'
                )
                all_matches.extend(matches)
                time.sleep(6)  # Respetar 10 req/min
            except Exception as e:
                print(f"[FBData] Error {code}: {e}")
        return all_matches

    def get_fixtures(self, days_ahead=7):
        """Partidos programados en los próximos N días."""
        date_from = datetime.utcnow().strftime('%Y-%m-%d')
        date_to   = (datetime.utcnow() + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
        fixtures = []
        for code in FREE_COMPETITIONS:
            try:
                matches = self.get_matches(
                    code,
                    date_from=date_from,
                    date_to=date_to,
                    status='SCHEDULED'
                )
                fixtures.extend(matches)
                time.sleep(6)
            except Exception as e:
                print(f"[FBData] Error fixtures {code}: {e}")
        return fixtures

    def scan(self, days_back=14, days_ahead=7, glai=None):
        """
        Scan completo:
        - Aprende resultados pasados en GLAI
        - Retorna fixtures futuros
        """
        learned = 0
        fixtures = []

        # ── Resultados pasados ──────────────────────────────────────
        print(f"[FBData] Escaneando {days_back} días de resultados...")
        past = self.get_past_results(days_back)
        for m in past:
            if glai and m['homeScore'] is not None and m['awayScore'] is not None:
                try:
                    ok = glai.learn(
                        sport='soccer',
                        league_id=m['league'],
                        home_team=m['homeTeam'],
                        away_team=m['awayTeam'],
                        home_goals=int(m['homeScore']),
                        away_goals=int(m['awayScore']),
                        source='footballdata',
                        event_id=m['eventId'],
                    )
                    if ok:
                        learned += 1
                except Exception:
                    pass

        # ── Fixtures futuros ────────────────────────────────────────
        print(f"[FBData] Buscando fixtures próximos {days_ahead} días...")
        fixtures = self.get_fixtures(days_ahead)

        print(f"[FBData] ✅ {learned} aprendidos · {len(fixtures)} fixtures")
        return learned, fixtures
