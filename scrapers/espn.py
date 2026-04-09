"""
scrapers/espn.py — Multi-sport: Soccer, NBA, MLB
El backend llama directo a ESPN API — sin CORS.
"""
import requests
import time
from datetime import datetime, timedelta

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports"

SOCCER_LEAGUES = [
    'eng.1','eng.2','esp.1','ger.1','ger.2','ita.1','fra.1',
    'por.1','ned.1','bel.1','tur.1','sco.1','aut.1','gre.1',
    'mex.1','usa.1','arg.1','bra.1','col.1','chi.1',
    'uefa.champions','uefa.europa','uefa.conference',
    'concacaf.champions','concacaf.league',
]

class ESPNScraper:
    def __init__(self, keys=None):
        self.keys = keys or {}
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Mozilla/5.0'})

    # ─── Helpers ──────────────────────────────────────────────────
    def _parse_competitors(self, competitors):
        home = next((c for c in competitors if c.get('homeAway') == 'home'), competitors[0])
        away = next((c for c in competitors if c.get('homeAway') == 'away'), competitors[1])
        return home, away

    def _event_dict(self, ev, lg_id, sport='soccer'):
        comp        = ev.get('competitions', [{}])[0]
        competitors = comp.get('competitors', [])
        if len(competitors) < 2:
            return None
        home, away = self._parse_competitors(competitors)
        sport_path  = 'soccer' if sport == 'soccer' else ('basketball' if sport == 'nba' else 'baseball')
        logo_base   = f"https://a.espncdn.com/i/teamlogos/{sport_path}/500"
        return {
            'id':        ev.get('id'),
            'name':      ev.get('name', ''),
            'date':      ev.get('date', ''),
            'status':    comp.get('status', {}).get('type', {}).get('name', ''),
            'league':    lg_id,
            'sport':     sport,
            'homeTeam':  home.get('team', {}).get('displayName', ''),
            'awayTeam':  away.get('team', {}).get('displayName', ''),
            'homeScore': home.get('score', ''),
            'awayScore': away.get('score', ''),
            'homeAbbr':  home.get('team', {}).get('abbreviation', ''),
            'awayAbbr':  away.get('team', {}).get('abbreviation', ''),
            'homeLogo':  (home.get('team', {}).get('logo') or
                          f"{logo_base}/{home.get('team',{}).get('id','')}.png"),
            'awayLogo':  (away.get('team', {}).get('logo') or
                          f"{logo_base}/{away.get('team',{}).get('id','')}.png"),
        }

    def _fetch_scoreboard(self, sport_path, league_id, date_str=None):
        url    = f"{ESPN_BASE}/{sport_path}/{league_id}/scoreboard"
        params = {}
        if date_str:
            params['dates'] = date_str
        try:
            r = self.session.get(url, params=params, timeout=8)
            if not r.ok:
                return []
            return r.json().get('events') or []
        except Exception:
            return []

    # ─── Soccer ───────────────────────────────────────────────────
    def get_today_matches(self):
        """Partidos de fútbol de hoy (todas las ligas)."""
        results = []
        for lg_id in SOCCER_LEAGUES:
            for ev in self._fetch_scoreboard('soccer', lg_id):
                d = self._event_dict(ev, lg_id, sport='soccer')
                if d:
                    results.append(d)
        return results

    # ─── NBA ──────────────────────────────────────────────────────
    def get_nba_matches(self, date_str=None):
        """Partidos NBA de hoy (o de una fecha específica)."""
        results = []
        for ev in self._fetch_scoreboard('basketball', 'nba', date_str):
            d = self._event_dict(ev, 'nba', sport='nba')
            if d:
                results.append(d)
        return results

    # ─── MLB ──────────────────────────────────────────────────────
    def get_mlb_matches(self, date_str=None):
        """Partidos MLB de hoy (o de una fecha específica)."""
        results = []
        for ev in self._fetch_scoreboard('baseball', 'mlb', date_str):
            d = self._event_dict(ev, 'mlb', sport='mlb')
            if d:
                results.append(d)
        return results

    # ─── Todos los deportes ───────────────────────────────────────
    def get_all_today(self):
        """Soccer + NBA + MLB — todos los partidos de hoy."""
        all_matches = []
        all_matches.extend(self.get_today_matches())
        all_matches.extend(self.get_nba_matches())
        all_matches.extend(self.get_mlb_matches())
        return all_matches

    def get_match_detail(self, league_id, event_id):
        """Detalles completos de un partido soccer."""
        try:
            url = f"{ESPN_BASE}/soccer/{league_id}/summary"
            r = self.session.get(url, params={'event': event_id}, timeout=10)
            if not r.ok:
                return None
            return r.json()
        except Exception:
            return None

    # ─── Scan histórico ───────────────────────────────────────────
    def scan(self, days_back=7, glai=None, on_progress=None):
        """Escanea resultados históricos — Soccer + NBA + MLB."""
        total = 0
        today = datetime.utcnow()
        dates = [(today - timedelta(days=i)).strftime('%Y%m%d') for i in range(1, days_back + 1)]
        bad_leagues = set()

        # ── Soccer ────────────────────────────────────────────────
        total_ops = len(SOCCER_LEAGUES) * len(dates)
        done = 0
        for date_str in dates:
            for lg_id in SOCCER_LEAGUES:
                if lg_id in bad_leagues:
                    done += 1
                    continue
                try:
                    url = f"{ESPN_BASE}/soccer/{lg_id}/scoreboard"
                    r = self.session.get(url, params={'dates': date_str, 'limit': 50}, timeout=6)
                    if r.status_code in (400, 404):
                        bad_leagues.add(lg_id)
                        done += 1
                        continue
                    if not r.ok:
                        done += 1
                        continue
                    for ev in (r.json().get('events') or []):
                        comp = ev.get('competitions', [{}])[0]
                        st   = comp.get('status', {}).get('type', {}).get('name', '')
                        if 'Final' not in st and 'STATUS_FINAL' not in st:
                            continue
                        competitors = comp.get('competitors', [])
                        if len(competitors) < 2:
                            continue
                        home, away = self._parse_competitors(competitors)
                        hg = home.get('score', '')
                        ag = away.get('score', '')
                        if hg == '' or ag == '':
                            continue
                        try:
                            hg, ag = int(hg), int(ag)
                        except (ValueError, TypeError):
                            continue
                        hn = home.get('team', {}).get('displayName', '')
                        an = away.get('team', {}).get('displayName', '')
                        if glai and hn and an:
                            glai.learn('soccer', lg_id, hn, an, hg, ag,
                                       source='espn', event_id=f'espn_{ev.get("id")}')
                            total += 1
                except Exception:
                    pass
                done += 1
                if on_progress and done % 10 == 0:
                    pct = round(min(50, done / total_ops * 50))
                    on_progress(pct, f'ESPN soccer {date_str} — {total} resultados', total)
                time.sleep(0.1)

        # ── NBA histórico ─────────────────────────────────────────
        for date_str in dates:
            try:
                url = f"{ESPN_BASE}/basketball/nba/scoreboard"
                r   = self.session.get(url, params={'dates': date_str, 'limit': 50}, timeout=6)
                if r.ok:
                    for ev in (r.json().get('events') or []):
                        comp = ev.get('competitions', [{}])[0]
                        st   = comp.get('status', {}).get('type', {}).get('name', '')
                        if 'Final' not in st and 'STATUS_FINAL' not in st:
                            continue
                        competitors = comp.get('competitors', [])
                        if len(competitors) < 2:
                            continue
                        home, away = self._parse_competitors(competitors)
                        hg = home.get('score', '')
                        ag = away.get('score', '')
                        if hg == '' or ag == '':
                            continue
                        try:
                            hg, ag = int(float(hg)), int(float(ag))
                        except (ValueError, TypeError):
                            continue
                        hn = home.get('team', {}).get('displayName', '')
                        an = away.get('team', {}).get('displayName', '')
                        if glai and hn and an:
                            glai.learn('nba', 'nba', hn, an, hg, ag,
                                       source='espn_nba', event_id=f'espn_nba_{ev.get("id")}')
                            total += 1
            except Exception:
                pass
            time.sleep(0.15)

        # ── MLB histórico ─────────────────────────────────────────
        for date_str in dates:
            try:
                url = f"{ESPN_BASE}/baseball/mlb/scoreboard"
                r   = self.session.get(url, params={'dates': date_str, 'limit': 50}, timeout=6)
                if r.ok:
                    for ev in (r.json().get('events') or []):
                        comp = ev.get('competitions', [{}])[0]
                        st   = comp.get('status', {}).get('type', {}).get('name', '')
                        if 'Final' not in st and 'STATUS_FINAL' not in st:
                            continue
                        competitors = comp.get('competitors', [])
                        if len(competitors) < 2:
                            continue
                        home, away = self._parse_competitors(competitors)
                        hg = home.get('score', '')
                        ag = away.get('score', '')
                        if hg == '' or ag == '':
                            continue
                        try:
                            hg, ag = int(float(hg)), int(float(ag))
                        except (ValueError, TypeError):
                            continue
                        hn = home.get('team', {}).get('displayName', '')
                        an = away.get('team', {}).get('displayName', '')
                        if glai and hn and an:
                            glai.learn('mlb', 'mlb', hn, an, hg, ag,
                                       source='espn_mlb', event_id=f'espn_mlb_{ev.get("id")}')
                            total += 1
            except Exception:
                pass
            time.sleep(0.15)

        if on_progress:
            on_progress(58, f'ESPN completo — {total} resultados aprendidos', total)
        return total
