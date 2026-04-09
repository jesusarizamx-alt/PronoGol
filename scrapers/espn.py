"""
scrapers/espn.py — Fetcher de ESPN sin restricciones de CORS
El backend llama directo a ESPN API — el browser ya no tiene que hacerlo.
"""
import requests
import time
from datetime import datetime, timedelta

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer"

LEAGUES = [
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

    def get_today_matches(self):
        """Retorna partidos de hoy de todas las ligas activas."""
        results = []
        for lg_id in LEAGUES:
            try:
                url = f"{ESPN_BASE}/{lg_id}/scoreboard"
                r = self.session.get(url, timeout=8)
                if not r.ok:
                    continue
                data = r.json()
                for ev in (data.get('events') or []):
                    comp = ev.get('competitions', [{}])[0]
                    competitors = comp.get('competitors', [])
                    if len(competitors) < 2:
                        continue
                    home = next((c for c in competitors if c.get('homeAway') == 'home'), competitors[0])
                    away = next((c for c in competitors if c.get('homeAway') == 'away'), competitors[1])
                    results.append({
                        'id':        ev.get('id'),
                        'name':      ev.get('name', ''),
                        'date':      ev.get('date', ''),
                        'status':    comp.get('status', {}).get('type', {}).get('name', ''),
                        'league':    lg_id,
                        'homeTeam':  home.get('team', {}).get('displayName', ''),
                        'awayTeam':  away.get('team', {}).get('displayName', ''),
                        'homeScore': home.get('score', ''),
                        'awayScore': away.get('score', ''),
                        'homeAbbr':  home.get('team', {}).get('abbreviation', ''),
                        'awayAbbr':  away.get('team', {}).get('abbreviation', ''),
                        'homeLogo':  (home.get('team', {}).get('logo') or
                                      f"https://a.espncdn.com/i/teamlogos/soccer/500/{home.get('team',{}).get('id','')}.png"),
                        'awayLogo':  (away.get('team', {}).get('logo') or
                                      f"https://a.espncdn.com/i/teamlogos/soccer/500/{away.get('team',{}).get('id','')}.png"),
                    })
            except Exception:
                continue
        return results

    def get_match_detail(self, league_id, event_id):
        """Detalles completos de un partido (boxscore, stats, etc.)"""
        try:
            url = f"{ESPN_BASE}/{league_id}/summary"
            r = self.session.get(url, params={'event': event_id}, timeout=10)
            if not r.ok:
                return None
            return r.json()
        except Exception:
            return None

    def scan(self, days_back=7, glai=None, on_progress=None):
        """Escanea resultados históricos de ESPN para alimentar GLAI."""
        total = 0
        today = datetime.utcnow()
        dates = [(today - timedelta(days=i)).strftime('%Y%m%d') for i in range(1, days_back + 1)]
        bad_leagues = set()

        total_ops = len(LEAGUES) * len(dates)
        done = 0

        for date_str in dates:
            for lg_id in LEAGUES:
                if lg_id in bad_leagues:
                    done += 1
                    continue
                try:
                    url = f"{ESPN_BASE}/{lg_id}/scoreboard"
                    r = self.session.get(url, params={'dates': date_str, 'limit': 50}, timeout=6)
                    if r.status_code == 400 or r.status_code == 404:
                        bad_leagues.add(lg_id)
                        done += 1
                        continue
                    if not r.ok:
                        done += 1
                        continue
                    data = r.json()
                    for ev in (data.get('events') or []):
                        comp = ev.get('competitions', [{}])[0]
                        st = comp.get('status', {}).get('type', {}).get('name', '')
                        if 'Final' not in st and 'STATUS_FINAL' not in st:
                            continue
                        competitors = comp.get('competitors', [])
                        if len(competitors) < 2:
                            continue
                        home = next((c for c in competitors if c.get('homeAway') == 'home'), competitors[0])
                        away = next((c for c in competitors if c.get('homeAway') == 'away'), competitors[1])
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
                    pct = round(min(55, done / total_ops * 55))
                    on_progress(pct, f'ESPN {date_str} — {total} resultados', total)
                time.sleep(0.1)  # cortés con ESPN

        return total
