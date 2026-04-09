"""
scrapers/besoccer.py — BeSoccer Pro API (requiere token)
"""
import requests
import time
from datetime import datetime, timedelta

BSC_BASE = "https://api.besoccer.com"

BSC_ESPN = {
    1:'esp.1', 2:'uefa.champions', 3:'ger.1', 4:'fra.1', 5:'ita.1',
    6:'uefa.europa', 7:'por.1', 9:'ned.1', 10:'sco.1', 11:'eng.1',
    12:'eng.2', 13:'ger.2', 14:'mex.1', 16:'bra.1', 17:'arg.1',
    18:'usa.1', 19:'bel.1', 20:'tur.1', 40:'concacaf.champions',
    848:'uefa.conference',
}

class BeSoccerScraper:
    def __init__(self, token):
        self.token = token
        self.session = requests.Session()

    def _get(self, path, timeout=8):
        try:
            url = f"{BSC_BASE}/{path}&token={self.token}"
            r = self.session.get(url, timeout=timeout)
            return r.json() if r.ok else None
        except Exception:
            return None

    def scan(self, days_back=30, glai=None):
        total = 0
        today = datetime.utcnow()
        for i in range(1, days_back + 1):
            day = (today - timedelta(days=i)).strftime('%Y-%m-%d')
            data = self._get(f"matches/day?day={day}")
            matches = (data or {}).get('data') or (data or {}).get('matches') or []
            for m in matches:
                st = str((m.get('status') or '')).lower()
                if not any(x in st for x in ['fin', 'end', 'complet', '2']):
                    continue
                hn = (m.get('local') or {}).get('name') or m.get('home_team') or ''
                an = (m.get('visitor') or {}).get('name') or m.get('away_team') or ''
                hg = (m.get('local') or {}).get('score')
                ag = (m.get('visitor') or {}).get('score')
                if hg is None: hg = m.get('goals_home')
                if ag is None: ag = m.get('goals_away')
                if None in (hg, ag, hn, an) or hn == '' or an == '':
                    continue
                try:
                    hg, ag = int(hg), int(ag)
                except (ValueError, TypeError):
                    continue
                comp_id = m.get('competition_id') or m.get('comp_id')
                espn_lg = BSC_ESPN.get(int(comp_id), f'bsc_{comp_id}') if comp_id else 'bsc_unknown'
                if glai:
                    glai.learn('soccer', espn_lg, hn, an, hg, ag,
                               source='besoccer', event_id=f'bsc_{m.get("id")}')
                    total += 1
            time.sleep(0.3)
        return total
