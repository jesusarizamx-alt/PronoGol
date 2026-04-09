"""
scrapers/thesportsdb.py — TheSportsDB (gratuito, sin key)
"""
import requests
import time

TSDB_BASE = "https://www.thesportsdb.com/api/v1/json/3"

# Top ligas para escaneo autónomo (TheSportDB league IDs)
TOP_LEAGUES = [4328, 4335, 4331, 4333, 4334, 4337, 4338, 4346, 4347, 4480, 4481]

# Mapeo TheSportDB league ID → ESPN league ID
TSDB_ESPN_MAP = {
    4328: 'eng.1', 4329: 'eng.2', 4331: 'ger.1', 4332: 'ger.2',
    4333: 'ita.1', 4334: 'fra.1', 4335: 'esp.1', 4337: 'por.1',
    4338: 'ned.1', 4339: 'bel.1', 4340: 'tur.1', 4341: 'sco.1',
    4346: 'usa.1', 4347: 'mex.1', 4399: 'arg.1', 4400: 'bra.1',
    4480: 'uefa.champions', 4481: 'uefa.europa', 4482: 'uefa.conference',
}

class TheSportsDBScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Mozilla/5.0'})

    def _get(self, path, timeout=6):
        try:
            r = self.session.get(f"{TSDB_BASE}/{path}", timeout=timeout)
            return r.json() if r.ok else None
        except Exception:
            return None

    def search_team(self, name):
        data = self._get(f"searchteams.php?t={requests.utils.quote(name)}")
        teams = data.get('teams') if data else None
        return teams[0] if teams else None

    def team_last_events(self, team_id):
        results = []
        for page in [1, 2]:
            suffix = '' if page == 1 else '&r=2'
            data = self._get(f"eventslast.php?id={team_id}{suffix}")
            if data and data.get('results'):
                results.extend(data['results'])
        return results

    def get_team_history(self, team_name, limit=5):
        """
        Busca el equipo por nombre y retorna últimos N partidos.
        Sin límite de fecha — trae los más recientes sin importar competencia.
        """
        team = self.search_team(team_name)
        if not team:
            return []
        events = self.team_last_events(team['idTeam'])
        hist = []
        seen = set()
        for ev in sorted(events, key=lambda x: x.get('dateEvent',''), reverse=True):
            eid = ev.get('idEvent')
            if eid in seen:
                continue
            seen.add(eid)
            hg = ev.get('intHomeScore')
            ag = ev.get('intAwayScore')
            if hg is None or ag is None:
                continue
            try:
                hg, ag = int(hg), int(ag)
            except (ValueError, TypeError):
                continue
            is_home = team_name.lower()[:6] in (ev.get('strHomeTeam','') or '').lower()
            my_g  = hg if is_home else ag
            opp_g = ag if is_home else hg
            opp   = ev.get('strAwayTeam') if is_home else ev.get('strHomeTeam')
            lg_raw = ev.get('idLeague')
            espn_lg = TSDB_ESPN_MAP.get(int(lg_raw), f'tsdb_{lg_raw}') if lg_raw else 'tsdb_unknown'
            result = 'G' if my_g > opp_g else ('P' if my_g < opp_g else 'E')
            hist.append({
                'result': result, 'myG': my_g, 'oppG': opp_g,
                'opp': opp, 'comp': ev.get('strLeague', espn_lg),
                'isHome': is_home, 'date': ev.get('dateEvent', ''),
            })
            if len(hist) >= limit:
                break
        return hist

    def learn_team_events(self, team_name, glai, league_id=None):
        """Busca equipo y alimenta GLAI con sus últimos partidos."""
        team = self.search_team(team_name)
        if not team:
            return 0
        events = self.team_last_events(team['idTeam'])
        total = 0
        for ev in events:
            hg = ev.get('intHomeScore')
            ag = ev.get('intAwayScore')
            hn = ev.get('strHomeTeam')
            an = ev.get('strAwayTeam')
            if None in (hg, ag, hn, an):
                continue
            try:
                hg, ag = int(hg), int(ag)
            except (ValueError, TypeError):
                continue
            lg_raw = ev.get('idLeague')
            espn_lg = TSDB_ESPN_MAP.get(int(lg_raw), f'tsdb_{lg_raw}') if lg_raw else (league_id or 'tsdb_unk')
            glai.learn('soccer', espn_lg, hn, an, hg, ag,
                       source='tsdb', event_id=f'tsdb_{ev.get("idEvent")}')
            total += 1
        return total

    def scan_top_leagues(self, days_back=7, glai=None):
        """Escanea top ligas en TheSportsDB para alimentar GLAI."""
        total = 0
        for lg_id in TOP_LEAGUES:
            try:
                data = self._get(f"eventspastleague.php?id={lg_id}")
                if not data or not data.get('events'):
                    continue
                espn_lg = TSDB_ESPN_MAP.get(lg_id, f'tsdb_{lg_id}')
                for ev in (data.get('events') or []):
                    hg = ev.get('intHomeScore')
                    ag = ev.get('intAwayScore')
                    hn = ev.get('strHomeTeam')
                    an = ev.get('strAwayTeam')
                    if None in (hg, ag, hn, an):
                        continue
                    try:
                        hg, ag = int(hg), int(ag)
                    except (ValueError, TypeError):
                        continue
                    if glai:
                        glai.learn('soccer', espn_lg, hn, an, hg, ag,
                                   source='tsdb', event_id=f'tsdb_{ev.get("idEvent")}')
                        total += 1
                time.sleep(0.3)
            except Exception:
                continue
        return total
