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
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
            'Origin': 'https://www.espn.com',
            'Referer': 'https://www.espn.com/',
        })

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
        _sport_path_map = {'soccer':'soccer','nba':'basketball','mlb':'baseball','nhl':'hockey'}
        sport_path  = _sport_path_map.get(sport, 'soccer')
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
        params = {'limit': 50}
        if date_str:
            params['dates'] = date_str
        try:
            r = self.session.get(url, params=params, timeout=10)
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

    # ─── NHL (API oficial NHL.com — sin key, gratis) ─────────────
    def get_nhl_matches(self, date_str=None):
        """
        Partidos NHL de hoy usando NHL API oficial (api-web.nhle.com).
        Fallback a ESPN con fecha explícita en formato YYYYMMDD.
        date_str: 'YYYY-MM-DD' o None para hoy.
        """
        from datetime import datetime, timezone
        today_dash  = date_str or datetime.now(timezone.utc).strftime('%Y-%m-%d')
        today_nodash = today_dash.replace('-', '')  # ESPN usa YYYYMMDD

        # Intento 1: NHL API oficial
        results = self._get_nhl_from_nhle(date_str)
        if results:
            return results

        # Intento 2: ESPN con fecha explícita YYYYMMDD
        fallback = []
        for ev in self._fetch_scoreboard('hockey', 'nhl', today_nodash):
            d = self._event_dict(ev, 'nhl', sport='nhl')
            if d:
                fallback.append(d)
        if fallback:
            return fallback

        # Intento 3: ESPN sin fecha (por si acaso)
        for ev in self._fetch_scoreboard('hockey', 'nhl', None):
            d = self._event_dict(ev, 'nhl', sport='nhl')
            if d:
                fallback.append(d)
        return fallback

    def _get_nhl_from_nhle(self, date_str=None):
        """
        NHL API oficial: https://api-web.nhle.com/v1/scoreboard/now
        Devuelve partidos del día en formato estándar del sistema.
        gameState values: FUT=futuro, PRE=previo, LIVE=en vivo, CRIT=en vivo crítico, FINAL=terminado, OFF=terminado
        """
        NHL_API = 'https://api-web.nhle.com/v1'
        from datetime import datetime, timezone
        today = date_str or datetime.now(timezone.utc).strftime('%Y-%m-%d')

        games = []
        try:
            if date_str:
                # /v1/schedule/YYYY-MM-DD
                url = f'{NHL_API}/schedule/{date_str}'
                r = self.session.get(url, timeout=8)
                if r.ok:
                    for day in r.json().get('gameWeek', []):
                        if date_str in day.get('date', ''):
                            games = day.get('games', [])
                            break
            else:
                # Primero intentar scoreboard/now (más actualizado)
                url = f'{NHL_API}/scoreboard/now'
                r = self.session.get(url, timeout=8)
                if r.ok:
                    data = r.json()
                    games = data.get('games', [])
                    # Si el scoreboard no tiene juegos de HOY, buscar por fecha explícita
                    today_games = [g for g in games
                                   if today in g.get('startTimeUTC', '')]
                    if not today_games and games:
                        # scoreboard está en otro día — buscar hoy explícito
                        games = []

                # Si no hay juegos o el scoreboard falló, usar schedule por fecha
                if not games:
                    url2 = f'{NHL_API}/schedule/{today}'
                    r2 = self.session.get(url2, timeout=8)
                    if r2.ok:
                        for day in r2.json().get('gameWeek', []):
                            if today in day.get('date', ''):
                                games = day.get('games', [])
                                break

            results = []
            for g in games:
                home = g.get('homeTeam', {})
                away = g.get('awayTeam', {})
                hn = home.get('name', {})
                an = away.get('name', {})
                # El API NHL retorna nombre como objeto {default, fr} o string
                home_name = hn.get('default', '') if isinstance(hn, dict) else str(hn)
                away_name = an.get('default', '') if isinstance(an, dict) else str(an)
                # Fallback: usar placeName + teamName
                if not home_name:
                    home_name = (home.get('placeName', {}).get('default', '') + ' ' +
                                 home.get('teamName', {}).get('default', '')).strip()
                if not away_name:
                    away_name = (away.get('placeName', {}).get('default', '') + ' ' +
                                 away.get('teamName', {}).get('default', '')).strip()

                # Estado del juego
                state = g.get('gameState', '')   # FUT, PRE, LIVE, CRIT, FINAL, OFF
                if state in ('FINAL', 'OFF'):
                    status = 'STATUS_FINAL'
                elif state in ('LIVE', 'CRIT'):
                    status = 'STATUS_IN_PROGRESS'
                else:
                    status = 'STATUS_SCHEDULED'

                # Scores
                home_score = home.get('score', '')
                away_score = away.get('score', '')
                if home_score == 0 and state not in ('LIVE', 'CRIT', 'FINAL', 'OFF'):
                    home_score = ''
                if away_score == 0 and state not in ('LIVE', 'CRIT', 'FINAL', 'OFF'):
                    away_score = ''

                # Logos
                home_logo = home.get('logo', '') or f"https://assets.nhle.com/logos/nhl/svg/{home.get('abbrev','')}_light.svg"
                away_logo = away.get('logo', '') or f"https://assets.nhle.com/logos/nhl/svg/{away.get('abbrev','')}_light.svg"

                if not home_name or not away_name:
                    continue

                results.append({
                    'id':        str(g.get('id', '')),
                    'name':      f"{home_name} vs {away_name}",
                    'date':      g.get('startTimeUTC', ''),
                    'status':    status,
                    'league':    'nhl',
                    'sport':     'nhl',
                    'homeTeam':  home_name,
                    'awayTeam':  away_name,
                    'homeScore': '' if home_score == '' else str(home_score),
                    'awayScore': '' if away_score == '' else str(away_score),
                    'homeAbbr':  home.get('abbrev', ''),
                    'awayAbbr':  away.get('abbrev', ''),
                    'homeLogo':  home_logo,
                    'awayLogo':  away_logo,
                    'source':    'nhle',
                })
            return results
        except Exception as e:
            print(f'[NHL API error] {e}')
            return []

    # ─── Todos los deportes ───────────────────────────────────────
    def get_all_today(self):
        """Soccer + NBA + MLB + NHL — todos los partidos de hoy."""
        all_matches = []
        all_matches.extend(self.get_today_matches())
        all_matches.extend(self.get_nba_matches())
        all_matches.extend(self.get_mlb_matches())
        all_matches.extend(self.get_nhl_matches())
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

        # ── NHL histórico — usa NHL API oficial (nhle.com) con fallback ESPN ──
        NHL_API = 'https://api-web.nhle.com/v1'
        for date_str in dates:
            learned_this_date = False
            # Intento 1: NHL API oficial /v1/schedule/YYYY-MM-DD
            try:
                url_nhl = f'{NHL_API}/schedule/{date_str}'
                rn = self.session.get(url_nhl, timeout=6)
                if rn.ok:
                    game_week = rn.json().get('gameWeek', [])
                    for day in game_week:
                        if date_str not in day.get('date', ''):
                            continue
                        for g in day.get('games', []):
                            state = g.get('gameState', '')
                            if state not in ('FINAL', 'OFF'):
                                continue
                            home = g.get('homeTeam', {})
                            away = g.get('awayTeam', {})
                            hg = home.get('score', '')
                            ag = away.get('score', '')
                            if hg == '' or ag == '':
                                continue
                            try:
                                hg, ag = int(hg), int(ag)
                            except (ValueError, TypeError):
                                continue
                            # Nombre completo: place + team name
                            hn_p = home.get('placeName', {})
                            hn_t = home.get('teamName', {})
                            an_p = away.get('placeName', {})
                            an_t = away.get('teamName', {})
                            hn = ((hn_p.get('default','') if isinstance(hn_p,dict) else str(hn_p)) + ' ' +
                                  (hn_t.get('default','') if isinstance(hn_t,dict) else str(hn_t))).strip()
                            an = ((an_p.get('default','') if isinstance(an_p,dict) else str(an_p)) + ' ' +
                                  (an_t.get('default','') if isinstance(an_t,dict) else str(an_t))).strip()
                            if glai and hn and an:
                                glai.learn('nhl', 'nhl', hn, an, hg, ag,
                                           source='nhle', event_id=f'nhle_{g.get("id","")}')
                                total += 1
                                learned_this_date = True
            except Exception:
                pass

            # Intento 2: fallback ESPN si NHL API no dio datos
            if not learned_this_date:
                try:
                    url = f"{ESPN_BASE}/hockey/nhl/scoreboard"
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
                                glai.learn('nhl', 'nhl', hn, an, hg, ag,
                                           source='espn_nhl', event_id=f'espn_nhl_{ev.get("id")}')
                                total += 1
                except Exception:
                    pass
            time.sleep(0.15)

        if on_progress:
            on_progress(58, f'ESPN completo — {total} resultados aprendidos', total)
        return total
