"""
glai.py — Motor de IA GLAI en Python
Aprende de resultados reales, predice con Poisson, calibra con historial.
"""
import math
import json
import threading
from datetime import datetime, timedelta

class GLAIEngine:
    def __init__(self, db):
        self.db = db
        self._scan_lock = threading.Lock()
        self._scan_running = False
        self._scan_progress = {'pct': 0, 'msg': 'Inactivo', 'learned': 0}

    # ─── Aprendizaje ──────────────────────────────────────────────
    def learn(self, sport, league_id, home_team, away_team,
              home_goals, away_goals, source='unknown', event_id=None):
        return self.db.learn_result(
            sport, league_id, home_team, away_team,
            home_goals, away_goals, source, event_id
        )

    def total_learned(self):
        return self.db.get_total_learned()

    def get_league_stats(self, sport, league_id):
        return self.db.get_league_stats(sport, league_id)

    def get_all_stats(self, sport='soccer'):
        return self.db.get_all_league_stats(sport)

    # ─── Predicción Poisson ───────────────────────────────────────
    def _poisson(self, lam, k):
        """P(X=k) para distribución Poisson."""
        try:
            return math.exp(-lam) * (lam ** k) / math.factorial(k)
        except Exception:
            return 0.0

    def predict(self, sport, league_id, home_xg, away_xg, max_goals=7):
        """
        Genera matriz de marcadores y probabilidades 1X2.
        home_xg / away_xg: goles esperados (del modelo base o del frontend).
        Opcionalmente ajusta con estadísticas históricas de la liga.
        """
        lg = self.get_league_stats(sport, league_id)

        # Ajustar xG con datos históricos de la liga (si hay suficientes)
        adj_home_xg = home_xg
        adj_away_xg = away_xg
        if lg and lg['n'] >= 10:
            blend = min(0.35, lg['n'] / 300)
            adj_home_xg = home_xg * (1 - blend) + lg['avgHG'] * blend
            adj_away_xg = away_xg * (1 - blend) + lg['avgAG'] * blend

        # Matriz de probabilidades de marcador
        matrix = []
        p_home = 0.0
        p_draw = 0.0
        p_away = 0.0

        for h in range(max_goals + 1):
            for a in range(max_goals + 1):
                p = self._poisson(adj_home_xg, h) * self._poisson(adj_away_xg, a)
                matrix.append({'a': h, 'b': a, 'p': round(p, 4)})
                if h > a:
                    p_home += p
                elif h == a:
                    p_draw += p
                else:
                    p_away += p

        # Normalizar
        total = p_home + p_draw + p_away
        if total > 0:
            p_home /= total
            p_draw /= total
            p_away /= total

        # Si hay stats históricas, blend con ellas
        if lg and lg['n'] >= 10:
            blend = min(0.35, lg['n'] / 300)
            p_home = p_home * (1 - blend) + lg['homeWinRate'] * blend
            p_away = p_away * (1 - blend) + lg['awayWinRate'] * blend
            p_draw = 1 - p_home - p_away

        matrix.sort(key=lambda x: x['p'], reverse=True)

        # BTTS
        p_btts = (1 - math.exp(-adj_home_xg)) * (1 - math.exp(-adj_away_xg))

        # Over/Under 2.5
        over_25 = sum(
            self._poisson(adj_home_xg, h) * self._poisson(adj_away_xg, a)
            for h in range(max_goals + 1)
            for a in range(max_goals + 1)
            if h + a > 2.5
        )

        return {
            'pctA':    round(p_home * 100),
            'pctD':    round(p_draw * 100),
            'pctB':    round(p_away * 100),
            'xgA':     round(adj_home_xg, 2),
            'xgB':     round(adj_away_xg, 2),
            'btts':    round(p_btts * 100),
            'over25':  round(over_25 * 100),
            'matrix':  matrix[:10],
            'lgStats': lg,
            'totalLearned': self.total_learned(),
        }

    # ─── Historial por equipo ─────────────────────────────────────
    def team_history(self, team_name, limit=5):
        """Últimos N partidos de un equipo — todas las competencias."""
        results = self.db.get_team_results(team_name, limit * 3)
        hist = []
        seen = set()
        for r in results:
            key = f"{r['event_id'] or r['id']}"
            if key in seen:
                continue
            seen.add(key)
            is_home = team_name.lower()[:6] in r['home_team'].lower()
            my_g   = r['home_goals'] if is_home else r['away_goals']
            opp_g  = r['away_goals'] if is_home else r['home_goals']
            opp    = r['away_team'] if is_home else r['home_team']
            result = 'G' if my_g > opp_g else ('P' if my_g < opp_g else 'E')
            hist.append({
                'result':  result,
                'myG':     my_g,
                'oppG':    opp_g,
                'opp':     opp,
                'comp':    r['league_id'],
                'isHome':  is_home,
                'source':  r['source'],
            })
            if len(hist) >= limit:
                break
        return hist

    # ─── Análisis de apuesta IA ───────────────────────────────────
    def ai_bet(self, hist_a, hist_b, xg_a, xg_b, team_a, team_b):
        """
        Genera recomendación de apuesta basada en historial real.
        hist_a / hist_b: listas de dicts con 'myG', 'oppG', 'result'.
        """
        def avg(arr, key):
            return sum(m[key] for m in arr) / len(arr) if arr else 0

        avg_scored_a = avg(hist_a, 'myG')
        avg_conceded_a = avg(hist_a, 'oppG')
        avg_scored_b = avg(hist_b, 'myG')
        avg_conceded_b = avg(hist_b, 'oppG')

        has_real = len(hist_a) >= 2 or len(hist_b) >= 2

        cross_xg_a = (avg_scored_a + avg_conceded_b) / 2 if has_real else xg_a
        cross_xg_b = (avg_scored_b + avg_conceded_a) / 2 if has_real else xg_b
        cross_total = cross_xg_a + cross_xg_b

        def rate(arr, fn):
            return sum(1 for m in arr if fn(m)) / len(arr) if arr else 0.5

        btts_a = rate(hist_a, lambda m: m['myG'] > 0 and m['oppG'] > 0)
        btts_b = rate(hist_b, lambda m: m['myG'] > 0 and m['oppG'] > 0)
        btts_comb = round((btts_a + btts_b) / 2 * 100)

        o25_a = rate(hist_a, lambda m: m['myG'] + m['oppG'] > 2.5)
        o25_b = rate(hist_b, lambda m: m['myG'] + m['oppG'] > 2.5)
        o25_comb = round((o25_a + o25_b) / 2 * 100)

        o15_a = rate(hist_a, lambda m: m['myG'] + m['oppG'] > 1.5)
        o15_b = rate(hist_b, lambda m: m['myG'] + m['oppG'] > 1.5)
        o15_comb = round((o15_a + o15_b) / 2 * 100)

        win_a = rate(hist_a, lambda m: m['result'] == 'G')
        win_b = rate(hist_b, lambda m: m['result'] == 'G')

        bets = []
        if btts_comb >= 60:
            bets.append({'bet': 'Ambos Anotan — SÍ', 'conf': btts_comb,
                         'reason': f'{btts_comb}% de sus partidos recientes terminaron con goles de ambos lados'})
        if btts_comb <= 38:
            bets.append({'bet': 'Ambos Anotan — NO', 'conf': 100 - btts_comb,
                         'reason': f'Solo {btts_comb}% de sus partidos tuvieron BTTS — portería a cero es frecuente'})
        if o25_comb >= 62:
            bets.append({'bet': 'Más de 2.5 Goles', 'conf': o25_comb,
                         'reason': f'{o25_comb}% de sus partidos recientes superaron 2.5 goles · xG cruzado: {cross_total:.2f}'})
        if o25_comb <= 36:
            bets.append({'bet': 'Menos de 2.5 Goles', 'conf': 100 - o25_comb,
                         'reason': f'Solo {o25_comb}% de sus partidos superaron 2.5 goles — partidos cerrados son la norma'})
        if o15_comb >= 80:
            bets.append({'bet': 'Más de 1.5 Goles', 'conf': o15_comb,
                         'reason': f'{o15_comb}% de sus partidos tienen al menos 2 goles — apuesta de alta seguridad'})
        if win_a >= 0.70 and len(hist_a) >= 4:
            bets.append({'bet': f'{team_a} Gana o Empate', 'conf': min(88, round(win_a * 100) + 8),
                         'reason': f'{team_a} gana {round(win_a*100)}% de sus últimos partidos'})
        if win_b >= 0.70 and len(hist_b) >= 4:
            bets.append({'bet': f'{team_b} Gana (Sorpresa)', 'conf': min(85, round(win_b * 100)),
                         'reason': f'{team_b} gana {round(win_b*100)}% de sus últimos partidos — visita en racha'})

        # Fallback si no hay señal fuerte
        if not bets:
            if cross_total >= 2.6:
                bets.append({'bet': 'Más de 2.5 Goles', 'conf': round(50 + (cross_total - 2.5) * 18),
                             'reason': f'xG cruzado IA: {cross_total:.2f} goles proyectados'})
            else:
                bets.append({'bet': 'Menos de 2.5 Goles', 'conf': round(50 + (2.5 - cross_total) * 18),
                             'reason': f'xG cruzado IA: {cross_total:.2f} — partido de bajo puntaje esperado'})

        bets.sort(key=lambda x: x['conf'], reverse=True)
        return {
            'best':       bets[0] if bets else None,
            'alt':        bets[1] if len(bets) > 1 else None,
            'crossXgA':   round(cross_xg_a, 2),
            'crossXgB':   round(cross_xg_b, 2),
            'crossTotal': round(cross_total, 2),
            'btts':       btts_comb,
            'over25':     o25_comb,
            'avgScoredA': round(avg_scored_a, 1),
            'avgConcededA': round(avg_conceded_a, 1),
            'avgScoredB': round(avg_scored_b, 1),
            'avgConcededB': round(avg_conceded_b, 1),
            'hasRealData': has_real,
        }

    # ─── Estado del scan autónomo ──────────────────────────────────
    def scan_status(self):
        return {
            'running':  self._scan_running,
            'progress': self._scan_progress.copy(),
            'lastScan': self.db.get_last_scan(),
            'total':    self.total_learned(),
        }

    def set_progress(self, pct, msg, learned=None):
        self._scan_progress['pct'] = pct
        self._scan_progress['msg'] = msg
        if learned is not None:
            self._scan_progress['learned'] = learned

    def auto_scan(self, days_back=7, sources=None):
        """
        Escaneo autónomo en segundo plano.
        Llamado por APScheduler — no bloquea el servidor.
        """
        if self._scan_lock.locked():
            print("[GLAI] Auto-scan ya en curso, saltando.")
            return
        with self._scan_lock:
            self._scan_running = True
            self.set_progress(0, 'Iniciando escaneo autónomo...', 0)
            scan_id = self.db.start_scan_log()
            total_learned = 0
            used_sources = []
            try:
                from scrapers.espn import ESPNScraper
                from scrapers.thesportsdb import TheSportsDBScraper

                espn = ESPNScraper(self.db.get_all_keys())
                tsdb = TheSportsDBScraper()

                self.set_progress(5, 'Escaneando ESPN...', 0)
                n = espn.scan(days_back=days_back, glai=self, on_progress=self.set_progress)
                total_learned += n
                used_sources.append('ESPN')
                self.set_progress(60, f'ESPN: {n} resultados · Escaneando TheSportsDB...', total_learned)

                n2 = tsdb.scan_top_leagues(days_back=days_back, glai=self)
                total_learned += n2
                used_sources.append('TheSportsDB')
                self.set_progress(90, f'TheSportsDB: {n2} resultados · Finalizando...', total_learned)

                # BeSoccer si hay key
                if self.db.get_key('besoccer'):
                    from scrapers.besoccer import BeSoccerScraper
                    bsc = BeSoccerScraper(self.db.get_key('besoccer'))
                    n3 = bsc.scan(days_back=days_back, glai=self)
                    total_learned += n3
                    used_sources.append('BeSoccer')

                self.db.finish_scan_log(scan_id, total_learned, used_sources)
                self.set_progress(100, f'✅ Completado — {total_learned} resultados aprendidos', total_learned)
                print(f"[GLAI] Auto-scan completado: {total_learned} resultados · {used_sources}")
            except Exception as e:
                print(f"[GLAI] Error en auto-scan: {e}")
                self.set_progress(0, f'Error: {str(e)[:80]}', total_learned)
            finally:
                self._scan_running = False
