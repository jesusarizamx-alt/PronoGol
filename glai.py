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
        self._fixtures = []  # Partidos futuros de ResultadosFutbol

    def get_fixtures(self):
        """Retorna los fixtures futuros guardados en el último scan."""
        return self._fixtures

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

    def _dc_correction(self, h, a, lam_h, lam_a, rho=-0.10):
        """
        Dixon-Coles (1997) ρ-correction para marcadores bajos.
        Corrige la sobrestimación / subestimación del Poisson puro
        en partidos de 0-0, 1-0, 0-1 y 1-1.
        rho negativo es el valor estándar calibrado en fútbol.
        """
        if h == 0 and a == 0:
            return max(0.01, 1 - lam_h * lam_a * rho)
        elif h == 0 and a == 1:
            return max(0.01, 1 + lam_h * rho)
        elif h == 1 and a == 0:
            return max(0.01, 1 + lam_a * rho)
        elif h == 1 and a == 1:
            return max(0.01, 1 - rho)
        return 1.0

    def _weighted_avg(self, arr, key, decay=0.82):
        """
        Promedio con decaimiento exponencial.
        El partido más reciente (índice 0) tiene peso 1.0,
        el siguiente 0.82, luego 0.67, 0.55, ...
        Esto da ~60% del peso a los últimos 3 partidos.
        """
        if not arr:
            return 0.0
        total_w = total_v = 0.0
        for i, item in enumerate(arr):
            w = decay ** i
            v = item.get(key) or 0
            total_w += w
            total_v += w * float(v)
        return total_v / total_w if total_w else 0.0

    # ── Mejoras IA v3 ─────────────────────────────────────────────

    def _momentum_factor(self, hist, decay=0.82):
        """
        Factor de momentum basado en forma reciente.
        Retorna un multiplicador entre 0.94 y 1.06:
          - Equipo en racha ganadora → hasta +6% sobre λ Poisson
          - Equipo en racha perdedora → hasta -6%
        Usa decay para que victorias recientes pesen más.
        """
        if not hist:
            return 1.0
        result_val = {'G': 1.0, 'E': 0.5, 'P': 0.0}
        total_w = total_v = 0.0
        for i, m in enumerate(hist[:7]):
            w = decay ** i
            total_w += w
            total_v += w * result_val.get(m.get('result', 'E'), 0.5)
        form = total_v / total_w if total_w else 0.5
        # form=1.0 → factor=1.06, form=0.5 → factor=1.0, form=0.0 → factor=0.94
        return round(1.0 + (form - 0.5) * 0.12, 4)

    def _h2h_decay_avg(self, h2h, team_a, decay=0.75):
        """
        Promedio H2H con decay exponencial (partidos más recientes pesan más).
        decay=0.75 es más agresivo que el decay de forma general (0.82)
        para penalizar más los H2H lejanos en el tiempo.
        Retorna: {'avgGoals': float, 'winRateA': float 0-1, 'n': int}
        """
        if not h2h:
            return {}
        total_w = total_goals = total_wins_a = 0.0
        for i, r in enumerate(h2h):
            w = decay ** i
            total_w    += w
            total_goals += w * (r['home_goals'] + r['away_goals'])
            won = (
                (team_a[:6].lower() in r['home_team'].lower() and r['home_goals'] > r['away_goals']) or
                (team_a[:6].lower() in r['away_team'].lower() and r['away_goals'] > r['home_goals'])
            )
            total_wins_a += w * (1.0 if won else 0.0)
        if not total_w:
            return {}
        return {
            'n':        len(h2h),
            'avgGoals': round(total_goals / total_w, 2),
            'winRateA': round(total_wins_a / total_w, 3),   # 0.0-1.0 con decay
        }

    def _opposition_quality(self, team, sport, recent_n=10):
        """
        Estima la calidad media de los rivales enfrentados recientemente.
        Un rival con muchos goles anotados = rival peligroso.
        Retorna ratio: 1.0 = media, >1.0 = rivales fuertes, <1.0 = rivales débiles.
        Permite ponderar victorias/derrotas según la dificultad.
        """
        rows = self.db.get_team_results(team, recent_n * 2)
        if not rows:
            return 1.0
        opp_scores = []
        seen = set()
        for r in rows:
            key = str(r.get('event_id') or r.get('id'))
            if key in seen:
                continue
            seen.add(key)
            is_home = team.lower()[:6] in r['home_team'].lower()
            opp_g = r['away_goals'] if is_home else r['home_goals']
            opp_scores.append(float(opp_g))
            if len(opp_scores) >= recent_n:
                break
        if not opp_scores:
            return 1.0
        GENERIC_AVG = 1.35
        ratio = (sum(opp_scores) / len(opp_scores)) / GENERIC_AVG
        return round(max(0.5, min(2.0, ratio)), 3)

    def predict(self, sport, league_id, home_xg, away_xg, max_goals=7,
                team_a=None, team_b=None, hist_a=None, hist_b=None):
        """
        Genera matriz de marcadores y probabilidades 1X2.
        home_xg / away_xg: goles esperados base.

        Mejoras v3:
        - Ajuste por estadísticas de liga con blend proporcional al tamaño
        - Fuerza relativa del equipo (attack/defense ratings) cuando hay historial
        - Corrección Dixon-Coles para marcadores bajos (0-0, 1-0, 0-1, 1-1)
        - Blend final con tasas históricas de victorias de la liga
        - Momentum de forma reciente (decay ponderado) ajusta λ Poisson directamente
        """
        lg = self.get_league_stats(sport, league_id)

        # ── Paso 1: xG base ajustado con media de liga ────────────────
        adj_home_xg = home_xg
        adj_away_xg = away_xg
        if lg and lg['n'] >= 10:
            blend_lg = min(0.30, lg['n'] / 350)
            adj_home_xg = home_xg * (1 - blend_lg) + lg['avgHG'] * blend_lg
            adj_away_xg = away_xg * (1 - blend_lg) + lg['avgAG'] * blend_lg

        # ── Paso 2: Ajuste por fuerza del equipo (si hay historial) ───
        data_quality = 0
        strength_info = {}
        if team_a and team_b and lg and lg['n'] >= 15:
            str_a = self.get_team_strength(team_a, sport, league_id, lg)
            str_b = self.get_team_strength(team_b, sport, league_id, lg)
            n_data = min(str_a['n'], str_b['n'])

            if n_data >= 5:
                blend_str = min(0.55, n_data / 40)
                lg_avg = (lg['avgHG'] + lg['avgAG']) / 2
                str_home = str_a['attack'] * str_b['defense'] * lg['avgHG']
                str_away = str_b['attack'] * str_a['defense'] * lg['avgAG']
                adj_home_xg = adj_home_xg * (1 - blend_str) + str_home * blend_str
                adj_away_xg = adj_away_xg * (1 - blend_str) + str_away * blend_str
                data_quality = min(95, 30 + n_data * 4)
                strength_info = {
                    'attackA': str_a['attack'], 'defA': str_a['defense'],
                    'attackB': str_b['attack'], 'defB': str_b['defense'],
                    'nA': str_a['n'], 'nB': str_b['n'],
                }
            elif n_data > 0:
                data_quality = min(30, n_data * 6)
        elif lg and lg['n'] >= 10:
            data_quality = min(25, lg['n'] // 5)

        adj_home_xg = max(0.15, adj_home_xg)
        adj_away_xg = max(0.10, adj_away_xg)

        # ── Paso 2b: Momentum de forma reciente ───────────────────────
        if hist_a:
            mom_a = max(0.92, min(1.08, self._momentum_factor(hist_a)))
            adj_home_xg *= mom_a
        if hist_b:
            mom_b = max(0.92, min(1.08, self._momentum_factor(hist_b)))
            adj_away_xg *= mom_b
        adj_home_xg = max(0.15, adj_home_xg)
        adj_away_xg = max(0.10, adj_away_xg)

        # ── Paso 3: Matriz Poisson con corrección Dixon-Coles ──────────
        matrix = []
        p_home = p_draw = p_away = 0.0

        for h in range(max_goals + 1):
            for a in range(max_goals + 1):
                raw_p = self._poisson(adj_home_xg, h) * self._poisson(adj_away_xg, a)
                dc    = self._dc_correction(h, a, adj_home_xg, adj_away_xg, rho=-0.10)
                p     = raw_p * dc
                matrix.append({'a': h, 'b': a, 'p': round(p, 4)})
                if   h > a: p_home += p
                elif h == a: p_draw += p
                else:        p_away += p

        total = p_home + p_draw + p_away or 1
        p_home /= total; p_draw /= total; p_away /= total

        # ── Paso 4: Blend con tasas históricas reales de liga ──────────
        if lg and lg['n'] >= 20:
            blend_hist = min(0.30, lg['n'] / 400)
            p_home = p_home * (1 - blend_hist) + lg['homeWinRate'] * blend_hist
            p_away = p_away * (1 - blend_hist) + lg['awayWinRate'] * blend_hist
            p_draw = max(0.02, 1 - p_home - p_away)

        matrix.sort(key=lambda x: x['p'], reverse=True)

        # ── Asian Handicap (AH) — desde la matriz Poisson ─────────────
        _diff_p = {}
        for _it in matrix:
            _d = _it['a'] - _it['b']
            _diff_p[_d] = _diff_p.get(_d, 0.0) + _it['p']
        _diff_total = sum(_diff_p.values()) or 1.0
        _diff_p = {k: v / _diff_total for k, v in _diff_p.items()}

        def _ah(line):
            """P(home cubre AH -line): home - away > line (push si igual)."""
            _w = _pp = 0.0
            for _d2, _pr2 in _diff_p.items():
                _eff = _d2 - line
                if _eff > 0:    _w  += _pr2
                elif _eff == 0: _pp += _pr2
            return _w + _pp * 0.5

        asian_handicap = {
            'home-2.5': round(_ah(2.5) * 100),   # local gana por 3+
            'home-1.5': round(_ah(1.5) * 100),   # local gana por 2+
            'home-0.5': round(_ah(0.5) * 100),   # local gana (cualquier margen)
            'home+0.5': round(_ah(-0.5) * 100),  # local no pierde (gana o empata)
            'home+1.5': round(_ah(-1.5) * 100),  # local cubre +1.5
            'home+2.5': round(_ah(-2.5) * 100),  # local cubre +2.5
            'away-2.5': round((1 - _ah(-2.5)) * 100),  # visita gana por 3+
            'away-1.5': round((1 - _ah(-1.5)) * 100),  # visita gana por 2+
            'away-0.5': round((1 - _ah(-0.5)) * 100),  # visita gana (cualquier margen)
            'away+0.5': round((1 - _ah(0.5)) * 100),   # visita no pierde
            'away+1.5': round((1 - _ah(1.5)) * 100),   # visita cubre +1.5
            'away+2.5': round((1 - _ah(2.5)) * 100),   # visita cubre +2.5
        }

        # ── BTTS y Over/Under ──────────────────────────────────────────
        p_btts  = (1 - math.exp(-adj_home_xg)) * (1 - math.exp(-adj_away_xg))
        over_25 = sum(
            self._poisson(adj_home_xg, h) * self._poisson(adj_away_xg, a)
            for h in range(max_goals + 1)
            for a in range(max_goals + 1)
            if h + a > 2.5
        )
        over_15 = sum(
            self._poisson(adj_home_xg, h) * self._poisson(adj_away_xg, a)
            for h in range(max_goals + 1)
            for a in range(max_goals + 1)
            if h + a > 1.5
        )

        return {
            'pctA':           round(p_home * 100),
            'pctD':           round(p_draw * 100),
            'pctB':           round(p_away * 100),
            'xgA':            round(adj_home_xg, 2),
            'xgB':            round(adj_away_xg, 2),
            'btts':           round(p_btts * 100),
            'over25':         round(over_25 * 100),
            'over15':         round(over_15 * 100),
            'matrix':         matrix[:10],
            'lgStats':        lg,
            'totalLearned':   self.total_learned(),
            'dataQuality':    data_quality,
            'strength':       strength_info,
            'asianHandicap':  asian_handicap,
            'model':          'poisson-dc-v2',
        }

    def get_team_strength(self, team, sport, league_id, lg=None):
        """
        Calcula fuerza relativa del equipo (ataque/defensa) vs. media de liga.
        attack > 1.0 = ataca mejor que la media
        defense < 1.0 = concede menos que la media (buena defensa)
        """
        if lg is None:
            lg = self.get_league_stats(sport, league_id)
        if not lg or lg['n'] < 5:
            return {'attack': 1.0, 'defense': 1.0, 'n': 0}

        rows = self.db.get_team_results(team, 20)
        if not rows:
            return {'attack': 1.0, 'defense': 1.0, 'n': 0}

        league_avg = (lg['avgHG'] + lg['avgAG']) / 2 or 1.3

        # ── Ponderación decay + calidad de rival ─────────────────────
        total_w_sc = total_w_cc = 0.0
        total_v_sc = total_v_cc = 0.0
        for i, r in enumerate(rows):
            ih       = team[:6].lower() in r['home_team'].lower()
            my_g     = r['home_goals'] if ih else r['away_goals']
            opp_g    = r['away_goals'] if ih else r['home_goals']
            opp_str  = r['home_goals'] if not ih else r['away_goals']

            opp_quality_bonus = 1.0 + max(0, (opp_str - league_avg) / league_avg) * 0.3
            w = (0.82 ** i) * opp_quality_bonus

            total_w_sc += w;  total_v_sc += w * my_g
            total_w_cc += w;  total_v_cc += w * opp_g

        avg_scored   = total_v_sc / total_w_sc if total_w_sc else 0.0
        avg_conceded = total_v_cc / total_w_cc if total_w_cc else 0.0
        n = len(rows)

        attack  = avg_scored   / league_avg
        defense = avg_conceded / league_avg

        recent = rows[:5]
        if recent:
            r_wgt = sum(0.82**i * (r['home_goals'] if team[:6].lower() in r['home_team'].lower() else r['away_goals'])
                        for i, r in enumerate(recent))
            r_w   = sum(0.82**i for i in range(len(recent)))
            form_attack = (r_wgt / r_w) / league_avg if r_w else attack
            attack = attack * 0.5 + form_attack * 0.5

        return {
            'attack':       round(max(0.1, attack),  3),
            'defense':      round(max(0.1, defense), 3),
            'avgScored':    round(avg_scored, 2),
            'avgConceded':  round(avg_conceded, 2),
            'n':            n,
        }

    # ─── Historial por equipo ─────────────────────────────────────
    def team_history(self, team_name, limit=5):
        """
        Últimos N partidos de un equipo — todas las competencias.
        v2: incluye splits home/away y racha de forma.
        """
        results = self.db.get_team_results(team_name, limit * 4)
        hist = []
        seen = set()
        for r in results:
            key = str(r['event_id'] or r['id'])
            if key in seen:
                continue
            seen.add(key)
            is_home = team_name.lower()[:6] in r['home_team'].lower()
            my_g   = r['home_goals'] if is_home else r['away_goals']
            opp_g  = r['away_goals'] if is_home else r['home_goals']
            opp    = r['away_team'] if is_home else r['home_team']
            result = 'G' if my_g > opp_g else ('P' if my_g < opp_g else 'E')
            hist.append({
                'result':    result,
                'myG':       my_g,
                'oppG':      opp_g,
                'opp':       opp,
                'comp':      r['league_id'],
                'isHome':    is_home,
                'source':    r['source'],
                'learnedAt': r.get('learned_at', ''),
            })
            if len(hist) >= limit:
                break
        return hist

    def team_history_split(self, team_name, limit=10):
        """
        Historial separado: últimos N como local y N como visitante.
        Útil para calcular rendimiento home vs. away correctamente.
        """
        results = self.db.get_team_results(team_name, limit * 6)
        home_hist = []
        away_hist = []
        seen = set()
        for r in results:
            key = str(r['event_id'] or r['id'])
            if key in seen:
                continue
            seen.add(key)
            is_home = team_name.lower()[:6] in r['home_team'].lower()
            my_g  = r['home_goals'] if is_home else r['away_goals']
            opp_g = r['away_goals'] if is_home else r['home_goals']
            entry = {
                'result': 'G' if my_g > opp_g else ('P' if my_g < opp_g else 'E'),
                'myG':    my_g, 'oppG': opp_g,
            }
            if is_home and len(home_hist) < limit:
                home_hist.append(entry)
            elif not is_home and len(away_hist) < limit:
                away_hist.append(entry)
            if len(home_hist) >= limit and len(away_hist) >= limit:
                break
        return {'home': home_hist, 'away': away_hist}

    # ─── Análisis de apuesta IA v2 ────────────────────────────────
    def ai_bet(self, hist_a, hist_b, xg_a, xg_b, team_a, team_b, h2h=None):
        """
        Genera recomendación de apuesta basada en historial real.
        v2: promedio ponderado con decay exponencial, splits home/away,
        integración de H2H, y puntuación de confianza de datos.
        """
        wgt_scored_a   = self._weighted_avg(hist_a, 'myG')
        wgt_conceded_a = self._weighted_avg(hist_a, 'oppG')
        wgt_scored_b   = self._weighted_avg(hist_b, 'myG')
        wgt_conceded_b = self._weighted_avg(hist_b, 'oppG')

        def avg(arr, key):
            return sum(m[key] for m in arr) / len(arr) if arr else 0
        avg_scored_a   = avg(hist_a, 'myG')
        avg_conceded_a = avg(hist_a, 'oppG')
        avg_scored_b   = avg(hist_b, 'myG')
        avg_conceded_b = avg(hist_b, 'oppG')

        has_real = len(hist_a) >= 2 or len(hist_b) >= 2

        if has_real:
            cross_xg_a = (wgt_scored_a + wgt_conceded_b) / 2
            cross_xg_b = (wgt_scored_b + wgt_conceded_a) / 2
        else:
            cross_xg_a = xg_a
            cross_xg_b = xg_b
        cross_total = cross_xg_a + cross_xg_b

        # ── H2H: blend con decay (partidos recientes pesan más) ─────
        h2h_info = {}
        if h2h and len(h2h) >= 3:
            h2h_dec = self._h2h_decay_avg(h2h, team_a, decay=0.75)
            h2h_avg        = h2h_dec['avgGoals']
            h2h_win_rate_a = h2h_dec['winRateA']
            cross_total = cross_total * 0.70 + h2h_avg * 0.30
            cross_xg_a  = cross_xg_a  * 0.80 + (h2h_avg * 0.50) * 0.20
            cross_xg_b  = cross_xg_b  * 0.80 + (h2h_avg * 0.50) * 0.20
            h2h_info = {
                'n': len(h2h),
                'avgGoals': round(h2h_avg, 1),
                'winRateA': round(h2h_win_rate_a * 100),
            }

        def wgt_rate(arr, fn, decay=0.82):
            if not arr:
                return 0.5
            total_w = total_v = 0.0
            for i, m in enumerate(arr):
                w = decay ** i
                total_w += w
                total_v += w * (1.0 if fn(m) else 0.0)
            return total_v / total_w if total_w else 0.5

        btts_a = wgt_rate(hist_a, lambda m: m['myG'] > 0 and m['oppG'] > 0)
        btts_b = wgt_rate(hist_b, lambda m: m['myG'] > 0 and m['oppG'] > 0)
        btts_comb = round((btts_a + btts_b) / 2 * 100)

        o25_a = wgt_rate(hist_a, lambda m: m['myG'] + m['oppG'] > 2.5)
        o25_b = wgt_rate(hist_b, lambda m: m['myG'] + m['oppG'] > 2.5)
        o25_comb = round((o25_a + o25_b) / 2 * 100)

        o15_a = wgt_rate(hist_a, lambda m: m['myG'] + m['oppG'] > 1.5)
        o15_b = wgt_rate(hist_b, lambda m: m['myG'] + m['oppG'] > 1.5)
        o15_comb = round((o15_a + o15_b) / 2 * 100)

        win_a = wgt_rate(hist_a, lambda m: m['result'] == 'G')
        win_b = wgt_rate(hist_b, lambda m: m['result'] == 'G')

        home_a = [m for m in hist_a if m.get('isHome', True)]
        away_b = [m for m in hist_b if not m.get('isHome', True)]
        home_win_a = wgt_rate(home_a, lambda m: m['result'] == 'G') if home_a else win_a
        away_win_b = wgt_rate(away_b, lambda m: m['result'] == 'G') if away_b else win_b

        n_total = len(hist_a) + len(hist_b)
        h2h_bonus = min(10, (h2h_info.get('n', 0)) * 2) if h2h_info else 0
        confidence_score = min(95, round(
            (min(n_total, 20) / 20) * 70
            + h2h_bonus
            + (10 if has_real else 0)
            + (5  if len(hist_a) >= 3 and len(hist_b) >= 3 else 0)
        ))

        bets = []

        if btts_comb >= 60:
            bets.append({'bet': 'Ambos Anotan — SÍ', 'conf': btts_comb,
                         'reason': f'Tasa BTTS ponderada {btts_comb}% · xG total proyectado {cross_total:.2f}'})
        if btts_comb <= 36:
            bets.append({'bet': 'Ambos Anotan — NO', 'conf': 100 - btts_comb,
                         'reason': f'Solo {btts_comb}% BTTS en historial reciente — portería a cero probable'})
        if o25_comb >= 62:
            bets.append({'bet': 'Más de 2.5 Goles', 'conf': o25_comb,
                         'reason': f'{o25_comb}% Over 2.5 ponderado · xG cruzado: {cross_total:.2f}'})
        if o25_comb <= 36:
            bets.append({'bet': 'Menos de 2.5 Goles', 'conf': 100 - o25_comb,
                         'reason': f'Solo {o25_comb}% Over 2.5 — partidos cerrados son la norma'})
        if o15_comb >= 80:
            bets.append({'bet': 'Más de 1.5 Goles', 'conf': o15_comb,
                         'reason': f'{o15_comb}% de sus partidos recientes superan 1.5 — apuesta de alta seguridad'})

        if home_win_a >= 0.68 and len(home_a) >= 3:
            bets.append({'bet': f'{team_a} Gana (Local)', 'conf': min(84, round(home_win_a * 100) + 5),
                         'reason': f'{team_a} gana {round(home_win_a*100)}% en casa (historial ponderado)'})
        elif win_a >= 0.70 and len(hist_a) >= 4:
            bets.append({'bet': f'{team_a} Gana o Empate', 'conf': min(82, round(win_a * 100) + 6),
                         'reason': f'{team_a} gana {round(win_a*100)}% de sus últimos partidos (decay ponderado)'})

        if away_win_b >= 0.65 and len(away_b) >= 3:
            bets.append({'bet': f'{team_b} Gana (Visitante)', 'conf': min(80, round(away_win_b * 100)),
                         'reason': f'{team_b} gana {round(away_win_b*100)}% fuera de casa — visita sólida'})
        elif win_b >= 0.70 and len(hist_b) >= 4:
            bets.append({'bet': f'{team_b} Gana (Sorpresa)', 'conf': min(78, round(win_b * 100)),
                         'reason': f'{team_b} en racha: gana {round(win_b*100)}% de sus últimos partidos'})

        if h2h_info and h2h_info.get('winRateA', 50) >= 70:
            bets.append({'bet': f'{team_a} Gana (Historial H2H)', 'conf': h2h_info['winRateA'],
                         'reason': f'En los últimos {h2h_info["n"]} enfrentamientos directos, {team_a} ganó {h2h_info["winRateA"]}%'})

        # ── Asian Handicap desde xG cruzado ───────────────────────────
        _max_g = 7
        _dp = {}
        for _hh in range(_max_g + 1):
            for _aa in range(_max_g + 1):
                _dd = _hh - _aa
                _dp[_dd] = _dp.get(_dd, 0.0) + (
                    self._poisson(max(0.15, cross_xg_a), _hh) *
                    self._poisson(max(0.10, cross_xg_b), _aa)
                )
        def _ahb(line):
            _w2 = _pp2 = 0.0
            for _d3, _pr3 in _dp.items():
                _e2 = _d3 - line
                if _e2 > 0:    _w2  += _pr3
                elif _e2 == 0: _pp2 += _pr3
            return _w2 + _pp2 * 0.5

        ah_hm15 = round(_ahb(1.5) * 100)   # home -1.5 (gana por 2+)
        ah_ap15 = round((1 - _ahb(1.5)) * 100)  # away +1.5 (cubre +1.5)
        ah_hm05 = round(_ahb(0.5) * 100)   # home -0.5 (gana)
        ah_ap05 = round((1 - _ahb(0.5)) * 100)  # away +0.5 (no pierde)

        if ah_hm15 >= 62 and home_win_a >= 0.55:
            bets.append({'bet': f'{team_a} -1.5 (Asian Handicap)', 'conf': min(74, ah_hm15),
                         'reason': f'{ah_hm15}% de ganar por 2+ goles · xG {cross_xg_a:.2f} vs {cross_xg_b:.2f}'})
        elif ah_ap15 >= 65:
            bets.append({'bet': f'{team_b} +1.5 (Asian Handicap)', 'conf': min(74, ah_ap15),
                         'reason': f'{ah_ap15}% de cubrir +1.5 — visita defensiva proyectada'})

        if not bets:
            if cross_total >= 2.6:
                bets.append({'bet': 'Más de 2.5 Goles', 'conf': min(72, round(50 + (cross_total - 2.5) * 16)),
                             'reason': f'xG cruzado IA: {cross_total:.2f} goles proyectados'})
            else:
                bets.append({'bet': 'Menos de 2.5 Goles', 'conf': min(72, round(50 + (2.5 - cross_total) * 16)),
                             'reason': f'xG cruzado IA: {cross_total:.2f} — partido de bajo puntaje esperado'})

        bets.sort(key=lambda x: x['conf'], reverse=True)
        return {
            'best':            bets[0] if bets else None,
            'alt':             bets[1] if len(bets) > 1 else None,
            'crossXgA':        round(cross_xg_a, 2),
            'crossXgB':        round(cross_xg_b, 2),
            'crossTotal':      round(cross_total, 2),
            'btts':            btts_comb,
            'over25':          o25_comb,
            'over15':          o15_comb,
            'avgScoredA':      round(avg_scored_a, 1),
            'avgConcededA':    round(avg_conceded_a, 1),
            'avgScoredB':      round(avg_scored_b, 1),
            'avgConcededB':    round(avg_conceded_b, 1),
            'wgtScoredA':      round(wgt_scored_a, 2),
            'wgtScoredB':      round(wgt_scored_b, 2),
            'hasRealData':     has_real,
            'confidenceScore': confidence_score,
            'h2h':             h2h_info,
            'homeWinA':        round(home_win_a * 100),
            'awayWinB':        round(away_win_b * 100),
            'ahHome_m15':      ah_hm15,
            'ahAway_p15':      ah_ap15,
            'ahHome_m05':      ah_hm05,
            'ahAway_p05':      ah_ap05,
        }

    # ─── Predicción de esquinas y tarjetas ───────────────────────────
    def predict_corners_cards(self, xg_a, xg_b, league_id='soccer'):
        """
        Predice tiros de esquina y tarjetas usando modelo estadístico
        basado en xG, diferencial de fuerzas y tendencias por liga.
        """
        total_xg = xg_a + xg_b if (xg_a + xg_b) > 0 else 2.5

        # ── TIROS DE ESQUINA ──────────────────────────────────────────
        base_corners = 8.0 + total_xg * 1.3

        league_corner_factor = {
            'eng.1': 1.08, 'esp.1': 1.05, 'ger.1': 1.10,
            'ita.1': 0.95, 'fra.1': 1.00, 'ned.1': 1.08,
            'por.1': 1.02, 'bel.1': 1.05,
            'mex.1': 0.95, 'usa.1': 1.02, 'arg.1': 0.93,
            'bra.1': 0.95, 'col.1': 0.92,
            'uefa.champions': 1.06, 'uefa.europa': 1.04,
        }.get(league_id, 1.0)

        base_corners *= league_corner_factor

        share_a = (xg_a / total_xg) * 1.04
        share_a = min(max(share_a, 0.30), 0.70)
        share_b = 1 - share_a

        corners_a = round(base_corners * share_a, 1)
        corners_b = round(base_corners * share_b, 1)
        total_corners = round(corners_a + corners_b, 1)

        over_8_5  = min(88, max(15, round(45 + (total_xg - 2.0) * 14)))
        over_10_5 = min(78, max(10, round(35 + (total_xg - 2.0) * 12)))
        over_12_5 = min(60, max(5,  round(20 + (total_xg - 2.0) * 9)))

        # ── TARJETAS ──────────────────────────────────────────────────
        diff = abs(xg_a - xg_b)
        competitiveness = max(0.3, 1 - diff / (total_xg + 0.1))

        league_card_factor = {
            'esp.1': 1.22, 'ita.1': 1.18, 'tur.1': 1.25,
            'arg.1': 1.28, 'bra.1': 1.22, 'col.1': 1.20,
            'eng.1': 0.96, 'ger.1': 0.88, 'fra.1': 1.02,
            'por.1': 1.12, 'ned.1': 0.94, 'sco.1': 1.05,
            'mex.1': 1.10, 'usa.1': 0.92,
            'uefa.champions': 1.05, 'uefa.europa': 1.10, 'uefa.conference': 1.08,
        }.get(league_id, 1.0)

        base_yellows = 3.4 * league_card_factor * (0.75 + competitiveness * 0.45)

        losing_factor = 0.10
        if xg_a < xg_b:
            yellows_a = round(base_yellows * (0.52 + losing_factor), 1)
            yellows_b = round(base_yellows * (0.48 - losing_factor / 2), 1)
        else:
            yellows_a = round(base_yellows * (0.50 - losing_factor / 2), 1)
            yellows_b = round(base_yellows * (0.50 + losing_factor), 1)

        total_yellows = round(yellows_a + yellows_b, 1)
        red_prob = round(min(45, max(5, competitiveness * 20 * league_card_factor)))

        over_2_5 = min(90, max(25, round(55 + (total_yellows - 3.4) * 18)))
        over_3_5 = min(78, max(15, round(40 + (total_yellows - 3.4) * 16)))
        over_4_5 = min(62, max(8,  round(28 + (total_yellows - 3.4) * 13)))

        return {
            'corners': {
                'total':    total_corners,
                'teamA':    corners_a,
                'teamB':    corners_b,
                'over8_5':  over_8_5,
                'over10_5': over_10_5,
                'over12_5': over_12_5,
            },
            'cards': {
                'total':    total_yellows,
                'teamA':    yellows_a,
                'teamB':    yellows_b,
                'redProb':  int(red_prob),
                'over2_5':  over_2_5,
                'over3_5':  over_3_5,
                'over4_5':  over_4_5,
            }
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

    # ─── NBA — Predicción v3 ──────────────────────────────────────
    def predict_nba(self, home_ppg, away_ppg,
                    home_apg=None, away_apg=None, h2h=None,
                    confidence_score=0, hist_a=None, hist_b=None):
        """
        Predicción NBA v3 — distribución Normal con ratings ofensivos/defensivos.
        home_ppg / away_ppg : puntos anotados por partido (ponderados)
        home_apg / away_apg : puntos PERMITIDOS por partido (proxy defensa)
        h2h                 : lista de enfrentamientos directos (opcional)
        hist_a / hist_b     : historial reciente para momentum de forma (±8%)
        """
        HOME_ADV = 3.0
        SIGMA    = 11.5

        # ── Proyección cruzada: ataque vs. defensa rival ──────────────
        if home_apg and away_apg:
            cross_home = (home_ppg + away_apg) / 2
            cross_away = (away_ppg + home_apg) / 2
        else:
            cross_home = home_ppg
            cross_away = away_ppg

        adj_home = cross_home + HOME_ADV
        adj_away = cross_away

        # ── Blend H2H si está disponible ─────────────────────────────
        if h2h and len(h2h) >= 3:
            h2h_totals = [r['home_goals'] + r['away_goals'] for r in h2h]
            h2h_avg_pts = sum(h2h_totals) / len(h2h_totals)
            h2h_proj_total = (adj_home + adj_away) * 0.75 + h2h_avg_pts * 0.25
            ratio = h2h_proj_total / (adj_home + adj_away) if (adj_home + adj_away) > 0 else 1
            adj_home *= ratio
            adj_away *= ratio

        # ── Momentum de forma reciente (±8% sobre proyección) ────────
        if hist_a:
            adj_home *= max(0.92, min(1.08, self._momentum_factor(hist_a)))
        if hist_b:
            adj_away *= max(0.92, min(1.08, self._momentum_factor(hist_b)))

        spread = adj_home - adj_away
        z      = spread / (SIGMA * math.sqrt(2))
        p_home = 0.5 + 0.5 * math.erf(z)
        p_away = 1.0 - p_home

        # ── Líneas ATS (Against The Spread) ──────────────────────────
        def _ats(line):
            """P(home - away > line) usando dist. Normal(spread, SIGMA)."""
            _z = (line - spread) / (SIGMA * math.sqrt(2))
            return max(0.01, min(0.99, 0.5 - 0.5 * math.erf(_z)))

        def _ats_away(line):
            """P(away - home > line) — away gana por line+ pts."""
            _z = (line + spread) / (SIGMA * math.sqrt(2))
            return max(0.01, min(0.99, 0.5 - 0.5 * math.erf(_z)))

        handicap = {}
        for _ln in [1.5, 3.5, 5.5, 7.5, 10.5, 13.5]:
            handicap[f'home-{_ln}'] = round(_ats(_ln) * 100)         # local cubre -{_ln}
            handicap[f'away+{_ln}'] = round((1 - _ats(_ln)) * 100)   # visita cubre +{_ln}
            handicap[f'away-{_ln}'] = round(_ats_away(_ln) * 100)    # visita cubre -{_ln}
            handicap[f'home+{_ln}'] = round((1 - _ats_away(_ln)) * 100)  # local cubre +{_ln}

        total   = adj_home + adj_away
        ou_line = round(total / 2.5) * 2.5
        ou_over = min(78, max(22, round(50 + (total - ou_line) / (SIGMA * 0.35))))

        # ── Primera mitad (Half-Time) ─────────────────────────────────
        ht_home   = round(adj_home * 0.485, 1)
        ht_away   = round(adj_away * 0.485, 1)
        ht_total  = round(ht_home + ht_away, 1)
        ht_ou_ln  = round(ht_total / 2.5) * 2.5
        ht_ou_ov  = min(75, max(25, round(50 + (ht_total - ht_ou_ln) / (SIGMA * 0.28))))

        # ── Cuartos Q1-Q4 ────────────────────────────────────────────
        q_factors = [0.247, 0.253, 0.247, 0.253]
        quarters  = []
        for i, qf in enumerate(q_factors):
            qh   = round(adj_home * qf, 1)
            qa   = round(adj_away * qf, 1)
            qt   = round(qh + qa, 1)
            q_ln = round(qt / 2.5) * 2.5
            q_ov = min(72, max(28, round(50 + (qt - q_ln) * 7)))
            quarters.append({'q': i + 1, 'home': qh, 'away': qa,
                              'total': qt, 'ouLine': q_ln, 'overPct': q_ov})

        return {
            'pctA':          round(p_home * 100),
            'pctB':          round(p_away * 100),
            'projHome':      round(adj_home, 1),
            'projAway':      round(adj_away, 1),
            'rawPpgHome':    round(home_ppg, 1),
            'rawPpgAway':    round(away_ppg, 1),
            'apgHome':       round(home_apg, 1) if home_apg else None,
            'apgAway':       round(away_apg, 1) if away_apg else None,
            'total':         round(total, 1),
            'spread':        round(spread, 1),
            'ouLine':        ou_line,
            'ouOver':        ou_over,
            'halfTime':      {'projHome': ht_home, 'projAway': ht_away,
                              'projTotal': ht_total, 'ouLine': ht_ou_ln, 'ouOver': ht_ou_ov},
            'quarters':      quarters,
            'handicap':      handicap,
            'dataQuality':   confidence_score,
            'hasCrossStats': bool(home_apg and away_apg),
        }

    def ai_bet_nba(self, pred, team_a, team_b, hist_a=None, hist_b=None, h2h=None):
        """
        Recomendación de apuesta NBA v3.
        Usa historial real ponderado, splits home/away, H2H decay y tendencia de forma.
        """
        hist_a = hist_a or []
        hist_b = hist_b or []
        pA     = pred.get('pctA', 50)
        pB     = pred.get('pctB', 50)
        spread = pred.get('spread', 0)
        total  = pred.get('total', 220)
        ou_ln  = pred.get('ouLine', 220)
        ou_ov  = pred.get('ouOver', 50)
        ht     = pred.get('halfTime', {})
        abs_sp   = abs(spread)
        handicap = pred.get('handicap', {})

        def wgt_rate(arr, fn, decay=0.82):
            if not arr: return 0.5
            tw = tv = 0.0
            for i, m in enumerate(arr):
                w = decay ** i; tw += w; tv += w * (1.0 if fn(m) else 0.0)
            return tv / tw if tw else 0.5

        win_a    = wgt_rate(hist_a, lambda m: m['result'] == 'G')
        win_b    = wgt_rate(hist_b, lambda m: m['result'] == 'G')
        home_a   = [m for m in hist_a if m.get('isHome', True)]
        away_b   = [m for m in hist_b if not m.get('isHome', True)]
        home_win_a = wgt_rate(home_a, lambda m: m['result'] == 'G') if home_a else win_a
        away_win_b = wgt_rate(away_b, lambda m: m['result'] == 'G') if away_b else win_b

        def _trend(arr):
            if len(arr) < 5: return 0
            recent = sum(1 for m in arr[:3] if m['result']=='G') / 3
            older  = sum(1 for m in arr[3:6] if m['result']=='G') / max(len(arr[3:6]),1)
            return round((recent - older) * 100)

        trend_a = _trend(hist_a)
        trend_b = _trend(hist_b)

        wgt_total_a = self._weighted_avg(hist_a, 'myG') + self._weighted_avg(hist_a, 'oppG')
        wgt_total_b = self._weighted_avg(hist_b, 'myG') + self._weighted_avg(hist_b, 'oppG')

        # H2H — con decay exponencial (partidos recientes pesan más)
        h2h_info = {}
        if h2h and len(h2h) >= 3:
            h2h_decay = self._h2h_decay_avg(h2h, team_a, decay=0.75)
            h2h_info = {
                'n':        h2h_decay.get('n', len(h2h)),
                'winRateA': round(h2h_decay.get('winRateA', 0.5) * 100),
                'avgPts':   round(h2h_decay.get('avgGoals', 0), 1),
            }

        n_total = len(hist_a) + len(hist_b)
        conf_score = min(90, round((min(n_total,20)/20)*60 + (10 if h2h_info else 0)
                                   + (10 if len(hist_a)>=3 and len(hist_b)>=3 else 0)
                                   + (10 if pred.get('hasCrossStats') else 0)))

        bets = []

        if home_win_a >= 0.65 and len(home_a) >= 3:
            bets.append({'bet': f'{team_a} Gana (Local)', 'conf': min(82, round(home_win_a*100)+4),
                         'reason': f'{team_a} gana {round(home_win_a*100)}% en casa · spread {spread:+.1f} pts proyectado'})
        elif pA >= 62:
            bets.append({'bet': f'{team_a} Gana (Moneyline)', 'conf': min(80, pA),
                         'reason': f'{pA}% probabilidad · PPG {pred.get("projHome"):.1f} vs {pred.get("projAway"):.1f}'})
        if away_win_b >= 0.60 and len(away_b) >= 3:
            bets.append({'bet': f'{team_b} Gana (Visitante)', 'conf': min(78, round(away_win_b*100)),
                         'reason': f'{team_b} gana {round(away_win_b*100)}% de visita — fuerte en cancha ajena'})
        elif pB >= 62:
            bets.append({'bet': f'{team_b} Gana (Moneyline)', 'conf': min(78, pB),
                         'reason': f'{team_b} {pB}% · visitante en racha con spread {-spread:+.1f} pts'})

        if abs_sp >= 5:
            fav  = team_a if spread > 0 else team_b
            conf = min(74, round(50 + abs_sp * 2.0))
            bets.append({'bet': f'{fav} -{abs_sp:.1f} (Spread)', 'conf': conf,
                         'reason': f'Diferencia proyectada {abs_sp:.1f} pts · respaldado por historial cruzado'})

        # ── ATS (Against The Spread) — línea más cercana ──────────────
        if abs_sp >= 4:
            _ats_line = 3.5 if abs_sp >= 6 else 1.5
            if spread > 0:
                _p_ats = handicap.get(f'home-{_ats_line}', 50)
                if _p_ats >= 58:
                    bets.append({'bet': f'{team_a} -{_ats_line} (ATS)', 'conf': min(74, _p_ats),
                                 'reason': f'{_p_ats}% de cubrir -{_ats_line} · ventaja proj. {abs_sp:.1f} pts'})
            else:
                _p_ats = handicap.get(f'away-{_ats_line}', 50)
                if _p_ats >= 58:
                    bets.append({'bet': f'{team_b} -{_ats_line} (ATS)', 'conf': min(74, _p_ats),
                                 'reason': f'{_p_ats}% de cubrir -{_ats_line} · {team_b} domina {abs_sp:.1f} pts proj.'})

        if ou_ov >= 60:
            bets.append({'bet': f'Más de {ou_ln} pts (O/U)', 'conf': min(76, ou_ov),
                         'reason': f'Total proyectado {total:.1f} pts · promedio ponderado H {wgt_total_a:.0f} / A {wgt_total_b:.0f} pts'})
        elif ou_ov <= 40:
            bets.append({'bet': f'Menos de {ou_ln} pts (O/U)', 'conf': min(76, 100-ou_ov),
                         'reason': f'Total proyectado {total:.1f} pts — defensas sólidas esperadas'})

        if ht and ht.get('ouOver', 50) >= 62:
            bets.append({'bet': f'Primera Mitad Over {ht.get("ouLine")} pts', 'conf': min(72, ht['ouOver']),
                         'reason': f'1ª mitad proyectada {ht.get("projTotal"):.1f} pts — ritmo alto desde el inicio'})

        if h2h_info and h2h_info.get('winRateA', 50) >= 70:
            bets.append({'bet': f'{team_a} Gana (Domina H2H)', 'conf': min(78, h2h_info['winRateA']),
                         'reason': f'Gana {h2h_info["winRateA"]}% de los últimos {h2h_info["n"]} enfrentamientos directos'})

        if trend_a >= 20 and pA >= 55:
            best_bet = bets[0] if bets else {'bet': f'{team_a} Gana (Moneyline)', 'conf': pA}
            bets.append({'bet': best_bet['bet'] + ' [🔥 Forma]',
                         'conf': min(82, best_bet['conf'] + trend_a // 4),
                         'reason': f'{team_a} mejorando: +{trend_a}% en últimos 3 vs. anteriores'})

        if not bets:
            bets.append({'bet': f'{team_a} Gana (Moneyline)' if pA > pB else f'{team_b} Gana (Moneyline)',
                         'conf': max(pA, pB, 51),
                         'reason': f'Total proyectado {total:.1f} pts · ventaja de {abs_sp:.1f} pts para el favorito'})

        bets.sort(key=lambda x: x['conf'], reverse=True)

        narrative = (
            f"🏀 {team_a} (local) vs {team_b} (visita).\n\n"
            f"Proyección: {pred.get('projHome'):.1f} – {pred.get('projAway'):.1f} pts "
            f"(Total {total:.1f} · Spread {spread:+.1f}).\n"
            + (f"Ratings cruzados — {team_a} APG permitidos: {pred.get('apgHome')} · {team_b}: {pred.get('apgAway')}.\n" if pred.get('hasCrossStats') else '')
            + (f"H2H: {team_a} gana {h2h_info.get('winRateA')}% de {h2h_info.get('n')} encuentros directos.\n" if h2h_info else '')
            + f"\n{bets[0]['bet']} ({bets[0]['conf']}%) — {bets[0]['reason']}."
            f"\n\nℹ️ Análisis orientativo. Apuesta con responsabilidad."
        )
        return {
            'best':           bets[0] if bets else None,
            'alt':            bets[1] if len(bets) > 1 else None,
            'narrative':      narrative,
            'confidenceScore': conf_score,
            'hasRealData':    len(hist_a) >= 2 or len(hist_b) >= 2,
            'h2h':            h2h_info,
            'homeWinA':       round(home_win_a * 100),
            'awayWinB':       round(away_win_b * 100),
            'trendA':         trend_a,
            'trendB':         trend_b,
        }

    # ─── MLB — Predicción v3 ──────────────────────────────────────
    def predict_mlb(self, home_rpg, away_rpg,
                    home_rag=None, away_rag=None, h2h=None,
                    confidence_score=0, hist_a=None, hist_b=None):
        """
        Predicción MLB v3 — Poisson con ratings ofensivos/defensivos.
        home_rpg / away_rpg : carreras anotadas por partido (ponderadas)
        home_rag / away_rag : carreras PERMITIDAS por partido (proxy ERA/pitcheo)
        hist_a / hist_b     : historial reciente para momentum de forma (±6%)
        """
        HOME_ADV = 0.18
        max_r    = 15

        # ── Proyección cruzada: ofensiva vs. pitcheo rival ────────────
        if home_rag and away_rag:
            cross_home = (home_rpg + away_rag) / 2
            cross_away = (away_rpg + home_rag) / 2
        else:
            cross_home = home_rpg
            cross_away = away_rpg

        adj_home = max(0.5, cross_home + HOME_ADV)
        adj_away = max(0.5, cross_away)

        # ── Blend H2H ─────────────────────────────────────────────────
        h2h_used = {}
        if h2h and len(h2h) >= 3:
            h2h_totals = [r['home_goals'] + r['away_goals'] for r in h2h]
            h2h_avg = sum(h2h_totals) / len(h2h_totals)
            blend   = 0.20
            ratio   = ((adj_home + adj_away) * (1 - blend) + h2h_avg * blend) / (adj_home + adj_away)
            adj_home *= ratio
            adj_away *= ratio
            h2h_used = {'n': len(h2h), 'avgRuns': round(h2h_avg, 1)}

        # ── Momentum de forma reciente (±6% sobre λ Poisson) ─────────
        if hist_a:
            adj_home *= max(0.94, min(1.06, self._momentum_factor(hist_a)))
        if hist_b:
            adj_away *= max(0.94, min(1.06, self._momentum_factor(hist_b)))
        adj_home = max(0.5, adj_home)
        adj_away = max(0.5, adj_away)

        # ── Probabilidades Poisson ────────────────────────────────────
        p_home = p_away = p_tie = 0.0
        for h in range(max_r + 1):
            for a in range(max_r + 1):
                p = self._poisson(adj_home, h) * self._poisson(adj_away, a)
                if   h > a: p_home += p
                elif h < a: p_away += p
                else:       p_tie  += p

        p_home += p_tie * 0.52
        p_away += p_tie * 0.48
        tot = p_home + p_away or 1
        p_home /= tot; p_away /= tot

        total = adj_home + adj_away

        def _over(line):
            return sum(
                self._poisson(adj_home, h) * self._poisson(adj_away, a)
                for h in range(max_r+1) for a in range(max_r+1) if h+a > line
            )

        over_4_5  = _over(4.5)
        over_6_5  = _over(6.5)
        over_7_5  = _over(7.5)
        over_9_5  = _over(9.5)

        # ── Run line (-1.5) ───────────────────────────────────────────
        p_rl_home = sum(
            self._poisson(adj_home, h) * self._poisson(adj_away, a)
            for h in range(max_r+1) for a in range(max_r+1) if h - a >= 2
        )
        p_rl_away = sum(
            self._poisson(adj_home, h) * self._poisson(adj_away, a)
            for h in range(max_r+1) for a in range(max_r+1) if a - h >= 2
        )

        # ── Alt Run Lines (-2.5 y -0.5) ──────────────────────────────
        p_rl_home_25 = sum(
            self._poisson(adj_home, h) * self._poisson(adj_away, a)
            for h in range(max_r+1) for a in range(max_r+1) if h - a >= 3
        )
        p_rl_away_25 = sum(
            self._poisson(adj_home, h) * self._poisson(adj_away, a)
            for h in range(max_r+1) for a in range(max_r+1) if a - h >= 3
        )
        # -0.5 = gana por cualquier margen (moneyline sin empate posible)
        p_rl_home_05 = sum(
            self._poisson(adj_home, h) * self._poisson(adj_away, a)
            for h in range(max_r+1) for a in range(max_r+1) if h > a
        )
        p_rl_away_05 = sum(
            self._poisson(adj_home, h) * self._poisson(adj_away, a)
            for h in range(max_r+1) for a in range(max_r+1) if a > h
        )

        # ── 1er Inning — NRFI ────────────────────────────────────────
        F1        = 0.118
        first_h   = round(adj_home * F1, 3)
        first_a   = round(adj_away * F1, 3)
        first_tot = first_h + first_a
        p_nrfi    = self._poisson(first_h, 0) * self._poisson(first_a, 0)
        p_any     = 1.0 - p_nrfi
        p_f_ov05  = 1.0 - self._poisson(first_tot, 0)

        # ── Primeras 5 entradas (F5) ──────────────────────────────────
        F5_factor  = 0.535
        f5_home    = round(adj_home * F5_factor, 2)
        f5_away    = round(adj_away * F5_factor, 2)
        f5_total   = f5_home + f5_away
        f5_ou_ln   = round(f5_total / 0.5) * 0.5
        f5_p_home  = p_home * 0.52 + 0.48 * (f5_home / (f5_home + f5_away) if f5_home + f5_away > 0 else 0.5)
        over_f5_45 = sum(
            self._poisson(f5_home, h) * self._poisson(f5_away, a)
            for h in range(10) for a in range(10) if h+a > 4.5
        )

        # ── Proyección entrada por entrada ────────────────────────────
        inning_factors = [0.118, 0.108, 0.112, 0.108, 0.112,
                          0.108, 0.105, 0.112, 0.117]
        innings = []
        for i, f in enumerate(inning_factors):
            ih  = round(adj_home * f, 2)
            ia  = round(adj_away * f, 2)
            it  = round(ih + ia, 3)
            psc = round((1.0 - self._poisson(it, 0)) * 100)
            innings.append({'inning': i+1, 'home': ih, 'away': ia,
                            'total': round(it, 2), 'scorePct': psc})

        # ── Matriz de marcadores ──────────────────────────────────────
        matrix = []
        for h in range(12):
            for a in range(12):
                p = self._poisson(adj_home, h) * self._poisson(adj_away, a)
                matrix.append({'h': h, 'a': a, 'p': round(p, 4)})
        matrix.sort(key=lambda x: x['p'], reverse=True)

        return {
            'pctA':         round(p_home * 100),
            'pctB':         round(p_away * 100),
            'projHome':     round(adj_home, 2),
            'projAway':     round(adj_away, 2),
            'rawRpgHome':   round(home_rpg, 2),
            'rawRpgAway':   round(away_rpg, 2),
            'ragHome':      round(home_rag, 2) if home_rag else None,
            'ragAway':      round(away_rag, 2) if away_rag else None,
            'total':        round(total, 2),
            'over4_5':      round(over_4_5 * 100),
            'over6_5':      round(over_6_5 * 100),
            'over7_5':      round(over_7_5 * 100),
            'over9_5':      round(over_9_5 * 100),
            'runLineHome':  round(p_rl_home * 100),    # local -1.5
            'runLineAway':  round(p_rl_away * 100),    # visita -1.5
            'runLine25Home': round(p_rl_home_25 * 100), # local -2.5
            'runLine25Away': round(p_rl_away_25 * 100), # visita -2.5
            'runLine05Home': round(p_rl_home_05 * 100), # local -0.5 (gana)
            'runLine05Away': round(p_rl_away_05 * 100), # visita -0.5 (gana)
            'first': {
                'projHome':  round(first_h, 2),
                'projAway':  round(first_a, 2),
                'projTotal': round(first_tot, 2),
                'scoreAny':  round(p_any * 100),
                'nrfi':      round(p_nrfi * 100),
                'over05':    round(p_f_ov05 * 100),
            },
            'f5': {
                'projHome':  f5_home, 'projAway': f5_away,
                'projTotal': round(f5_total, 2),
                'ouLine':    f5_ou_ln, 'over4_5': round(over_f5_45 * 100),
                'pctA':      round(f5_p_home * 100),
            },
            'innings':       innings,
            'matrix':        matrix[:10],
            'h2h':           h2h_used,
            'hasCrossStats': bool(home_rag and away_rag),
            'dataQuality':   confidence_score,
        }

    def ai_bet_mlb(self, pred, team_a, team_b, hist_a=None, hist_b=None, h2h=None):
        """
        Recomendación de apuesta MLB v3.
        Usa historial real ponderado, splits home/away, H2H decay, NRFI y run line.
        """
        hist_a = hist_a or []
        hist_b = hist_b or []
        pA     = pred.get('pctA', 50)
        pB     = pred.get('pctB', 50)
        total  = pred.get('total', 8.5)
        ov45   = pred.get('over4_5', 50)
        ov65   = pred.get('over6_5', 50)
        ov75   = pred.get('over7_5', 50)
        rl_h    = pred.get('runLineHome', 40)
        rl_a    = pred.get('runLineAway', 40)
        rl_h25  = pred.get('runLine25Home', 20)
        rl_a25  = pred.get('runLine25Away', 20)
        rl_h05  = pred.get('runLine05Home', 50)
        rl_a05  = pred.get('runLine05Away', 50)
        first   = pred.get('first', {})
        f5      = pred.get('f5', {})

        def wgt_rate(arr, fn, decay=0.82):
            if not arr: return 0.5
            tw = tv = 0.0
            for i, m in enumerate(arr):
                w = decay**i; tw += w; tv += w*(1.0 if fn(m) else 0.0)
            return tv/tw if tw else 0.5

        win_a      = wgt_rate(hist_a, lambda m: m['result']=='G')
        win_b      = wgt_rate(hist_b, lambda m: m['result']=='G')
        home_a     = [m for m in hist_a if m.get('isHome', True)]
        away_b     = [m for m in hist_b if not m.get('isHome', True)]
        home_win_a = wgt_rate(home_a, lambda m: m['result']=='G') if home_a else win_a
        away_win_b = wgt_rate(away_b, lambda m: m['result']=='G') if away_b else win_b

        wgt_total_a = self._weighted_avg(hist_a,'myG') + self._weighted_avg(hist_a,'oppG')
        wgt_total_b = self._weighted_avg(hist_b,'myG') + self._weighted_avg(hist_b,'oppG')
        hist_avg_total = (wgt_total_a + wgt_total_b) / 2 if (hist_a and hist_b) else total

        def streak(arr):
            s = ''
            for m in arr[:5]:
                s += ('W' if m['result']=='G' else ('D' if m['result']=='E' else 'L'))
            return s

        streak_a = streak(hist_a)
        streak_b = streak(hist_b)

        # H2H — con decay exponencial (partidos recientes pesan más)
        h2h_info = {}
        if h2h and len(h2h) >= 3:
            h2h_decay = self._h2h_decay_avg(h2h, team_a, decay=0.75)
            h2h_info = {
                'n':        h2h_decay.get('n', len(h2h)),
                'winRateA': round(h2h_decay.get('winRateA', 0.5) * 100),
                'avgRuns':  round(h2h_decay.get('avgGoals', 0), 1),
            }

        n_total    = len(hist_a) + len(hist_b)
        conf_score = min(90, round((min(n_total,20)/20)*60 + (10 if h2h_info else 0)
                                   + (10 if len(hist_a)>=3 and len(hist_b)>=3 else 0)
                                   + (10 if pred.get('hasCrossStats') else 0)))

        bets = []

        if home_win_a >= 0.62 and len(home_a) >= 3:
            bets.append({'bet': f'{team_a} Gana (Local)', 'conf': min(78, round(home_win_a*100)+3),
                         'reason': f'{team_a} gana {round(home_win_a*100)}% en casa · racha: {streak_a}'})
        elif pA >= 58:
            bets.append({'bet': f'{team_a} Gana (Moneyline)', 'conf': min(74, pA),
                         'reason': f'{pA}% probabilidad · RPG {pred.get("projHome"):.2f} vs {pred.get("projAway"):.2f}'})

        if away_win_b >= 0.58 and len(away_b) >= 3:
            bets.append({'bet': f'{team_b} Gana (Visitante)', 'conf': min(74, round(away_win_b*100)),
                         'reason': f'{team_b} gana {round(away_win_b*100)}% de visita · racha: {streak_b}'})
        elif pB >= 58:
            bets.append({'bet': f'{team_b} Gana (Moneyline)', 'conf': min(74, pB),
                         'reason': f'{team_b} {pB}% — pitcheo sólido como visitante'})

        if rl_h >= 55:
            bets.append({'bet': f'{team_a} -1.5 (Run Line)', 'conf': min(72, rl_h),
                         'reason': f'{rl_h}% de ganar por ≥2 carreras · proyección {pred.get("projHome"):.1f} vs {pred.get("projAway"):.1f}'})
        elif rl_a >= 55:
            bets.append({'bet': f'{team_b} +1.5 (Run Line)', 'conf': min(72, rl_a),
                         'reason': f'{rl_a}% de mantenerse a ≤1 carrera de diferencia'})

        # ── Alt Run Lines (-2.5 / -0.5) ──────────────────────────────
        if rl_h25 >= 42:
            bets.append({'bet': f'{team_a} -2.5 (Alt Run Line)', 'conf': min(68, rl_h25),
                         'reason': f'{rl_h25}% de ganar por 3+ · ofensiva dominante proyectada'})
        elif rl_a25 >= 42:
            bets.append({'bet': f'{team_b} -2.5 (Alt Run Line)', 'conf': min(68, rl_a25),
                         'reason': f'{rl_a25}% de que visita gane por 3+ carreras'})
        if rl_h05 >= 65 and rl_h < 55:
            bets.append({'bet': f'{team_a} -0.5 (Gana partido)', 'conf': min(70, rl_h05),
                         'reason': f'{rl_h05}% de ganar por cualquier margen — moneyline reforzado'})
        elif rl_a05 >= 65 and rl_a < 55:
            bets.append({'bet': f'{team_b} -0.5 (Gana partido)', 'conf': min(70, rl_a05),
                         'reason': f'{rl_a05}% de que visita gane — pitcheo sólido proyectado'})

        if ov65 >= 62:
            bets.append({'bet': 'Más de 6.5 carreras', 'conf': min(74, ov65),
                         'reason': f'{ov65}% Over 6.5 · total ponderado {hist_avg_total:.1f} · proyección {total:.1f}'})
        elif ov45 <= 40:
            bets.append({'bet': 'Menos de 4.5 carreras', 'conf': min(74, 100-ov45),
                         'reason': f'Solo {ov45}% Over 4.5 — pitcheo dominante de ambos lados'})
        elif ov75 >= 58:
            bets.append({'bet': 'Más de 7.5 carreras', 'conf': min(72, ov75),
                         'reason': f'{ov75}% Over 7.5 · ofensivas productivas proyectadas'})

        nrfi = first.get('nrfi', 50)
        if nrfi >= 60:
            bets.append({'bet': 'NRFI — Sin Carreras en 1er Inning', 'conf': min(70, nrfi),
                         'reason': f'{nrfi}% de que el 1er inning termine 0-0 — pitcheo abridor sólido'})
        elif first.get('scoreAny', 50) >= 68:
            bets.append({'bet': '1er Inning — Alguien Anota', 'conf': min(70, first.get('scoreAny',50)),
                         'reason': f'{first.get("scoreAny")}% de anotar en 1er inning — ofensivas activas'})

        if f5.get('over4_5', 50) >= 58:
            bets.append({'bet': f'F5 Más de {f5.get("ouLine",4.5)} carreras', 'conf': min(70, f5['over4_5']),
                         'reason': f'{f5["over4_5"]}% Over {f5.get("ouLine")} en primeras 5 entradas'})

        if h2h_info.get('winRateA', 50) >= 70:
            bets.append({'bet': f'{team_a} Gana (Domina H2H)', 'conf': min(76, h2h_info['winRateA']),
                         'reason': f'Gana {h2h_info["winRateA"]}% de los últimos {h2h_info["n"]} enfrentamientos directos'})

        if not bets:
            bets.append({'bet': 'Más de 4.5 carreras' if ov45 >= 50 else 'Menos de 4.5 carreras',
                         'conf': max(ov45, 100-ov45, 51),
                         'reason': f'Total proyectado {total:.1f} carreras'})

        bets.sort(key=lambda x: x['conf'], reverse=True)

        narrative = (
            f"⚾ {team_a} (local) vs {team_b} (visita).\n\n"
            f"Proyección: {pred.get('projHome'):.2f} – {pred.get('projAway'):.2f} carreras (Total {total:.2f}).\n"
            + (f"Ratings cruzados — ERA proxy {team_a}: {pred.get('ragHome')} · {team_b}: {pred.get('ragAway')} RPG permitidas.\n"
               if pred.get('hasCrossStats') else '')
            + (f"H2H: {team_a} gana {h2h_info.get('winRateA')}% · Promedio {h2h_info.get('avgRuns')} carreras/partido.\n"
               if h2h_info else '')
            + f"NRFI: {first.get('nrfi',50)}% · Over 6.5: {ov65}% · Run Line local: {rl_h}%\n"
            + f"\n{bets[0]['bet']} ({bets[0]['conf']}%) — {bets[0]['reason']}."
            f"\n\nℹ️ Análisis orientativo. Apuesta con responsabilidad."
        )
        return {
            'best':            bets[0] if bets else None,
            'alt':             bets[1] if len(bets) > 1 else None,
            'narrative':       narrative,
            'confidenceScore': conf_score,
            'hasRealData':     len(hist_a) >= 2 or len(hist_b) >= 2,
            'h2h':             h2h_info,
            'homeWinA':        round(home_win_a * 100),
            'awayWinB':        round(away_win_b * 100),
            'streakA':         streak_a,
            'streakB':         streak_b,
            'runLine25Home':   rl_h25,
            'runLine25Away':   rl_a25,
            'runLine05Home':   rl_h05,
            'runLine05Away':   rl_a05,
        }

    # ─── NHL — Predicción v2 ──────────────────────────────────────
    def predict_nhl(self, home_gpg, away_gpg,
                    home_gag=None, away_gag=None, h2h=None,
                    confidence_score=0, hist_a=None, hist_b=None):
        """
        Predicción NHL — Poisson sobre goles por partido.
        home_gpg / away_gpg : goles anotados por partido (ponderados con decay)
        home_gag / away_gag : goles PERMITIDOS (proxy GAA del portero)
        h2h                 : lista de enfrentamientos directos
        hist_a / hist_b     : historial reciente para momentum de forma (±6%)
        """
        HOME_ADV = 0.15
        max_g    = 12

        # ── Proyección cruzada: ataque vs. portero rival (GAA) ────────
        if home_gag and away_gag:
            cross_home = (home_gpg + away_gag) / 2
            cross_away = (away_gpg + home_gag) / 2
        else:
            cross_home = home_gpg
            cross_away = away_gpg

        adj_home = max(0.5, cross_home + HOME_ADV)
        adj_away = max(0.5, cross_away)

        # ── Blend H2H ─────────────────────────────────────────────────
        h2h_used = {}
        if h2h and len(h2h) >= 3:
            h2h_info = self._h2h_decay_avg(h2h, '', decay=0.75)
            h2h_avg  = h2h_info.get('avgGoals', sum(r['home_goals']+r['away_goals'] for r in h2h)/len(h2h))
            blend    = 0.22
            ratio    = ((adj_home + adj_away) * (1 - blend) + h2h_avg * blend) / max(0.01, adj_home + adj_away)
            adj_home *= ratio
            adj_away *= ratio
            h2h_used = {'n': len(h2h), 'avgGoals': round(h2h_avg, 2)}

        # ── Momentum de forma reciente (±6% sobre λ) ─────────────────
        if hist_a:
            adj_home *= max(0.94, min(1.06, self._momentum_factor(hist_a)))
        if hist_b:
            adj_away *= max(0.94, min(1.06, self._momentum_factor(hist_b)))
        adj_home = max(0.5, adj_home)
        adj_away = max(0.5, adj_away)

        # ── Probabilidades Poisson ────────────────────────────────────
        p_home = p_away = p_reg_tie = 0.0
        for h in range(max_g + 1):
            for a in range(max_g + 1):
                p = self._poisson(adj_home, h) * self._poisson(adj_away, a)
                if   h > a: p_home     += p
                elif h < a: p_away     += p
                else:       p_reg_tie  += p

        p_home_ot = p_reg_tie * 0.515
        p_away_ot = p_reg_tie * 0.485
        p_home_ml = p_home + p_home_ot
        p_away_ml = p_away + p_away_ot

        tot = p_home_ml + p_away_ml or 1
        p_home_ml /= tot
        p_away_ml /= tot

        total = adj_home + adj_away

        def _over(line):
            return sum(
                self._poisson(adj_home, h) * self._poisson(adj_away, a)
                for h in range(max_g+1) for a in range(max_g+1) if h+a > line
            )

        over_5_5  = _over(5.5)
        over_6_0  = _over(6.0)
        over_6_5  = _over(6.5)
        over_7_5  = _over(7.5)

        # ── Puck Line (-1.5) ──────────────────────────────────────────
        p_pl_home = sum(
            self._poisson(adj_home, h) * self._poisson(adj_away, a)
            for h in range(max_g+1) for a in range(max_g+1) if h - a >= 2
        )
        p_pl_away = sum(
            self._poisson(adj_home, h) * self._poisson(adj_away, a)
            for h in range(max_g+1) for a in range(max_g+1) if a - h >= 2
        )

        p_ot = round(p_reg_tie * 100)

        # ── 1er Período (P1) ──────────────────────────────────────────
        P1_FACTOR = 0.305
        p1_home   = round(adj_home * P1_FACTOR, 2)
        p1_away   = round(adj_away * P1_FACTOR, 2)
        p1_total  = p1_home + p1_away
        p1_nrfp   = self._poisson(p1_home, 0) * self._poisson(p1_away, 0)
        p1_any    = 1 - p1_nrfp
        p1_ou_ln  = round(p1_total / 0.5) * 0.5

        over_p1_15 = sum(
            self._poisson(p1_home, h) * self._poisson(p1_away, a)
            for h in range(7) for a in range(7) if h+a > 1.5
        )

        # ── Proyección por período ────────────────────────────────────
        p_factors  = [0.305, 0.335, 0.360]
        periods    = []
        for i, pf in enumerate(p_factors):
            ph  = round(adj_home * pf, 2)
            pa  = round(adj_away * pf, 2)
            pt  = round(ph + pa, 2)
            pln = round(pt / 0.5) * 0.5
            pov = min(74, max(26, round(50 + (pt - pln) * 18)))
            periods.append({'period': i+1, 'projHome': ph, 'projAway': pa,
                            'projTotal': pt, 'ouLine': pln, 'overPct': pov})

        # ── Matriz de marcadores ──────────────────────────────────────
        matrix = []
        for h in range(10):
            for a in range(10):
                p = self._poisson(adj_home, h) * self._poisson(adj_away, a)
                matrix.append({'h': h, 'a': a, 'p': round(p, 4)})
        matrix.sort(key=lambda x: x['p'], reverse=True)

        return {
            'pctA':         round(p_home_ml * 100),
            'pctB':         round(p_away_ml * 100),
            'pOT':          p_ot,
            'projHome':     round(adj_home, 2),
            'projAway':     round(adj_away, 2),
            'rawGpgHome':   round(home_gpg, 2),
            'rawGpgAway':   round(away_gpg, 2),
            'gagHome':      round(home_gag, 2) if home_gag else None,
            'gagAway':      round(away_gag, 2) if away_gag else None,
            'total':        round(total, 2),
            'over5_5':      round(over_5_5 * 100),
            'over6_0':      round(over_6_0 * 100),
            'over6_5':      round(over_6_5 * 100),
            'over7_5':      round(over_7_5 * 100),
            'puckLineHome': round(p_pl_home * 100),
            'puckLineAway': round(p_pl_away * 100),
            'period1': {
                'projHome':  p1_home,   'projAway': p1_away,
                'projTotal': round(p1_total, 2),
                'ouLine':    p1_ou_ln,  'scoreAny': round(p1_any * 100),
                'nrfp':      round(p1_nrfp * 100),
                'over1_5':   round(over_p1_15 * 100),
            },
            'periods':       periods,
            'matrix':        matrix[:10],
            'h2h':           h2h_used,
            'hasCrossStats': bool(home_gag and away_gag),
            'dataQuality':   confidence_score,
        }

    def ai_bet_nhl(self, pred, team_a, team_b, hist_a=None, hist_b=None, h2h=None):
        """
        Recomendación de apuesta NHL v2.
        Moneyline, Puck Line, O/U, 1er Período, NRFP y señales de historial real.
        """
        hist_a = hist_a or []
        hist_b = hist_b or []
        pA     = pred.get('pctA', 50)
        pB     = pred.get('pctB', 50)
        p_ot   = pred.get('pOT', 20)
        total  = pred.get('total', 6.0)
        ov55   = pred.get('over5_5', 50)
        ov65   = pred.get('over6_5', 50)
        pl_h   = pred.get('puckLineHome', 35)
        pl_a   = pred.get('puckLineAway', 35)
        p1     = pred.get('period1', {})

        def wgt_rate(arr, fn, decay=0.82):
            if not arr: return 0.5
            tw = tv = 0.0
            for i, m in enumerate(arr):
                w = decay**i; tw += w; tv += w*(1.0 if fn(m) else 0.0)
            return tv/tw if tw else 0.5

        win_a      = wgt_rate(hist_a, lambda m: m['result']=='G')
        win_b      = wgt_rate(hist_b, lambda m: m['result']=='G')
        home_a     = [m for m in hist_a if m.get('isHome', True)]
        away_b     = [m for m in hist_b if not m.get('isHome', True)]
        home_win_a = wgt_rate(home_a, lambda m: m['result']=='G') if home_a else win_a
        away_win_b = wgt_rate(away_b, lambda m: m['result']=='G') if away_b else win_b

        wgt_total_a = self._weighted_avg(hist_a,'myG') + self._weighted_avg(hist_a,'oppG')
        wgt_total_b = self._weighted_avg(hist_b,'myG') + self._weighted_avg(hist_b,'oppG')
        hist_avg_total = (wgt_total_a + wgt_total_b) / 2 if (hist_a and hist_b) else total

        h2h_info = {}
        if h2h and len(h2h) >= 3:
            wins_a = sum(
                1 for r in h2h
                if (team_a[:6].lower() in r['home_team'].lower() and r['home_goals'] > r['away_goals'])
                or (team_a[:6].lower() in r['away_team'].lower() and r['away_goals'] > r['home_goals'])
            )
            goals = [r['home_goals']+r['away_goals'] for r in h2h]
            h2h_info = {
                'n': len(h2h),
                'winRateA': round(wins_a/len(h2h)*100),
                'avgGoals': round(sum(goals)/len(goals), 1),
            }

        n_total    = len(hist_a) + len(hist_b)
        conf_score = min(90, round((min(n_total,20)/20)*60
                         + (10 if h2h_info else 0)
                         + (10 if len(hist_a)>=3 and len(hist_b)>=3 else 0)
                         + (10 if pred.get('hasCrossStats') else 0)))

        bets = []

        if home_win_a >= 0.62 and len(home_a) >= 3:
            bets.append({'bet': f'{team_a} Gana (ML — Local)', 'conf': min(78, round(home_win_a*100)+3),
                         'reason': f'{team_a} gana {round(home_win_a*100)}% en casa · spread proyectado {pred.get("projHome",3.0):.2f} vs {pred.get("projAway",3.0):.2f}'})
        elif pA >= 60:
            bets.append({'bet': f'{team_a} Gana (Moneyline)', 'conf': min(76, pA),
                         'reason': f'{pA}% probabilidad · GPG {pred.get("projHome"):.2f} vs {pred.get("projAway"):.2f}'})

        if away_win_b >= 0.58 and len(away_b) >= 3:
            bets.append({'bet': f'{team_b} Gana (ML — Visitante)', 'conf': min(74, round(away_win_b*100)),
                         'reason': f'{team_b} gana {round(away_win_b*100)}% de visita — portería sólida'})
        elif pB >= 60:
            bets.append({'bet': f'{team_b} Gana (Moneyline)', 'conf': min(74, pB),
                         'reason': f'{team_b} {pB}% — visitante en buena forma'})

        if pl_h >= 52:
            bets.append({'bet': f'{team_a} -1.5 (Puck Line)', 'conf': min(70, pl_h),
                         'reason': f'{pl_h}% de ganar por 2+ goles · GPG {pred.get("projHome"):.2f} vs GAA rival {pred.get("gagAway") or "?"}'})
        elif pl_a >= 52:
            bets.append({'bet': f'{team_b} +1.5 (Puck Line)', 'conf': min(70, pl_a),
                         'reason': f'{pl_a}% de mantenerse a ≤1 gol de diferencia'})

        if ov55 >= 62:
            bets.append({'bet': 'Más de 5.5 goles', 'conf': min(74, ov55),
                         'reason': f'{ov55}% Over 5.5 · total proyectado {total:.2f} · historial ponderado {hist_avg_total:.1f}'})
        elif ov65 <= 36:
            bets.append({'bet': 'Menos de 6.5 goles', 'conf': min(74, 100-ov65),
                         'reason': f'Solo {ov65}% Over 6.5 — porteros sólidos proyectados'})
        elif ov65 >= 58:
            bets.append({'bet': 'Más de 6.5 goles', 'conf': min(72, ov65),
                         'reason': f'{ov65}% Over 6.5 · ofensivas en racha'})

        nrfp = p1.get('nrfp', 50)
        if nrfp >= 58:
            bets.append({'bet': 'NRFP — Sin goles en 1er período', 'conf': min(68, nrfp),
                         'reason': f'{nrfp}% de que el P1 termine 0-0 — porteros dominantes al inicio'})
        elif p1.get('scoreAny', 50) >= 72:
            bets.append({'bet': '1er Período — Alguien Anota', 'conf': min(70, p1.get('scoreAny',50)),
                         'reason': f'{p1.get("scoreAny")}% de anotar en P1 · total proyectado {p1.get("projTotal","?")}'})

        if p_ot >= 28:
            bets.append({'bet': f'Se Va al Tiempo Extra / Shootout', 'conf': min(65, p_ot),
                         'reason': f'{p_ot}% de probabilidad de OT/SO — equipos muy parejos'})

        if h2h_info.get('winRateA', 50) >= 70:
            bets.append({'bet': f'{team_a} Gana (Domina H2H)', 'conf': min(74, h2h_info['winRateA']),
                         'reason': f'Gana {h2h_info["winRateA"]}% de sus últimos {h2h_info["n"]} duelos directos'})

        if not bets:
            bets.append({'bet': 'Más de 5.5 goles' if ov55 >= 50 else 'Menos de 5.5 goles',
                         'conf': max(ov55, 100-ov55, 51),
                         'reason': f'Total proyectado {total:.2f} goles'})

        bets.sort(key=lambda x: x['conf'], reverse=True)

        narrative = (
            f"🏒 {team_a} (local) vs {team_b} (visita).\n\n"
            f"Proyección: {pred.get('projHome'):.2f} – {pred.get('projAway'):.2f} goles (Total {total:.2f}).\n"
            + (f"GAA cruzado — {team_a}: {pred.get('gagHome')} · {team_b}: {pred.get('gagAway')} goles permitidos.\n"
               if pred.get('hasCrossStats') else '')
            + f"Puck Line local: {pl_h}% · OT probable: {p_ot}% · Over 5.5: {ov55}%\n"
            + (f"H2H: {team_a} domina {h2h_info.get('winRateA')}% · Prom {h2h_info.get('avgGoals')} goles.\n"
               if h2h_info else '')
            + f"\n{bets[0]['bet']} ({bets[0]['conf']}%) — {bets[0]['reason']}."
            f"\n\nℹ️ Análisis orientativo. Apuesta con responsabilidad."
        )
        return {
            'best':            bets[0] if bets else None,
            'alt':             bets[1] if len(bets) > 1 else None,
            'narrative':       narrative,
            'confidenceScore': conf_score,
            'hasRealData':     len(hist_a) >= 2 or len(hist_b) >= 2,
            'h2h':             h2h_info,
            'homeWinA':        round(home_win_a * 100),
            'awayWinB':        round(away_win_b * 100),
        }

    # ─── Predicciones EN VIVO ─────────────────────────────────────

    def predict_live_soccer(self, home_score, away_score, minute, xg_a, xg_b):
        """
        Ajusta probabilidades en vivo según el marcador y el minuto.
        Mientras más avanza el partido, el marcador actual pesa más.
        """
        minute   = max(1, min(90, minute))
        pct_done = minute / 90.0
        pct_left = 1.0 - pct_done

        rem_a = xg_a * pct_left
        rem_b = xg_b * pct_left

        def p_goals(lam, k):
            try: return math.exp(-lam) * (lam**k) / math.factorial(k)
            except: return 0.0

        p_home = p_draw = p_away = 0.0
        for ha in range(6):
            for aa in range(6):
                p = p_goals(rem_a, ha) * p_goals(rem_b, aa)
                final_h = home_score + ha
                final_a = away_score + aa
                if   final_h > final_a: p_home += p
                elif final_h == final_a: p_draw += p
                else:                    p_away += p

        total = p_home + p_draw + p_away or 1
        p_home /= total; p_draw /= total; p_away /= total

        if home_score > away_score:
            p_home = min(0.97, p_home + 0.05)
            p_away = max(0.01, p_away - 0.03)
            p_draw = max(0.01, 1 - p_home - p_away)
        elif away_score > home_score:
            p_away = min(0.97, p_away + 0.05)
            p_home = max(0.01, p_home - 0.03)
            p_draw = max(0.01, 1 - p_home - p_away)

        mins_left = 90 - minute
        return {
            'pctA':      round(p_home * 100),
            'pctD':      round(p_draw * 100),
            'pctB':      round(p_away * 100),
            'minsLeft':  mins_left,
            'xgRemA':    round(rem_a, 2),
            'xgRemB':    round(rem_b, 2),
            'scoreA':    home_score,
            'scoreB':    away_score,
            'minute':    minute,
        }

    def predict_live_nba(self, home_pts, away_pts, period, home_ppg, away_ppg):
        """
        Proyección de partido NBA en vivo.
        period 1-4, home_ppg/away_ppg = puntos por partido histórico.
        """
        period    = max(1, min(4, period))
        pct_done  = (period - 1) / 4.0 + 0.5 / 4.0
        pct_left  = max(0.05, 1.0 - pct_done)

        proj_a = home_pts + home_ppg * pct_left
        proj_b = away_pts + away_ppg * pct_left
        diff   = proj_a - proj_b

        p_home = 1 / (1 + math.exp(-diff / 8))
        p_away = 1 - p_home

        quarters_left = 4 - period
        pts_left_a    = round(home_ppg * pct_left)
        pts_left_b    = round(away_ppg * pct_left)

        return {
            'pctA':        round(p_home * 100),
            'pctB':        round(p_away * 100),
            'projA':       round(proj_a),
            'projB':       round(proj_b),
            'ptsLeftA':    pts_left_a,
            'ptsLeftB':    pts_left_b,
            'quartersLeft': quarters_left,
            'scoreA':      home_pts,
            'scoreB':      away_pts,
            'period':      period,
        }

    def predict_live_mlb(self, home_runs, away_runs, inning, half, home_rpg, away_rpg):
        """
        Proyección de partido MLB en vivo.
        inning 1-9, half 'top'/'bottom', home_rpg/away_rpg = carreras por partido.
        """
        inning    = max(1, min(9, inning))
        half_val  = 0.5 if half == 'bottom' else 0.0
        pct_done  = ((inning - 1) + half_val) / 9.0
        pct_left  = max(0.05, 1.0 - pct_done)

        proj_a = home_runs + home_rpg * pct_left
        proj_b = away_runs + away_rpg * pct_left
        diff   = proj_a - proj_b

        p_home = 1 / (1 + math.exp(-diff / 2))
        p_away = 1 - p_home

        innings_left = 9 - inning

        return {
            'pctA':        round(p_home * 100),
            'pctB':        round(p_away * 100),
            'projA':       round(proj_a, 1),
            'projB':       round(proj_b, 1),
            'inningsLeft': innings_left,
            'scoreA':      home_runs,
            'scoreB':      away_runs,
            'inning':      inning,
            'half':        half,
        }

    # ─── TENIS — Predicción v1 ───────────────────────────────────
    def predict_tennis(self, rank_a, rank_b, surface='hard',
                       h2h=None, hist_a=None, hist_b=None,
                       confidence_score=0, fmt='bo3'):
        """
        Predicción Tenis v1 — modelo Elo por ranking + superficie + forma.

        rank_a / rank_b : ranking ATP/WTA (menor = mejor). Si es 0 → desconocido.
        surface         : 'hard' | 'clay' | 'grass' | 'indoor'
        fmt             : 'bo3' (mejor de 3, WTA/ATP 250-500) | 'bo5' (Grand Slams)
        h2h             : historial directo de partidos
        hist_a / hist_b : forma reciente [{'result': 'G'/'P', ...}]

        Modelo:
        1. Probabilidad base desde rankings → Elo logarítmico
        2. Factor de superficie (clay favorece defensa, grass favorece servicio)
        3. Momentum de forma reciente (±8%)
        4. Blend con H2H decay (peso 25% si hay ≥3 H2H)
        5. Set handicap, juegos O/U, breakpoint, tiebreak
        """
        max_rank = 500  # tope para jugadores sin ranking conocido

        # ── 1. Elo estimado desde ranking ────────────────────────────
        def rank_to_elo(r):
            r = max(1, r or max_rank)
            return 2500 - 350 * math.log(r)  # rank 1 → ~2500, rank 100 → ~1879

        elo_a = rank_to_elo(rank_a)
        elo_b = rank_to_elo(rank_b)

        # ── 2. Factor de superficie ──────────────────────────────────
        # Sin datos de especialidad de jugador, ajustamos muy levemente
        # (la especialidad real se aprende con el tiempo del historial)
        SURFACE_FACTORS = {
            'clay':   {'home_bonus': 0.0, 'variance': 0.85},   # más lento, menos ventaja servicio
            'grass':  {'home_bonus': 0.0, 'variance': 1.15},   # más rápido, favorece favorito
            'hard':   {'home_bonus': 0.0, 'variance': 1.00},   # neutro
            'indoor': {'home_bonus': 0.0, 'variance': 1.05},   # ligeramente más rápido
        }
        surf = SURFACE_FACTORS.get(surface.lower(), SURFACE_FACTORS['hard'])
        # En arcilla los partidos son más ajustados → reducir diferencia de Elo
        elo_diff = (elo_a - elo_b) * surf['variance']
        p_base_a = 1.0 / (1.0 + 10 ** (-elo_diff / 400))

        # ── 3. Momentum de forma reciente (±8%) ─────────────────────
        mom_a = max(0.92, min(1.08, self._momentum_factor(hist_a))) if hist_a else 1.0
        mom_b = max(0.92, min(1.08, self._momentum_factor(hist_b))) if hist_b else 1.0
        # Ajustar la probabilidad en logit-space para evitar extremos
        logit = math.log(p_base_a / (1 - p_base_a)) if 0 < p_base_a < 1 else 0.0
        logit += (mom_a - 1.0) * 2.0 - (mom_b - 1.0) * 2.0
        p_adj_a = 1.0 / (1.0 + math.exp(-logit))

        # ── 4. Blend H2H con decay ───────────────────────────────────
        h2h_info = {}
        if h2h and len(h2h) >= 3:
            h2h_dec = self._h2h_decay_avg(h2h, '', decay=0.75)
            h2h_win_a = h2h_dec.get('winRateA', p_adj_a)
            # Blend 25% H2H decay-weighted
            p_adj_a = p_adj_a * 0.75 + h2h_win_a * 0.25
            h2h_info = {
                'n':        h2h_dec.get('n', len(h2h)),
                'winRateA': round(h2h_win_a * 100),
                'avgGames': round(h2h_dec.get('avgGoals', 21), 1),
            }

        # Normalizar
        p_a = max(0.03, min(0.97, p_adj_a))
        p_b = 1.0 - p_a

        # ── 5. Set handicap — probabilidad de ganar por +1.5 sets ────
        # Modelo: si p_a = 0.75, P(2-0 bo3) ≈ p_a², P(2-1) ≈ 2*p_a*(1-p_a)*p_a
        if fmt == 'bo5':
            max_sets = 5
            win_target = 3
        else:
            max_sets = 3
            win_target = 2

        def p_win_clean(p, target, max_s):
            """P(ganar target-0)"""
            return p ** target

        def p_win_margin(p, target, max_s):
            """P(ganar con al menos 1 set perdido)"""
            return 1 - p_win_clean(p, target, max_s) - sum(
                math.comb(target - 1 + i, i) * (1 - p) ** i * p ** (target)
                for i in range(1, max_s - target + 1)
            ) if max_s > target else 0.0

        p_hl_a = p_win_clean(p_a, win_target, max_sets)   # gana limpio (handicap -1.5)
        p_hl_b = p_win_clean(p_b, win_target, max_sets)   # visitante gana limpio

        # ── 6. O/U juegos totales ────────────────────────────────────
        # Promedio ATP: ~21 juegos bo3, ~36 juegos bo5
        base_games = 35.5 if fmt == 'bo5' else 21.5
        # Partidos muy parejos → más juegos (tiebreaks)
        competitiveness = 1.0 - abs(p_a - 0.5) * 2   # 0 (one-sided) → 1 (50/50)
        proj_games = round(base_games + competitiveness * 4.5, 1)
        ou_line_games = round(proj_games / 0.5) * 0.5

        # ── 7. Probabilidad de tiebreak en el set decisivo ───────────
        p_tiebreak = round(min(75, max(15, (1.0 - abs(p_a - 0.5) * 2) * 60)), 1)

        # ── 8. Probabilidad Over 2.5 sets (en bo3 = duración del partido) ─
        # P(2-0 de cualquier lado) = limpio, P(2-1) = competido
        p_clean = p_a ** win_target + p_b ** win_target
        p_over_sets = max(0, round((1 - p_clean) * 100))   # P(partido va a sets)

        return {
            'pctA':          round(p_a * 100),
            'pctB':          round(p_b * 100),
            'surface':       surface,
            'format':        fmt,
            'eloA':          round(elo_a, 1),
            'eloB':          round(elo_b, 1),
            'rankA':         rank_a or 0,
            'rankB':         rank_b or 0,
            'handicapA':     round(p_hl_a * 100),   # gana sin perder sets
            'handicapB':     round(p_hl_b * 100),
            'projGames':     proj_games,
            'ouLineGames':   ou_line_games,
            'tiebreakPct':   p_tiebreak,
            'overSetsPct':   p_over_sets,            # P(>2 sets en bo3 / >4 en bo5)
            'h2h':           h2h_info,
            'dataQuality':   confidence_score,
            'model':         'tennis-elo-v1',
        }

    def ai_bet_tennis(self, pred, player_a, player_b,
                      hist_a=None, hist_b=None, h2h=None):
        """
        Recomendación de apuesta Tenis v1.
        Moneyline, Set Handicap, O/U Juegos, Tiebreak.
        """
        hist_a = hist_a or []
        hist_b = hist_b or []
        pA     = pred.get('pctA', 50)
        pB     = pred.get('pctB', 50)
        hl_a   = pred.get('handicapA', 30)   # gana sin ceder sets
        hl_b   = pred.get('handicapB', 30)
        games  = pred.get('projGames', 21.5)
        ou_ln  = pred.get('ouLineGames', 21.5)
        over_s = pred.get('overSetsPct', 40)   # P(>sets)
        tb_pct = pred.get('tiebreakPct', 35)
        surface = pred.get('surface', 'hard')
        fmt    = pred.get('format', 'bo3')

        def wgt_rate(arr, fn, decay=0.82):
            if not arr: return 0.5
            tw = tv = 0.0
            for i, m in enumerate(arr):
                w = decay**i; tw += w; tv += w*(1.0 if fn(m) else 0.0)
            return tv/tw if tw else 0.5

        win_a = wgt_rate(hist_a, lambda m: m['result']=='G')
        win_b = wgt_rate(hist_b, lambda m: m['result']=='G')

        h2h_info = pred.get('h2h', {})

        n_total    = len(hist_a) + len(hist_b)
        conf_score = min(88, round((min(n_total,16)/16)*55
                         + (15 if h2h_info else 0)
                         + (10 if len(hist_a)>=3 and len(hist_b)>=3 else 0)
                         + (8  if pred.get('rankA') and pred.get('rankB') else 0)))

        surf_label = {'clay': 'Arcilla 🟠', 'grass': 'Hierba 🟢',
                      'hard': 'Dura 🔵', 'indoor': 'Indoor 🏟️'}.get(surface, surface)
        fmt_label  = 'Mejor de 5' if fmt == 'bo5' else 'Mejor de 3'

        bets = []

        # ── Moneyline ─────────────────────────────────────────────────
        if pA >= 65:
            bets.append({'bet': f'{player_a} Gana', 'conf': min(82, pA),
                         'reason': f'{pA}% probabilidad · Ranking #{pred.get("rankA","?")} vs #{pred.get("rankB","?")} · {surf_label}'})
        elif pB >= 65:
            bets.append({'bet': f'{player_b} Gana', 'conf': min(82, pB),
                         'reason': f'{pB}% probabilidad · {surf_label} favorece al visitante'})
        elif pA >= 55:
            bets.append({'bet': f'{player_a} Gana (Moneyline)', 'conf': min(72, pA),
                         'reason': f'{pA}% · ligera ventaja por ranking y forma reciente'})

        # ── Set Handicap (ganar 2-0 bo3 / 3-0 bo5) ───────────────────
        if hl_a >= 48:
            bets.append({'bet': f'{player_a} -{1 if fmt=="bo3" else 2}.5 Sets', 'conf': min(74, hl_a),
                         'reason': f'{hl_a}% de ganar sin ceder sets — dominio esperado en {surf_label}'})
        elif hl_b >= 45:
            bets.append({'bet': f'{player_b} +{1 if fmt=="bo3" else 2}.5 Sets', 'conf': min(70, 100-hl_a),
                         'reason': f'{100-hl_a}% de que el partido no sea limpio — {player_b} lucha'})

        # ── O/U Juegos totales ─────────────────────────────────────────
        if games > ou_ln + 1.0:
            bets.append({'bet': f'Más de {ou_ln} juegos', 'conf': min(74, round(55 + (games - ou_ln) * 5)),
                         'reason': f'Proyección {games} juegos — partido competido en {surf_label}'})
        elif games < ou_ln - 1.0:
            bets.append({'bet': f'Menos de {ou_ln} juegos', 'conf': min(74, round(55 + (ou_ln - games) * 5)),
                         'reason': f'Proyección {games} juegos — dominio del favorito esperado'})

        # ── Over sets (partido largo) ──────────────────────────────────
        if over_s >= 62 and fmt == 'bo3':
            bets.append({'bet': f'Partido a 3 Sets', 'conf': min(72, over_s),
                         'reason': f'{over_s}% de que llegue al set decisivo — {fmt_label} muy igualado'})

        # ── Tiebreak ──────────────────────────────────────────────────
        if tb_pct >= 52:
            bets.append({'bet': 'Al menos 1 Tiebreak', 'conf': min(68, tb_pct),
                         'reason': f'{tb_pct}% de probabilidad de tiebreak — jugadores muy equilibrados'})

        # ── H2H ───────────────────────────────────────────────────────
        if h2h_info and h2h_info.get('winRateA', 50) >= 70:
            bets.append({'bet': f'{player_a} Gana (Domina H2H)', 'conf': min(78, h2h_info['winRateA']),
                         'reason': f'Gana {h2h_info["winRateA"]}% de los últimos {h2h_info["n"]} enfrentamientos directos'})

        # ── Forma reciente ─────────────────────────────────────────────
        if win_a >= 0.78 and len(hist_a) >= 4 and pA >= 55:
            bets.append({'bet': f'{player_a} Gana [🔥 Racha]', 'conf': min(80, round(win_a*100)+2),
                         'reason': f'{player_a} gana {round(win_a*100)}% de sus últimos partidos — en racha'})

        if not bets:
            fav = player_a if pA >= pB else player_b
            p_fav = max(pA, pB)
            bets.append({'bet': f'{fav} Gana', 'conf': max(p_fav, 52),
                         'reason': f'Favorito por ranking · {surf_label} · {fmt_label}'})

        bets.sort(key=lambda x: x['conf'], reverse=True)

        narrative = (
            f"🎾 {player_a} vs {player_b} · {surf_label} · {fmt_label}.\n\n"
            f"Ranking: #{pred.get('rankA','?')} vs #{pred.get('rankB','?')} "
            f"(Elo estimado {pred.get('eloA',0):.0f} vs {pred.get('eloB',0):.0f}).\n"
            f"Probabilidad: {player_a} {pA}% — {player_b} {pB}%.\n"
            f"Juegos proyectados: {games} · Tiebreak: {tb_pct}%\n"
            + (f"H2H: {player_a} gana {h2h_info.get('winRateA')}% de {h2h_info.get('n')} encuentros.\n"
               if h2h_info else '')
            + f"\n{bets[0]['bet']} ({bets[0]['conf']}%) — {bets[0]['reason']}."
            f"\n\nℹ️ Análisis orientativo. Apuesta con responsabilidad."
        )
        return {
            'best':            bets[0] if bets else None,
            'alt':             bets[1] if len(bets) > 1 else None,
            'narrative':       narrative,
            'confidenceScore': conf_score,
            'hasRealData':     len(hist_a) >= 2 or len(hist_b) >= 2,
            'h2h':             h2h_info,
            'winRateA':        round(win_a * 100),
            'winRateB':        round(win_b * 100),
        }

    def auto_scan(self, days_back=7, sources=None):
        """
        Escaneo autónomo en segundo plano.
        Llamado por APScheduler — no bloquea el servidor.
        Incluye: ESPN, TheSportsDB, Sportradar Soccer/NBA/MLB (si hay key).
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

                self.set_progress(5, 'Escaneando ESPN (Soccer/NBA/MLB)...', 0)
                n = espn.scan(days_back=days_back, glai=self, on_progress=self.set_progress)
                total_learned += n
                used_sources.append('ESPN')
                self.set_progress(30, f'ESPN: {n} resultados · Escaneando TheSportsDB...', total_learned)

                n2 = tsdb.scan_top_leagues(days_back=days_back, glai=self)
                total_learned += n2
                used_sources.append('TheSportsDB')
                self.set_progress(50, f'TheSportsDB: {n2} resultados · Verificando Sportradar...', total_learned)

                # ── SPORTRADAR Soccer ─────────────────────────────────────────
                try:
                    from scrapers.sportradar import SportradarScraper
                    sprt = SportradarScraper(db=self.db)
                    if sprt._ok():
                        self.set_progress(52, 'Sportradar Soccer — escaneando...', total_learned)
                        ns = sprt.scan(days_back=days_back, glai=self)
                        total_learned += ns
                        used_sources.append('Sportradar Soccer')
                        print(f"[GLAI] Sportradar Soccer: {ns} resultados")
                except Exception as e_sr:
                    print(f"[GLAI] Sportradar Soccer skipped: {e_sr}")

                # ── SPORTRADAR NBA ────────────────────────────────────────────
                try:
                    try:
                        from scrapers.sportradar_nba import SportradarNBAScraper as _SNBA
                    except ImportError:
                        from sportradar_nba import SportradarNBAScraper as _SNBA
                    snba = _SNBA(db=self.db)
                    if snba._ok():
                        nba_st = snba.status()
                        if nba_st.get('not_subscribed'):
                            print("[GLAI] Sportradar NBA no suscrito — omitiendo scan")
                        else:
                            self.set_progress(62, 'Sportradar NBA — escaneando...', total_learned)
                            nn = snba.scan(days_back=days_back, glai=self)
                            total_learned += nn
                            used_sources.append('Sportradar NBA')
                            print(f"[GLAI] Sportradar NBA: {nn} resultados")
                except Exception as e_nba:
                    print(f"[GLAI] Sportradar NBA skipped: {e_nba}")

                # ── SPORTRADAR MLB ────────────────────────────────────────────
                try:
                    try:
                        from scrapers.sportradar_mlb import SportradarMLBScraper as _SMLB
                    except ImportError:
                        from sportradar_mlb import SportradarMLBScraper as _SMLB
                    smlb = _SMLB(db=self.db)
                    if smlb._ok():
                        mlb_st = smlb.status()
                        if mlb_st.get('not_subscribed'):
                            print("[GLAI] Sportradar MLB no suscrito — omitiendo scan")
                        else:
                            self.set_progress(72, 'Sportradar MLB — escaneando...', total_learned)
                            nm = smlb.scan(days_back=days_back, glai=self)
                            total_learned += nm
                            used_sources.append('Sportradar MLB')
                            print(f"[GLAI] Sportradar MLB: {nm} resultados")
                except Exception as e_mlb:
                    print(f"[GLAI] Sportradar MLB skipped: {e_mlb}")

                self.set_progress(85, f'Sportradar completo · Finalizando...', total_learned)

                # BeSoccer si hay key
                if self.db.get_key('besoccer'):
                    from scrapers.besoccer import BeSoccerScraper
                    bsc = BeSoccerScraper(self.db.get_key('besoccer'))
                    n3 = bsc.scan(days_back=days_back, glai=self)
                    total_learned += n3
                    used_sources.append('BeSoccer')

                # TheSportsDB fixtures futuros
                try:
                    self.set_progress(88, 'Buscando partidos futuros (TheSportsDB)...', total_learned)
                    tsdb_fixtures = tsdb.get_next_fixtures(days_ahead=7)
                    if tsdb_fixtures:
                        existing_keys = {(f['homeTeam'], f['awayTeam']) for f in self._fixtures}
                        for f in tsdb_fixtures:
                            k = (f['homeTeam'], f['awayTeam'])
                            if k not in existing_keys:
                                self._fixtures.append(f)
                                existing_keys.add(k)
                    print(f"[GLAI] {len(self._fixtures)} fixtures futuros guardados")
                except Exception as e:
                    print(f"[GLAI] TSDB fixtures skipped: {e}")

                # football-data.org (si hay token configurado)
                try:
                    fd_token = self.db.get_key('footballdata')
                    if fd_token:
                        from scrapers.footballdata import FootballDataScraper
                        fd = FootballDataScraper(fd_token)
                        self.set_progress(92, 'Escaneando football-data.org...', total_learned)
                        n_fd, fd_fixtures = fd.scan(days_back=days_back, days_ahead=7, glai=self)
                        total_learned += n_fd
                        used_sources.append('FootballData')
                        existing_keys = {(f['homeTeam'], f['awayTeam']) for f in self._fixtures}
                        for f in fd_fixtures:
                            k = (f['homeTeam'], f['awayTeam'])
                            if k not in existing_keys:
                                self._fixtures.append(f)
                                existing_keys.add(k)
                except Exception as e:
                    print(f"[GLAI] FootballData skipped: {e}")

                self.db.finish_scan_log(scan_id, total_learned, used_sources)
                self.set_progress(100, f'✅ Completado — {total_learned} resultados aprendidos', total_learned)
                print(f"[GLAI] Auto-scan completado: {total_learned} resultados · {used_sources}")
            except Exception as e:
                print(f"[GLAI] Error en auto-scan: {e}")
                self.set_progress(0, f'Error: {str(e)[:80]}', total_learned)
            finally:
                self._scan_running = False
