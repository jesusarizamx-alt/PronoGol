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

    def predict(self, sport, league_id, home_xg, away_xg, max_goals=7,
                team_a=None, team_b=None):
        """
        Genera matriz de marcadores y probabilidades 1X2.
        home_xg / away_xg: goles esperados base.

        Mejoras v2:
        - Ajuste por estadísticas de liga con blend proporcional al tamaño
        - Fuerza relativa del equipo (attack/defense ratings) cuando hay historial
        - Corrección Dixon-Coles para marcadores bajos (0-0, 1-0, 0-1, 1-1)
        - Blend final con tasas históricas de victorias de la liga
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
        data_quality = 0  # 0-100
        strength_info = {}
        if team_a and team_b and lg and lg['n'] >= 15:
            str_a = self.get_team_strength(team_a, sport, league_id, lg)
            str_b = self.get_team_strength(team_b, sport, league_id, lg)
            n_data = min(str_a['n'], str_b['n'])

            if n_data >= 5:
                # xG basado en: ataque propio × defensa rival × media de liga
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
            'pctA':         round(p_home * 100),
            'pctD':         round(p_draw * 100),
            'pctB':         round(p_away * 100),
            'xgA':          round(adj_home_xg, 2),
            'xgB':          round(adj_away_xg, 2),
            'btts':         round(p_btts * 100),
            'over25':       round(over_25 * 100),
            'over15':       round(over_15 * 100),
            'matrix':       matrix[:10],
            'lgStats':      lg,
            'totalLearned': self.total_learned(),
            'dataQuality':  data_quality,
            'strength':     strength_info,
            'model':        'poisson-dc-v2',
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

        is_home_flags = [team[:6].lower() in r['home_team'].lower() for r in rows]
        scored   = [r['home_goals'] if ih else r['away_goals']  for r, ih in zip(rows, is_home_flags)]
        conceded = [r['away_goals'] if ih else r['home_goals']  for r, ih in zip(rows, is_home_flags)]

        n = len(rows)
        avg_scored   = sum(scored) / n
        avg_conceded = sum(conceded) / n
        league_avg   = (lg['avgHG'] + lg['avgAG']) / 2 or 1.3

        attack  = avg_scored   / league_avg
        defense = avg_conceded / league_avg   # < 1 es bueno (conceden menos que media)

        # Forma reciente (últimos 5) con peso extra
        recent = rows[:5]
        if recent:
            r_scored = [r['home_goals'] if team[:6].lower() in r['home_team'].lower()
                        else r['away_goals'] for r in recent]
            r_wgt = sum(0.82 ** i * g for i, g in enumerate(r_scored))
            r_w   = sum(0.82 ** i      for i in range(len(recent)))
            form_attack = (r_wgt / r_w) / league_avg if r_w else attack
            attack = attack * 0.5 + form_attack * 0.5  # blend: historial + forma

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
    def ai_bet(self, hist_a, hist_b, xg_a, xg_b, team_a, team_b,
               h2h=None):
        """
        Genera recomendación de apuesta basada en historial real.
        v2: promedio ponderado con decay exponencial, splits home/away,
        integración de H2H, y puntuación de confianza de datos.

        hist_a / hist_b: listas con 'myG', 'oppG', 'result', 'isHome'
        h2h: lista de enfrentamientos directos (opcional)
        """
        # ── Promedios ponderados con decay ──────────────────────────
        wgt_scored_a   = self._weighted_avg(hist_a, 'myG')
        wgt_conceded_a = self._weighted_avg(hist_a, 'oppG')
        wgt_scored_b   = self._weighted_avg(hist_b, 'myG')
        wgt_conceded_b = self._weighted_avg(hist_b, 'oppG')

        # Simple avg para compatibilidad con respuesta
        def avg(arr, key):
            return sum(m[key] for m in arr) / len(arr) if arr else 0
        avg_scored_a   = avg(hist_a, 'myG')
        avg_conceded_a = avg(hist_a, 'oppG')
        avg_scored_b   = avg(hist_b, 'myG')
        avg_conceded_b = avg(hist_b, 'oppG')

        has_real = len(hist_a) >= 2 or len(hist_b) >= 2

        # ── xG cruzado (ataque A vs. defensa B y viceversa) ─────────
        # Usa promedios ponderados para más precisión
        if has_real:
            cross_xg_a = (wgt_scored_a + wgt_conceded_b) / 2
            cross_xg_b = (wgt_scored_b + wgt_conceded_a) / 2
        else:
            cross_xg_a = xg_a
            cross_xg_b = xg_b
        cross_total = cross_xg_a + cross_xg_b

        # ── H2H: blend directo si hay suficientes ───────────────────
        h2h_info = {}
        if h2h and len(h2h) >= 3:
            h2h_goals = [(r['home_goals'] + r['away_goals']) for r in h2h]
            h2h_avg   = sum(h2h_goals) / len(h2h_goals)
            h2h_wins_a = sum(
                1 for r in h2h
                if (team_a[:6].lower() in r['home_team'].lower() and r['home_goals'] > r['away_goals'])
                or (team_a[:6].lower() in r['away_team'].lower() and r['away_goals'] > r['home_goals'])
            )
            h2h_win_rate_a = h2h_wins_a / len(h2h)
            # Blend del total esperado con H2H
            cross_total = cross_total * 0.70 + h2h_avg * 0.30
            cross_xg_a  = cross_xg_a  * 0.80 + (h2h_avg * 0.50) * 0.20
            cross_xg_b  = cross_xg_b  * 0.80 + (h2h_avg * 0.50) * 0.20
            h2h_info = {
                'n': len(h2h),
                'avgGoals': round(h2h_avg, 1),
                'winRateA': round(h2h_win_rate_a * 100),
            }

        # ── Tasas ponderadas con decay ───────────────────────────────
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

        # ── Splits home/away ─────────────────────────────────────────
        home_a = [m for m in hist_a if m.get('isHome', True)]
        away_b = [m for m in hist_b if not m.get('isHome', True)]
        home_win_a = wgt_rate(home_a, lambda m: m['result'] == 'G') if home_a else win_a
        away_win_b = wgt_rate(away_b, lambda m: m['result'] == 'G') if away_b else win_b

        # ── Puntuación de confianza (0-100) ──────────────────────────
        n_total = len(hist_a) + len(hist_b)
        h2h_bonus = min(10, (h2h_info.get('n', 0)) * 2) if h2h_info else 0
        confidence_score = min(95, round(
            (min(n_total, 20) / 20) * 70     # hasta 70 pts por volumen de datos
            + h2h_bonus                       # hasta 10 pts por H2H
            + (10 if has_real else 0)         # 10 pts por datos reales
            + (5  if len(hist_a) >= 3 and len(hist_b) >= 3 else 0)
        ))

        # ── Señales de apuesta ───────────────────────────────────────
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

        # Ventaja local ponderada por splits home/away
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

        # Señal H2H directa
        if h2h_info and h2h_info.get('winRateA', 50) >= 70:
            bets.append({'bet': f'{team_a} Gana (Historial H2H)', 'conf': h2h_info['winRateA'],
                         'reason': f'En los últimos {h2h_info["n"]} enfrentamientos directos, {team_a} ganó {h2h_info["winRateA"]}%'})

        # Fallback si no hay señal fuerte
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
        }

    # ─── Predicción de esquinas y tarjetas ───────────────────────────
    def predict_corners_cards(self, xg_a, xg_b, league_id='soccer'):
        """
        Predice tiros de esquina y tarjetas usando modelo estadístico
        basado en xG, diferencial de fuerzas y tendencias por liga.
        """
        total_xg = xg_a + xg_b if (xg_a + xg_b) > 0 else 2.5

        # ── TIROS DE ESQUINA ──────────────────────────────────────────
        # Total correlacionado con intensidad ofensiva (xG total)
        base_corners = 8.0 + total_xg * 1.3

        # Liga: algunas tienen más esquinas por estilo de juego
        league_corner_factor = {
            'eng.1': 1.08, 'esp.1': 1.05, 'ger.1': 1.10,
            'ita.1': 0.95, 'fra.1': 1.00, 'ned.1': 1.08,
            'por.1': 1.02, 'bel.1': 1.05,
            'mex.1': 0.95, 'usa.1': 1.02, 'arg.1': 0.93,
            'bra.1': 0.95, 'col.1': 0.92,
            'uefa.champions': 1.06, 'uefa.europa': 1.04,
        }.get(league_id, 1.0)

        base_corners *= league_corner_factor

        # Reparto proporcional al xG (equipo más dominante gana más esquinas)
        share_a = (xg_a / total_xg) * 1.04  # leve ventaja local
        share_a = min(max(share_a, 0.30), 0.70)
        share_b = 1 - share_a

        corners_a = round(base_corners * share_a, 1)
        corners_b = round(base_corners * share_b, 1)
        total_corners = round(corners_a + corners_b, 1)

        # Over/Under esquinas
        over_8_5  = min(88, max(15, round(45 + (total_xg - 2.0) * 14)))
        over_10_5 = min(78, max(10, round(35 + (total_xg - 2.0) * 12)))
        over_12_5 = min(60, max(5,  round(20 + (total_xg - 2.0) * 9)))

        # ── TARJETAS ──────────────────────────────────────────────────
        # Rivalidad / competitividad: partidos ajustados = más tarjetas
        diff = abs(xg_a - xg_b)
        competitiveness = max(0.3, 1 - diff / (total_xg + 0.1))

        # Factor por liga
        league_card_factor = {
            'esp.1': 1.22, 'ita.1': 1.18, 'tur.1': 1.25,
            'arg.1': 1.28, 'bra.1': 1.22, 'col.1': 1.20,
            'eng.1': 0.96, 'ger.1': 0.88, 'fra.1': 1.02,
            'por.1': 1.12, 'ned.1': 0.94, 'sco.1': 1.05,
            'mex.1': 1.10, 'usa.1': 0.92,
            'uefa.champions': 1.05, 'uefa.europa': 1.10, 'uefa.conference': 1.08,
        }.get(league_id, 1.0)

        base_yellows = 3.4 * league_card_factor * (0.75 + competitiveness * 0.45)

        # El equipo que va perdiendo (menor xG) tiende a acumular más tarjetas
        losing_factor = 0.10  # 10% extra para el equipo más débil
        if xg_a < xg_b:
            yellows_a = round(base_yellows * (0.52 + losing_factor), 1)
            yellows_b = round(base_yellows * (0.48 - losing_factor / 2), 1)
        else:
            yellows_a = round(base_yellows * (0.50 - losing_factor / 2), 1)
            yellows_b = round(base_yellows * (0.50 + losing_factor), 1)

        total_yellows = round(yellows_a + yellows_b, 1)

        # Probabilidad de tarjeta roja
        red_prob = round(min(45, max(5, competitiveness * 20 * league_card_factor)))

        # Over/Under tarjetas amarillas
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

    # ─── NBA — Predicción ─────────────────────────────────────────
    def predict_nba(self, home_ppg, away_ppg):
        """
        Predicción NBA — distribución Normal basada en PPG.
        Incluye proyección por cuartos Q1-Q4.
        """
        HOME_ADV = 3.0
        SIGMA    = 12.0

        adj_home = home_ppg + HOME_ADV
        spread   = adj_home - away_ppg
        z        = spread / (SIGMA * math.sqrt(2))
        p_home   = 0.5 + 0.5 * math.erf(z)
        p_away   = 1.0 - p_home

        total   = adj_home + away_ppg
        ou_line = round(total / 5) * 5          # línea redondeada al 5 más cercano
        ou_over = min(75, max(25, round(50 + (total - ou_line) / (SIGMA * 0.4))))

        # Proyecciones por cuarto
        q_factors = [0.245, 0.250, 0.245, 0.260]
        quarters  = []
        for i, qf in enumerate(q_factors):
            qh    = round(adj_home * qf, 1)
            qa    = round(away_ppg * qf, 1)
            qt    = round(qh + qa, 1)
            q_ou  = round(qt / 2.5) * 2.5
            q_ov  = min(72, max(28, round(50 + (qt - q_ou) * 8)))
            quarters.append({'q': i + 1, 'home': qh, 'away': qa,
                              'total': qt, 'ouLine': q_ou, 'overPct': q_ov})

        return {
            'pctA':     round(p_home * 100),
            'pctB':     round(p_away * 100),
            'projHome': round(adj_home, 1),
            'projAway': round(away_ppg, 1),
            'total':    round(total, 1),
            'spread':   round(spread, 1),
            'ouLine':   ou_line,
            'ouOver':   ou_over,
            'quarters': quarters,
        }

    def ai_bet_nba(self, pred, team_a, team_b, hist_a=None, hist_b=None):
        """Genera recomendación de apuesta NBA."""
        pA     = pred.get('pctA', 50)
        pB     = pred.get('pctB', 50)
        spread = pred.get('spread', 0)
        total  = pred.get('total', 220)
        ou_ln  = pred.get('ouLine', 220)
        ou_ov  = pred.get('ouOver', 50)
        abs_sp = abs(spread)

        bets = []
        if pA >= 62:
            bets.append({'bet': f'{team_a} Gana (Moneyline)', 'conf': min(80, pA),
                         'reason': f'{team_a} tiene {pA}% de prob. · spread proyectado +{abs_sp:.1f} pts'})
        elif pB >= 62:
            bets.append({'bet': f'{team_b} Gana (Moneyline)', 'conf': min(80, pB),
                         'reason': f'{team_b} tiene {pB}% de prob. como visitante'})

        if abs_sp >= 4:
            fav  = team_a if spread > 0 else team_b
            conf = min(72, round(50 + abs_sp * 2.5))
            bets.append({'bet': f'{fav} -{abs_sp:.1f} (Spread)', 'conf': conf,
                         'reason': f'Diferencia proyectada {abs_sp:.1f} pts — ventaja significativa'})

        if ou_ov >= 58:
            bets.append({'bet': f'Más de {ou_ln} pts (O/U)', 'conf': ou_ov,
                         'reason': f'Total proyectado {total:.1f} pts · favorece el Over {ou_ln}'})
        elif ou_ov <= 42:
            bets.append({'bet': f'Menos de {ou_ln} pts (O/U)', 'conf': 100 - ou_ov,
                         'reason': f'Total proyectado {total:.1f} pts · partidos de bajo puntaje'})

        if not bets:
            bets.append({'bet': f'{team_a} Gana (Moneyline)', 'conf': max(pA, 51),
                         'reason': 'Partido parejo — ventaja local ligera favorece al local'})

        bets.sort(key=lambda x: x['conf'], reverse=True)
        narrative = (
            f"🏀 GLAI analiza el partido NBA entre {team_a} y {team_b}.\n\n"
            f"PPG ajustado: {team_a} {pred.get('projHome')} pts · {team_b} {pred.get('projAway')} pts. "
            f"Total proyectado: {total:.1f} pts.\n\n"
            f"{team_a} tiene {pA}% de probabilidades de ganar con spread de {spread:.1f} pts.\n\n"
            f"La apuesta de mayor confianza es: \"{bets[0]['bet']}\" ({bets[0]['conf']}%). "
            f"{bets[0]['reason']}.\n\n"
            f"ℹ️ Este análisis es orientativo. Apuesta con responsabilidad."
        )
        return {
            'best':      bets[0] if bets else None,
            'alt':       bets[1] if len(bets) > 1 else None,
            'narrative': narrative,
        }

    # ─── MLB — Predicción ─────────────────────────────────────────
    def predict_mlb(self, home_rpg, away_rpg):
        """
        Predicción MLB — Poisson sobre carreras por partido.
        Incluye análisis 1er inning y proyección entrada por entrada.
        """
        HOME_ADV = 0.20
        adj_home = home_rpg + HOME_ADV
        max_r    = 15

        # Probabilidad de victoria (Poisson)
        p_home = p_away = p_tie = 0.0
        for h in range(max_r + 1):
            for a in range(max_r + 1):
                p = self._poisson(adj_home, h) * self._poisson(away_rpg, a)
                if   h > a: p_home += p
                elif h < a: p_away += p
                else:       p_tie  += p
        # Extra innings: ligera ventaja local
        p_home += p_tie * 0.52
        p_away += p_tie * 0.48
        tot = p_home + p_away
        p_home /= tot
        p_away /= tot

        total = adj_home + away_rpg

        over_4_5 = sum(
            self._poisson(adj_home, h) * self._poisson(away_rpg, a)
            for h in range(max_r + 1) for a in range(max_r + 1) if h + a > 4.5
        )
        over_7_5 = sum(
            self._poisson(adj_home, h) * self._poisson(away_rpg, a)
            for h in range(max_r + 1) for a in range(max_r + 1) if h + a > 7.5
        )

        # ── 1er Inning ────────────────────────────────────────────
        F1 = 0.12
        first_h   = round(adj_home * F1, 2)
        first_a   = round(away_rpg * F1, 2)
        first_tot = round(first_h + first_a, 2)
        p_any     = 1.0 - self._poisson(first_h, 0) * self._poisson(first_a, 0)
        p_f_ov05  = 1.0 - self._poisson(first_tot, 0)

        # ── Proyección entrada por entrada ────────────────────────
        inning_factors = [0.120, 0.105, 0.112, 0.108, 0.115,
                          0.112, 0.105, 0.110, 0.113]
        innings = []
        for i, f in enumerate(inning_factors):
            ih  = round(adj_home * f, 2)
            ia  = round(away_rpg * f, 2)
            it  = round(ih + ia, 2)
            psc = round((1.0 - self._poisson(it, 0)) * 100)
            innings.append({'inning': i + 1, 'home': ih, 'away': ia,
                             'total': it, 'scorePct': psc})

        # ── Matriz de marcadores ──────────────────────────────────
        matrix = []
        for h in range(12):
            for a in range(12):
                p = self._poisson(adj_home, h) * self._poisson(away_rpg, a)
                matrix.append({'h': h, 'a': a, 'p': round(p, 4)})
        matrix.sort(key=lambda x: x['p'], reverse=True)

        return {
            'pctA':     round(p_home * 100),
            'pctB':     round(p_away * 100),
            'projHome': round(adj_home, 2),
            'projAway': round(away_rpg, 2),
            'total':    round(total, 2),
            'over4_5':  round(over_4_5 * 100),
            'over7_5':  round(over_7_5 * 100),
            'first': {
                'projHome':   first_h,
                'projAway':   first_a,
                'projTotal':  first_tot,
                'scoreAny':   round(p_any * 100),
                'over05':     round(p_f_ov05 * 100),
            },
            'innings': innings,
            'matrix':  matrix[:10],
        }

    def ai_bet_mlb(self, pred, team_a, team_b, hist_a=None, hist_b=None):
        """Genera recomendación de apuesta MLB."""
        pA      = pred.get('pctA', 50)
        pB      = pred.get('pctB', 50)
        total   = pred.get('total', 8.5)
        ov45    = pred.get('over4_5', 50)
        ov75    = pred.get('over7_5', 50)
        first   = pred.get('first', {})

        bets = []
        if pA >= 58:
            bets.append({'bet': f'{team_a} Gana (Moneyline)', 'conf': min(75, pA),
                         'reason': f'{team_a} tiene {pA}% de prob. con ventaja local (+0.20 carreras)'})
        elif pB >= 58:
            bets.append({'bet': f'{team_b} Gana (Moneyline)', 'conf': min(75, pB),
                         'reason': f'{team_b} tiene {pB}% de prob. visitando'})

        if ov45 >= 60:
            bets.append({'bet': 'Más de 4.5 carreras', 'conf': ov45,
                         'reason': f'{ov45}% de probabilidad · total proyectado {total:.1f} carreras'})
        elif ov45 <= 42:
            bets.append({'bet': 'Menos de 4.5 carreras', 'conf': 100 - ov45,
                         'reason': f'Solo {ov45}% para más de 4.5 — pitcheo dominante esperado'})

        if ov75 >= 58:
            bets.append({'bet': 'Más de 7.5 carreras', 'conf': ov75,
                         'reason': f'{ov75}% probabilidad · ofensivas productivas proyectadas'})

        if first.get('scoreAny', 0) >= 65:
            bets.append({'bet': '1er Inning — Alguien Anota', 'conf': first.get('scoreAny', 0),
                         'reason': f'{first.get("scoreAny")}% de que algún equipo anote en 1er inning'})

        if not bets:
            bets.append({'bet': 'Más de 4.5 carreras', 'conf': ov45,
                         'reason': f'Total proyectado {total:.1f} carreras'})

        bets.sort(key=lambda x: x['conf'], reverse=True)
        narrative = (
            f"⚾ GLAI analiza el partido MLB entre {team_a} y {team_b}.\n\n"
            f"Carreras proyectadas: {team_a} {pred.get('projHome'):.2f} · "
            f"{team_b} {pred.get('projAway'):.2f} · Total: {total:.2f}.\n\n"
            f"{team_a} tiene {pA}% de probabilidades de ganar. "
            f"Over 4.5: {ov45}% · Over 7.5: {ov75}%.\n\n"
            f"1er Inning — total proyectado {first.get('projTotal','?')} carreras · "
            f"{first.get('scoreAny',0)}% probabilidad de que alguien anote.\n\n"
            f"ℹ️ Este análisis es orientativo. Apuesta con responsabilidad."
        )
        return {
            'best':      bets[0] if bets else None,
            'alt':       bets[1] if len(bets) > 1 else None,
            'narrative': narrative,
        }

    # ─── Predicciones EN VIVO ─────────────────────────────────────

    def predict_live_soccer(self, home_score, away_score, minute, xg_a, xg_b):
        """
        Ajusta probabilidades en vivo según el marcador y el minuto.
        Mientras más avanza el partido, el marcador actual pesa más.
        """
        minute   = max(1, min(90, minute))
        pct_done = minute / 90.0          # fracción del partido jugada
        pct_left = 1.0 - pct_done

        # xG restante proporcional al tiempo que queda
        rem_a = xg_a * pct_left
        rem_b = xg_b * pct_left

        # Probabilidades Poisson para goles adicionales (máx 5 extra)
        def p_goals(lam, k):
            import math
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

        # Tendencia: equipo que va ganando tiene momentum extra (5%)
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
        pct_done  = (period - 1) / 4.0 + 0.5 / 4.0   # asumimos mitad del período
        pct_left  = max(0.05, 1.0 - pct_done)

        proj_a = home_pts + home_ppg * pct_left
        proj_b = away_pts + away_ppg * pct_left
        diff   = proj_a - proj_b

        # Probabilidad de victoria local basada en proyección
        import math
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

        import math
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
                    from scrapers.sportradar_nba import SportradarNBAScraper
                    snba = SportradarNBAScraper(db=self.db)
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
                    from scrapers.sportradar_mlb import SportradarMLBScraper
                    smlb = SportradarMLBScraper(db=self.db)
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

                # TheSportsDB fixtures futuros (gratis, sin key extra)
                try:
                    self.set_progress(88, 'Buscando partidos futuros (TheSportsDB)...', total_learned)
                    tsdb_fixtures = tsdb.get_next_fixtures(days_ahead=7)
                    if tsdb_fixtures:
                        # Merge con fixtures existentes, sin duplicados
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
                        # Agregar fixtures sin duplicar
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
