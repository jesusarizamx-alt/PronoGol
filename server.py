"""
server.py — Soccer Predictor Pro — Backend Python/Flask
Corre con: python server.py
Abre: http://localhost:5000
"""
from flask import Flask, request, jsonify, session, send_from_directory
from flask_cors import CORS
from apscheduler.schedulers.background import BackgroundScheduler
import os, threading, secrets, string, random
from datetime import datetime, timedelta
from pathlib import Path

from database import Database
from users import UserManager
from glai import GLAIEngine
from scrapers.espn import ESPNScraper
from scrapers.thesportsdb import TheSportsDBScraper
from scrapers.sportradar import SportradarScraper          # ← Soccer

# NBA y MLB scrapers — opcionales (no crashean si faltan los archivos)
try:
    from scrapers.sportradar_nba import SportradarNBAScraper
    _NBA_OK = True
except ImportError:
    _NBA_OK = False
    print("[Server] ⚠️ sportradar_nba.py no encontrado — NBA deshabilitado")

try:
    from scrapers.sportradar_mlb import SportradarMLBScraper
    _MLB_OK = True
except ImportError:
    _MLB_OK = False
    print("[Server] ⚠️ sportradar_mlb.py no encontrado — MLB deshabilitado")

# ── App setup ────────────────────────────────────────────────────
BASE = Path(__file__).parent
app = Flask(__name__, static_folder=str(BASE / 'frontend'), static_url_path='')
app.secret_key = os.environ.get('SECRET_KEY', 'pronogol-default-key-cambiame-en-render')
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = os.environ.get('RENDER', False)  # HTTPS en Render
# Permitir cualquier origen (para que funcione en Render.com y local)
CORS(app, supports_credentials=True, origins='*')

# ── Core services ─────────────────────────────────────────────────
db    = Database()
users = UserManager(db)
glai  = GLAIEngine(db)
espn  = ESPNScraper(db.get_all_keys())
tsdb  = TheSportsDBScraper()
sprt  = SportradarScraper(db=db)                          # Soccer
snba  = SportradarNBAScraper(db=db) if _NBA_OK else None  # NBA (opcional)
smlb  = SportradarMLBScraper(db=db) if _MLB_OK else None  # MLB (opcional)

# ── Helper: calcular xG automático (Sportradar DB + historial ponderado) ──
def calc_auto_xg(team, sport='soccer', default=1.4):
    """
    Calcula el xG estimado de un equipo usando:
    1. Resultados de Sportradar (fuente más confiable) — peso mayor
    2. Resultados de ESPN y otras fuentes — peso menor
    Juegos más recientes pesan más (decaimiento exponencial).
    """
    # Normaliza: el frontend manda 'nba'/'mlb' pero la DB guarda 'basketball'/'baseball'
    _SPORT_MAP = {'nba': 'basketball', 'mlb': 'baseball', 'soccer': 'soccer',
                  'basketball': 'basketball', 'baseball': 'baseball'}
    sport_db = _SPORT_MAP.get(sport, sport)

    try:
        results = db.get_team_results(team, limit=15)
        if not results:
            return default

        weighted_sum = 0.0
        weight_total = 0.0
        key = team.lower()[:7]

        for i, r in enumerate(results):
            # Acepta tanto el nombre del frontend ('nba') como el de la DB ('basketball')
            r_sport = r.get('sport', 'soccer')
            if r_sport != sport_db and r_sport != sport:
                continue
            hn = r.get('home_team', '')
            an = r.get('away_team', '')

            if key in hn.lower():
                goles = r['home_goals']
            elif key in an.lower():
                goles = r['away_goals']
            else:
                continue

            # Juegos más recientes pesan más (posición 0 = más reciente)
            recency_weight = 1.0 / (1.0 + i * 0.15)
            # Sportradar pesa el doble que otras fuentes
            source_weight = 2.0 if r.get('source', '') == 'sportradar' else 1.0

            w = recency_weight * source_weight
            weighted_sum += goles * w
            weight_total += w

        if weight_total == 0:
            return default

        xg = weighted_sum / weight_total
        return round(max(0.1, xg), 2)
    except Exception:
        return default

# ── Helpers de códigos e IP ───────────────────────────────────────
def get_client_ip():
    """Obtiene la IP real del cliente (considera proxies de Render)."""
    return (request.headers.get('X-Forwarded-For', '') or '').split(',')[0].strip() \
           or request.remote_addr or 'unknown'

def generate_invite_code():
    """Genera un código único tipo PRONO-XXXX-XXXX."""
    chars = string.ascii_uppercase + string.digits
    part1 = ''.join(random.choices(chars, k=4))
    part2 = ''.join(random.choices(chars, k=4))
    return f"PRONO-{part1}-{part2}"

# ── Scheduler — GLAI aprende sola cada 2 horas ───────────────────
scheduler = BackgroundScheduler(daemon=True)
scheduler.add_job(
    func=lambda: glai.auto_scan(days_back=7),
    trigger='interval',
    hours=1,
    id='glai_auto_learn',
    replace_existing=True
)
scheduler.start()
print("[Server] ✅ APScheduler activo — GLAI aprende sola cada 2 horas")
if sprt._ok():
    print("[Server] ✅ Sportradar Soccer conectado")
else:
    print("[Server] ⚠️  Sportradar Soccer sin API key")
if snba and snba._ok():
    print("[Server] ✅ Sportradar NBA conectado")
if smlb and smlb._ok():
    print("[Server] ✅ Sportradar MLB conectado")

# ── Auth helpers ─────────────────────────────────────────────────
def current_user():
    return session.get('user')

def require_login(fn):
    from functools import wraps
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not current_user():
            return jsonify({'error': 'No autenticado'}), 401
        return fn(*args, **kwargs)
    return wrapper

def require_admin(fn):
    from functools import wraps
    @wraps(fn)
    def wrapper(*args, **kwargs):
        u = current_user()
        if not u or u.get('role') != 'admin':
            return jsonify({'error': 'Acceso solo para administradores'}), 403
        return fn(*args, **kwargs)
    return wrapper

def require_premium(fn):
    from functools import wraps
    @wraps(fn)
    def wrapper(*args, **kwargs):
        u = current_user()
        if not u:
            return jsonify({'error': 'No autenticado'}), 401
        if u.get('role') == 'free':
            return jsonify({'error': 'premium_required', 'msg': 'Esta función es solo Premium'}), 403
        return fn(*args, **kwargs)
    return wrapper

# ════════════════════════════════════════════════════════════════
# RUTAS — AUTENTICACIÓN
# ════════════════════════════════════════════════════════════════
@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.get_json() or {}
    username = (data.get('username') or '').strip()
    password = (data.get('password') or '').strip()
    if not username or not password:
        return jsonify({'error': 'Completa usuario y contraseña'}), 400
    user = users.login(username, password)
    if not user:
        db.add_log(username, 'login_failed', ip=get_client_ip(), details='Contraseña incorrecta')
        return jsonify({'error': 'Usuario o contraseña incorrectos'}), 401
    session['user'] = user
    session.permanent = True
    db.add_log(username, 'login', ip=get_client_ip())
    return jsonify({'ok': True, 'user': user})

@app.route('/api/auth/register', methods=['POST'])
def register():
    """Registro solo con código de invitación válido."""
    data     = request.get_json() or {}
    code     = (data.get('code') or '').strip().upper()
    password = (data.get('password') or '').strip()
    ip       = get_client_ip()

    if not code:
        return jsonify({'error': 'Se requiere un código de invitación'}), 400
    if len(password) < 6:
        return jsonify({'error': 'La contraseña debe tener al menos 6 caracteres'}), 400

    # Validar código
    invite = db.get_invite_code(code)
    if not invite:
        db.add_log('unknown', 'register_failed', ip=ip, details=f'Código inválido: {code}')
        return jsonify({'error': 'Código de invitación inválido'}), 400
    if invite['used_at']:
        db.add_log('unknown', 'register_failed', ip=ip, details=f'Código ya usado: {code}')
        return jsonify({'error': 'Este código ya fue utilizado'}), 400
    if invite['expires_at'] and invite['expires_at'] < datetime.utcnow().isoformat():
        db.add_log('unknown', 'register_failed', ip=ip, details=f'Código expirado: {code}')
        return jsonify({'error': 'Este código ha expirado'}), 400

    username = invite['username']
    try:
        user = users.create_user(username, password, invite['role'], invite['tokens'])
        db.use_invite_code(code, ip)
        db.add_log(username, 'register', ip=ip, details=f'Código: {code} | Rol: {invite["role"]}')
        session['user'] = user
        session.permanent = True
        return jsonify({'ok': True, 'user': user})
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/auth/validate-code', methods=['POST'])
def validate_code():
    """Verifica si un código es válido y devuelve el username asignado."""
    data = request.get_json() or {}
    code = (data.get('code') or '').strip().upper()
    if not code:
        return jsonify({'valid': False}), 400
    invite = db.get_invite_code(code)
    if not invite or invite['used_at']:
        return jsonify({'valid': False, 'error': 'Código inválido o ya usado'}), 400
    if invite['expires_at'] and invite['expires_at'] < datetime.utcnow().isoformat():
        return jsonify({'valid': False, 'error': 'Código expirado'}), 400
    return jsonify({'valid': True, 'username': invite['username'], 'role': invite['role']})

@app.route('/api/auth/logout', methods=['POST'])
def logout():
    u = current_user()
    if u:
        db.add_log(u['username'], 'logout', ip=get_client_ip())
    session.clear()
    return jsonify({'ok': True})

@app.route('/api/auth/me')
def me():
    u = current_user()
    if not u:
        return jsonify({'user': None}), 200
    # Refresh token count from DB
    fresh = users.get_user(u['username'])
    if fresh:
        session['user'] = fresh
    return jsonify({'user': fresh or u})

@app.route('/api/auth/change-password', methods=['POST'])
@require_login
def change_password():
    data = request.get_json() or {}
    current_pw  = data.get('currentPassword', '')
    new_pw      = data.get('newPassword', '')
    confirm_pw  = data.get('confirmPassword', '')

    if not current_pw or not new_pw or not confirm_pw:
        return jsonify({'ok': False, 'error': 'Completa todos los campos'}), 400
    if new_pw != confirm_pw:
        return jsonify({'ok': False, 'error': 'Las contraseñas nuevas no coinciden'}), 400
    if len(new_pw) < 6:
        return jsonify({'ok': False, 'error': 'La nueva contraseña debe tener al menos 6 caracteres'}), 400

    u = current_user()
    ok = users.change_password(u['username'], current_pw, new_pw)
    if not ok:
        return jsonify({'ok': False, 'error': 'La contraseña actual es incorrecta'}), 403

    return jsonify({'ok': True, 'msg': '✅ Contraseña actualizada correctamente'})

# ════════════════════════════════════════════════════════════════
# RUTAS — PARTIDOS (ESPN + Sportradar)
# ════════════════════════════════════════════════════════════════
@app.route('/api/matches')
@require_login
def matches():
    try:
        # ── ESPN: Soccer + NBA + MLB ──────────────────────────────
        data = espn.get_all_today()

        # ── Sportradar: partidos en vivo (si hay key configurada) ─
        if sprt._ok():
            sr_live = sprt.get_live_matches()
            if sr_live:
                existing = {(m['homeTeam'], m['awayTeam']) for m in data}
                for m in sr_live:
                    key = (m['homeTeam'], m['awayTeam'])
                    if key not in existing:
                        data.append(m)
                        existing.add(key)
                    else:
                        # Actualizar score con datos de Sportradar (más precisos)
                        for i, em in enumerate(data):
                            if (em['homeTeam'], em['awayTeam']) == key:
                                data[i]['homeScore'] = m['homeScore']
                                data[i]['awayScore'] = m['awayScore']
                                data[i]['clock']     = m.get('clock', '')
                                data[i]['period']    = m.get('period', 0)
                                break

        # ── Fixtures futuros (soccer de TheSportsDB/FootballData) ─
        fixtures = glai.get_fixtures()
        if fixtures:
            existing = {(m['homeTeam'], m['awayTeam']) for m in data}
            for f in fixtures:
                key = (f.get('homeTeam',''), f.get('awayTeam',''))
                if key not in existing:
                    data.append({
                        'homeTeam':  f.get('homeTeam', ''),
                        'awayTeam':  f.get('awayTeam', ''),
                        'homeScore': '',
                        'awayScore': '',
                        'homeLogo':  f.get('homeLogo', ''),
                        'awayLogo':  f.get('awayLogo', ''),
                        'status':    'pre',
                        'date':      f.get('date', ''),
                        'league':    f.get('league', 'soccer'),
                        'sport':     'soccer',
                    })
                    existing.add(key)

        return jsonify({'matches': data, 'total': len(data)})
    except Exception as e:
        return jsonify({'error': str(e), 'matches': []}), 500

@app.route('/api/matches/<league>/<event_id>')
@require_login
def match_detail(league, event_id):
    try:
        data = espn.get_match_detail(league, event_id)
        return jsonify(data or {})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ════════════════════════════════════════════════════════════════
# RUTAS — SPORTRADAR (nuevo)
# ════════════════════════════════════════════════════════════════
@app.route('/api/sportradar/debug')
@require_admin
def sportradar_debug():
    """Diagnóstico completo — muestra exactamente por qué la API funciona o no."""
    import os, requests as req

    env_key   = os.environ.get('SPORTRADAR_API_KEY', '')
    all_keys  = {}
    try:
        all_keys = db.get_all_keys()
    except Exception as e:
        all_keys = {'error': str(e)}

    db_key    = ''
    db_key_name = ''
    for name, val in (all_keys.items() if isinstance(all_keys, dict) else {}.items()):
        if 'sportradar' in str(name).lower() and val:
            db_key = val
            db_key_name = name
            break

    active_key = env_key or db_key
    key_source = 'env var SPORTRADAR_API_KEY' if env_key else (f'DB key "{db_key_name}"' if db_key else 'ninguna')

    # Prueba múltiples URLs para encontrar la correcta
    results = []
    if active_key:
        from datetime import datetime
        today = datetime.utcnow().strftime('%Y-%m-%d')
        hdrs  = {'Accept': 'application/json', 'x-api-key': active_key}
        urls_to_try = [
            f"https://api.sportradar.com/soccer/trial/v4/en/schedules/{today}/schedule.json",
            f"https://api.sportradar.com/soccer/production/v4/en/schedules/{today}/schedule.json",
            f"https://api.sportradar.com/soccer/trial/v4/en/competitions.json",
            f"https://api.sportradar.com/soccer/trial/v4/en/schedules/live/schedule.json",
            f"https://api.sportradar.com/soccer-extended/trial/v4/en/competitions.json",
            f"https://api.sportradar.com/soccer/trial/v4/en/seasons.json",
        ]
        import time as _time
        for url in urls_to_try:
            try:
                r = req.get(url, headers=hdrs, timeout=8)
                results.append({'url': url, 'status': r.status_code, 'body': r.text[:120]})
                if r.status_code == 200:
                    break  # Encontramos la URL correcta
            except Exception as ex:
                results.append({'url': url, 'status': 'error', 'body': str(ex)})
            _time.sleep(1.1)  # Respetar rate limit

    return jsonify({
        'env_key_set':    bool(env_key),
        'env_key_prefix': env_key[:6] + '...' if env_key else '',
        'db_key_set':     bool(db_key),
        'db_key_name':    db_key_name,
        'key_source':     key_source,
        'active_key_set': bool(active_key),
        'scraper_ok':     sprt._ok(),
        'all_db_keys':    list(all_keys.keys()) if isinstance(all_keys, dict) else [],
        'url_tests':      results,
    })

@app.route('/api/sportradar/status')
@require_admin
def sportradar_status():
    """Verifica si Sportradar está conectado y funcionando."""
    return jsonify(sprt.status())

@app.route('/api/sportradar/live')
@require_login
def sportradar_live():
    """Partidos en vivo de Sportradar."""
    if not sprt._ok():
        return jsonify({'error': 'Sportradar no configurado', 'matches': []}), 503
    try:
        matches = sprt.get_live_matches()
        return jsonify({'matches': matches, 'total': len(matches), 'source': 'sportradar'})
    except Exception as e:
        return jsonify({'error': str(e), 'matches': []}), 500

@app.route('/api/sportradar/today')
@require_login
def sportradar_today():
    """Partidos de hoy según Sportradar."""
    if not sprt._ok():
        return jsonify({'error': 'Sportradar no configurado', 'matches': []}), 503
    try:
        matches = sprt.get_today_schedule()
        return jsonify({'matches': matches, 'total': len(matches), 'source': 'sportradar'})
    except Exception as e:
        return jsonify({'error': str(e), 'matches': []}), 500

@app.route('/api/sportradar/scan', methods=['POST'])
@require_admin
def sportradar_scan():
    """Lanza scan histórico de Sportradar para entrenar GLAI."""
    if not sprt._ok():
        return jsonify({'ok': False, 'msg': 'Sportradar no configurado — agrega la API key'}), 503
    data = request.get_json() or {}
    days = int(data.get('days', 7))
    def _run():
        sprt.scan(days_back=days, glai=glai)
    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return jsonify({'ok': True, 'msg': f'Scan Sportradar iniciado ({days} días)'})

# ════════════════════════════════════════════════════════════════
# RUTA — PARLAY / ANÁLISIS MÚLTIPLE
# ════════════════════════════════════════════════════════════════
@app.route('/api/glai/parlay', methods=['POST'])
@require_login
def glai_parlay():
    """
    Analiza hasta 10 partidos juntos para armar un parlay personalizado.
    Body: { matches: [{teamA, teamB, sport, league}, ...] }
    Respuesta: { ok, matches: [...análisis individual...], combinedProb, count }
    """
    u = current_user()
    # Admin = ilimitado. Todos los demás consumen 1 token por parlay
    if u.get('role') != 'admin':
        ok = users.use_token(u['username'])
        if not ok:
            return jsonify({'error': 'no_tokens',
                           'msg': 'Sin tokens disponibles. Contacta al admin.'}), 403
        session['user'] = users.get_user(u['username'])

    data    = request.get_json() or {}
    matches = data.get('matches', [])[:10]  # máx 10 partidos

    if not matches:
        return jsonify({'ok': False, 'msg': 'No se enviaron partidos'}), 400

    results      = []
    combined_prob = 1.0

    for m in matches:
        team_a = m.get('teamA', '').strip()
        team_b = m.get('teamB', '').strip()
        sport  = m.get('sport', 'soccer').lower()
        league = m.get('league', '')
        if not team_a or not team_b:
            continue

        _SPORT_MAP = {'nba': 'basketball', 'mlb': 'baseball',
                      'soccer': 'soccer', 'basketball': 'basketball', 'baseball': 'baseball'}
        _def_a = 115.0 if sport == 'nba' else (4.5 if sport == 'mlb' else 1.4)
        _def_b = 112.0 if sport == 'nba' else (4.2 if sport == 'mlb' else 1.1)
        val_a  = calc_auto_xg(team_a, sport, _def_a)
        val_b  = calc_auto_xg(team_b, sport, _def_b)

        entry = {
            'teamA': team_a, 'teamB': team_b,
            'sport': sport,  'league': league,
            'valA': val_a,   'valB': val_b,
        }

        try:
            if sport == 'nba':
                hist_a = glai.team_history(team_a, limit=5)
                hist_b = glai.team_history(team_b, limit=5)
                def _avg(h, d):
                    s = [x['myG'] for x in h if x.get('myG') is not None]
                    return round(sum(s)/len(s), 1) if s else d
                val_a = _avg(hist_a, val_a)
                val_b = _avg(hist_b, val_b)
                pred  = glai.predict_nba(val_a, val_b)
                bet   = glai.ai_bet_nba(pred, team_a, team_b, hist_a, hist_b)
                entry.update({'prediction': pred, 'bet': bet, 'valA': val_a, 'valB': val_b})
                best_p = max(pred.get('pctA', 50), pred.get('pctB', 50)) / 100

            elif sport == 'mlb':
                hist_a = glai.team_history(team_a, limit=5)
                hist_b = glai.team_history(team_b, limit=5)
                def _avg(h, d):
                    s = [x['myG'] for x in h if x.get('myG') is not None]
                    return round(sum(s)/len(s), 2) if s else d
                val_a = _avg(hist_a, val_a)
                val_b = _avg(hist_b, val_b)
                pred  = glai.predict_mlb(val_a, val_b)
                bet   = glai.ai_bet_mlb(pred, team_a, team_b, hist_a, hist_b)
                entry.update({'prediction': pred, 'bet': bet, 'valA': val_a, 'valB': val_b})
                best_p = max(pred.get('pHome', 50), pred.get('pAway', 50)) / 100

            else:  # soccer
                hist_a = glai.team_history(team_a, limit=5)
                hist_b = glai.team_history(team_b, limit=5)
                pred    = glai.predict('soccer', league, val_a, val_b)
                bet     = glai.ai_bet(hist_a, hist_b, val_a, val_b, team_a, team_b)
                corners = glai.predict_corners_cards(val_a, val_b, league)
                entry.update({'prediction': pred, 'bet': bet,
                               'corners': corners, 'histA': hist_a, 'histB': hist_b})
                pA = pred.get('pctA', 33)
                pB = pred.get('pctB', 33)
                best_p = max(pA, pB) / 100

            combined_prob *= best_p

        except Exception as ex:
            entry['error'] = str(ex)

        results.append(entry)

    try:
        db.add_log(u['username'], 'parlay', ip=get_client_ip(),
                   details=f'{len(results)} partidos analizados')
    except Exception:
        pass

    return jsonify({
        'ok':           True,
        'matches':      results,
        'combinedProb': round(combined_prob * 100, 1),
        'count':        len(results),
        'tokensLeft':   session['user'].get('tokens', 0) if u.get('role') != 'admin' else -1,
    })


# ════════════════════════════════════════════════════════════════
# RUTA — STATS POR EQUIPO (PPG / RPG / xG)
# ════════════════════════════════════════════════════════════════
@app.route('/api/team/stats')
def team_stats():
    """
    Devuelve el promedio de goles/puntos/carreras de un equipo.
    GET /api/team/stats?team=Lakers&sport=nba
    Respuesta: { ok, team, sport, avg, games, source }
    """
    team  = request.args.get('team', '').strip()
    sport = request.args.get('sport', 'soccer').lower()
    if not team:
        return jsonify({'ok': False, 'msg': 'Falta parámetro team'})

    _SPORT_MAP = {'nba': 'basketball', 'mlb': 'baseball',
                  'soccer': 'soccer', 'basketball': 'basketball', 'baseball': 'baseball'}
    sport_db = _SPORT_MAP.get(sport, sport)

    try:
        results = db.get_team_results(team, limit=20)
        key     = team.lower()[:7]
        scores  = []
        for r in results:
            r_sport = r.get('sport', 'soccer')
            if r_sport != sport_db and r_sport != sport:
                continue
            hn = r.get('home_team', '')
            an = r.get('away_team', '')
            if key in hn.lower():
                scores.append(r['home_goals'])
            elif key in an.lower():
                scores.append(r['away_goals'])

        if not scores:
            # Sin datos propios → devuelve el default del deporte
            defaults = {'nba': 115.0, 'basketball': 115.0,
                        'mlb': 4.5,   'baseball': 4.5,
                        'soccer': 1.4}
            return jsonify({'ok': True, 'team': team, 'sport': sport,
                            'avg': defaults.get(sport, 1.4), 'games': 0,
                            'source': 'default'})

        avg = round(sum(scores) / len(scores), 2)
        return jsonify({'ok': True, 'team': team, 'sport': sport,
                        'avg': avg, 'games': len(scores), 'source': 'db'})
    except Exception as e:
        return jsonify({'ok': False, 'msg': str(e)}), 500


# ════════════════════════════════════════════════════════════════
# RUTAS — SPORTRADAR NBA
# ════════════════════════════════════════════════════════════════
@app.route('/api/nba/sportradar/status')
@require_admin
def nba_sr_status():
    if not snba: return jsonify({'ok': False, 'msg': 'sportradar_nba.py no instalado'})
    return jsonify(snba.status())

@app.route('/api/nba/sportradar/live')
@require_login
def nba_sr_live():
    if not snba or not snba._ok():
        return jsonify({'error': 'NBA Sportradar no configurado', 'matches': []}), 503
    try:
        matches = snba.get_live_matches()
        return jsonify({'matches': matches, 'total': len(matches), 'source': 'sportradar_nba'})
    except Exception as e:
        return jsonify({'error': str(e), 'matches': []}), 500

@app.route('/api/nba/sportradar/today')
@require_login
def nba_sr_today():
    if not snba or not snba._ok():
        return jsonify({'error': 'NBA Sportradar no configurado', 'matches': []}), 503
    try:
        matches = snba.get_today_schedule()
        return jsonify({'matches': matches, 'total': len(matches), 'source': 'sportradar_nba'})
    except Exception as e:
        return jsonify({'error': str(e), 'matches': []}), 500

@app.route('/api/nba/sportradar/scan', methods=['POST'])
@require_admin
def nba_sr_scan():
    if not snba or not snba._ok():
        return jsonify({'ok': False, 'msg': 'NBA Sportradar no configurado'}), 503
    data = request.get_json() or {}
    days = int(data.get('days', 7))
    threading.Thread(target=lambda: snba.scan(days_back=days, glai=glai), daemon=True).start()
    return jsonify({'ok': True, 'msg': f'Scan NBA iniciado ({days} días)'})

# ════════════════════════════════════════════════════════════════
# RUTAS — SPORTRADAR MLB
# ════════════════════════════════════════════════════════════════
@app.route('/api/mlb/sportradar/status')
@require_admin
def mlb_sr_status():
    if not smlb: return jsonify({'ok': False, 'msg': 'sportradar_mlb.py no instalado'})
    return jsonify(smlb.status())

@app.route('/api/mlb/sportradar/live')
@require_login
def mlb_sr_live():
    if not smlb or not smlb._ok():
        return jsonify({'error': 'MLB Sportradar no configurado', 'matches': []}), 503
    try:
        matches = smlb.get_live_matches()
        return jsonify({'matches': matches, 'total': len(matches), 'source': 'sportradar_mlb'})
    except Exception as e:
        return jsonify({'error': str(e), 'matches': []}), 500

@app.route('/api/mlb/sportradar/today')
@require_login
def mlb_sr_today():
    if not smlb or not smlb._ok():
        return jsonify({'error': 'MLB Sportradar no configurado', 'matches': []}), 503
    try:
        matches = smlb.get_today_schedule()
        return jsonify({'matches': matches, 'total': len(matches), 'source': 'sportradar_mlb'})
    except Exception as e:
        return jsonify({'error': str(e), 'matches': []}), 500

@app.route('/api/mlb/sportradar/scan', methods=['POST'])
@require_admin
def mlb_sr_scan():
    if not smlb or not smlb._ok():
        return jsonify({'ok': False, 'msg': 'MLB Sportradar no configurado'}), 503
    data = request.get_json() or {}
    days = int(data.get('days', 7))
    threading.Thread(target=lambda: smlb.scan(days_back=days, glai=glai), daemon=True).start()
    return jsonify({'ok': True, 'msg': f'Scan MLB iniciado ({days} días)'})

# ════════════════════════════════════════════════════════════════
# RUTAS — GLAI IA
# ════════════════════════════════════════════════════════════════
@app.route('/api/glai/status')
@require_login
def glai_status():
    return jsonify(glai.scan_status())

@app.route('/api/glai/stats')
@require_login
def glai_stats():
    soccer_stats = glai.get_all_stats('soccer')
    nba_stats    = glai.get_all_stats('nba')
    mlb_stats    = glai.get_all_stats('mlb')
    return jsonify({
        'total':      glai.total_learned(),
        'bySport': {
            'soccer': sum(v['n'] for v in soccer_stats.values()),
            'nba':    sum(v['n'] for v in nba_stats.values()),
            'mlb':    sum(v['n'] for v in mlb_stats.values()),
        },
        'byLeague':  soccer_stats,
        'byLeagueNba': nba_stats,
        'byLeagueMlb': mlb_stats,
        'lastScan':  db.get_last_scan(),
    })

@app.route('/api/glai/scan', methods=['POST'])
@require_admin
def glai_scan():
    if glai._scan_running:
        return jsonify({'ok': False, 'msg': 'Scan ya en curso'}), 409
    days = int((request.get_json() or {}).get('days', 7))
    t = threading.Thread(target=glai.auto_scan, args=(days,), daemon=True)
    t.start()
    return jsonify({'ok': True, 'msg': f'Scan iniciado ({days} días)'})

@app.route('/api/glai/analyze', methods=['POST'])
@require_login
def glai_analyze():
    """Análisis GLAI — consume 1 token para todos (admin = ilimitado)."""
    u = current_user()
    # Admin = ilimitado. Todos los demás consumen 1 token.
    if u.get('role') != 'admin':
        ok = users.use_token(u['username'])
        if not ok:
            db.add_log(u['username'], 'analyze_failed', ip=get_client_ip(), details='Sin tokens')
            return jsonify({'error': 'no_tokens',
                           'msg': 'Sin tokens disponibles. Contacta al admin.'}), 403
        session['user'] = users.get_user(u['username'])

    try:
        data   = request.get_json() or {}
        team_a = data.get('teamA', '')
        team_b = data.get('teamB', '')
        league = data.get('league', 'soccer')
        sport  = data.get('sport', 'soccer')   # soccer | nba | mlb
        _def_a = 115.0 if sport == 'nba' else (4.5 if sport == 'mlb' else 1.4)
        _def_b = 112.0 if sport == 'nba' else (4.2 if sport == 'mlb' else 1.1)
        val_a  = float(data['xgA']) if 'xgA' in data else calc_auto_xg(team_a, sport, _def_a)
        val_b  = float(data['xgB']) if 'xgB' in data else calc_auto_xg(team_b, sport, _def_b)
    except Exception as e:
        return jsonify({'error': f'Datos inválidos: {e}'}), 400

    # Tokens restantes
    tokens_left = session['user'].get('tokens', 0) if u.get('role') != 'admin' else -1

    try:
        # ── NBA ─────────────────────────────────────────────────────────
        if sport == 'nba':
            # Historial real de cada equipo (se usa cuando ya hay datos de Sportradar)
            hist_a = glai.team_history(team_a, limit=5) if team_a else []
            hist_b = glai.team_history(team_b, limit=5) if team_b else []

            # Si tenemos historial, recalcula PPG desde resultados reales
            def _avg_scored(hist, default):
                scores = [h['myG'] for h in hist if h.get('myG') is not None]
                return round(sum(scores) / len(scores), 1) if scores else default

            real_ppg_a = _avg_scored(hist_a, val_a)
            real_ppg_b = _avg_scored(hist_b, val_b)

            prediction = glai.predict_nba(real_ppg_a, real_ppg_b)
            bet        = glai.ai_bet_nba(prediction, team_a, team_b, hist_a, hist_b)
            try:
                db.add_log(u['username'], 'analyze', ip=get_client_ip(),
                           details=f'{team_a} vs {team_b} | nba | PPG:{real_ppg_a}-{real_ppg_b}')
            except Exception:
                pass
            return jsonify({
                'ok':         True,
                'sport':      'nba',
                'tokensLeft': tokens_left,
                'prediction': prediction,
                'bet':        bet,
                'histA':      hist_a,
                'histB':      hist_b,
                'ppgA':       real_ppg_a,
                'ppgB':       real_ppg_b,
                'total':      glai.total_learned(),
            })

        # ── MLB ─────────────────────────────────────────────────────────
        if sport == 'mlb':
            # Historial real de cada equipo
            hist_a = glai.team_history(team_a, limit=5) if team_a else []
            hist_b = glai.team_history(team_b, limit=5) if team_b else []

            def _avg_scored(hist, default):
                scores = [h['myG'] for h in hist if h.get('myG') is not None]
                return round(sum(scores) / len(scores), 2) if scores else default

            real_rpg_a = _avg_scored(hist_a, val_a)
            real_rpg_b = _avg_scored(hist_b, val_b)

            prediction = glai.predict_mlb(real_rpg_a, real_rpg_b)
            bet        = glai.ai_bet_mlb(prediction, team_a, team_b, hist_a, hist_b)
            try:
                db.add_log(u['username'], 'analyze', ip=get_client_ip(),
                           details=f'{team_a} vs {team_b} | mlb | RPG:{real_rpg_a}-{real_rpg_b}')
            except Exception:
                pass
            return jsonify({
                'ok':         True,
                'sport':      'mlb',
                'tokensLeft': tokens_left,
                'prediction': prediction,
                'bet':        bet,
                'histA':      hist_a,
                'histB':      hist_b,
                'rpgA':       real_rpg_a,
                'rpgB':       real_rpg_b,
                'total':      glai.total_learned(),
            })

        # ── SOCCER (default) ────────────────────────────────────────────
        # Historial desde DB local (rápido, sin llamadas externas)
        hist_a = glai.team_history(team_a, limit=5) if team_a else []
        hist_b = glai.team_history(team_b, limit=5) if team_b else []

        # TheSportsDB en hilo separado — no bloquea el análisis
        def _fetch_tsdb():
            try:
                if team_a: tsdb.learn_team_events(team_a, glai, league_id=league)
                if team_b: tsdb.learn_team_events(team_b, glai, league_id=league)
            except Exception:
                pass
        threading.Thread(target=_fetch_tsdb, daemon=True).start()

        prediction    = glai.predict('soccer', league, val_a, val_b)
        # ✅ ai_bet NO recibe h2h — se quitó el parámetro que no existía
        bet           = glai.ai_bet(hist_a, hist_b, val_a, val_b, team_a, team_b)
        corners_cards = glai.predict_corners_cards(val_a, val_b, league)
        lg_stats      = glai.get_league_stats('soccer', league)

    except Exception as e:
        import traceback
        print(f"[GLAI analyze error] {traceback.format_exc()}")
        return jsonify({'error': f'Error en análisis: {str(e)}'}), 500

    try:
        db.add_log(u['username'], 'analyze', ip=get_client_ip(),
                   details=f'{team_a} vs {team_b} | soccer | xG:{val_a}-{val_b}')
    except Exception:
        pass  # No crashear si DB está temporalmente bloqueada
    return jsonify({
        'ok':         True,
        'sport':      'soccer',
        'tokensLeft': tokens_left,
        'prediction': prediction,
        'bet':        bet,
        'corners':    corners_cards['corners'],
        'cards':      corners_cards['cards'],
        'histA':      hist_a,
        'histB':      hist_b,
        'h2h':        None,
        'lgStats':    lg_stats,
        'total':      glai.total_learned(),
    })

@app.route('/api/glai/live', methods=['POST'])
@require_login
def glai_live():
    """Análisis en vivo — gratis para todos los usuarios logueados."""
    data       = request.get_json() or {}
    team_a     = data.get('teamA', '')
    team_b     = data.get('teamB', '')
    sport      = data.get('sport', 'soccer')
    home_score = int(data.get('homeScore', 0))
    away_score = int(data.get('awayScore', 0))
    val_a      = float(data.get('xgA', 1.4))
    val_b      = float(data.get('xgB', 1.1))

    if sport == 'soccer':
        minute = int(data.get('minute', 45))
        pred   = glai.predict_live_soccer(home_score, away_score, minute, val_a, val_b)
    elif sport == 'nba':
        period = int(data.get('period', 2))
        pred   = glai.predict_live_nba(home_score, away_score, period, val_a, val_b)
    elif sport == 'mlb':
        inning = int(data.get('inning', 5))
        half   = data.get('half', 'bottom')
        pred   = glai.predict_live_mlb(home_score, away_score, inning, half, val_a, val_b)
    else:
        return jsonify({'error': 'Sport no soportado'}), 400

    return jsonify({
        'ok':        True,
        'sport':     sport,
        'teamA':     team_a,
        'teamB':     team_b,
        'homeScore': home_score,
        'awayScore': away_score,
        'prediction': pred,
    })

@app.route('/api/glai/team-history')
@require_login
def team_history():
    team = request.args.get('team', '')
    limit = int(request.args.get('limit', 5))
    if not team:
        return jsonify({'history': []})
    hist = tsdb.get_team_history(team, limit=limit)
    return jsonify({'team': team, 'history': hist})

# ════════════════════════════════════════════════════════════════
# RUTAS — ADMIN
# ════════════════════════════════════════════════════════════════
@app.route('/api/admin/users')
@require_admin
def admin_list_users():
    return jsonify({'users': users.list_users()})

@app.route('/api/admin/users', methods=['POST'])
@require_admin
def admin_create_user():
    data = request.get_json() or {}
    try:
        user = users.create_user(
            data.get('username',''),
            data.get('password',''),
            data.get('role','free'),
            int(data.get('tokens', 10))
        )
        return jsonify({'ok': True, 'user': user})
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/admin/users/<username>', methods=['DELETE'])
@require_admin
def admin_delete_user(username):
    try:
        users.delete_user(username)
        return jsonify({'ok': True})
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/admin/users/<username>/role', methods=['PATCH'])
@require_admin
def admin_change_role(username):
    data = request.get_json() or {}
    try:
        users.change_role(username, data.get('role','free'))
        return jsonify({'ok': True, 'user': users.get_user(username)})
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/admin/users/<username>/tokens', methods=['PATCH'])
@require_admin
def admin_grant_tokens(username):
    data = request.get_json() or {}
    try:
        new_total = users.grant_tokens(username, int(data.get('amount', 0)))
        return jsonify({'ok': True, 'tokens': new_total})
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/admin/users/<username>/password', methods=['PATCH'])
@require_admin
def admin_reset_password(username):
    data = request.get_json() or {}
    new_pw = data.get('password', '').strip()
    if len(new_pw) < 6:
        return jsonify({'error': 'La contraseña debe tener al menos 6 caracteres'}), 400
    try:
        users.admin_set_password(username, new_pw)
        return jsonify({'ok': True, 'msg': f'Contraseña de "{username}" actualizada'})
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/admin/scan', methods=['POST'])
@require_admin
def admin_trigger_scan():
    data = request.get_json() or {}
    days = int(data.get('days', 30))
    if glai._scan_running:
        return jsonify({'ok': False, 'msg': 'Scan ya en curso'}), 409
    t = threading.Thread(target=glai.auto_scan, args=(days,), daemon=True)
    t.start()
    return jsonify({'ok': True, 'msg': f'Scan histórico iniciado ({days} días)'})

# ════════════════════════════════════════════════════════════════
# RUTAS — CÓDIGOS DE INVITACIÓN
# ════════════════════════════════════════════════════════════════
@app.route('/api/admin/codes', methods=['GET'])
@require_admin
def admin_list_codes():
    codes = db.list_invite_codes()
    return jsonify({'codes': codes})

@app.route('/api/admin/codes', methods=['POST'])
@require_admin
def admin_create_code():
    data     = request.get_json() or {}
    username = (data.get('username') or '').strip()
    role     = data.get('role', 'premium')
    tokens   = int(data.get('tokens', 10))
    days     = data.get('expires_days')  # None = no expira
    admin    = current_user()['username']

    if not username or len(username) < 3:
        return jsonify({'error': 'Username inválido (mín. 3 caracteres)'}), 400
    if users.get_user(username):
        return jsonify({'error': f'El usuario "{username}" ya existe'}), 400

    # Verificar que no haya un código activo para ese username
    activos = [c for c in db.list_invite_codes()
               if c['username'] == username and not c['used_at']]
    if activos:
        return jsonify({'error': f'Ya existe un código activo para "{username}"'}), 400

    # Generar código único
    code = generate_invite_code()
    while db.get_invite_code(code):
        code = generate_invite_code()

    expires_at = None
    if days:
        expires_at = (datetime.utcnow() + timedelta(days=int(days))).isoformat()

    db.create_invite_code(code, username, role, tokens, admin, expires_at)
    db.add_log(admin, 'code_generated', ip=get_client_ip(),
               details=f'Código {code} para "{username}" | rol={role} | tokens={tokens}')

    return jsonify({'ok': True, 'code': code, 'username': username,
                    'role': role, 'tokens': tokens, 'expires_at': expires_at})

@app.route('/api/admin/codes/<code>', methods=['DELETE'])
@require_admin
def admin_revoke_code(code):
    invite = db.get_invite_code(code)
    if not invite:
        return jsonify({'error': 'Código no encontrado'}), 404
    if invite['used_at']:
        return jsonify({'error': 'No se puede revocar un código ya usado'}), 400
    db.revoke_invite_code(code)
    admin = current_user()['username']
    db.add_log(admin, 'code_revoked', ip=get_client_ip(), details=f'Código revocado: {code}')
    return jsonify({'ok': True})

# ════════════════════════════════════════════════════════════════
# RUTAS — LOGS
# ════════════════════════════════════════════════════════════════
@app.route('/api/admin/logs')
@require_admin
def admin_all_logs():
    limit = int(request.args.get('limit', 200))
    logs = db.get_all_logs(limit=limit)
    return jsonify({'logs': logs})

@app.route('/api/admin/logs/<username>')
@require_admin
def admin_user_logs(username):
    logs = db.get_user_logs(username, limit=100)
    return jsonify({'username': username, 'logs': logs})

# ════════════════════════════════════════════════════════════════
# RUTAS — PARTIDOS CLAVE (alertas VIP)
# ════════════════════════════════════════════════════════════════
@app.route('/api/glai/key-matches', methods=['POST'])
@require_admin
def key_matches():
    """Analiza los partidos de hoy y devuelve los que tienen prob > umbral."""
    data      = request.get_json() or {}
    threshold = float(data.get('threshold', 0.72))  # 72% default
    matches   = espn.get_all_today()
    results   = []
    for m in matches:
        if m.get('status') not in ('pre', 'scheduled'):
            continue  # solo partidos futuros
        sport  = m.get('sport', 'soccer')
        home   = m.get('homeTeam', '')
        away   = m.get('awayTeam', '')
        try:
            if sport == 'nba':
                pred = glai.predict_nba(115.0, 112.0)
                prob = max(pred.get('home_win', 0), pred.get('away_win', 0))
            elif sport == 'mlb':
                pred = glai.predict_mlb(4.5, 4.2)
                prob = max(pred.get('home_win', 0), pred.get('away_win', 0))
            else:
                pred = glai.predict('soccer', m.get('league', 'soccer'), 1.4, 1.1)
                prob = max(
                    pred.get('home_win', 0),
                    pred.get('draw', 0),
                    pred.get('away_win', 0)
                )
            if prob >= threshold:
                winner = home if pred.get('home_win', 0) == prob else (
                    'Empate' if pred.get('draw', 0) == prob else away
                )
                results.append({
                    'sport':    sport,
                    'home':     home,
                    'away':     away,
                    'league':   m.get('league', ''),
                    'date':     m.get('date', ''),
                    'prob':     round(prob * 100, 1),
                    'winner':   winner,
                    'waText':   f'🔥 Partido clave hoy: {home} vs {away} — Pronóstico: {winner} ({round(prob*100,1)}%) 📊 PronoGol IA'
                })
        except Exception:
            continue
    results.sort(key=lambda x: x['prob'], reverse=True)
    return jsonify({'ok': True, 'matches': results})

# RUTAS — API KEYS
# ════════════════════════════════════════════════════════════════
@app.route('/api/keys', methods=['GET'])
@require_admin
def get_keys():
    keys = db.get_all_keys()
    # Enmascarar valores
    masked = {k: '•' * 12 + v[-4:] if len(v) > 4 else '****'
              for k, v in keys.items()}
    return jsonify({'keys': masked, 'configured': list(keys.keys())})

@app.route('/api/keys', methods=['POST'])
@require_admin
def save_key():
    data = request.get_json() or {}
    name  = data.get('name','').strip()
    value = data.get('value','').strip()
    if not name or not value:
        return jsonify({'error': 'Nombre y valor requeridos'}), 400
    db.save_key(name, value)
    # Hot-reload scrapers si se guarda la key de Sportradar
    if 'sportradar' in name.lower():
        sprt.api_key = value
        sprt._update_session_headers()
        if snba:
            snba.api_key = value
            snba._update_session_headers()
        if smlb:
            smlb.api_key = value
            smlb._update_session_headers()
        print(f"[Server] ✅ Sportradar key actualizada")
    return jsonify({'ok': True, 'msg': f'Key "{name}" guardada'})

# ════════════════════════════════════════════════════════════════
# FRONTEND
# ════════════════════════════════════════════════════════════════
@app.route('/')
@app.route('/<path:path>')
def serve_frontend(path=''):
    frontend = BASE / 'frontend'
    target = frontend / path
    if path and target.exists() and target.is_file():
        return send_from_directory(str(frontend), path)
    # SPA fallback — siempre sirve index.html
    return send_from_directory(str(frontend), 'index.html')

# ════════════════════════════════════════════════════════════════
# INICIO
# ════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("\n" + "="*55)
    print("  ⚽ Soccer Predictor Pro — Backend Python")
    print("="*55)
    print(f"  🌐 URL: http://localhost:5000")
    print(f"  🧠 GLAI aprende sola cada 2 horas (APScheduler)")
    print(f"  💾 Base de datos: soccer_predictor.db")
    print(f"  👤 Admin: usuario=admin / contraseña=admin123")
    print(f"  🏟️  Sportradar: {'✅ conectado' if sprt._ok() else '⚠️ sin key'}")
    print("="*55 + "\n")
    # Primer scan al iniciar (en background, no bloquea)
    import threading
    threading.Thread(target=lambda: glai.auto_scan(days_back=7), daemon=True).start()
    PORT = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=PORT, debug=False, use_reloader=False)
