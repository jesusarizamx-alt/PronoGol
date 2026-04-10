"""
server.py — Soccer Predictor Pro — Backend Python/Flask
Corre con: python server.py
Abre: http://localhost:5000
"""
from flask import Flask, request, jsonify, session, send_from_directory
from flask_cors import CORS
from apscheduler.schedulers.background import BackgroundScheduler
import os, threading, secrets
from pathlib import Path

from database import Database
from users import UserManager
from glai import GLAIEngine
from scrapers.espn import ESPNScraper
from scrapers.thesportsdb import TheSportsDBScraper
from scrapers.sportradar import SportradarScraper  # ← NUEVO

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
sprt  = SportradarScraper(db=db)  # ← NUEVO — lee key desde env var o DB

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
    print("[Server] ✅ Sportradar conectado — API key detectada")
else:
    print("[Server] ⚠️  Sportradar sin API key — usando solo ESPN")

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
        return jsonify({'error': 'Usuario o contraseña incorrectos'}), 401
    session['user'] = user
    session.permanent = True
    return jsonify({'ok': True, 'user': user})

@app.route('/api/auth/register', methods=['POST'])
def register():
    data     = request.get_json() or {}
    username = (data.get('username') or '').strip()
    password = (data.get('password') or '').strip()
    if not username or not password:
        return jsonify({'error': 'Completa usuario y contraseña'}), 400
    if len(username) < 3:
        return jsonify({'error': 'El usuario debe tener al menos 3 caracteres'}), 400
    if len(password) < 6:
        return jsonify({'error': 'La contraseña debe tener al menos 6 caracteres'}), 400
    try:
        # 5 tokens gratis de bienvenida, rol free
        user = users.create_user(username, password, 'free', 5)
        session['user'] = user
        session.permanent = True
        return jsonify({'ok': True, 'user': user})
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/auth/logout', methods=['POST'])
def logout():
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
            return jsonify({'error': 'no_tokens',
                           'msg': 'Sin tokens disponibles. Contacta al admin.'}), 403
        session['user'] = users.get_user(u['username'])

    try:
        data   = request.get_json() or {}
        team_a = data.get('teamA', '')
        team_b = data.get('teamB', '')
        league = data.get('league', 'soccer')
        sport  = data.get('sport', 'soccer')   # soccer | nba | mlb
        val_a  = float(data.get('xgA', 1.4))  # xG / PPG / RPG
        val_b  = float(data.get('xgB', 1.1))
    except Exception as e:
        return jsonify({'error': f'Datos inválidos: {e}'}), 400

    # Tokens restantes: -1 = ilimitado (admin o legacy premium), número real para los demás
    raw_tokens  = session['user'].get('tokens', 0)
    tokens_left = raw_tokens if u.get('role') != 'admin' else -1

    try:
        # ── NBA ─────────────────────────────────────────────────────────
        if sport == 'nba':
            prediction = glai.predict_nba(val_a, val_b)
            bet        = glai.ai_bet_nba(prediction, team_a, team_b)
            return jsonify({
                'ok':         True,
                'sport':      'nba',
                'tokensLeft': tokens_left,
                'prediction': prediction,
                'bet':        bet,
                'total':      glai.total_learned(),
            })

        # ── MLB ─────────────────────────────────────────────────────────
        if sport == 'mlb':
            prediction = glai.predict_mlb(val_a, val_b)
            bet        = glai.ai_bet_mlb(prediction, team_a, team_b)
            return jsonify({
                'ok':         True,
                'sport':      'mlb',
                'tokensLeft': tokens_left,
                'prediction': prediction,
                'bet':        bet,
                'total':      glai.total_learned(),
            })

        # ── SOCCER (default) ────────────────────────────────────────────
        hist_a = tsdb.get_team_history(team_a, limit=5) if team_a else []
        hist_b = tsdb.get_team_history(team_b, limit=5) if team_b else []
        try:
            if team_a: tsdb.learn_team_events(team_a, glai, league_id=league)
            if team_b: tsdb.learn_team_events(team_b, glai, league_id=league)
        except Exception:
            pass  # Si falla TheSportsDB, continúa sin historial extra

        # Head-to-Head directo entre los dos equipos
        h2h = glai.get_h2h(team_a, team_b) if team_a and team_b else None

        prediction    = glai.predict('soccer', league, val_a, val_b)
        bet           = glai.ai_bet(hist_a, hist_b, val_a, val_b, team_a, team_b, h2h=h2h)
        corners_cards = glai.predict_corners_cards(val_a, val_b, league)
        lg_stats      = glai.get_league_stats('soccer', league)

    except Exception as e:
        return jsonify({'error': f'Error en análisis: {str(e)}'}), 500

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
        'h2h':        h2h,
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
    # Si guardaron la key de Sportradar, recargar el scraper
    if name.lower() in ('sportradar', 'sportradar_api_key'):
        sprt.api_key = value
        print(f"[Server] ✅ Sportradar API key actualizada desde panel admin")
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
