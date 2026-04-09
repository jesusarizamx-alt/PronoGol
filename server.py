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

# ── App setup ────────────────────────────────────────────────────
BASE = Path(__file__).parent
app = Flask(__name__, static_folder=str(BASE / 'frontend'), static_url_path='')
app.secret_key = os.environ.get('SECRET_KEY', secrets.token_hex(32))
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

# ── Scheduler — GLAI aprende sola cada 2 horas ───────────────────
scheduler = BackgroundScheduler(daemon=True)
scheduler.add_job(
    func=lambda: glai.auto_scan(days_back=7),
    trigger='interval',
    hours=2,
    id='glai_auto_learn',
    replace_existing=True
)
scheduler.start()
print("[Server] ✅ APScheduler activo — GLAI aprende sola cada 2 horas")

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

# ════════════════════════════════════════════════════════════════
# RUTAS — PARTIDOS (ESPN proxy — sin CORS)
# ════════════════════════════════════════════════════════════════
@app.route('/api/matches')
@require_login
def matches():
    try:
        data = espn.get_today_matches()
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
# RUTAS — GLAI IA
# ════════════════════════════════════════════════════════════════
@app.route('/api/glai/status')
@require_login
def glai_status():
    return jsonify(glai.scan_status())

@app.route('/api/glai/stats')
@require_login
def glai_stats():
    return jsonify({
        'total':      glai.total_learned(),
        'byLeague':   glai.get_all_stats('soccer'),
        'lastScan':   db.get_last_scan(),
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
@require_premium
def glai_analyze():
    """Análisis GLAI completo — consume 1 token si es premium."""
    u = current_user()
    # Deducir token si es premium
    if u.get('role') == 'premium':
        ok = users.use_token(u['username'])
        if not ok:
            return jsonify({'error': 'no_tokens',
                           'msg': 'Sin tokens disponibles. Contacta al admin.'}), 403
        # Refrescar sesión con nuevo balance
        session['user'] = users.get_user(u['username'])

    data = request.get_json() or {}
    team_a   = data.get('teamA', '')
    team_b   = data.get('teamB', '')
    league   = data.get('league', 'soccer')
    xg_a    = float(data.get('xgA', 1.4))
    xg_b    = float(data.get('xgB', 1.1))

    # Historial real por equipo (TheSportsDB — sin límite de fecha)
    hist_a = tsdb.get_team_history(team_a, limit=5) if team_a else []
    hist_b = tsdb.get_team_history(team_b, limit=5) if team_b else []

    # También aprende de estos partidos
    if team_a:
        tsdb.learn_team_events(team_a, glai, league_id=league)
    if team_b:
        tsdb.learn_team_events(team_b, glai, league_id=league)

    # Predicción Poisson ajustada con historial
    prediction = glai.predict('soccer', league, xg_a, xg_b)

    # Apuesta IA
    bet = glai.ai_bet(hist_a, hist_b, xg_a, xg_b, team_a, team_b)

    # Stats de liga
    lg_stats = glai.get_league_stats('soccer', league)

    return jsonify({
        'ok':        True,
        'tokensLeft': session['user'].get('tokens') if u.get('role') == 'premium' else -1,
        'prediction': prediction,
        'bet':        bet,
        'histA':      hist_a,
        'histB':      hist_b,
        'lgStats':    lg_stats,
        'total':      glai.total_learned(),
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
    print("="*55 + "\n")
    # Primer scan al iniciar (en background, no bloquea)
    import threading
    threading.Thread(target=lambda: glai.auto_scan(days_back=7), daemon=True).start()
    PORT = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=PORT, debug=False, use_reloader=False)
