"""
database.py — SQLite layer para Soccer Predictor Pro
Reemplaza localStorage con persistencia real y sin límite de espacio.
"""
import sqlite3
import json
import threading
from datetime import datetime
from pathlib import Path

# Usa disco persistente en Render (/data) si existe, si no usa la carpeta local
import os
_PERSISTENT = Path("/data")
if _PERSISTENT.exists():
    DB_PATH = _PERSISTENT / "soccer_predictor.db"
else:
    DB_PATH = Path(__file__).parent / "soccer_predictor.db"

class Database:
    def __init__(self, path=DB_PATH):
        self.path = str(path)
        self._local = threading.local()
        self._write_lock = threading.Lock()  # evita "database is locked"
        self._init_tables()

    def _conn(self):
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            # timeout=30 → espera hasta 30s antes de lanzar OperationalError
            self._local.conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA synchronous=NORMAL")   # más rápido y seguro
            self._local.conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn.execute("PRAGMA busy_timeout=30000")   # 30s busy timeout
        return self._local.conn

    def _init_tables(self):
        c = self._conn()
        c.executescript("""
        -- Usuarios del sistema
        CREATE TABLE IF NOT EXISTS users (
            username    TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            role        TEXT NOT NULL DEFAULT 'free',  -- free | premium | admin
            tokens      INTEGER NOT NULL DEFAULT 0,    -- -1 = ilimitado (admin)
            created_at  TEXT NOT NULL,
            last_login  TEXT
        );

        -- Resultados aprendidos por GLAI
        CREATE TABLE IF NOT EXISTS match_results (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            sport       TEXT NOT NULL DEFAULT 'soccer',
            league_id   TEXT NOT NULL,
            home_team   TEXT NOT NULL,
            away_team   TEXT NOT NULL,
            home_goals  INTEGER NOT NULL,
            away_goals  INTEGER NOT NULL,
            source      TEXT,           -- espn | tsdb | besoccer | apifb etc.
            event_id    TEXT UNIQUE,    -- deduplica por evento
            learned_at  TEXT NOT NULL
        );

        -- Estadísticas por liga (calculadas por GLAI)
        CREATE TABLE IF NOT EXISTS league_stats (
            sport       TEXT NOT NULL,
            league_id   TEXT NOT NULL,
            n           INTEGER NOT NULL DEFAULT 0,
            home_wins   INTEGER NOT NULL DEFAULT 0,
            draws       INTEGER NOT NULL DEFAULT 0,
            away_wins   INTEGER NOT NULL DEFAULT 0,
            total_home_goals REAL NOT NULL DEFAULT 0,
            total_away_goals REAL NOT NULL DEFAULT 0,
            updated_at  TEXT NOT NULL,
            PRIMARY KEY (sport, league_id)
        );

        -- API keys guardadas
        CREATE TABLE IF NOT EXISTS api_keys (
            key_name    TEXT PRIMARY KEY,
            key_value   TEXT NOT NULL,
            saved_at    TEXT NOT NULL
        );

        -- Log de escaneos automáticos
        CREATE TABLE IF NOT EXISTS scan_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at  TEXT NOT NULL,
            finished_at TEXT,
            results     INTEGER DEFAULT 0,
            sources     TEXT,
            status      TEXT DEFAULT 'running'  -- running | done | failed
        );

        -- Predicciones registradas (para calibración de GLAI)
        CREATE TABLE IF NOT EXISTS predictions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            sport       TEXT NOT NULL,
            league_id   TEXT NOT NULL,
            home_team   TEXT NOT NULL,
            away_team   TEXT NOT NULL,
            predicted   TEXT,    -- H | D | A
            actual      TEXT,    -- H | D | A (se llena después)
            confidence  REAL,
            created_at  TEXT NOT NULL
        );

        -- Índices para búsquedas frecuentes
        CREATE INDEX IF NOT EXISTS idx_results_league ON match_results(sport, league_id);
        CREATE INDEX IF NOT EXISTS idx_results_teams  ON match_results(home_team, away_team);
        CREATE INDEX IF NOT EXISTS idx_results_source ON match_results(source);
        """)
        c.commit()

    # ─── match_results ───────────────────────────────────────────
    def learn_result(self, sport, league_id, home_team, away_team,
                     home_goals, away_goals, source='unknown', event_id=None):
        """Guarda un resultado. Ignora duplicados por event_id."""
        with self._write_lock:
            c = self._conn()
            try:
                c.execute("""
                    INSERT OR IGNORE INTO match_results
                    (sport, league_id, home_team, away_team, home_goals, away_goals, source, event_id, learned_at)
                    VALUES (?,?,?,?,?,?,?,?,?)
                """, (sport, league_id, home_team.strip(), away_team.strip(),
                      int(home_goals), int(away_goals), source, event_id,
                      datetime.utcnow().isoformat()))
                inserted = c.rowcount > 0
                c.commit()
                if inserted:
                    self._update_league_stats(c, sport, league_id, home_goals, away_goals)
                return inserted
            except Exception as e:
                print(f"[DB] learn_result error: {e}")
                return False

    def _update_league_stats(self, c, sport, league_id, hg, ag):
        hg, ag = int(hg), int(ag)
        result = 'H' if hg > ag else ('A' if ag > hg else 'D')
        c.execute("""
            INSERT INTO league_stats (sport, league_id, n, home_wins, draws, away_wins,
                                      total_home_goals, total_away_goals, updated_at)
            VALUES (?,?,1,?,?,?,?,?,?)
            ON CONFLICT(sport, league_id) DO UPDATE SET
                n = n + 1,
                home_wins = home_wins + ?,
                draws = draws + ?,
                away_wins = away_wins + ?,
                total_home_goals = total_home_goals + ?,
                total_away_goals = total_away_goals + ?,
                updated_at = ?
        """, (sport, league_id,
              1 if result=='H' else 0, 1 if result=='D' else 0, 1 if result=='A' else 0,
              hg, ag, datetime.utcnow().isoformat(),
              1 if result=='H' else 0, 1 if result=='D' else 0, 1 if result=='A' else 0,
              hg, ag, datetime.utcnow().isoformat()))
        c.commit()

    def get_total_learned(self):
        r = self._conn().execute("SELECT COUNT(*) FROM match_results").fetchone()
        return r[0] if r else 0

    def get_league_stats(self, sport, league_id):
        r = self._conn().execute(
            "SELECT * FROM league_stats WHERE sport=? AND league_id=?",
            (sport, league_id)).fetchone()
        if not r or r['n'] == 0:
            return None
        n = r['n']
        return {
            'n': n,
            'homeWinRate': round(r['home_wins'] / n, 4),
            'drawRate':    round(r['draws'] / n, 4),
            'awayWinRate': round(r['away_wins'] / n, 4),
            'avgHG':       round(r['total_home_goals'] / n, 2),
            'avgAG':       round(r['total_away_goals'] / n, 2),
            'avgTG':       round((r['total_home_goals'] + r['total_away_goals']) / n, 2),
        }

    def get_all_league_stats(self, sport='soccer'):
        rows = self._conn().execute(
            "SELECT * FROM league_stats WHERE sport=? AND n>=3 ORDER BY n DESC",
            (sport,)).fetchall()
        result = {}
        for r in rows:
            n = r['n']
            result[r['league_id']] = {
                'n': n,
                'homeWinRate': round(r['home_wins'] / n, 4),
                'drawRate':    round(r['draws'] / n, 4),
                'awayWinRate': round(r['away_wins'] / n, 4),
                'avgHG':       round(r['total_home_goals'] / n, 2),
                'avgAG':       round(r['total_away_goals'] / n, 2),
            }
        return result

    def get_team_results(self, team_name, limit=10):
        """Últimos N resultados de un equipo (local o visita)."""
        like = f"%{team_name[:8]}%"
        rows = self._conn().execute("""
            SELECT * FROM match_results
            WHERE home_team LIKE ? OR away_team LIKE ?
            ORDER BY learned_at DESC LIMIT ?
        """, (like, like, limit)).fetchall()
        return [dict(r) for r in rows]

    def get_h2h_results(self, team_a, team_b, limit=10):
        """
        Historial directo entre team_a y team_b (en cualquier dirección).
        Retorna los últimos N enfrentamientos del más reciente al más viejo.
        """
        like_a = f"%{team_a[:8]}%"
        like_b = f"%{team_b[:8]}%"
        rows = self._conn().execute("""
            SELECT * FROM match_results
            WHERE (home_team LIKE ? AND away_team LIKE ?)
               OR (home_team LIKE ? AND away_team LIKE ?)
            ORDER BY learned_at DESC LIMIT ?
        """, (like_a, like_b, like_b, like_a, limit)).fetchall()
        return [dict(r) for r in rows]

    # ─── api_keys ────────────────────────────────────────────────
    def save_key(self, name, value):
        with self._write_lock:
            c = self._conn()
            c.execute("""
                INSERT OR REPLACE INTO api_keys (key_name, key_value, saved_at)
                VALUES (?,?,?)
            """, (name, value, datetime.utcnow().isoformat()))
            c.commit()

    def get_key(self, name):
        r = self._conn().execute(
            "SELECT key_value FROM api_keys WHERE key_name=?", (name,)).fetchone()
        return r[0] if r else None

    def get_all_keys(self):
        rows = self._conn().execute("SELECT key_name, key_value FROM api_keys").fetchall()
        return {r['key_name']: r['key_value'] for r in rows}

    # ─── scan_log ────────────────────────────────────────────────
    def start_scan_log(self):
        c = self._conn()
        c.execute("INSERT INTO scan_log (started_at, status) VALUES (?,?)",
                  (datetime.utcnow().isoformat(), 'running'))
        c.commit()
        return c.lastrowid

    def finish_scan_log(self, scan_id, results, sources):
        self._conn().execute("""
            UPDATE scan_log SET finished_at=?, results=?, sources=?, status='done'
            WHERE id=?
        """, (datetime.utcnow().isoformat(), results, json.dumps(sources), scan_id))
        self._conn().commit()

    def get_last_scan(self):
        r = self._conn().execute(
            "SELECT * FROM scan_log WHERE status='done' ORDER BY finished_at DESC LIMIT 1"
        ).fetchone()
        return dict(r) if r else None
