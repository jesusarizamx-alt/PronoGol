"""
users.py — Gestión de usuarios, roles y tokens
"""
import hashlib
import secrets
from datetime import datetime

class UserManager:
    def __init__(self, db):
        self.db = db
        self._init_admin()

    def _hash(self, password: str) -> str:
        return hashlib.sha256(password.encode()).hexdigest()

    def _init_admin(self):
        """Crea el admin por defecto si no existe."""
        c = self.db._conn()
        exists = c.execute(
            "SELECT 1 FROM users WHERE username='admin'"
        ).fetchone()
        if not exists:
            c.execute("""
                INSERT INTO users (username, password_hash, role, tokens, created_at)
                VALUES ('admin', ?, 'admin', -1, ?)
            """, (self._hash('admin123'), datetime.utcnow().isoformat()))
            c.commit()
            print("[Users] ✅ Admin creado — usuario: admin / contraseña: admin123")

    # ─── Auth ────────────────────────────────────────────────────
    def login(self, username: str, password: str):
        """Valida credenciales. Retorna dict del usuario o None."""
        c = self.db._conn()
        row = c.execute(
            "SELECT * FROM users WHERE username=?", (username.lower().strip(),)
        ).fetchone()
        if not row:
            return None
        if row['password_hash'] != self._hash(password):
            return None
        # Update last login
        c.execute("UPDATE users SET last_login=? WHERE username=?",
                  (datetime.utcnow().isoformat(), username))
        c.commit()
        return {
            'username': row['username'],
            'role':     row['role'],
            'tokens':   row['tokens'],
        }

    def get_user(self, username: str):
        row = self.db._conn().execute(
            "SELECT * FROM users WHERE username=?", (username,)
        ).fetchone()
        if not row:
            return None
        return {
            'username':   row['username'],
            'role':       row['role'],
            'tokens':     row['tokens'],
            'created_at': row['created_at'],
            'last_login': row['last_login'],
        }

    # ─── User CRUD ───────────────────────────────────────────────
    def create_user(self, username: str, password: str, role='free', tokens=10):
        username = username.lower().strip()
        if len(username) < 2:
            raise ValueError("Usuario muy corto (mín 2 caracteres)")
        c = self.db._conn()
        existing = c.execute(
            "SELECT 1 FROM users WHERE username=?", (username,)
        ).fetchone()
        if existing:
            raise ValueError(f'El usuario "{username}" ya existe')
        if role not in ('free', 'premium', 'admin'):
            raise ValueError("Rol inválido")
        real_tokens = -1 if role == 'admin' else int(tokens)
        c.execute("""
            INSERT INTO users (username, password_hash, role, tokens, created_at)
            VALUES (?,?,?,?,?)
        """, (username, self._hash(password), role, real_tokens,
              datetime.utcnow().isoformat()))
        c.commit()
        return self.get_user(username)

    def delete_user(self, username: str):
        if username == 'admin':
            raise ValueError("No puedes eliminar al admin principal")
        self.db._conn().execute("DELETE FROM users WHERE username=?", (username,))
        self.db._conn().commit()

    def change_role(self, username: str, new_role: str):
        if new_role not in ('free', 'premium', 'admin'):
            raise ValueError("Rol inválido")
        c = self.db._conn()
        new_tokens = -1 if new_role == 'admin' else None
        if new_tokens is not None:
            c.execute("UPDATE users SET role=?, tokens=? WHERE username=?",
                      (new_role, new_tokens, username))
        else:
            c.execute("UPDATE users SET role=? WHERE username=?",
                      (new_role, username))
        c.commit()

    def grant_tokens(self, username: str, amount: int):
        if amount <= 0:
            raise ValueError("Cantidad debe ser mayor a 0")
        c = self.db._conn()
        row = c.execute("SELECT role, tokens FROM users WHERE username=?",
                        (username,)).fetchone()
        if not row:
            raise ValueError("Usuario no encontrado")
        if row['role'] == 'admin':
            raise ValueError("Los admins tienen tokens ilimitados")
        c.execute("UPDATE users SET tokens = tokens + ? WHERE username=?",
                  (amount, username))
        c.commit()
        new_total = c.execute(
            "SELECT tokens FROM users WHERE username=?", (username,)
        ).fetchone()[0]
        return new_total

    def use_token(self, username: str) -> bool:
        """Descuenta 1 token. Retorna True si ok, False si sin tokens."""
        c = self.db._conn()
        row = c.execute(
            "SELECT role, tokens FROM users WHERE username=?", (username,)
        ).fetchone()
        if not row:
            return False
        if row['role'] == 'admin':
            return True  # ilimitado
        if row['tokens'] <= 0:
            return False
        c.execute("UPDATE users SET tokens = tokens - 1 WHERE username=?",
                  (username,))
        c.commit()
        return True

    def change_password(self, username: str, current_password: str, new_password: str) -> bool:
        """Cambia la contraseña verificando la actual. Retorna True si exitoso."""
        c = self.db._conn()
        row = c.execute(
            "SELECT password_hash FROM users WHERE username=?", (username,)
        ).fetchone()
        if not row:
            return False
        if row['password_hash'] != self._hash(current_password):
            return False
        c.execute(
            "UPDATE users SET password_hash=? WHERE username=?",
            (self._hash(new_password), username)
        )
        c.commit()
        return True

    def admin_set_password(self, username: str, new_password: str):
        """Admin fuerza nueva contraseña sin verificar la actual."""
        c = self.db._conn()
        exists = c.execute("SELECT 1 FROM users WHERE username=?", (username,)).fetchone()
        if not exists:
            raise ValueError(f'Usuario "{username}" no encontrado')
        c.execute(
            "UPDATE users SET password_hash=? WHERE username=?",
            (self._hash(new_password), username)
        )
        c.commit()

    def list_users(self):
        rows = self.db._conn().execute(
            "SELECT * FROM users ORDER BY created_at"
        ).fetchall()
        return [{
            'username':   r['username'],
            'role':       r['role'],
            'tokens':     r['tokens'],
            'created_at': r['created_at'],
            'last_login': r['last_login'],
        } for r in rows]
