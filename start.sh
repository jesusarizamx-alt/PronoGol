#!/bin/bash
# ══════════════════════════════════════════════════════
#  Soccer Predictor Pro — Inicio rápido
#  Uso: bash start.sh
# ══════════════════════════════════════════════════════
set -e

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║   ⚽  Soccer Predictor Pro — Setup & Start   ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# ── Python 3 ──────────────────────────────────────
if ! command -v python3 &>/dev/null; then
  echo "❌  Python 3 no encontrado. Instálalo desde https://python.org"
  exit 1
fi
echo "✅  Python $(python3 --version)"

# ── pip ───────────────────────────────────────────
if ! python3 -m pip --version &>/dev/null; then
  echo "❌  pip no encontrado."
  exit 1
fi

# ── Entorno virtual (opcional pero recomendado) ───
if [ ! -d ".venv" ]; then
  echo "🔧  Creando entorno virtual .venv ..."
  python3 -m venv .venv
fi

# Activar venv
source .venv/bin/activate 2>/dev/null || true

# ── Instalar dependencias ─────────────────────────
echo "📦  Instalando dependencias..."
pip install -q -r requirements.txt

echo ""
echo "══════════════════════════════════════════════"
echo "  🌐  URL:  http://localhost:5000"
echo "  👤  Admin: usuario=admin / contraseña=admin123"
echo "  🧠  GLAI aprende sola cada 2 horas"
echo "  💾  Base de datos: soccer_predictor.db"
echo "══════════════════════════════════════════════"
echo ""

# ── Arrancar servidor ─────────────────────────────
python3 server.py
