# 🚀 Cómo subir Soccer Predictor Pro a la nube (Render.com)
## Sin instalar nada en tu PC — Windows 11

---

## PASO 1 — Crear cuenta GitHub (gratis)
1. Ve a → https://github.com
2. Clic en **Sign up** (arriba a la derecha)
3. Escribe tu email, crea una contraseña, elige un username
4. Verifica tu email

---

## PASO 2 — Subir el código a GitHub

1. Inicia sesión en GitHub
2. Clic en el **+** (arriba a la derecha) → **New repository**
3. Nombre: `soccer-predictor` → clic en **Create repository**
4. En la página que aparece, busca el enlace que dice **"uploading an existing file"** y haz clic
5. Arrastra **TODOS los archivos** de la carpeta `soccer-predictor` (incluyendo la subcarpeta `frontend/` y `scrapers/`)
6. Clic en **Commit changes** (botón verde abajo)

> ⚠️ Asegúrate de subir TAMBIÉN las subcarpetas:
> - `frontend/index.html`
> - `scrapers/espn.py`
> - `scrapers/thesportsdb.py`
> - `scrapers/besoccer.py`
> - `scrapers/__init__.py`

---

## PASO 3 — Crear cuenta Render.com (gratis)

1. Ve a → https://render.com
2. Clic en **Get Started for Free**
3. Escoge **Sign up with GitHub** (así conecta directo tu repo)
4. Autoriza a Render acceder a GitHub

---

## PASO 4 — Desplegar la app

1. En Render, clic en **New +** → **Web Service**
2. Selecciona tu repositorio `soccer-predictor`
3. Render detectará automáticamente el `render.yaml` y configurará todo
4. Clic en **Create Web Service**
5. Espera ~3 minutos mientras se construye
6. Render te dará una URL tipo: `https://soccer-predictor-pro.onrender.com`

---

## PASO 5 — ¡Listo! Usar la app

- Abre la URL en cualquier navegador (PC, celular, lo que sea)
- **Usuario:** `admin`
- **Contraseña:** `admin123`
- ⚠️ **Cambia la contraseña** desde el panel de Admin al entrar

---

## ¿Qué pasa con el aprendizaje autónomo?

- ✅ GLAI aprende sola **cada 2 horas** en el servidor de Render
- ✅ Los datos se guardan aunque cierres la pestaña
- ✅ Funciona 24/7 en la nube
- ⚠️ En el plan **gratis**, la app "duerme" si nadie la usa por 15 minutos
  - Al volver a entrar, tarda ~30 segundos en despertar
  - Si quieres que esté siempre activa, el plan Starter cuesta $7/mes

---

## ¿Cómo actualizar el código después?

Solo sube los archivos nuevos a GitHub (arrastra y suelta igual que antes) y Render se actualiza automáticamente.

---

## Resumen de archivos a subir a GitHub

```
soccer-predictor/
├── server.py
├── database.py
├── glai.py
├── users.py
├── requirements.txt
├── render.yaml
├── scrapers/
│   ├── __init__.py
│   ├── espn.py
│   ├── thesportsdb.py
│   └── besoccer.py
└── frontend/
    └── index.html
```
