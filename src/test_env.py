import sys

# ── 1. Imports ────────────────────────────────────────────────────────────────
libs = {
    "pymupdf4llm": "pymupdf4llm",
    "pytesseract": "pytesseract",
    "Pillow": "PIL",
    "chromadb": "chromadb",
    "langchain": "langchain",
    "langchain-community": "langchain_community",
    "langchain-ollama": "langchain_ollama",
}

all_ok = True
for name, module in libs.items():
    try:
        __import__(module)
        print(f"  [OK] {name}")
    except ImportError as e:
        print(f"  [FAIL] {name}: {e}")
        all_ok = False

print()
if all_ok:
    print("Todas las librerías se importaron correctamente.\n")
else:
    print("Algunas librerías fallaron. Revisa la instalación.\n")
    sys.exit(1)

# ── 2. Petición a Ollama ──────────────────────────────────────────────────────
import urllib.request
import json

OLLAMA_URL = "http://localhost:11434/api/tags"
print(f"Conectando a Ollama en {OLLAMA_URL} ...")

try:
    with urllib.request.urlopen(OLLAMA_URL, timeout=5) as resp:
        data = json.loads(resp.read())
    modelos = [m["name"] for m in data.get("models", [])]
    print(f"Modelos disponibles en Ollama: {modelos}\n")
except Exception as e:
    print(f"  [ERROR] No se pudo conectar a Ollama: {e}")
    print("  Asegúrate de que Ollama esté corriendo: ollama serve\n")
    sys.exit(1)

# ── 3. Verificar modelos requeridos ───────────────────────────────────────────
REQUIRED = {
    "nomic-embed-text": ["nomic-embed-text"],
    "llama3":           ["llama3", "llama3.1", "llama3:latest", "llama3.1:latest"],
}

faltantes = []
for label, variantes in REQUIRED.items():
    encontrado = any(
        any(v in m for m in modelos) for v in variantes
    )
    if encontrado:
        match = next(m for m in modelos if any(v in m for v in variantes))
        print(f"  [OK] {label} → encontrado como '{match}'")
    else:
        print(f"  [MISSING] {label} no está disponible")
        faltantes.append(label)

# ── 4. Comandos para descargar los que faltan ─────────────────────────────────
if faltantes:
    print("\nEjecuta los siguientes comandos para descargar los modelos faltantes:")
    for m in faltantes:
        print(f"  ollama pull {m}")
else:
    print("\nTodos los modelos requeridos están listos.")
