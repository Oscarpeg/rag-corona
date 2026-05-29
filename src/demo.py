"""
Demo CLI interactiva — RAG-CORONA · Auditoría Semántica GHS
Uso:
    python src/demo.py
"""
import os
import sys
import textwrap

import chromadb

sys.path.insert(0, os.path.dirname(__file__))
from rag_pipeline import query_rag

# ── Paths ─────────────────────────────────────────────────────────────────────
CHROMA_DIR      = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db"))
COLLECTION_NAME = "fichas_seguridad_corona"

# ── Helpers ───────────────────────────────────────────────────────────────────

def _productos_disponibles() -> list[str]:
    client     = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(COLLECTION_NAME)
    items      = collection.get(include=["metadatas"])
    return sorted(set(m["producto"] for m in items["metadatas"]))


def _limpiar(texto: str) -> str:
    """Elimina etiquetas HTML y normaliza saltos de línea para consola."""
    import re
    texto = re.sub(r'<br\s*/?>', ' ', texto, flags=re.IGNORECASE)
    texto = re.sub(r'<[^>]+>', '', texto)          # cualquier otra etiqueta
    texto = re.sub(r'\s{2,}', ' ', texto)           # espacios múltiples
    texto = texto.strip()
    # Truncar títulos muy largos (tablas completas en seccion_titulo)
    if len(texto) > 80:
        texto = texto[:77] + "..."
    return texto


def _wrap(text: str, width: int = 72, indent: str = "  ") -> str:
    lines = []
    for paragraph in text.split("\n"):
        if paragraph.strip() == "":
            lines.append("")
        else:
            lines.extend(textwrap.wrap(paragraph, width=width, initial_indent=indent,
                                       subsequent_indent=indent))
    return "\n".join(lines)


def _print_banner(productos: list[str]) -> None:
    SEP = "═" * 72
    print(f"\n{SEP}")
    print("  RAG-CORONA // SISTEMA DE AUDITORÍA SEMÁNTICA GHS")
    print("  Grupo J · LLM: llama3 via Ollama · Embeddings: nomic-embed-text")
    print(SEP)
    print(f"\n  {len(productos)} productos indexados:\n")
    for i, p in enumerate(productos, 1):
        print(f"    {i:>2}. {p}")
    print()


def _print_ayuda() -> None:
    print("""
  Comandos disponibles:
    salir      — termina la sesión
    productos  — lista todos los productos indexados
    ayuda      — muestra este mensaje
    <pregunta> — consulta en lenguaje natural
""")


def _print_resultado(resultado: dict) -> None:
    SEP = "─" * 72
    print(f"\n{SEP}")
    print("  RESPUESTA:")
    print(_wrap(resultado["respuesta"]))
    print()
    for f in resultado["fuentes"]:
        titulo  = _limpiar(f["seccion_titulo"])
        producto = _limpiar(f["producto"])
        print(
            f"  Fuente: Sección {f['seccion_num']} — {titulo}"
            f" | Producto: {producto}"
            f" | Score: {f['score']:.4f}"
        )
    print(f"\n  Tiempo: {resultado['tiempo_seg']}s")
    print(f"{SEP}\n")


def _seleccionar_producto(productos: list[str]) -> str | None:
    """Devuelve el nombre del producto elegido o None para 'todos'."""
    print("  ¿Sobre qué producto desea consultar?")
    print("    0. Todos los productos")
    for i, p in enumerate(productos, 1):
        print(f"   {i:>2}. {p}")
    print()
    while True:
        raw = input("  Selección (número): ").strip()
        if raw == "0":
            return None
        if raw.isdigit() and 1 <= int(raw) <= len(productos):
            return productos[int(raw) - 1]
        print("  Opción no válida. Ingrese un número de la lista.")


# ── Prueba automática con mock ────────────────────────────────────────────────

MOCK_RESULTADOS = [
    {
        "pregunta":  "¿Qué EPP se recomienda para manipular pinturas CORONA?",
        "respuesta": (
            "Se recomienda la utilización de equipos de protección individual "
            "básicos: guantes de protección química (nitrilo o neopreno), "
            "gafas de seguridad, mascarilla de vapores orgánicos y ropa de "
            "trabajo adecuada. Para más información consulte la sección 8 de "
            "la ficha de seguridad correspondiente.\n"
            "Fuente: Sección 8 — CONTROLES DE EXPOSICIÓN/PROTECCIÓN PERSONAL "
            "| Producto: PINTURA PRIMERA MANO & ACABADO"
        ),
        "fuentes": [
            {"seccion_num": 8,  "seccion_titulo": "CONTROLES DE EXPOSICIÓN/PROTECCIÓN PERSONAL",
             "producto": "PINTURA PRIMERA MANO & ACABADO", "score": 0.6986},
            {"seccion_num": 7,  "seccion_titulo": "MANIPULACIÓN Y ALMACENAMIENTO",
             "producto": "FDS 31 - 407141521 - RECUBRIMIENTO ANTIGRAFFITI - CORONA", "score": 0.7395},
            {"seccion_num": 5,  "seccion_titulo": "MEDIDAS DE LUCHA CONTRA INCENDIOS",
             "producto": "FDS 61 - PINTURA SEÑALIZACIÓN Y DEMARCACIÓN - CORONA", "score": 0.6819},
        ],
        "tiempo_seg": 0.0,
    },
    {
        "pregunta":  "¿Cuáles son los primeros auxilios en caso de ingestión?",
        "respuesta": (
            "En caso de ingestión NO provocar el vómito. Enjuagar la boca con "
            "agua. Dar a beber agua en pequeñas cantidades. Acudir al médico "
            "lo más rápidamente posible llevando consigo la FDS del producto. "
            "Los síntomas pueden presentarse con posterioridad a la exposición.\n"
            "Fuente: Sección 4 — PRIMEROS AUXILIOS | Producto: FDS 76 - PINTURA TOTAL - CORONA"
        ),
        "fuentes": [
            {"seccion_num": 4, "seccion_titulo": "PRIMEROS AUXILIOS",
             "producto": "FDS 76 - PINTURA TOTAL - CORONA", "score": 0.7632},
            {"seccion_num": 4, "seccion_titulo": "PRIMEROS AUXILIOS",
             "producto": "FDS 93 - PINTURA FACHADA FLEXIBLE - CORONA", "score": 0.7592},
            {"seccion_num": 4, "seccion_titulo": "PRIMEROS AUXILIOS",
             "producto": "FDS 91 - PINTURA EXTERIORES - CORONA", "score": 0.7575},
        ],
        "tiempo_seg": 0.0,
    },
    {
        "pregunta":  "¿Cómo se almacena correctamente la pintura?",
        "respuesta": (
            "Almacenar en lugar fresco, seco y bien ventilado, alejado de "
            "fuentes de calor, radiación y electricidad estática. Temperatura "
            "recomendada entre 5 °C y 30 °C. Tiempo máximo de almacenamiento: "
            "6 meses. Evitar el contacto con alimentos.\n"
            "Fuente: Sección 7 — MANIPULACIÓN Y ALMACENAMIENTO "
            "| Producto: FDS 42 - PINTURA LAVABLE ANTIBACTERIAL - CORONA"
        ),
        "fuentes": [
            {"seccion_num": 7, "seccion_titulo": "MANIPULACIÓN Y ALMACENAMIENTO",
             "producto": "FDS 42 - PINTURA LAVABLE ANTIBACTERIAL - CORONA", "score": 0.7630},
            {"seccion_num": 7, "seccion_titulo": "MANIPULACIÓN Y ALMACENAMIENTO",
             "producto": "FDS 91 - PINTURA EXTERIORES - CORONA", "score": 0.7679},
            {"seccion_num": 16, "seccion_titulo": "OTRAS INFORMACIONES",
             "producto": "PINTURA PRIMERA MANO & ACABADO", "score": 0.7649},
        ],
        "tiempo_seg": 0.0,
    },
]


def _demo_mock() -> None:
    SEP2 = "─" * 72
    print(f"\n{'═'*72}")
    print("  PRUEBA AUTOMÁTICA — 3 consultas con respuestas indexadas")
    print(f"{'═'*72}")
    for i, r in enumerate(MOCK_RESULTADOS, 1):
        print(f"\n  [{i}/3] {r['pregunta']}")
        _print_resultado(r)
    print("  Prueba automática completada. Iniciando modo interactivo...\n")


# ── Loop interactivo ──────────────────────────────────────────────────────────

def main() -> None:
    productos = _productos_disponibles()
    _print_banner(productos)
    _demo_mock()

    filtro = _seleccionar_producto(productos)
    scope  = filtro if filtro else "TODOS LOS PRODUCTOS"

    print(f"\n  Scope activo: {scope}")
    _print_ayuda()

    while True:
        try:
            raw = input("Pregunta RAG > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Sesión terminada.\n")
            break

        if not raw:
            continue

        cmd = raw.lower()
        if cmd == "salir":
            print("\n  Sesión terminada.\n")
            break
        elif cmd == "productos":
            print()
            for i, p in enumerate(productos, 1):
                print(f"  {i:>2}. {p}")
            print()
        elif cmd == "ayuda":
            _print_ayuda()
        else:
            print("  Consultando... (puede tardar ~2 min en CPU)\n")
            resultado = query_rag(raw, filtro_producto=filtro)
            _print_resultado(resultado)


if __name__ == "__main__":
    main()
