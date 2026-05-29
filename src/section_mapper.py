"""
Detecta y mapea las 16 secciones GHS/SGA en el texto completo de una FDS CORONA.
"""
import re
import logging

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# Títulos canónicos de las 16 secciones GHS
SECCIONES_GHS = {
    1:  "Identificación del producto",
    2:  "Identificación de peligros",
    3:  "Composición/información sobre componentes",
    4:  "Primeros auxilios",
    5:  "Lucha contra incendios",
    6:  "Medidas en caso de vertido accidental",
    7:  "Manipulación y almacenamiento",
    8:  "Controles de exposición/protección personal",
    9:  "Propiedades físicas y químicas",
    10: "Estabilidad y reactividad",
    11: "Información toxicológica",
    12: "Información ecotoxicológica",
    13: "Eliminación de los productos",
    14: "Información relativa al transporte",
    15: "Información sobre la reglamentación",
    16: "Otras informaciones",
}

# Regex: captura "SECCIÓN N: TÍTULO" ignorando variantes de acento y casing.
# Excluye las líneas de continuación "(continúa)".
_PATTERN = re.compile(
    r'^[ \t]*SECCI[ÓO]N\s+(\d+)\s*:\s*(.+?)[ \t]*$',
    re.IGNORECASE | re.MULTILINE,
)
_CONTINUA = re.compile(r'\(contin[uú]a\)', re.IGNORECASE)


def detect_sections(texto_completo: str) -> dict:
    """
    Detecta las secciones GHS en el texto extraído de una FDS CORONA.

    Returns
    -------
    dict  {num_seccion (int): {"titulo": str, "inicio": int, "fin": int}}
          inicio/fin son índices de carácter en texto_completo.
          Las secciones no encontradas emiten WARNING y no aparecen en el dict.
    """
    # Recoger todas las ocurrencias de encabezados (ignorar continuaciones)
    encontradas: dict[int, dict] = {}

    for match in _PATTERN.finditer(texto_completo):
        titulo_raw = match.group(2).strip()
        # Saltar repeticiones de "continúa"
        if _CONTINUA.search(titulo_raw):
            continue

        num = int(match.group(1))
        # Si ya teníamos esta sección, conservar la primera aparición
        if num not in encontradas:
            encontradas[num] = {
                "titulo": titulo_raw,
                "inicio": match.start(),
                "fin":    -1,       # se rellena en el paso siguiente
            }

    # Calcular fin de cada sección = inicio de la siguiente - 1
    nums_ordenados = sorted(encontradas.keys())
    for i, num in enumerate(nums_ordenados):
        siguiente = nums_ordenados[i + 1] if i + 1 < len(nums_ordenados) else None
        if siguiente is not None:
            encontradas[num]["fin"] = encontradas[siguiente]["inicio"] - 1
        else:
            encontradas[num]["fin"] = len(texto_completo) - 1

    # Emitir WARNING por cada sección GHS no encontrada
    for num_ghs, titulo_canonico in SECCIONES_GHS.items():
        if num_ghs not in encontradas:
            logger.warning(
                "Sección %d (%s) NO encontrada en el documento.", num_ghs, titulo_canonico
            )

    return encontradas


# ── Ejecución de prueba ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import fitz
    import sys
    import os

    RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    pdfs = sorted(
        [f for f in os.listdir(RAW_DIR) if f.lower().endswith(".pdf")]
    )
    if not pdfs:
        print("No hay PDFs en data/raw/")
        sys.exit(1)

    pdf_path = os.path.join(RAW_DIR, pdfs[0])
    print(f"Analizando: {pdfs[0]}\n")

    doc = fitz.open(pdf_path)
    texto_completo = "\n".join(page.get_text("text") for page in doc)

    resultado = detect_sections(texto_completo)

    print(f"Secciones encontradas: {len(resultado)} / {len(SECCIONES_GHS)}\n")
    print(f"{'N°':<4} {'TÍTULO EXACTO EN EL DOCUMENTO':<60} {'INICIO':>8} {'FIN':>8} {'CHARS':>7}")
    print("-" * 90)
    for num in sorted(resultado.keys()):
        s = resultado[num]
        chars = s["fin"] - s["inicio"] + 1
        print(f"{num:<4} {s['titulo']:<60} {s['inicio']:>8} {s['fin']:>8} {chars:>7}")
