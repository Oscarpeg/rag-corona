"""
Pipeline principal: PDF → Markdown estructurado con secciones GHS.
Flujo único: pymupdf4llm (tablas nativas en pipes) + detect_sections().
"""
import re
import os
import sys
import logging

import fitz
import pymupdf4llm

sys.path.insert(0, os.path.dirname(__file__))
from section_mapper import detect_sections, SECCIONES_GHS

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

# Captura encabezados de sección en el markdown generado por pymupdf4llm,
# que puede envolverlos en **, ## o dejarlos planos.
_SEC_PAT = re.compile(
    r'^(?:#{1,6}\s*)?(?:\*{1,2})?[ \t]*SECCI[ÓO]N\s+(\d+)\s*:\s*(.*?)(?:\*{1,2})?[ \t]*$',
    re.IGNORECASE | re.MULTILINE,
)
_CONTINUA = re.compile(r'\(contin[uú]a\)', re.IGNORECASE)


def _normalizar_encabezados(md: str) -> str:
    """Reemplaza cualquier variante de 'SECCIÓN N: TÍTULO' por '## Sección N: TÍTULO'.
    Las líneas de continuación se eliminan para no duplicar el encabezado.
    """
    def _reemplazar(m: re.Match) -> str:
        titulo = re.sub(r'\*+', '', m.group(2)).strip()
        if _CONTINUA.search(titulo):
            return ""                        # elimina la línea de continuación
        num = int(m.group(1))
        return f"## Sección {num}: {titulo}"

    return _SEC_PAT.sub(_reemplazar, md)


def extract_to_markdown(pdf_path: str) -> str:
    """
    Convierte un PDF de FDS CORONA a Markdown estructurado.

    Parameters
    ----------
    pdf_path : str  Ruta absoluta al PDF.

    Returns
    -------
    str  Contenido Markdown con ## por cada sección GHS.
         También persiste el archivo en data/processed/.
    """
    pdf_path = os.path.abspath(pdf_path)
    nombre = os.path.splitext(os.path.basename(pdf_path))[0]
    logger.info("Procesando: %s", os.path.basename(pdf_path))

    # ── 1. pymupdf4llm: texto + tablas en pipes ───────────────────────────────
    md_raw = pymupdf4llm.to_markdown(pdf_path, write_images=False)
    logger.info("pymupdf4llm generó %d caracteres de Markdown.", len(md_raw))

    # ── 2. detect_sections() sobre el texto plano para auditoría ─────────────
    doc = fitz.open(pdf_path)
    texto_plano = "\n".join(page.get_text("text") for page in doc)
    secciones = detect_sections(texto_plano)
    logger.info("Secciones detectadas: %d / %d", len(secciones), len(SECCIONES_GHS))

    # ── 3. Normalizar encabezados → ## Sección N: [Título] ───────────────────
    md_final = _normalizar_encabezados(md_raw)

    # ── 5. Guardar en data/processed/ ─────────────────────────────────────────
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    out_path = os.path.join(PROCESSED_DIR, f"{nombre}.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md_final)
    logger.info("Guardado en: %s", out_path)

    return md_final


# ── Ejecución directa ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    pdfs = sorted(f for f in os.listdir(RAW_DIR) if f.lower().endswith(".pdf"))

    if not pdfs:
        print("No hay PDFs en data/raw/")
        sys.exit(1)

    pdf_path = os.path.join(RAW_DIR, pdfs[0])
    md = extract_to_markdown(pdf_path)

    # ── Primeras 80 líneas ────────────────────────────────────────────────────
    lineas = md.splitlines()
    print("\n" + "=" * 70)
    print("PRIMERAS 80 LÍNEAS")
    print("=" * 70)
    for i, linea in enumerate(lineas[:80], 1):
        print(f"{i:3d}  {linea}")

    # ── Primera tabla en formato pipes ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PRIMERA TABLA DETECTADA (pipes)")
    print("=" * 70)
    tabla_lineas = []
    en_tabla = False
    for linea in lineas:
        if re.match(r'^\s*\|', linea):
            en_tabla = True
            tabla_lineas.append(linea)
        elif en_tabla:
            break       # fin de la tabla

    if tabla_lineas:
        print("\n".join(tabla_lineas))
    else:
        print("(ninguna tabla en formato pipes detectada en el Markdown)")
