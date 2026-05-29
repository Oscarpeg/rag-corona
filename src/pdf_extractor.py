"""
Pipeline principal: PDF → Markdown estructurado con secciones GHS + imágenes con trazabilidad.
Flujo: pymupdf4llm (texto + tablas pipes) → normalizar encabezados → inyectar imágenes OCR.
"""
import re
import os
import io
import sys
import logging

import fitz
import pymupdf4llm
import pytesseract
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from section_mapper import detect_sections, SECCIONES_GHS

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

PROCESSED_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "processed")
)
IMAGES_DIR = os.path.join(PROCESSED_DIR, "images")

# Captura encabezados de sección en el markdown generado por pymupdf4llm.
_SEC_PAT = re.compile(
    r'^(?:#{1,6}\s*)?(?:\*{1,2})?[ \t]*SECCI[ÓO]N\s+(\d+)\s*:\s*(.*?)(?:\*{1,2})?[ \t]*$',
    re.IGNORECASE | re.MULTILINE,
)
_CONTINUA = re.compile(r'\(contin[uú]a\)', re.IGNORECASE)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalizar_encabezados(md: str) -> str:
    """Reemplaza cualquier variante de 'SECCIÓN N: TÍTULO' por '## Sección N: TÍTULO'.
    Elimina las repeticiones de continuación.
    """
    def _reemplazar(m: re.Match) -> str:
        titulo = re.sub(r'\*+', '', m.group(2)).strip()
        if _CONTINUA.search(titulo):
            return ""
        return f"## Sección {num}: {titulo}" if (num := int(m.group(1))) else ""

    return _SEC_PAT.sub(_reemplazar, md)


def _pagina_a_seccion(page_idx: int, page_starts: list[int], sections_map: dict) -> tuple[int, str]:
    """Devuelve (num_sección, título) para la página dada, usando los offsets de carácter."""
    char_pos = page_starts[page_idx]
    # Buscar la sección cuyo rango [inicio, fin] contiene char_pos
    for num in sorted(sections_map.keys()):
        info = sections_map[num]
        if info["inicio"] <= char_pos <= info["fin"]:
            return num, info["titulo"]
    # Si la página cae antes de la primera sección o entre huecos, tomar la
    # sección más reciente cuyo inicio no supera char_pos.
    candidato = None
    for num in sorted(sections_map.keys()):
        if sections_map[num]["inicio"] <= char_pos:
            candidato = num
    if candidato is not None:
        return candidato, sections_map[candidato]["titulo"]
    # Fallback: primera sección
    first = sorted(sections_map.keys())[0]
    return first, sections_map[first]["titulo"]


# ── Función principal de imágenes ─────────────────────────────────────────────

def extract_images_with_traceability(
    pdf_path: str,
    md_content: str,
    sections_map: dict,
) -> str:
    """
    Extrae todas las imágenes del PDF, aplica OCR y las inyecta en el Markdown
    con un bloque de trazabilidad posicionado en la sección GHS correspondiente.

    Parameters
    ----------
    pdf_path     : ruta absoluta al PDF.
    md_content   : Markdown ya normalizado (con ## Sección N:).
    sections_map : salida de detect_sections() — {num: {titulo, inicio, fin}}.

    Returns
    -------
    str  Markdown enriquecido con bloques de imagen.
    """
    doc = fitz.open(pdf_path)
    nombre = os.path.splitext(os.path.basename(pdf_path))[0]
    os.makedirs(IMAGES_DIR, exist_ok=True)

    # Calcular el offset de carácter inicial de cada página en el texto plano
    page_starts: list[int] = []
    pos = 0
    for page in doc:
        page_starts.append(pos)
        pos += len(page.get_text("text")) + 1   # +1 por el "\n" del join

    # Acumular bloques Markdown por sección
    bloques_por_seccion: dict[int, list[str]] = {}
    contador_global = 0
    img_por_pagina: dict[int, int] = {}

    for page_idx, page in enumerate(doc):
        page_num = page_idx + 1
        imagenes = page.get_images(full=True)
        if not imagenes:
            continue

        sec_num, sec_titulo = _pagina_a_seccion(page_idx, page_starts, sections_map)
        img_por_pagina.setdefault(page_num, 0)

        for img_info in imagenes:
            xref = img_info[0]
            img_por_pagina[page_num] += 1
            contador_global += 1
            img_idx = img_por_pagina[page_num]

            # ── Extraer y guardar como PNG ────────────────────────────────────
            try:
                pix = fitz.Pixmap(doc, xref)
                if pix.n - pix.alpha > 3:       # CMYK → RGB
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                img_bytes = pix.tobytes("png")
            except Exception as exc:
                logger.warning("No se pudo extraer imagen xref=%d p%d: %s", xref, page_num, exc)
                continue

            img_filename = f"{nombre}_img_p{page_num}_i{img_idx}.png"
            img_path = os.path.join(IMAGES_DIR, img_filename)
            with open(img_path, "wb") as f:
                f.write(img_bytes)

            # ── OCR ───────────────────────────────────────────────────────────
            try:
                pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                ocr_raw = pytesseract.image_to_string(pil_img, lang="spa").strip()
                ocr_texto = " ".join(ocr_raw.split())   # colapsar espacios/saltos
            except Exception as exc:
                logger.warning("OCR falló para %s: %s", img_filename, exc)
                ocr_texto = ""

            if len(ocr_texto) > 10:
                nota = ocr_texto
            else:
                nota = "Elemento visual: Pictograma o gráfico de seguridad sin texto extenso"

            # ── Bloque Markdown de trazabilidad ───────────────────────────────
            bloque = (
                f"\n![Imagen {contador_global}](images/{img_filename})\n"
                f"> **Nota de trazabilidad:** {nota}\n"
                f"> Imagen en Sección {sec_num}: {sec_titulo}.\n"
                f"> Información relacionada en la sección correspondiente.\n"
            )
            bloques_por_seccion.setdefault(sec_num, []).append(bloque)

    logger.info(
        "Imágenes procesadas: %d en %d secciones.",
        contador_global,
        len(bloques_por_seccion),
    )

    # ── Inyectar bloques después del encabezado ## Sección N: ────────────────
    for sec_num, bloques in bloques_por_seccion.items():
        bloque_combinado = "".join(bloques)
        patron = re.compile(
            rf'^(## Sección {sec_num}: .+)$',
            re.MULTILINE | re.IGNORECASE,
        )
        # Solo reemplazar la primera aparición del encabezado
        md_content = patron.sub(
            lambda m, b=bloque_combinado: m.group(0) + b,
            md_content,
            count=1,
        )

    return md_content


# ── Pipeline principal ────────────────────────────────────────────────────────

def extract_to_markdown(pdf_path: str) -> str:
    """
    Convierte un PDF de FDS CORONA a Markdown estructurado con imágenes trazadas.

    Returns
    -------
    str  Contenido Markdown final. También persiste en data/processed/.
    """
    pdf_path = os.path.abspath(pdf_path)
    nombre = os.path.splitext(os.path.basename(pdf_path))[0]
    logger.info("Procesando: %s", os.path.basename(pdf_path))

    # 1. pymupdf4llm: texto + tablas en pipes
    md_raw = pymupdf4llm.to_markdown(pdf_path, write_images=False)
    logger.info("pymupdf4llm: %d caracteres.", len(md_raw))

    # 2. Texto plano → detect_sections()
    doc = fitz.open(pdf_path)
    texto_plano = "\n".join(page.get_text("text") for page in doc)
    secciones = detect_sections(texto_plano)
    logger.info("Secciones: %d / %d", len(secciones), len(SECCIONES_GHS))

    # 3. Normalizar encabezados → ## Sección N: [Título]
    md_normalizado = _normalizar_encabezados(md_raw)

    # 4. Extraer imágenes + OCR + inyectar bloques de trazabilidad
    md_final = extract_images_with_traceability(pdf_path, md_normalizado, secciones)

    # 5. Guardar en data/processed/
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    out_path = os.path.join(PROCESSED_DIR, f"{nombre}.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md_final)
    logger.info("Guardado: %s", out_path)

    return md_final


# ── Ejecución directa ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    RAW_DIR = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    )
    pdfs = sorted(f for f in os.listdir(RAW_DIR) if f.lower().endswith(".pdf"))

    if not pdfs:
        print("No hay PDFs en data/raw/")
        sys.exit(1)

    pdf_path = os.path.join(RAW_DIR, pdfs[0])
    md = extract_to_markdown(pdf_path)
    lineas = md.splitlines()

    # Mostrar primer bloque de imagen inyectado
    print("\n" + "=" * 70)
    print("PRIMER BLOQUE DE IMAGEN INYECTADO EN EL MARKDOWN")
    print("=" * 70)
    for i, linea in enumerate(lineas):
        if linea.startswith("![Imagen"):
            # Mostrar contexto: encabezado previo + bloque completo (hasta 8 líneas)
            inicio = max(0, i - 2)
            fragmento = lineas[inicio: i + 6]
            for l in fragmento:
                print(l)
            break
    else:
        print("(ningún bloque de imagen encontrado)")

    # Conteo resumen
    total_imgs = sum(1 for l in lineas if l.startswith("![Imagen"))
    print(f"\nTotal bloques de imagen en el .md: {total_imgs}")
