"""
Paso 8 — Pipeline batch: convierte todos los PDFs de data/raw/ a Markdown estructurado.
Ejecutar desde la raíz del proyecto:
    ~/.conda/envs/rag-env/bin/python src/run_pipeline.py
"""
import os
import sys
import logging

sys.path.insert(0, os.path.dirname(__file__))
from pdf_extractor import extract_to_markdown

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.WARNING)

RAW_DIR       = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw"))
PROCESSED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed"))


# ── 1. Validación del Markdown generado ──────────────────────────────────────

def validate_markdown(md_path: str) -> dict:
    """
    Analiza el .md generado y devuelve métricas de calidad.

    Returns
    -------
    dict con claves:
        secciones_encontradas : int   — encabezados ## Sección N detectados
        tablas_markdown       : int   — líneas que contienen "|"
        notas_trazabilidad    : int   — bloques "> **Nota de trazabilidad:**"
        secciones_faltantes   : list  — números del 1 al 16 ausentes
    """
    with open(md_path, encoding="utf-8") as f:
        lineas = f.readlines()

    nums_encontrados = set()
    tablas = 0
    notas  = 0

    for linea in lineas:
        # Detectar ## Sección N:
        stripped = linea.strip()
        if stripped.lower().startswith("## sección"):
            import re
            m = re.search(r'##\s+secci[oó]n\s+(\d+)', stripped, re.IGNORECASE)
            if m:
                nums_encontrados.add(int(m.group(1)))
        # Contar filas de tabla (pipes)
        if "|" in stripped:
            tablas += 1
        # Contar notas de trazabilidad
        if "Nota de trazabilidad:" in linea:
            notas += 1

    faltantes = [n for n in range(1, 17) if n not in nums_encontrados]

    return {
        "secciones_encontradas": len(nums_encontrados),
        "tablas_markdown":       tablas,
        "notas_trazabilidad":    notas,
        "secciones_faltantes":   faltantes,
    }


# ── 2. Procesar un PDF individual ────────────────────────────────────────────

def process_pdf(pdf_path: str) -> bool:
    """
    Ejecuta el pipeline completo sobre un PDF y valida el resultado.

    Returns
    -------
    True si procesó sin excepciones, False si hubo error.
    """
    nombre_pdf = os.path.basename(pdf_path)
    nombre_md  = os.path.splitext(nombre_pdf)[0] + ".md"
    md_path    = os.path.join(PROCESSED_DIR, nombre_md)

    extract_to_markdown(pdf_path)

    metricas = validate_markdown(md_path)
    s = metricas["secciones_encontradas"]
    t = metricas["tablas_markdown"]
    z = metricas["notas_trazabilidad"]

    print(
        f"✓ {nombre_md} | "
        f"Secciones: {s}/16 | "
        f"Tablas: {t} | "
        f"Trazabilidad: {z} notas"
    )

    if s < 14:
        falt = metricas["secciones_faltantes"]
        print(f"  ⚠ WARNING: solo {s}/16 secciones — faltan: {falt}")

    return True


# ── 3. Batch sobre todos los PDFs ────────────────────────────────────────────

def process_all() -> None:
    """Recorre data/raw/, procesa cada PDF y resume los resultados."""
    pdfs = sorted(
        os.path.join(RAW_DIR, f)
        for f in os.listdir(RAW_DIR)
        if f.lower().endswith(".pdf")
    )

    if not pdfs:
        print("No se encontraron PDFs en data/raw/")
        return

    total   = len(pdfs)
    exitoso = 0
    fallido = []

    print(f"\n{'='*70}")
    print(f"  RAG-CORONA — Pipeline batch ({total} PDFs)")
    print(f"{'='*70}\n")

    for i, pdf_path in enumerate(pdfs, 1):
        print(f"[{i:02d}/{total}] {os.path.basename(pdf_path)}")
        try:
            process_pdf(pdf_path)
            exitoso += 1
        except Exception as exc:
            nombre = os.path.basename(pdf_path)
            print(f"  ✗ ERROR en {nombre}: {exc}")
            fallido.append(nombre)
        print()

    # Resumen final
    print(f"{'='*70}")
    print(f"  Completados: {exitoso}/{total}")
    if fallido:
        print(f"  Fallidos ({len(fallido)}):")
        for nombre in fallido:
            print(f"    - {nombre}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    process_all()
