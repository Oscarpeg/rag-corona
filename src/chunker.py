"""
Paso 10 — Chunker semántico para fichas de seguridad CORONA.
Unidad base: sección GHS (## Sección N:). Sub-chunking por ### o ventana deslizante.
"""
import os
import re
from collections import defaultdict

# ── Constantes ────────────────────────────────────────────────────────────────
TOKEN_THRESHOLD = 400   # umbral para activar sub-chunking
CHUNK_WORDS     = 300   # tamaño de bloque en sub-chunking por palabras
OVERLAP_WORDS   = 50    # solapamiento entre bloques

_SEC_PAT    = re.compile(r'^## Sección (\d+): (.+)$', re.MULTILINE | re.IGNORECASE)
_SUBSEC_PAT = re.compile(r'^### .+$',                  re.MULTILINE)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _approx_tokens(text: str) -> float:
    """Estimación rápida de tokens: palabras × 1.3."""
    return len(text.split()) * 1.3


def _try_subsection_split(
    header_line: str,
    body: str,
    meta: dict,
) -> list[dict] | None:
    """
    Intenta dividir por sub-encabezados ###.
    Devuelve None si no existen ### en el cuerpo.
    """
    sub_matches = list(_SUBSEC_PAT.finditer(body))
    if not sub_matches:
        return None

    meta_sub = {**meta, "es_sub_chunk": True}
    chunks   = []

    # Texto entre el encabezado de sección y el primer ###
    preamble = body[: sub_matches[0].start()].strip()
    if preamble:
        chunks.append({
            "texto":    f"{header_line}\n{preamble}",
            "metadata": meta_sub,
        })

    # Cada bloque ### → siguiente ###
    for i, sm in enumerate(sub_matches):
        end      = sub_matches[i + 1].start() if i + 1 < len(sub_matches) else len(body)
        sub_text = body[sm.start() : end].strip()
        if sub_text:
            chunks.append({"texto": sub_text, "metadata": meta_sub})

    return chunks or None


def _word_split(text: str, meta: dict) -> list[dict]:
    """
    Divide el texto en bloques de CHUNK_WORDS palabras con OVERLAP_WORDS de solapamiento.
    Usado como fallback cuando no existen ### en secciones densas.
    """
    meta_sub = {**meta, "es_sub_chunk": True}
    words    = text.split()
    chunks   = []
    start    = 0

    while start < len(words):
        end        = min(start + CHUNK_WORDS, len(words))
        chunk_text = " ".join(words[start:end])
        chunks.append({"texto": chunk_text, "metadata": meta_sub})
        if end >= len(words):
            break
        start += CHUNK_WORDS - OVERLAP_WORDS

    return chunks


# ── Función principal ─────────────────────────────────────────────────────────

def chunk_markdown(md_path: str, producto: str) -> list[dict]:
    """
    Segmenta un archivo .md de FDS CORONA en chunks semánticos.

    Parameters
    ----------
    md_path  : ruta al archivo .md limpio en data/processed/
    producto : nombre del producto (va a metadata.producto)

    Returns
    -------
    list[dict]  cada elemento tiene claves "texto" y "metadata":
        {
          "texto": str,
          "metadata": {
            "producto":       str,
            "seccion_num":    int,
            "seccion_titulo": str,
            "fuente":         str,
            "es_sub_chunk":   bool
          }
        }
    """
    with open(md_path, encoding="utf-8") as f:
        content = f.read()

    fuente     = os.path.basename(md_path)
    all_chunks = []
    matches    = list(_SEC_PAT.finditer(content))

    for idx, m in enumerate(matches):
        sec_num    = int(m.group(1))
        sec_titulo = m.group(2).strip()
        header_line = m.group(0)

        # Extraer cuerpo hasta el siguiente encabezado de sección
        start     = m.start()
        end       = matches[idx + 1].start() if idx + 1 < len(matches) else len(content)
        full_text = content[start:end].strip()
        body      = content[m.end():end].strip()

        metadata_base = {
            "producto":       producto,
            "seccion_num":    sec_num,
            "seccion_titulo": sec_titulo,
            "fuente":         fuente,
            "es_sub_chunk":   False,
        }

        tokens = _approx_tokens(full_text)

        if tokens <= TOKEN_THRESHOLD:
            # ── Chunk único ──────────────────────────────────────────────────
            all_chunks.append({"texto": full_text, "metadata": metadata_base})

        else:
            # ── Sección densa: intentar sub-chunking ─────────────────────────
            sub_by_headers = _try_subsection_split(header_line, body, metadata_base)

            if sub_by_headers:
                # División por ### encontrada
                all_chunks.extend(sub_by_headers)
            else:
                # Fallback: ventana deslizante de palabras
                all_chunks.extend(_word_split(full_text, metadata_base))

    # ── Enriquecimiento de contexto en texto plano ────────────────────────────
    for chunk in all_chunks:
        prod    = chunk["metadata"]["producto"].strip()
        sec_num = chunk["metadata"]["seccion_num"]
        sec_tit = chunk["metadata"]["seccion_titulo"].strip()
        chunk["texto"] = (
            f"DOCUMENTO: {prod}\n"
            f"CONTEXTO SEGURIDAD GHS: {sec_num} {sec_tit}\n\n"
            f"{chunk['texto']}"
        )

    return all_chunks


# ── Prueba unitaria ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    MD_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed"))
    MD_FILE = "FDS 29 - PINTURA PRIMERA MANO & ACABADO - CORONA .md"
    md_path = os.path.join(MD_DIR, MD_FILE)

    if not os.path.exists(md_path):
        print(f"Archivo no encontrado: {md_path}")
        sys.exit(1)

    producto = "PINTURA PRIMERA MANO & ACABADO"
    chunks   = chunk_markdown(md_path, producto)

    # ── Recalcular tokens por sección para la tabla ───────────────────────────
    with open(md_path, encoding="utf-8") as f:
        content = f.read()

    matches  = list(_SEC_PAT.finditer(content))
    sec_info = {}

    for idx, m in enumerate(matches):
        num    = int(m.group(1))
        titulo = m.group(2).strip()
        start  = m.start()
        end    = matches[idx + 1].start() if idx + 1 < len(matches) else len(content)
        tokens = int(_approx_tokens(content[start:end].strip()))
        sec_info[num] = {"titulo": titulo, "tokens": tokens, "n_chunks": 0, "sub": False}

    for chunk in chunks:
        num = chunk["metadata"]["seccion_num"]
        if num in sec_info:
            sec_info[num]["n_chunks"] += 1
            if chunk["metadata"]["es_sub_chunk"]:
                sec_info[num]["sub"] = True

    # ── Imprimir tabla ────────────────────────────────────────────────────────
    COL = 42
    print(f"\n{'='*80}")
    print(f"  Chunker — {MD_FILE}")
    print(f"{'='*80}")
    print(f"{'Sec':>4}  {'Sección (título)':<{COL}}  {'Tokens':>8}  {'Chunks':>7}  Sub-chunk")
    print(f"{'-'*80}")

    for num in sorted(sec_info.keys()):
        d   = sec_info[num]
        sub = "Sí" if d["sub"] else "No"
        titulo_corto = d["titulo"][:COL]
        print(f"{num:>4}  {titulo_corto:<{COL}}  {d['tokens']:>8}  {d['n_chunks']:>7}  {sub}")

    print(f"{'='*80}")
    print(f"  Total chunks generados: {len(chunks)}")
    print(f"{'='*80}\n")
