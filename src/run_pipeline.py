"""
Paso 8 — Pipeline batch con limpieza genérica + correcciones específicas.
Ejecutar desde la raíz del proyecto:
    ~/.conda/envs/rag-env/bin/python src/run_pipeline.py
"""
import os
import re
import sys
import logging
import unicodedata

sys.path.insert(0, os.path.dirname(__file__))
from pdf_extractor import extract_to_markdown

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.WARNING)

RAW_DIR       = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw"))
PROCESSED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed"))


# ═══════════════════════════════════════════════════════════════════════════════
# LOGS GENÉRICOS — PASE 1: Deduplicación de bloques
# ═══════════════════════════════════════════════════════════════════════════════

def _dedup_paragraph_blocks(md: str) -> str:
    """
    Divide el doc en párrafos (bloques separados por líneas vacías).
    Si un bloque de 3+ líneas aparece más de una vez, conserva solo la primera
    aparición. Bloques de ≤2 líneas o puramente vacíos se ignoran.
    """
    # Separar en párrafos preservando los separadores vacíos
    raw_blocks = re.split(r'(\n{2,})', md)
    seen: set[str] = set()
    result: list[str] = []

    for block in raw_blocks:
        lines = [l for l in block.splitlines() if l.strip()]
        if len(lines) < 3:
            result.append(block)
            continue
        key = '\n'.join(l.strip() for l in lines)
        if key in seen:
            result.append('')          # reemplazar bloque duplicado con vacío
        else:
            seen.add(key)
            result.append(block)

    return ''.join(result)


def _dedup_table_vs_plaintext(md: str) -> str:
    """
    Dentro de ventanas de 50 líneas que contienen una tabla pipe (|),
    si existe un bloque de texto plano que comparte >60 % de sus palabras
    clave con la tabla, elimina el bloque de texto plano.
    """
    lines  = md.split('\n')
    n      = len(lines)
    remove = set()

    def _keywords(text: str) -> set:
        stop = {'para', 'que', 'con', 'como', 'este', 'esta', 'los', 'las',
                'del', 'por', 'una', 'uno', 'más', 'sus', 'sin', 'son'}
        return {w for w in re.findall(r'\b\w{4,}\b', text.lower()) if w not in stop}

    # Identificar rangos de tablas pipe
    table_ranges: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if lines[i].strip().startswith('|'):
            s = i
            while i < n and lines[i].strip().startswith('|'):
                i += 1
            table_ranges.append((s, i))
        else:
            i += 1

    for t_start, t_end in table_ranges:
        table_kw = _keywords(re.sub(r'[|*`]', '', '\n'.join(lines[t_start:t_end])))
        if len(table_kw) < 4:
            continue
        # Ventana de 50 líneas ANTES de la tabla
        win_start = max(0, t_start - 50)
        j = win_start
        while j < t_start:
            line = lines[j]
            if (not line.strip()
                    or line.strip().startswith(('|', '#', '>', '!', '*'))
                    or j in remove):
                j += 1
                continue
            # Inicio de bloque de texto plano
            p_start = j
            while j < t_start and lines[j].strip() and not lines[j].strip().startswith('|'):
                j += 1
            p_end = j
            if p_end - p_start < 2:
                continue
            plain_kw = _keywords('\n'.join(lines[p_start:p_end]))
            if not plain_kw:
                continue
            overlap = len(table_kw & plain_kw) / len(plain_kw)
            if overlap > 0.60:
                for k in range(p_start, p_end):
                    remove.add(k)

    return '\n'.join(l for idx, l in enumerate(lines) if idx not in remove)


# ═══════════════════════════════════════════════════════════════════════════════
# LOGS GENÉRICOS — PASE 2: Contaminación cruzada de sub-secciones
# ═══════════════════════════════════════════════════════════════════════════════

_SUBSEC_LINE = re.compile(r'^(?:\*{1,2})?\s*(\d+)\.(\d+)(?:\*{1,2})?\s')

def _fix_cross_contamination(md: str) -> str:
    """
    Dentro de cada ## Sección N:, elimina líneas cuyo número de sub-sección
    pertenece a una sección ANTERIOR (ej: "6.4" dentro de Sección 7).
    El bloque intruso se descarta hasta:
      - la siguiente línea vacía, O
      - una sub-sección de la sección ACTUAL (ej: "7.3" dentro de Sección 7).
    Esta segunda condición evita eliminar contenido propio de la sección cuando
    no hay línea vacía entre el bloque intruso y la siguiente sub-sección válida.
    """
    lines    = md.split('\n')
    result   = []
    cur_sec  = 0
    skipping = False

    for line in lines:
        sec_m = re.match(r'^## Sección (\d+):', line, re.IGNORECASE)
        if sec_m:
            cur_sec  = int(sec_m.group(1))
            skipping = False
            result.append(line)
            continue

        if not line.strip():
            skipping = False       # línea vacía cierra el bloque intruso
            result.append(line)
            continue

        if skipping:
            # Cerrar el bloque intruso si encontramos una sub-sección
            # que pertenece a la sección ACTUAL (no debe ser descartada)
            sub_m2 = _SUBSEC_LINE.match(line.strip())
            if sub_m2 and int(sub_m2.group(1)) == cur_sec:
                skipping = False
                result.append(line)
            # else: seguir descartando la línea intrusa
            continue

        if cur_sec > 1:
            sub_m = _SUBSEC_LINE.match(line.strip())
            if sub_m:
                ref_sec = int(sub_m.group(1))
                if 0 < ref_sec < cur_sec:
                    # Si la misma línea también contiene contenido de la sección
                    # actual (caso: 1.x y 2.x concatenados en un solo bloque de
                    # texto), conservar la línea en lugar de descartarla.
                    if re.search(rf'\b{cur_sec}\.\d+', line):
                        result.append(line)
                    else:
                        skipping = True
                    continue

        result.append(line)

    return '\n'.join(result)


# ═══════════════════════════════════════════════════════════════════════════════
# CORRECCIONES ESPECÍFICAS — Secciones duplicadas
# ═══════════════════════════════════════════════════════════════════════════════

def _fix_duplicate_sections(md: str) -> str:
    """Conserva solo la primera aparición de cada ## Sección N: y su contenido."""
    lines    = md.split('\n')
    seen     = set()
    result   = []
    skipping = False

    for line in lines:
        m = re.match(r'^## Sección (\d+):', line, re.IGNORECASE)
        if m:
            num = int(m.group(1))
            if num in seen:
                skipping = True
            else:
                seen.add(num)
                skipping = False
                result.append(line)
        elif not skipping:
            result.append(line)

    return '\n'.join(result)


# ═══════════════════════════════════════════════════════════════════════════════
# CORRECCIONES ESPECÍFICAS — Tablas duplicadas
# ═══════════════════════════════════════════════════════════════════════════════

def _fix_duplicate_tables(md: str) -> str:
    """Elimina bloques de tabla pipe con la misma fila de encabezado."""
    lines        = md.split('\n')
    result       = []
    seen_headers: set[str] = set()
    i = 0

    while i < len(lines):
        line = lines[i]
        if (line.strip().startswith('|')
                and i + 1 < len(lines)
                and re.match(r'^\s*\|[\s\-|:]+\|\s*$', lines[i + 1])):
            header_key = re.sub(r'[\s\*`<>]+', '', line.lower())
            if header_key in seen_headers:
                while i < len(lines) and lines[i].strip().startswith('|'):
                    i += 1
                continue
            seen_headers.add(header_key)

        result.append(line)
        i += 1

    return '\n'.join(result)


# ═══════════════════════════════════════════════════════════════════════════════
# CORRECCIONES ESPECÍFICAS — Contaminación de página (reforzada)
# ═══════════════════════════════════════════════════════════════════════════════

_NOISE = [
    # Encabezado de página repetido
    re.compile(r'^.*según Decreto 1496 de 2018.*$',              re.IGNORECASE | re.MULTILINE),
    # Pie de página limpio
    re.compile(r'^.*Emisión:.*?Versión:.*?Página\s+\d+/\d+.*$', re.IGNORECASE | re.MULTILINE),
    # Pie de página con codificación rota (ió + dígitos + Pá/Pa + dígitos)
    re.compile(
        r'^.{0,40}(?:[Ee]misi[oó][nñ]|[Vv]ersi[oó][nñ]).{0,50}'
        r'(?:[Pp][aá]gin[ao]|[Pp][aá]g\.?)\s*\d+.{0,30}$',
        re.IGNORECASE | re.MULTILINE,
    ),
    # "CONTINÚA EN LA SIGUIENTE PÁGINA" — con o sin guión de cierre
    re.compile(
        r'^.*-\s*CONTIN[ÚU]A\s+EN\s+LA\s+SIGUIENTE\s+P[ÁA]GINA.*$',
        re.IGNORECASE | re.MULTILINE,
    ),
    # Pie corrupto: letra suelta + espacios + fragmentos de fecha (p.ej. "E m i s i ó n  05/06/2019")
    re.compile(
        r'^[A-Za-z]\s+[a-z]\s+[a-z][oó]+\s+\d{2}/\d{2}/\d{4}.*$',
        re.MULTILINE,
    ),
    # Líneas de guiones largos (separadores de página)
    re.compile(r'^-{4,}\s*$', re.MULTILINE),
    # "Ficha de datos de seguridad" repetida como encabezado de página
    re.compile(r'^Ficha de datos de seguridad.*$', re.MULTILINE | re.IGNORECASE),
    # Marcadores de imagen omitida
    re.compile(r'^\*\*==> picture \[\d+ x \d+\] intentionally omitted <==\*\*$', re.MULTILINE),
    # Encabezado de producto repetido como H2 bold (p.ej. ## **PINTURA LAVABLE**)
    re.compile(r'^## \*\*[A-ZÁÉÍÓÚÑ &]+\*\*\s*$', re.MULTILINE),
    # Pie de página: "Emisión: ... Versión: ..." como texto plano
    re.compile(r'^Emisión:.*Versión:.*$', re.MULTILINE | re.IGNORECASE),
    # Pie de página: "**Página X/Y**" como texto plano bold
    re.compile(r'^\*\*Página \d+/\d+\*\*\s*$', re.MULTILINE),
    # Nombre de producto repetido como bold sin ## (p.ej. **PINTURA PRIMERA MANO**)
    re.compile(r'^\*\*[A-ZÁÉÍÓÚÑ &]+\*\*\s*$', re.MULTILINE),
]

def _fix_page_contamination(md: str) -> str:
    """Elimina ruido de cabeceras/pies de página; conserva solo el primer H1."""
    for pat in _NOISE:
        md = pat.sub('', md)

    # Conservar solo el primer título H1 del producto
    h1_matches = list(re.finditer(r'^(# .+)$', md, re.MULTILINE))
    if len(h1_matches) > 1:
        for m in reversed(h1_matches[1:]):
            md = md[:m.start()] + md[m.end():]

    return re.sub(r'\n{3,}', '\n\n', md)


# ═══════════════════════════════════════════════════════════════════════════════
# CORRECCIONES ESPECÍFICAS — Imágenes (general + EPP Sección 8)
# ═══════════════════════════════════════════════════════════════════════════════

_H_CODES   = re.compile(r'\b(H\d{3}(?:\+H\d{3})*)\b')
_NFPA_VALS = re.compile(
    r'Salud:\s*(\d+).*?Inflamabilidad:\s*(\d+).*?Inestabilidad:\s*(\d+)',
    re.DOTALL | re.IGNORECASE,
)
_EPP_GENERIC = [
    ('máscara autofiltrante',   'Mascarilla/respirador autofiltrante'),
    ('máscara',                 'Mascarilla/respirador'),
    ('mascara',                 'Mascarilla/respirador'),
    ('guantes no desechables',  'Guantes de protección química no desechables'),
    ('guantes',                 'Guantes de protección química'),
    ('pantalla facial',         'Pantalla facial'),
    ('calzado de seguridad',    'Calzado de seguridad contra riesgo químico'),
    ('calzado',                 'Calzado de seguridad'),
    ('prenda de protección',    'Prenda de protección química'),
    ('gafas',                   'Gafas de seguridad'),
]

_IMG_BLOCK = re.compile(
    r'(!\[Imagen (\d+)\]\(images/([^)\n]+_img_p(\d+)_i(\d+)\.png)\)\n'
    r'> \*\*Nota de trazabilidad:\*\* (.*?)\n'
    r'> Imagen en Sección (\d+): (.+?)\.\n'
    r'> Información relacionada en la sección correspondiente\.\n)',
    re.MULTILINE,
)


def _build_block(img_line: str, nota: str, sec_num: str, sec_title: str) -> str:
    return (
        f"{img_line}\n"
        f"> **Nota de trazabilidad:** {nota}\n"
        f"> Imagen en Sección {sec_num}: {sec_title}.\n"
        f"> Información relacionada en la sección correspondiente.\n"
    )


def _fix_image_descriptions(md: str) -> str:
    """Enriquece o elimina bloques de imagen según contexto (±400 chars)."""

    def _classify(match: re.Match) -> str:
        img_num   = match.group(2)
        filename  = match.group(3)
        p_num     = int(match.group(4))
        i_num     = int(match.group(5))
        ocr_note  = match.group(6).strip()
        sec_num   = match.group(7)
        sec_title = match.group(8)
        img_line  = f"![Imagen {img_num}](images/{filename})"

        pos          = match.start()
        ctx_before   = md[max(0, pos - 400): pos]
        ctx_after    = md[match.end(): min(len(md), match.end() + 400)]
        ctx_full     = ctx_before + ctx_after
        ctx_low      = ctx_full.lower()
        near_before  = md[max(0, pos - 120): pos].lower()

        # Regla 1: Logos de cabecera repetidos (páginas 2+ imagen i1 en header)
        if (i_num == 1 and p_num >= 2
                and any(kw in near_before for kw in ('corona.co', 'corlanc', 'decreto 1496'))):
            return ''

        # Regla 2: Diamante NFPA
        nfpa_m = _NFPA_VALS.search(ctx_full)
        if nfpa_m and 'nfpa' in ctx_low:
            s, inf, inst = nfpa_m.group(1), nfpa_m.group(2), nfpa_m.group(3)
            nota = f"Diamante NFPA 704: Salud {s} / Inflamabilidad {inf} / Inestabilidad {inst}."
            return _build_block(img_line, nota, sec_num, sec_title)

        # Regla 3: Pictograma GHS/SGA
        h_codes = list(dict.fromkeys(_H_CODES.findall(ctx_full)))
        if h_codes or ('sga' in ctx_low and ('pictograma' in ctx_low or 'atención' in ctx_low)):
            codigos = ', '.join(h_codes) if h_codes else 'SGA'
            return _build_block(img_line, nota=f"Pictograma(s) GHS: {codigos}.",
                                sec_num=sec_num, sec_title=sec_title)

        # Regla 4: EPP genérico (±400 chars)
        for kw, label in _EPP_GENERIC:
            if kw in ctx_low:
                return _build_block(img_line,
                                    nota=f"Pictograma EPP: {label} — uso obligatorio.",
                                    sec_num=sec_num, sec_title=sec_title)

        # Fallback
        nota = ocr_note if len(ocr_note) > 15 and 'pictograma' not in ocr_note.lower() \
               else "Elemento visual sin texto identificable."
        return _build_block(img_line, nota, sec_num, sec_title)

    return _IMG_BLOCK.sub(_classify, md)


def _fix_epp_images_sec8(md: str) -> str:
    """
    Mejora específica para imágenes de Sección 8:
    analiza 300 chars POSTERIORES a cada imagen; si detecta el EPP exacto,
    reemplaza la nota de trazabilidad con la descripción normativa completa.
    Como fallback recorre el cuerpo completo de la sección en orden de aparición.
    """
    # Límites de Sección 8
    s8_m = re.search(r'^## Sección 8:', md, re.MULTILINE | re.IGNORECASE)
    s9_m = re.search(r'^## Sección 9:', md, re.MULTILINE | re.IGNORECASE)
    if not s8_m:
        return md

    s8_start = s8_m.start()
    s8_end   = s9_m.start() if s9_m else len(md)
    sec8     = md[s8_start:s8_end]

    # EPP esperados en Sección 8 con sus notas normativas exactas
    EPP_RULES: list[tuple[str, str]] = [
        (
            'máscara autofiltrante',
            "Pictograma EPP: Protección respiratoria — "
            "Máscara autofiltrante para gases y vapores. Uso obligatorio.",
        ),
        (
            'guantes no desechables',
            "Pictograma EPP: Protección de manos — "
            "Guantes no desechables de protección química. Uso obligatorio.",
        ),
        (
            'pantalla facial',
            "Pictograma EPP: Protección ocular y facial — "
            "Pantalla facial. Uso obligatorio en caso de riesgo de salpicaduras.",
        ),
    ]

    # Orden de aparición de EPP en el cuerpo de Sección 8 (fallback)
    sec8_low   = sec8.lower()
    epp_queue  = [nota for kw, nota in EPP_RULES if kw in sec8_low]

    # Reemplazar nota de cada imagen de Sección 8
    img_count  = [0]   # mutable counter para closure

    def _replace_sec8(match: re.Match) -> str:
        img_num   = match.group(2)
        filename  = match.group(3)
        sec_num   = match.group(7)
        sec_title = match.group(8)
        img_line  = f"![Imagen {img_num}](images/{filename})"

        # 300 chars DESPUÉS del bloque de imagen (dentro de sec8 relativo)
        rel_end   = match.end()
        after_300 = sec8[rel_end: rel_end + 300].lower()

        nota = None
        for kw, nota_epp in EPP_RULES:
            if kw in after_300:
                nota = nota_epp
                break

        # Fallback: asignar en orden de aparición
        if nota is None and epp_queue:
            nota = epp_queue.pop(0)

        if nota is None:
            nota = match.group(6).strip() or "Elemento visual sin texto identificable."

        return _build_block(img_line, nota, sec_num, sec_title)

    new_sec8 = _IMG_BLOCK.sub(_replace_sec8, sec8)
    return md[:s8_start] + new_sec8 + md[s8_end:]


# ═══════════════════════════════════════════════════════════════════════════════
# CORRECCIONES ESPECÍFICAS — Sección 2 duplicada
# ═══════════════════════════════════════════════════════════════════════════════

# Señales de contenido NFPA o cabecera 2.1/2.2 en texto plano
_NFPA_SIGNALS = re.compile(
    r'^(?:\*{0,2}NFPA\*{0,2}:\s*$'
    r'|Salud:\s*\d'
    r'|Inflamabilidad:\s*\d'
    r'|Inestabilidad:\s*\d'
    r'|Especiales:\s*'
    r'|\*{0,2}2\.[12]\b)',
    re.IGNORECASE,
)


def _fix_sec2_duplicate(md: str) -> str:
    """
    Dentro de Sección 2, elimina la segunda aparición de los bloques 2.1 NFPA/SGA
    y 2.2 elementos de etiqueta que reaparece después de '2.3 Otros peligros'.
    Este duplicado surge del corte de página que repite el contenido de la página
    anterior. Se detecta por cabeceras 2.1/2.2 o por valores NFPA en texto plano
    (solo actúa sobre líneas sin '|' ni '<br>' para no afectar celdas de tabla).
    """
    lines = md.split('\n')
    result: list[str] = []
    in_sec2 = False
    past_23 = False
    skipping = False

    for line in lines:
        sec_m = re.match(r'^## Sección (\d+)\b', line, re.IGNORECASE)
        if sec_m:
            in_sec2 = (int(sec_m.group(1)) == 2)
            past_23 = False
            skipping = False
            result.append(line)
            continue

        # Marcar cuando hemos pasado la cabecera 2.3
        if in_sec2 and not past_23:
            if (re.match(r'^\*{0,2}2\.3\b', line.strip())
                    and '<br>' not in line and '|' not in line):
                past_23 = True

        # Después de 2.3: detectar inicio del bloque duplicado (NFPA o 2.1/2.2)
        if (in_sec2 and past_23 and not skipping
                and line.strip()
                and '<br>' not in line
                and '|' not in line
                and _NFPA_SIGNALS.match(line.strip())):
            skipping = True
            continue

        if skipping:
            # Terminar al llegar al siguiente ## Sección
            if re.match(r'^## Sección \d+\b', line, re.IGNORECASE):
                in_sec2 = False
                skipping = False
                result.append(line)
            # else: descartar línea duplicada
        else:
            result.append(line)

    return '\n'.join(result)


# ═══════════════════════════════════════════════════════════════════════════════
# CORRECCIONES ESPECÍFICAS — Problemas residuales
# ═══════════════════════════════════════════════════════════════════════════════

def _fix_sec9_olor_duplicate(md: str) -> str:
    """
    Elimina el bloque parcial de Sección 9 que comienza con 'Olor: <valor>'
    en texto plano (artefacto de continuación de página: el PDF repite
    propiedades físicas desde 'Olor:' en la página siguiente).
    Solo actúa sobre líneas que NO están dentro de celda de tabla (sin | ni <br>).
    """
    lines = md.split('\n')
    result: list[str] = []
    in_sec9 = False
    past_main_block = False
    skipping = False

    for line in lines:
        sec_m = re.match(r'^## Sección (\d+)\b', line, re.IGNORECASE)
        if sec_m:
            in_sec9 = (int(sec_m.group(1)) == 9)
            past_main_block = False
            skipping = False
            result.append(line)
            continue

        # El bloque principal de propiedades termina con esta frase
        if in_sec9 and not past_main_block:
            if 'No relevante debido a la naturaleza del producto' in line:
                past_main_block = True

        # Detectar duplicado: "Olor:" como línea sola, fuera de tabla/br
        if (in_sec9 and past_main_block and not skipping
                and re.match(r'^Olor\s*:', line.strip())
                and '|' not in line and '<br>' not in line):
            skipping = True
            continue

        if skipping:
            # Detener al llegar al siguiente ## Sección N:
            if re.match(r'^## Sección \d+\b', line, re.IGNORECASE):
                in_sec9 = False
                skipping = False
                result.append(line)
        else:
            result.append(line)

    return '\n'.join(result)


def _fix_sec10_4_plain_dup(md: str) -> str:
    """
    Elimina la segunda aparición de '10.4 Condiciones que deben evitarse:'
    (versión texto plano), conservando solo la primera (que va seguida de tabla pipe).
    Ignora ocurrencias embebidas en bloques <br> (picture text).
    """
    _PAT_104 = re.compile(
        r'^\*{0,2}10\.4\s+Condiciones[^\n]*\*{0,2}\s*$',
        re.IGNORECASE | re.MULTILINE,
    )
    # Filtrar matches que estén dentro de líneas con <br> (picture text)
    lines = md.split('\n')
    standalone_positions: list[int] = []
    offset = 0
    for line in lines:
        if _PAT_104.match(line.strip()) and '<br>' not in line:
            standalone_positions.append(offset)
        offset += len(line) + 1  # +1 para el '\n'

    if len(standalone_positions) < 2:
        return md

    # Eliminar desde el inicio del segundo match hasta el siguiente 10.5 o ## Sección
    start = standalone_positions[1]
    _END = re.compile(
        r'(?:^\*{0,2}10\.5\b|^## Sección \d+\b)',
        re.IGNORECASE | re.MULTILINE,
    )
    end_m = _END.search(md, start + 1)
    end = end_m.start() if end_m else len(md)

    return md[:start] + md[end:]


def _fix_fin_ficha_duplicate(md: str) -> str:
    """
    Conserva solo la primera aparición de 'FIN DE LA FICHA DE DATOS DE SEGURIDAD'
    y elimina todo lo que viene después (incluyendo la segunda aparición).
    """
    m = re.search(
        r'^FIN DE LA FICHA DE DATOS DE SEGURIDAD\s*$',
        md,
        re.MULTILINE | re.IGNORECASE,
    )
    if m:
        md = md[: m.end()].rstrip('\n') + '\n'
    return md


_RE_PICTURE_TEXT = re.compile(
    r'\*\*-{5} Start of picture text -{5}\*\*(?:<br>)?\n(.*?)\n?\*\*-{5} End of picture text -{5}\*\*(?:<br>)?',
    re.DOTALL,
)

def _extract_picture_text_blocks(md: str) -> str:
    """Extrae contenido atrapado entre marcadores picture text, convierte <br>
    a saltos de línea y elimina los marcadores."""
    def _replace(m):
        content = m.group(1).replace('<br>', '\n').strip()
        return content + '\n'
    return _RE_PICTURE_TEXT.sub(_replace, md)


_RE_SECTION_CONTINUATION = re.compile(
    r'^\|.*SECCI[ÓO]N\s+\d+.*contin[úu].*\|.*$',
    re.MULTILINE | re.IGNORECASE,
)

def _fix_section_continuation_tables(md: str) -> str:
    """Elimina filas de tabla que son bloques 'SECCIÓN X (continúa)' filtrados
    desde Sección 8 dentro de otra sección."""
    return _RE_SECTION_CONTINUATION.sub('', md)


# ═══════════════════════════════════════════════════════════════════════════════
# CORRECCIONES ESPECÍFICAS — Sub-secciones letradas duplicadas (Sección 11 y similares)
# ═══════════════════════════════════════════════════════════════════════════════

_LETTERED_SUBSEC = re.compile(r'(^[A-H][-–]\s+\w)', re.MULTILINE)

def _fix_duplicate_lettered_subsections(md: str) -> str:
    """
    Elimina apariciones duplicadas de sub-secciones letradas del tipo
    'A- Título', 'B– Título', etc. (comunes en Sección 11 y 16).
    Conserva la primera aparición completa; descarta la siguiente y su contenido
    hasta el próximo encabezado ## o sub-sección letrada.
    """
    lines  = md.split('\n')
    seen   = set()
    result = []
    skip   = False

    for line in lines:
        m = _LETTERED_SUBSEC.match(line)
        if m:
            key = re.sub(r'\s+', ' ', line.strip().lower())
            if key in seen:
                skip = True
                continue
            seen.add(key)
            skip = False
            result.append(line)
        elif skip:
            # Cerrar bloque saltado al llegar a un nuevo encabezado o sub-sección
            if re.match(r'^##', line) or _LETTERED_SUBSEC.match(line) or not line.strip():
                skip = False
                result.append(line)
            # else: seguir descartando
        else:
            result.append(line)

    return '\n'.join(result)


# ═══════════════════════════════════════════════════════════════════════════════
# CORRECCIONES ESPECÍFICAS — Encabezados H2 falsos
# ═══════════════════════════════════════════════════════════════════════════════

_FALSE_H2 = re.compile(
    r'^## (\*\*.+|[A-H]- .+)$',
    re.MULTILINE,
)
_REAL_SECTION = re.compile(r'Secci[oó]n\s+\d+', re.IGNORECASE)


def _fix_false_headers(md: str) -> str:
    """
    Convierte encabezados H2 falsos a texto negrita normal.
    Un H2 es falso si empieza con '## **' y NO contiene 'Sección N'.
    Ejemplo: '## **Por inhalación:**' → '**Por inhalación:**'
    Los encabezados reales (## Sección N:) se conservan intactos.
    """
    def _replace(m: re.Match) -> str:
        content = m.group(1)          # todo lo que sigue a "## "
        if _REAL_SECTION.search(content):
            return m.group(0)         # encabezado real → no tocar
        return content                # falso → quitar "## "

    return _FALSE_H2.sub(_replace, md)


# ═══════════════════════════════════════════════════════════════════════════════
# ORQUESTADOR
# ═══════════════════════════════════════════════════════════════════════════════

def clean_markdown(md: str) -> str:
    """
    Aplica en orden los pases genéricos de NLP y luego las correcciones específicas.

    Orden:
    0.   Normalización Unicode NFC (blindar regex con tildes)
    0b.  Encabezados H2 falsos → texto negrita (## **X** → **X**)
    1.   Deduplicación genérica de párrafos
    2.   Deduplicación tabla vs. texto plano
    3.   Contaminación cruzada de sub-secciones [corregido: no elimina sub-sección propia]
    4.   Secciones duplicadas
    5.   Contaminación de página (cabeceras/pies)
    5b.  Sección 2 duplicada (2.1/2.2 que reaparecen tras 2.3)
    5c.  Sub-secciones letradas duplicadas (A-, B-, ...)
    6.   Tablas pipe duplicadas
    7.   Descripción general de imágenes
    8.   Mejora EPP específica Sección 8
    9.   Bloque parcial Sección 9 (Olor duplicado por corte de página)
    10.  Sección 10.4 texto plano duplicado
    11.  FIN DE LA FICHA duplicado — truncar tras primera aparición
    """
    md = unicodedata.normalize('NFC', md)
    md = _fix_false_headers(md)
    md = _dedup_paragraph_blocks(md)
    md = _dedup_table_vs_plaintext(md)
    md = _fix_cross_contamination(md)
    md = _fix_duplicate_sections(md)
    md = _fix_page_contamination(md)
    md = _extract_picture_text_blocks(md)
    md = _fix_section_continuation_tables(md)
    md = _fix_sec2_duplicate(md)
    md = _fix_duplicate_lettered_subsections(md)
    md = _fix_duplicate_tables(md)
    md = _fix_image_descriptions(md)
    md = _fix_epp_images_sec8(md)
    md = _fix_sec9_olor_duplicate(md)
    md = _fix_sec10_4_plain_dup(md)
    md = _fix_fin_ficha_duplicate(md)
    return md


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

def validate_markdown(md_path: str) -> dict:
    with open(md_path, encoding='utf-8') as f:
        lineas = f.readlines()

    nums = set()
    tablas = notas = 0

    for linea in lineas:
        s = linea.strip()
        if s.lower().startswith('## sección'):
            m = re.search(r'##\s+secci[oó]n\s+(\d+)', s, re.IGNORECASE)
            if m:
                nums.add(int(m.group(1)))
        if '|' in s:
            tablas += 1
        if 'Nota de trazabilidad:' in linea:
            notas += 1

    return {
        'secciones_encontradas': len(nums),
        'tablas_markdown':       tablas,
        'notas_trazabilidad':    notas,
        'secciones_faltantes':   [n for n in range(1, 17) if n not in nums],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE POR PDF
# ═══════════════════════════════════════════════════════════════════════════════

def process_pdf(pdf_path: str) -> bool:
    nombre_pdf = os.path.basename(pdf_path)
    nombre_md  = os.path.splitext(nombre_pdf)[0] + '.md'
    md_path    = os.path.join(PROCESSED_DIR, nombre_md)

    md_raw   = extract_to_markdown(pdf_path)
    md_clean = clean_markdown(md_raw)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_clean)

    m = validate_markdown(md_path)
    s, t, z = m['secciones_encontradas'], m['tablas_markdown'], m['notas_trazabilidad']
    print(f"  ✓ {nombre_md} | Secciones: {s}/16 | Tablas: {t} | Trazabilidad: {z} notas")

    if s < 14:
        print(f"    ⚠ WARNING: solo {s}/16 secciones — faltan: {m['secciones_faltantes']}")

    return True


# ═══════════════════════════════════════════════════════════════════════════════
# BATCH
# ═══════════════════════════════════════════════════════════════════════════════

def process_all() -> None:
    pdfs = sorted(
        os.path.join(RAW_DIR, f)
        for f in os.listdir(RAW_DIR)
        if f.lower().endswith('.pdf')
    )
    if not pdfs:
        print('No se encontraron PDFs en data/raw/')
        return

    total, exitoso, fallido = len(pdfs), 0, []

    print(f"\n{'='*70}")
    print(f"  RAG-CORONA — Pipeline batch v2 — NLP genérico + correcciones ({total} PDFs)")
    print(f"{'='*70}\n")

    for i, pdf_path in enumerate(pdfs, 1):
        print(f"[{i:02d}/{total}] {os.path.basename(pdf_path)}")
        try:
            process_pdf(pdf_path)
            exitoso += 1
        except Exception as exc:
            print(f"  ✗ ERROR: {exc}")
            fallido.append(os.path.basename(pdf_path))
        print()

    print(f"{'='*70}")
    print(f"  Completados: {exitoso}/{total}")
    if fallido:
        print(f"  Fallidos ({len(fallido)}):")
        for n in fallido:
            print(f"    - {n}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    process_all()
