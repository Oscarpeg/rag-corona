"""
Fase 3 — Evaluación automatizada con métricas científicas.
Lee ground_truth.csv, ejecuta query_rag() sobre las 20 preguntas,
exporta resultados a resultados_rag.csv e imprime métricas finales.

Uso:
    python evaluation/run_evaluation.py

Modo piloto (primeras N preguntas):
    python evaluation/run_evaluation.py --limite 3
"""
import os
import sys
import csv
import re
import json
import time
import argparse

# Asegurar que src/ esté en el path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rag_pipeline import query_rag

EVAL_DIR     = os.path.dirname(__file__)
GROUND_TRUTH = os.path.join(EVAL_DIR, "ground_truth.csv")
RESULTADOS   = os.path.join(EVAL_DIR, "resultados_rag.csv")
METRICAS_JSON = os.path.join(EVAL_DIR, "metricas_rag.json")

_SEC_RE = re.compile(r"Secci[oó]n\s+(\d+)", re.IGNORECASE)
FALLBACK_PHRASE = "no encontré esta información"


def _extraer_secciones(texto: str) -> set[int]:
    """Extrae números de sección de un string como 'Sección 3 y Sección 8'."""
    return {int(m) for m in _SEC_RE.findall(texto)}


def _secciones_rag(fuentes: list[dict]) -> set[int]:
    """Números de sección citados por el RAG en sus fuentes recuperadas."""
    return {f["seccion_num"] for f in fuentes}


def _secciones_str(secciones: set[int]) -> str:
    return ", ".join(f"Sección {n}" for n in sorted(secciones))


def _cito_fuente(sec_esperadas: set[int], sec_rag: set[int]) -> bool:
    """True si al menos una sección esperada aparece entre las citadas por el RAG."""
    return bool(sec_esperadas & sec_rag)


def ejecutar_evaluacion(limite: int | None = None) -> list[dict]:
    filas = []
    with open(GROUND_TRUTH, encoding="utf-8", newline="") as f:
        filas = list(csv.DictReader(f))
    if limite:
        filas = filas[:limite]

    total = len(filas)
    SEP   = "=" * 72

    print(f"\n{SEP}")
    print(f"  RAG-CORONA — Evaluación automatizada · {total} pregunta(s)")
    print(f"{SEP}\n")

    registros = []

    for i, fila in enumerate(filas, 1):
        pregunta           = fila["pregunta"]
        respuesta_esperada = fila["respuesta_esperada"]
        seccion_fuente_str = fila["seccion_fuente"]

        print(f"[{i:02d}/{total:02d}] {pregunta[:70]}...")

        resultado   = query_rag(pregunta)
        respuesta_r = resultado["respuesta"]
        fuentes_r   = resultado["fuentes"]
        tiempo      = resultado["tiempo_seg"]

        sec_esperadas = _extraer_secciones(seccion_fuente_str)
        sec_rag       = _secciones_rag(fuentes_r)
        cito          = _cito_fuente(sec_esperadas, sec_rag)
        es_fallback   = FALLBACK_PHRASE in respuesta_r.lower()

        registro = {
            "pregunta":            pregunta,
            "respuesta_esperada":  respuesta_esperada,
            "respuesta_rag":       respuesta_r,
            "seccion_esperada":    seccion_fuente_str,
            "seccion_citada_por_rag": _secciones_str(sec_rag),
            "cito_fuente":         "Sí" if cito else "No",
            "es_fallback":         "Sí" if es_fallback else "No",
            "tiempo_seg":          tiempo,
        }
        registros.append(registro)

        estado = "✓" if cito else "✗"
        print(f"  {estado} Sección esperada: {seccion_fuente_str} | "
              f"RAG citó: {_secciones_str(sec_rag)} | "
              f"{'FALLBACK' if es_fallback else 'Respondido'} | {tiempo}s\n")

    return registros


def exportar_csv(registros: list[dict]) -> None:
    columnas = [
        "pregunta", "respuesta_esperada", "respuesta_rag",
        "seccion_esperada", "seccion_citada_por_rag", "cito_fuente",
    ]
    with open(RESULTADOS, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columnas, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(registros)
    print(f"  Exportado: {RESULTADOS}")


def calcular_metricas(registros: list[dict]) -> dict:
    total      = len(registros)
    cito_ok    = sum(1 for r in registros if r["cito_fuente"] == "Sí")
    fallbacks  = sum(1 for r in registros if r["es_fallback"] == "Sí")

    # Acierto exacto de sección: todas las secciones esperadas están en las citadas
    acierto_exacto = 0
    for r in registros:
        sec_esp = _extraer_secciones(r["seccion_esperada"])
        sec_rag = _extraer_secciones(r["seccion_citada_por_rag"])
        if sec_esp and sec_esp <= sec_rag:   # subset: todas esperadas están en RAG
            acierto_exacto += 1

    pct_cito    = round(cito_ok    / total * 100, 1) if total else 0.0
    pct_exacto  = round(acierto_exacto / total * 100, 1) if total else 0.0

    # 3 peores casos: los que no citaron fuente; desempate por más secciones perdidas
    def _gap(r: dict) -> int:
        sec_esp = _extraer_secciones(r["seccion_esperada"])
        sec_rag = _extraer_secciones(r["seccion_citada_por_rag"])
        return len(sec_esp - sec_rag)

    fallos = [r for r in registros if r["cito_fuente"] == "No"]
    peores = sorted(fallos, key=_gap, reverse=True)[:3]

    return {
        "total_preguntas":       total,
        "cito_fuente_correcta":  cito_ok,
        "pct_cito_fuente":       pct_cito,
        "acierto_exacto_seccion": acierto_exacto,
        "pct_acierto_exacto":    pct_exacto,
        "fallbacks":             fallbacks,
        "peores_casos":          [
            {
                "pregunta":         p["pregunta"][:80],
                "esperada":         p["seccion_esperada"],
                "citada_por_rag":   p["seccion_citada_por_rag"],
                "gap_secciones":    _gap(p),
            }
            for p in peores
        ],
    }


def imprimir_metricas(m: dict) -> None:
    SEP = "=" * 72
    print(f"\n{SEP}")
    print("  MÉTRICAS DE EVALUACIÓN RAG-CORONA")
    print(SEP)
    print(f"  Total preguntas evaluadas     : {m['total_preguntas']}")
    print(f"  Citó fuente correctamente     : {m['cito_fuente_correcta']} "
          f"({m['pct_cito_fuente']}%)")
    print(f"  Acierto exacto de sección     : {m['acierto_exacto_seccion']} "
          f"({m['pct_acierto_exacto']}%)")
    print(f"  Respuestas con fallback       : {m['fallbacks']}")
    print()
    print("  TOP 3 PEORES CASOS (mayor desfase de sección):")
    if not m["peores_casos"]:
        print("    Ninguno — todas las respuestas citaron fuente correctamente.")
    for i, p in enumerate(m["peores_casos"], 1):
        print(f"\n  [{i}] {p['pregunta']}")
        print(f"       Esperada   : {p['esperada']}")
        print(f"       RAG citó   : {p['citada_por_rag'] or '(ninguna)'}")
        print(f"       Gap        : {p['gap_secciones']} sección(es) perdida(s)")
    print(f"\n{SEP}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluador RAG — métricas científicas")
    parser.add_argument("--limite", type=int, default=None,
                        help="Limitar a las primeras N preguntas (default: todas)")
    args = parser.parse_args()

    registros = ejecutar_evaluacion(limite=args.limite)
    exportar_csv(registros)
    metricas  = calcular_metricas(registros)
    imprimir_metricas(metricas)

    with open(METRICAS_JSON, "w", encoding="utf-8") as f:
        json.dump(metricas, f, ensure_ascii=False, indent=2)
    print(f"  Métricas guardadas en: {METRICAS_JSON}\n")
