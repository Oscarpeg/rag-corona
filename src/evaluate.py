"""
Fase 3 — Evaluación del pipeline RAG contra ground truth.
Modo piloto: procesa las primeras 3 preguntas y guarda en
evaluation/results_preview.json para verificar el pipeline sin colapsar CPU.

Uso completo (20 preguntas):
    python src/evaluate.py --all
"""
import os
import sys
import csv
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(__file__))
from rag_pipeline import query_rag

GROUND_TRUTH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "evaluation", "ground_truth.csv")
)
RESULTS_PREVIEW = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "evaluation", "results_preview.json")
)
RESULTS_FULL = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "evaluation", "results_full.json")
)


def cargar_ground_truth(limite: int | None = None) -> list[dict]:
    rows = []
    with open(GROUND_TRUTH, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limite is not None and i >= limite:
                break
            rows.append(row)
    return rows


def evaluar(filas: list[dict], salida: str) -> list[dict]:
    SEP = "=" * 72
    total = len(filas)
    resultados = []

    print(f"\n{SEP}")
    print(f"  RAG-CORONA — Evaluación · {total} pregunta(s)")
    print(f"{SEP}\n")

    for i, fila in enumerate(filas, 1):
        pregunta          = fila["pregunta"]
        respuesta_esperada = fila["respuesta_esperada"]
        seccion_fuente    = fila["seccion_fuente"]

        print(f"[{i:02d}/{total:02d}] {pregunta}")
        t0 = time.time()

        resultado = query_rag(pregunta)
        tiempo    = round(time.time() - t0, 2)

        entrada = {
            "id":                 i,
            "pregunta":           pregunta,
            "respuesta_esperada": respuesta_esperada,
            "seccion_fuente":     seccion_fuente,
            "respuesta_rag":      resultado["respuesta"],
            "fuentes_rag":        resultado["fuentes"],
            "tiempo_seg":         tiempo,
        }
        resultados.append(entrada)

        print(f"  Esperada : {respuesta_esperada[:120]}...")
        print(f"  Obtenida : {resultado['respuesta'][:120]}...")
        print(f"  Tiempo   : {tiempo}s\n")

    os.makedirs(os.path.dirname(salida), exist_ok=True)
    with open(salida, "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    print(f"{SEP}")
    print(f"  Resultados guardados en: {salida}")
    print(f"{SEP}\n")

    return resultados


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluador RAG — FDS CORONA")
    parser.add_argument(
        "--all", action="store_true",
        help="Procesar las 20 preguntas completas (lento en CPU)"
    )
    args = parser.parse_args()

    if args.all:
        filas   = cargar_ground_truth(limite=None)
        salida  = RESULTS_FULL
    else:
        filas   = cargar_ground_truth(limite=3)
        salida  = RESULTS_PREVIEW
        print("\n  Modo piloto: procesando las primeras 3 preguntas.")
        print("  Para evaluación completa usa: python src/evaluate.py --all\n")

    evaluar(filas, salida)
