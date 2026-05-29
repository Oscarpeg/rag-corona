"""
Fase 2 — Pipeline RAG completo: retrieve → prompt → LLM.
Uso:
    from rag_pipeline import query_rag
    resultado = query_rag("¿Qué EPP se requiere para PINTURA TOTAL?")
"""
import os
import sys
import time

from langchain_ollama import OllamaLLM

sys.path.insert(0, os.path.dirname(__file__))
from retriever import retrieve
from rag_prompt import build_prompt

# GPU disponible → "llama3.1" (mejor español, más contexto)
# Solo CPU      → "llama3"   (más rápido para la demo)
MODEL = "llama3"

_llm = OllamaLLM(model=MODEL, temperature=0.1)


def query_rag(
    pregunta: str,
    top_k: int = 3,
    filtro_producto: str | None = None,
) -> dict:
    """
    Ejecuta el pipeline RAG completo.

    Parameters
    ----------
    pregunta         : pregunta en lenguaje natural.
    top_k            : número de chunks a recuperar.
    filtro_producto  : nombre exacto del producto para filtrar (opcional).

    Returns
    -------
    dict con claves: pregunta, respuesta, fuentes, tiempo_seg.
    """
    t0 = time.time()

    chunks  = retrieve(pregunta, top_k=top_k, filtro_producto=filtro_producto)
    prompt  = build_prompt(pregunta, chunks)
    respuesta = _llm.invoke(prompt)

    fuentes = [
        {
            "producto":      c["producto"],
            "seccion_num":   c["seccion_num"],
            "seccion_titulo": c["seccion_titulo"],
            "score":         c["score"],
        }
        for c in chunks
    ]

    return {
        "pregunta":   pregunta,
        "respuesta":  respuesta.strip(),
        "fuentes":    fuentes,
        "tiempo_seg": round(time.time() - t0, 2),
    }


# ── Prueba con 5 preguntas ────────────────────────────────────────────────────
if __name__ == "__main__":
    PREGUNTAS = [
        "¿Qué equipos de protección personal se requieren para manipular la pintura?",
        "¿Cuáles son los primeros auxilios en caso de contacto con los ojos?",
        "¿Cómo se debe almacenar correctamente el producto?",
        "¿Qué hacer en caso de derrame o vertido accidental?",
        "¿Cuáles son los peligros de inflamabilidad del producto?",
    ]

    SEP = "=" * 72

    print(f"\n{SEP}")
    print("  RAG-CORONA — Pipeline completo · 5 preguntas de prueba")
    print(f"  Modelo: {MODEL}  |  top_k=3")
    print(f"{SEP}\n")

    for i, pregunta in enumerate(PREGUNTAS, 1):
        print(f"[{i}/5] {pregunta}")
        print("-" * 72)

        resultado = query_rag(pregunta)

        print(f"RESPUESTA:\n{resultado['respuesta']}\n")
        print("FUENTES:")
        for f in resultado["fuentes"]:
            print(f"  · Sección {f['seccion_num']}: {f['seccion_titulo']}"
                  f" | {f['producto']}  (score={f['score']:.4f})")
        print(f"TIEMPO: {resultado['tiempo_seg']} seg")
        print(f"{SEP}\n")
