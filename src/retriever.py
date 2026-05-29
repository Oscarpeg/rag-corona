"""
Fase 2 — Retriever semántico sobre la colección ChromaDB.
Uso:
    from retriever import retrieve
    resultados = retrieve("¿Qué EPP se recomienda para PINTURA TOTAL?", top_k=3)
"""
import os
import sys

import chromadb
from langchain_community.embeddings import OllamaEmbeddings

# ── Paths ─────────────────────────────────────────────────────────────────────
CHROMA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")
)
COLLECTION_NAME = "fichas_seguridad_corona"
EMBED_MODEL     = "nomic-embed-text"

# ── Singleton de embeddings ───────────────────────────────────────────────────
_embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url="http://localhost:11434")


def _get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection(COLLECTION_NAME)


def retrieve(
    query: str,
    top_k: int = 3,
    filtro_producto: str | None = None,
) -> list[dict]:
    """
    Recupera los chunks más relevantes para una consulta.

    Parameters
    ----------
    query            : pregunta en lenguaje natural.
    top_k            : número de resultados a retornar.
    filtro_producto  : nombre exacto del producto (campo 'producto' en metadata).

    Returns
    -------
    list[dict] con claves: texto, score, producto, seccion_num, seccion_titulo.
    """
    collection   = _get_collection()
    query_vector = _embeddings.embed_query(query)

    kwargs = {}
    if filtro_producto:
        kwargs["where"] = {"producto": filtro_producto}

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
        **kwargs,
    )

    salida = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        salida.append({
            "texto":         doc,
            "score":         round(1 - dist, 4),
            "producto":      meta.get("producto", ""),
            "seccion_num":   meta.get("seccion_num", 0),
            "seccion_titulo": meta.get("seccion_titulo", ""),
        })

    return salida


# ── Prueba de ejecución ───────────────────────────────────────────────────────
if __name__ == "__main__":
    CONSULTAS = [
        "¿Qué equipos de protección personal se recomiendan?",
        "¿Cuáles son los primeros auxilios en caso de ingestión?",
        "¿Cómo se debe almacenar y manipular el producto?",
    ]

    print(f"\n{'='*72}")
    print("  RAG-CORONA — Retriever · 3 consultas de prueba")
    print(f"{'='*72}\n")

    for i, consulta in enumerate(CONSULTAS, 1):
        print(f"[{i}] {consulta}")
        print("-" * 72)
        resultados = retrieve(consulta, top_k=3)
        if not resultados:
            print("  (sin resultados)\n")
            continue
        for r in resultados:
            print(f"  Score={r['score']:.4f} | {r['producto']} — Sección {r['seccion_num']}: {r['seccion_titulo']}")
            preview = r["texto"][:300].replace("\n", " ")
            print(f"  {preview}...\n")
        print()

    print(f"{'='*72}\n")
    sys.exit(0)
