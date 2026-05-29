"""
Paso 9 — Indexador ChromaDB con embeddings nomic-embed-text vía Ollama.
Ejecutar desde la raíz del proyecto:
    ~/.conda/envs/rag-env/bin/python src/indexer.py
"""
import os
import sys
import re
import hashlib
import logging

import chromadb
from chromadb.errors import NotFoundError as ChromaNotFoundError
import ollama

sys.path.insert(0, os.path.dirname(__file__))
from chunker import chunk_markdown

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.WARNING)

PROCESSED_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "processed")
)
CHROMA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")
)

EMBED_MODEL      = "nomic-embed-text"
COLLECTION_NAME  = "fichas_seguridad_corona"
BATCH_SIZE       = 32   # chunks por lote al insertar en ChromaDB

_PRODUCT_PAT = re.compile(r'^#\s+(.+)$', re.MULTILINE)


def _extract_product_name(md_path: str) -> str:
    """Extrae el nombre del producto del H1 del markdown; fallback al nombre de archivo."""
    with open(md_path, encoding="utf-8") as f:
        content = f.read(2000)
    m = _PRODUCT_PAT.search(content)
    if m:
        return m.group(1).strip()
    return os.path.splitext(os.path.basename(md_path))[0]


def _chunk_id(producto: str, sec_num: int, chunk_idx: int) -> str:
    """ID determinista por (producto, sección, índice)."""
    raw = f"{producto}|{sec_num}|{chunk_idx}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def get_embedding(text: str) -> list[float]:
    """Genera embedding con nomic-embed-text vía Ollama."""
    resp = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return resp["embedding"]


def _upsert(collection: chromadb.Collection, client: chromadb.PersistentClient,
            ids, documents, metadatas, embeddings) -> chromadb.Collection:
    """Upsert con un reintento si ChromaDB devuelve referencia stale."""
    try:
        collection.upsert(ids=ids, documents=documents,
                          metadatas=metadatas, embeddings=embeddings)
    except ChromaNotFoundError:
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        collection.upsert(ids=ids, documents=documents,
                          metadatas=metadatas, embeddings=embeddings)
    return collection


def build_index(reset: bool = False) -> chromadb.Collection:
    """
    Indexa todos los .md de data/processed/ en ChromaDB.

    Parameters
    ----------
    reset : si True, elimina la colección existente antes de indexar.

    Returns
    -------
    chromadb.Collection  colección ya populada.
    """
    os.makedirs(CHROMA_DIR, exist_ok=True)

    if reset:
        # Crear cliente temporal solo para borrar; luego instanciar uno nuevo
        _tmp = chromadb.PersistentClient(path=CHROMA_DIR)
        if COLLECTION_NAME in [c.name for c in _tmp.list_collections()]:
            _tmp.delete_collection(COLLECTION_NAME)
            print(f"  Colección '{COLLECTION_NAME}' eliminada.")
        del _tmp   # liberar antes de crear el nuevo cliente

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    md_files = sorted(
        os.path.join(PROCESSED_DIR, f)
        for f in os.listdir(PROCESSED_DIR)
        if f.endswith(".md") and not f.startswith(".")
    )

    if not md_files:
        print("No se encontraron archivos .md en data/processed/")
        return collection

    # Checkpointing: IDs ya presentes en la colección → se saltarán
    existing_ids = set(collection.get(include=[])["ids"])
    if existing_ids:
        print(f"  Checkpointing: {len(existing_ids)} chunks ya indexados — se saltarán.\n")

    total_docs  = 0
    total_files = len(md_files)

    print(f"\n{'='*70}")
    print(f"  RAG-CORONA — Indexador ChromaDB ({total_files} archivos)")
    print(f"  Modelo de embedding: {EMBED_MODEL}")
    print(f"  Colección: {COLLECTION_NAME}")
    print(f"{'='*70}\n")

    for file_idx, md_path in enumerate(md_files, 1):
        nombre = os.path.basename(md_path)
        producto = _extract_product_name(md_path)

        chunks = chunk_markdown(md_path, producto)
        if not chunks:
            print(f"[{file_idx:02d}/{total_files}] ⚠ Sin chunks: {nombre}")
            continue

        # Preparar lotes
        ids_batch: list[str]        = []
        docs_batch: list[str]       = []
        meta_batch: list[dict]      = []
        emb_batch:  list[list[float]] = []

        for chunk_idx, chunk in enumerate(chunks):
            cid  = _chunk_id(producto, chunk["metadata"]["seccion_num"], chunk_idx)
            if cid in existing_ids:
                continue  # ya indexado
            emb  = get_embedding(chunk["texto"])

            # ChromaDB requiere valores de metadata de tipos simples
            meta = {
                "producto":       str(chunk["metadata"]["producto"]),
                "seccion_num":    int(chunk["metadata"]["seccion_num"]),
                "seccion_titulo": str(chunk["metadata"]["seccion_titulo"]),
                "fuente":         str(chunk["metadata"]["fuente"]),
                "es_sub_chunk":   int(chunk["metadata"]["es_sub_chunk"]),
            }

            ids_batch.append(cid)
            docs_batch.append(chunk["texto"])
            meta_batch.append(meta)
            emb_batch.append(emb)

            # Insertar en lotes
            if len(ids_batch) >= BATCH_SIZE:
                collection = _upsert(collection, client,
                                     ids_batch, docs_batch, meta_batch, emb_batch)
                ids_batch  = []
                docs_batch = []
                meta_batch = []
                emb_batch  = []

        # Insertar el lote final
        if ids_batch:
            collection = _upsert(collection, client,
                                 ids_batch, docs_batch, meta_batch, emb_batch)

        total_docs += len(chunks)
        print(
            f"  ✓ [{file_idx:02d}/{total_files}] {nombre[:55]:<55} "
            f"| {len(chunks):>3} chunks"
        )

    print(f"\n{'='*70}")
    print(f"  Total documentos indexados: {total_docs}")
    print(f"  Colección ChromaDB:         {COLLECTION_NAME}")
    print(f"  Persistido en:              {CHROMA_DIR}")
    print(f"{'='*70}\n")

    return collection


def query_index(
    pregunta: str,
    n_results: int = 5,
    filtro_producto: str | None = None,
) -> list[dict]:
    """
    Consulta semántica sobre la colección ChromaDB.

    Parameters
    ----------
    pregunta        : texto de la pregunta en lenguaje natural.
    n_results       : número de resultados a retornar.
    filtro_producto : si se indica, filtra por nombre de producto exacto.

    Returns
    -------
    list[dict]  resultados con claves 'texto', 'score', 'metadata'.
    """
    client     = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(COLLECTION_NAME)

    emb    = get_embedding(pregunta)
    where  = {"producto": filtro_producto} if filtro_producto else None

    resultados = collection.query(
        query_embeddings=[emb],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    salida = []
    for doc, meta, dist in zip(
        resultados["documents"][0],
        resultados["metadatas"][0],
        resultados["distances"][0],
    ):
        salida.append({
            "texto":    doc,
            "score":    round(1 - dist, 4),   # similitud coseno
            "metadata": meta,
        })

    return salida


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Indexador ChromaDB — FDS CORONA")
    parser.add_argument("--reset",  action="store_true", help="Eliminar colección antes de indexar")
    parser.add_argument("--query",  type=str, default=None, help="Consulta de prueba tras indexar")
    parser.add_argument("--top-k",  type=int, default=3,    help="Número de resultados (default: 3)")
    args = parser.parse_args()

    build_index(reset=args.reset)

    if args.query:
        print(f"\nConsulta: «{args.query}»\n")
        resultados = query_index(args.query, n_results=args.top_k)
        for i, r in enumerate(resultados, 1):
            m = r["metadata"]
            print(f"  [{i}] Score={r['score']:.4f} | {m['producto']} — Sección {m['seccion_num']}: {m['seccion_titulo']}")
            print(f"      {r['texto'][:200].replace(chr(10), ' ')}...")
            print()
