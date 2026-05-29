"""
Fase 2 — Constructor de prompt RAG para fichas de seguridad CORONA.
"""

SYSTEM_PROMPT = """\
Eres un experto en fichas de seguridad de pinturas CORONA.
Responde ÚNICAMENTE basándote en el contexto provisto.
Si la información no está en el contexto, di:
'No encontré esta información en las fichas de seguridad disponibles.'
Al final incluye SIEMPRE:
Fuente: Sección [N] — [Título] | Producto: [nombre]\
"""


def build_prompt(query: str, chunks: list[dict]) -> str:
    """
    Construye el prompt completo para el LLM.

    Parameters
    ----------
    query  : pregunta del usuario.
    chunks : lista de dicts con claves texto, seccion_num, seccion_titulo, producto.

    Returns
    -------
    str  prompt listo para enviar al LLM.
    """
    context_blocks = []
    for chunk in chunks:
        header = (
            f"--- Sección {chunk['seccion_num']}: {chunk['seccion_titulo']}"
            f" | {chunk['producto']} ---"
        )
        context_blocks.append(f"{header}\n{chunk['texto']}")

    context = "\n\n".join(context_blocks)

    return (
        f"[SYSTEM]\n{SYSTEM_PROMPT}\n\n"
        f"[CONTEXT]\n{context}\n\n"
        f"[QUESTION]\n{query}"
    )


if __name__ == "__main__":
    from retriever import retrieve

    query = "¿Qué EPP se requiere?"
    chunks = retrieve(query, top_k=3)
    prompt = build_prompt(query, chunks)
    print(prompt)
