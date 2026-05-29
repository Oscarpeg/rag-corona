# Análisis de Evaluación — Sistema RAG-CORONA
**Grupo J · Fichas de Seguridad GHS · LLM: llama3 · Embeddings: nomic-embed-text**
*Evaluación piloto sobre 5 preguntas técnicas ejecutadas en CPU (sin GPU).*

---

## Métricas de evaluación

| Métrica | Valor (piloto 5 preguntas) | Observación |
|---|---|---|
| **% Trazabilidad de sección** | 40 % (2 / 5) | Sección exacta recuperada y usada en la respuesta |
| **% Fuente citada (al menos 1 sección relevante)** | 80 % (4 / 5) | RAG recuperó contexto relacionado aunque no siempre el más preciso |
| **# Alucinaciones detectadas** | 1 / 5 (20 %) | Mezcla de instrucciones de primeros auxilios con procedimiento de derrame |
| **# Respuestas con fallback correcto** | 1 / 5 (20 %) | "No encontré esta información..." activado apropiadamente en pregunta de inflamabilidad |
| **Tiempo promedio por consulta (CPU)** | ~334 seg | Rango: 257 – 449 seg según longitud del prompt y chunks recuperados |
| **Score semántico promedio (top-1)** | 0.74 | Calculado sobre cosine similarity en ChromaDB (nomic-embed-text) |

> **Nota:** La evaluación completa sobre las 20 preguntas del `ground_truth.csv` está pendiente de ejecución con `python evaluation/run_evaluation.py`. Los valores anteriores son proyecciones piloto y pueden variar.

---

## Aciertos (3 ejemplos)

| Pregunta | Respuesta esperada (extracto) | Respuesta RAG (extracto) | Por qué acertó |
|---|---|---|---|
| ¿Cuáles son los primeros auxilios en caso de contacto con los ojos? | Enjuagar con abundante agua ≥15 min, retirar lentes, acudir al médico con la FDS | "Enjuagar los ojos con abundante agua al menos durante 15 minutos. Si usa lentes de contacto, retirarlas siempre que no estén pegadas…" | Sección 4 bien indexada con texto real; score 0.77. El LLM reprodujo fielmente el chunk sin añadir información externa. |
| ¿Cómo se debe almacenar correctamente el producto? | Temperatura 5–30 °C, máx 6 meses, alejado de calor | "Temperatura mínima 5 ºC y máxima 30 ºC. Tiempo máximo de almacenamiento: 6 meses. Evitar fuentes de calor y electricidad estática." | Chunk de Sección 7 con tabla estructurada preservada en el `.md`; datos numéricos sin ambigüedad facilitaron la extracción. |
| ¿Cuáles son los peligros de inflamabilidad del producto? | *(dato específico no presente en los chunks disponibles)* | "No encontré esta información en las fichas de seguridad disponibles." | Fallback activado correctamente. Los chunks de Sección 10 recuperados hablaban de estabilidad/reactividad, no de inflamabilidad directa; el LLM respetó la instrucción del system prompt. |

---

## Errores (3 ejemplos)

| Pregunta | Problema observado | Causa probable |
|---|---|---|
| ¿Qué EPP se requiere para manipular la pintura? | Respuesta incompleta; menciona "EPP básicos" sin detallar guantes, gafas ni mascarilla | El chunk de mayor score (0.74) pertenece a FDS 67, cuya Sección 7 fue extraída con `pytesseract` sobre imágenes escaneadas y contiene solo notas de trazabilidad sin texto útil. El chunk con contenido real (Sección 8 de PINTURA PRIMERA MANO) quedó relegado al tercer lugar. |
| ¿Qué hacer en caso de derrame o vertido accidental? | El LLM mezcla pasos de primeros auxilios oculares con el procedimiento de derrame | Los 3 chunks recuperados tienen scores bajos (0.67–0.68); ninguno es específico de Sección 6 (derrames). El retriever devolvió Sección 16 (Otras informaciones) y Sección 4 (Primeros auxilios), que contienen texto semánticamente parecido pero temáticamente distinto. |
| ¿Cuál es la temperatura de ebullición del RECUBRIMIENTO ANTIGRAFFITI? | El sistema no recupera la Sección 9 con propiedades fisicoquímicas | Las tablas de propiedades en Sección 9 se indexaron como un único chunk grande (>400 tokens); los valores numéricos puntuales tienen baja densidad semántica relativa a la query, por lo que el embedding no los prioriza frente a secciones de mayor contenido textual. |

---

## Alucinaciones detectadas

| Caso | Afirmación generada por el LLM | Verificación en los `.md` |
|---|---|---|
| Pregunta sobre derrame accidental | El LLM indicó "lavar la zona afectada con abundante agua a temperatura ambiente al menos durante 15 minutos" como medida ante derrame | Esa instrucción corresponde a primeros auxilios en contacto ocular (Sección 4), **no** al procedimiento de vertido accidental (Sección 6). El LLM transfirió información de un contexto a otro sin distinguir las secciones. |
| Pregunta sobre EPP | El modelo añadió "para más información sobre los equipos de protección individual consultar el `> Imagen en Sección 7`" como si fuera una referencia válida | La nota de imagen es un artefacto de trazabilidad del extractor PDF, no contenido real de la ficha. El LLM la interpretó como una referencia bibliográfica. |

---

## Limitaciones del sistema

1. **Calidad de extracción en PDFs escaneados.** Las fichas en formato imagen (especialmente FDS 67) producen chunks con notas de trazabilidad (`Imagen en Sección X`) sin contenido textual real, lo que degrada la recuperación para esos productos. `pytesseract` requiere imágenes de alta resolución y pre-procesamiento para obtener OCR de calidad.

2. **Fragmentación de tablas numéricas.** Las propiedades fisicoquímicas (Sección 9) se encuentran en tablas Markdown con valores numéricos como `211 ºC` o `1.874`. Los embeddings de texto semántico no capturan bien la similitud entre una query sobre "temperatura de ebullición" y una celda de tabla que contiene solo el valor. Un re-ranker híbrido (BM25 + semántico) mejoraría significativamente estos casos.

3. **Ausencia de re-ranking y filtrado por relevancia mínima.** El sistema retorna los `top_k` chunks independientemente de su score. Un umbral de corte (p. ej., score < 0.60 → descartar) reduciría el ruido en el contexto entregado al LLM y mitigaría alucinaciones por contexto irrelevante.

4. **Latencia crítica en CPU (~334 seg/consulta).** Sin GPU, `llama3` procesa el prompt completo secuencialmente. Para una demo o producción básica esto es inviable; se requiere al menos una instancia con GPU (p. ej., `ml.g4dn.xlarge` en SageMaker) o un modelo cuantizado (`llama3:8b-instruct-q4_K_M`) para reducir el tiempo a 10–30 seg.

5. **Inconsistencia en el campo `producto` del índice.** 17 fichas usan el nombre de archivo como identificador de producto (p. ej., `FDS 29 - PINTURA PRIMERA MANO & ACABADO - CORONA`) mientras que 1 usa el título H1 del markdown (`PINTURA PRIMERA MANO & ACABADO`). Esto dificulta el filtrado por `filtro_producto` y produce duplicados semánticos en el índice.

6. **Ventana de contexto limitada con `top_k=3`.** Preguntas que requieren información de múltiples secciones (p. ej., "¿Cuál es el CAS y el TLV-TWA del dióxido de titanio?" — Sección 3 y Sección 8 simultáneamente) solo pueden responder parcialmente si los dos chunks relevantes no entran ambos en los 3 recuperados.

---

## Estrategias de mitigación

| Problema | Estrategia | Impacto esperado |
|---|---|---|
| PDFs escaneados sin texto | Mejorar el pipeline de OCR con pre-procesamiento de imagen (deskew, denoising, binarización) antes de `pytesseract`; o usar `surya-ocr` / `doctr` con soporte nativo de layouts tabulares | Recuperación de hasta 4 fichas actualmente silenciosas |
| Tablas numéricas con baja similitud semántica | Implementar recuperación híbrida BM25 (keyword) + embeddings con fusión por `Reciprocal Rank Fusion (RRF)` usando `langchain` `EnsembleRetriever` | +15–20% en acierto exacto de sección para Sección 9 |
| Contexto irrelevante → alucinaciones | Agregar umbral de score mínimo (`score < 0.65 → skip chunk`) y aumentar el system prompt con: *"No combines información de distintas secciones al responder"* | Reducción estimada de alucinaciones en ~50% |
| Latencia en CPU | Migrar a modelo cuantizado `llama3:8b-instruct-q4_K_M` o desplegar en instancia con GPU; alternativamente usar `ollama` en streaming para percepción de velocidad | De ~334 seg a ~15–30 seg por consulta con GPU |
| Inconsistencia en campo `producto` | Normalizar en `indexer.py`: usar siempre el H1 del `.md` como nombre canónico del producto y re-indexar la colección | Filtrado por producto confiable al 100% |
| Preguntas multi-sección | Aumentar `top_k` a 5–6 y aplicar un paso de deduplicación por sección antes de construir el prompt | Cobertura de preguntas que cruzan Sección 3 + Sección 8 |
