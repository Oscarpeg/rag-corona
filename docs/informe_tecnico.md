# Informe Técnico — Sistema RAG para Fichas de Datos de Seguridad CORONA
**Grupo J · 2026 · Stack: pymupdf4llm · ChromaDB · llama3 · nomic-embed-text**

---

## 1. Resumen ejecutivo

Se diseñó e implementó un sistema de Recuperación Aumentada por Generación (RAG) para consultar en lenguaje natural las fichas de datos de seguridad (FDS) GHS de 17 productos de pinturas CORONA. El pipeline extrae texto estructurado de PDFs heterogéneos —nativos y escaneados— usando `pymupdf4llm` como ruta principal y `pytesseract` como fallback OCR para bloques de imagen. Los documentos se segmentan respetando la unidad normativa GHS (sección H2), se indexan en ChromaDB con embeddings locales `nomic-embed-text` y se consultan mediante `llama3` ejecutado en Ollama sin dependencia de APIs externas. La evaluación piloto sobre 5 preguntas técnicas mostró un 80 % de trazabilidad de fuente y un 40 % de acierto exacto de sección, con un tiempo promedio de 334 segundos por consulta en CPU. El sistema es completamente reproducible en local con hardware modesto y sin costos variables de inferencia.

---

## 2. Pipeline de extracción PDF → Markdown

### 2.1 Elección de `pymupdf4llm` frente a alternativas

La extracción de fichas de seguridad impone requisitos estrictos: preservación de tablas de propiedades fisicoquímicas, identificación de encabezados jerárquicos (`## Sección N:`) y manejo de PDFs mixtos (texto nativo + páginas escaneadas). Se evaluaron tres alternativas:

| Herramienta | Tablas | Imágenes | Costo | Veredicto |
|---|---|---|---|---|
| `pymupdf4llm` | Markdown pipe nativo | Delegado a Tesseract | Gratis, local | **Elegida** |
| `pdfplumber` | Extracción por coordenadas, frágil en FDS tabulares | No | Gratis | Descartada: pierde estructura de tablas multicolumna |
| `unstructured` | Buena pero requiere configuración compleja | Sí | Gratis (modo local pesado) | Descartada: tiempo de setup excesivo para el alcance del proyecto |
| APIs (AWS Textract, Azure DI) | Excelente | Sí | Pago por página | Descartada: criterio explícito del proyecto (sin APIs pagas) |

Se usó `pymupdf4llm.to_markdown(doc, write_images=False)` con `write_images=False` para evitar volcar las imágenes de pictogramas GHS al disco —archivos PNG innecesarios para el pipeline de texto— y conservar únicamente el texto y las tablas. Las referencias a figuras se capturan en bloques de trazabilidad (ver §2.3).

### 2.2 Flujo único de extracción

El extractor sigue un único camino de procesamiento, sin bifurcaciones condicionales sobre el tipo de PDF:

```
PDF de entrada
    │
    ▼
pymupdf4llm.to_markdown()     ← texto nativo + tablas Markdown pipe
    │
    ▼
Normalización de encabezados  ← H1 para nombre de producto, H2 para secciones GHS
    │
    ▼
Detección de bloques imagen   ← fitz.Page.get_images() por página
    │
    ▼
OCR con pytesseract           ← sobre cada bloque imagen (fallback si no hay texto)
    │
    ▼
Inyección de bloques de       ← nota de trazabilidad con contexto de sección
trazabilidad en el .md
    │
    ▼
.md final con 16 secciones H2
```

`find_tables()` de PyMuPDF se evaluó como complemento para extraer tablas en PDFs donde `pymupdf4llm` producía celdas fusionadas incorrectamente, pero se descartó como ruta principal por introducir inconsistencias en la numeración de filas. Se reservó como parche manual post-revisión para los 2 productos con tablas especialmente complejas (FDS 44 TEXTUCO y FDS 94 ESMALTE).

### 2.3 Imágenes y bloques de trazabilidad

Los PDFs de CORONA incluyen pictogramas GHS (rombos de peligro) y diagramas de etiquetado. `pymupdf4llm` con `write_images=False` los omite del Markdown. Para no perder la referencia de ubicación, el extractor inyecta un bloque de trazabilidad por cada imagen detectada:

```markdown
> **Nota de trazabilidad:** Pictograma(s) GHS: SGA.
> Imagen en Sección 2: IDENTIFICACIÓN DEL PELIGRO.
> Información relacionada en la sección correspondiente.
```

Esto permite al LLM reconocer que existía contenido visual sin fabricar datos. En la evaluación se detectó un caso de alucinación en el que el modelo interpretó esta nota como una referencia bibliográfica válida (ver §4.3), lo que motivó añadir en el system prompt la instrucción explícita de no citar notas de imagen como fuente.

### 2.4 Errores reales encontrados y mitigaciones

| Error | FDS afectada | Causa | Mitigación aplicada |
|---|---|---|---|
| Secciones GHS sin encabezado `## Sección N:` | FDS 67, FDS 75 | PDF escaneado; OCR de baja calidad en encabezados | `section_mapper.py` reescribe encabezados mal formateados usando regex sobre texto bruto |
| Tablas de propiedades con celdas fusionadas | FDS 44, FDS 94 | PyMuPDF no puede separar celdas fusionadas en PDFs de diseño complejo | Revisión manual + corrección en el `.md` procesado |
| Caracteres corruptos en nombres de archivo | FDS 61 (SEÑALIZACIÓN) | Codificación UTF-8 rota en el nombre del PDF original | `.gitignore` y rutas manejadas con `os.fsencode` / apertura explícita en UTF-8 |
| Chunks sin contenido textual real (solo notas de imagen) | FDS 67 | PDF 100 % escaneado, OCR insuficiente en resolución original | Identificados en la evaluación; candidatos a re-extracción con Tesseract preconfigurado (`--psm 6`, preprocesamiento de imagen) |

---

## 3. Arquitectura del sistema RAG

### 3.1 Chunking semántico

La unidad de segmentación es la **sección normativa GHS** (`## Sección N: Título`), no el párrafo ni la ventana deslizante fija. Esta decisión se justifica porque:

- Las preguntas técnicas sobre FDS siempre referencian secciones específicas (EPP → §8, almacenamiento → §7, primeros auxilios → §4).
- Mezclar contenido de secciones distintas en un mismo chunk introduce ruido semántico que degrada la precisión del retriever.

**Umbral de sub-chunking:** 400 tokens estimados (`palabras × 1.3`). Secciones por debajo del umbral se indexan como un único chunk; secciones densas se dividen por sub-encabezados `###` (primera prioridad) o por ventana deslizante de 300 palabras con solapamiento de 50 (fallback). El prefijo de contexto añadido a cada chunk antes de generar el embedding:

```
DOCUMENTO: <nombre_producto>
CONTEXTO SEGURIDAD GHS: <N> <TÍTULO_SECCIÓN>

<texto de la sección>
```

Este prefijo ancla semánticamente el chunk al producto y la sección, mejorando la discriminación entre fichas distintas que comparten texto normativo casi idéntico.

### 3.2 Embeddings: nomic-embed-text

Se eligió `nomic-embed-text` (local, vía Ollama) por su ventana de contexto de **8 192 tokens** —suficiente para cualquier chunk generado por el pipeline— y por su rendimiento superior a `all-MiniLM-L6-v2` en tareas de recuperación técnica en español. No requiere GPU para generar embeddings; la latencia por chunk es de ~80–120 ms en CPU.

Dimensión del vector: **768**. Métrica de similitud en ChromaDB: **cosine** (`hnsw:space: cosine`). Los scores observados en la evaluación piloto oscilan entre 0.67 y 0.77 para recuperaciones relevantes; umbrales por debajo de 0.65 corresponden a contexto ruidoso.

### 3.3 ChromaDB con patrón `**kwargs` para filtros opcionales

El retriever implementa el filtro de producto mediante un patrón `**kwargs` que evita pasar `where=None` a ChromaDB (lo que genera un error de validación en algunas versiones):

```python
kwargs = {}
if filtro_producto:
    kwargs["where"] = {"producto": filtro_producto}

results = collection.query(
    query_embeddings=[query_vector],
    n_results=top_k,
    include=["documents", "metadatas", "distances"],
    **kwargs,
)
```

El campo de filtrado es `producto` (nombre canónico del producto extraído del H1 del `.md`), no `fuente` (nombre del archivo). La colección `fichas_seguridad_corona` contiene 18 valores de `producto` distintos; 17 corresponden al nombre del archivo y 1 al H1 limpio —inconsistencia identificada como limitación y documentada en el análisis de evaluación.

El indexer implementa **checkpointing** mediante `existing_ids = set(collection.get(include=[])["ids"])` para evitar re-embeber chunks ya indexados en ejecuciones sucesivas, reduciendo el tiempo de re-indexación parcial de ~45 min a segundos.

### 3.4 LLM: llama3 / llama3.1 — criterio GPU/CPU

| Modelo | Contexto | Español | RAM requerida | Tiempo/consulta (CPU) | Recomendado para |
|---|---|---|---|---|---|
| `llama3` (8B Q4) | 8 192 tokens | Aceptable | ~5 GB | ~300–450 seg | Demo en CPU, parcial |
| `llama3.1` (8B Q4) | **128 000 tokens** | **Mejor** | ~5 GB | ~300–450 seg (CPU) / ~15 seg (GPU) | Producción con GPU |
| `llama3:8b-instruct-q4_K_M` | 8 192 tokens | Bueno | ~4.5 GB | ~200 seg | CPU optimizado |

El system prompt instruye al modelo a responder únicamente desde el contexto provisto y a activar el fallback `"No encontré esta información en las fichas de seguridad disponibles."` cuando ningún chunk contiene la respuesta. La temperatura se fija en `0.1` para maximizar la fidelidad reproductiva y minimizar variaciones entre ejecuciones.

---

## 4. Resultados de evaluación

### 4.1 Métricas cuantitativas (evaluación piloto — 5 preguntas)

| Métrica | Valor | Cálculo |
|---|---|---|
| % Fuente citada (≥1 sección relevante recuperada) | **80 %** (4/5) | Intersección entre secciones esperadas y secciones en `fuentes_rag` |
| % Acierto exacto de sección | **40 %** (2/5) | Secciones esperadas ⊆ secciones citadas por el RAG |
| # Respuestas con fallback correcto | **1** (20 %) | Pregunta sobre inflamabilidad sin dato en chunks disponibles |
| # Alucinaciones detectadas | **1** (20 %) | Mezcla de Sección 4 (primeros auxilios) en respuesta sobre Sección 6 (derrames) |
| Score semántico promedio (top-1 chunk) | **0.74** | Cosine similarity sobre `nomic-embed-text` |
| Tiempo promedio por consulta (CPU) | **334 seg** | Rango: 257–449 seg; variación por longitud del prompt |

### 4.2 Casos de éxito

**Primeros auxilios — contacto ocular** (score 0.77): la Sección 4 de FDS 76 está bien indexada con texto estructurado completo. El LLM reprodujo fielmente los tres pasos clínicos (enjuague ≥15 min, retirar lentes, acudir al médico con la FDS) sin añadir información externa.

**Condiciones de almacenamiento** (score 0.77): la tabla de Sección 7 con valores numéricos (`5 ºC`, `30 ºC`, `6 meses`) fue preservada como Markdown pipe en la extracción; el embedding la recuperó correctamente y el LLM extrajo los valores sin parafrasear.

**Fallback en pregunta de inflamabilidad**: los chunks de Sección 10 (estabilidad/reactividad) recuperados con score 0.75 no contenían el dato específico de inflamabilidad. El LLM respetó la instrucción del system prompt y activó el fallback en lugar de inventar un valor.

### 4.3 Casos de fallo y alucinaciones

**EPP incompleto** (score 0.74 para FDS 67 Sección 7): el chunk de mayor similitud pertenece a FDS 67, cuya extracción produjo únicamente notas de trazabilidad de imagen. El contenido real de EPP (Sección 8 de PINTURA PRIMERA MANO) quedó relegado al tercer lugar en el ranking. Causa raíz: PDFs escaneados de baja resolución producen chunks vacíos que el retriever no puede descartar por ausencia de umbral de score mínimo.

**Derrame accidental — alucinación por cruce de secciones**: el retriever devolvió Sección 4 (primeros auxilios oculares) con score 0.67 al no encontrar Sección 6 (vertido) en el top-3. El LLM transfirió los pasos de lavado ocular como si fueran procedimiento de derrame. Causa raíz: scores bajos (0.67–0.68) indican que el contexto entregado no era el correcto; la ausencia de umbral mínimo permitió que ese contexto ruidoso llegara al LLM.

---

## 5. Limitaciones y trabajo futuro

### 5.1 Limitaciones identificadas

1. **PDFs escaneados de baja calidad.** FDS 67 y parcialmente FDS 75 fueron extraídas con OCR insuficiente, produciendo chunks sin contenido real. La resolución de los PDFs originales (~150 DPI en algunas páginas) está por debajo del umbral recomendado para Tesseract (≥300 DPI). Afecta directamente la cobertura del sistema para esos dos productos.

2. **Baja densidad semántica en tablas numéricas.** Los valores fisicoquímicos de Sección 9 (punto de ebullición, densidad, pH) son números aislados con escaso contexto textual circundante. Los embeddings densos no capturan bien la similitud entre la query `"temperatura de ebullición"` y una celda de tabla que contiene solo `211 ºC`. BM25 sobre el texto de la celda resolvería este caso.

3. **Latencia crítica en CPU.** ~334 seg/consulta hace inviable el uso interactivo en una demo presencial sin GPU. En la arquitectura actual no hay streaming; el usuario espera en blanco hasta que el LLM completa la generación.

4. **Inconsistencia en el identificador de producto.** 17 de las 18 entradas en el índice usan el nombre del archivo como `producto`; 1 usa el H1 del markdown. Esto impide que `filtro_producto` funcione de manera consistente a menos que el usuario conozca el valor exacto almacenado.

5. **`top_k=3` insuficiente para preguntas multi-sección.** Preguntas que cruzan Sección 3 (composición) y Sección 8 (límites de exposición) simultáneamente requieren al menos 2 chunks de secciones distintas; con `top_k=3` hay riesgo de que uno quede fuera si otro chunk irrelevante ocupa su lugar.

6. **Sin validación de integridad post-extracción.** El pipeline no verifica automáticamente que cada `.md` tenga exactamente las 16 secciones H2 requeridas antes de indexar. Una sección faltante por error de extracción pasa desapercibida hasta que una consulta específica falla.

### 5.2 Trabajo futuro

| Mejora | Impacto estimado | Complejidad |
|---|---|---|
| Recuperación híbrida BM25 + semántica con RRF (`EnsembleRetriever`) | +15–20% acierto en Sección 9 (tablas numéricas) | Media |
| Umbral de score mínimo (`< 0.65 → descartar chunk`) | –50% alucinaciones por contexto ruidoso | Baja |
| Pre-procesamiento de imagen antes de OCR (deskew, denoising, upscale a 300 DPI) | Recuperación de FDS 67 y FDS 75 | Media |
| Streaming de respuesta del LLM (`OllamaLLM` con `streaming=True`) | Percepción de latencia × 10 en demo | Baja |
| Normalización del campo `producto` en el indexer (siempre usar H1) | Filtrado por producto confiable al 100% | Baja |
| Re-ranker cruzado (`cross-encoder/ms-marco-MiniLM`) sobre los top-10 antes de enviar al LLM | +10–15% acierto global | Alta |
| Evaluación completa sobre las 20 preguntas del ground truth | Métricas estadísticamente representativas | — (pendiente de ejecución) |
