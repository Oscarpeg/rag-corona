# RAG-CORONA — Sistema de Auditoría Semántica GHS
**Grupo J · Universidad · 2026**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![LLM](https://img.shields.io/badge/LLM-llama3%20·%20Ollama-8A2BE2)
![VectorStore](https://img.shields.io/badge/VectorStore-ChromaDB-orange)
![Embeddings](https://img.shields.io/badge/Embeddings-nomic--embed--text-teal)
![Sin APIs de pago](https://img.shields.io/badge/APIs%20de%20pago-ninguna-brightgreen)

Sistema RAG (*Retrieval-Augmented Generation*) para consultar fichas de datos de seguridad (FDS) de pinturas CORONA en formato GHS. Extrae, indexa y responde preguntas técnicas sobre 17 productos utilizando únicamente modelos locales y sin APIs de pago.

---

## ✨ Features

- **Extracción estructurada de PDFs** — `pymupdf4llm` para texto nativo + `pytesseract` como fallback OCR para páginas escaneadas
- **16 secciones GHS por ficha** — normalización y mapeo automático de encabezados (`section_mapper.py`)
- **Indexado vectorial persistente** — ChromaDB con cosine similarity; checkpointing para re-indexados parciales
- **Consulta en lenguaje natural** — CLI interactiva con listado de productos, historial y comandos de ayuda
- **Pipeline RAG completo** — retrieve → build prompt → LLM local (llama3 / llama3.1 según hardware)
- **Trazabilidad de fuentes** — cada respuesta cita producto, sección GHS y score semántico
- **Evaluación automatizada** — 20 preguntas de ground truth con métricas de acierto exacto y fallback

---

## Stack tecnológico

| Componente | Herramienta |
|---|---|
| Extracción PDF → Markdown | `pymupdf4llm` + `pytesseract` (OCR) |
| Vector store | `ChromaDB` (persistente, cosine similarity) |
| Embeddings | `nomic-embed-text` vía Ollama |
| LLM generativo | `llama3` vía Ollama (CPU) / `llama3.1` (GPU) |
| Orquestación | `langchain-ollama` |
| Evaluación | CSV ground truth + métricas de trazabilidad |

---

## Requisitos

- **Python 3.10+**
- **Ollama** instalado y corriendo en `localhost:11434`
  - Instalación: https://ollama.com/download
- **Tesseract OCR** instalado en el sistema
  - Amazon Linux / CentOS: `sudo yum install tesseract`
  - Ubuntu: `sudo apt install tesseract-ocr`
- Al menos **8 GB RAM** (16 GB recomendado para llama3 en CPU)
- GPU opcional pero recomendada para inferencia rápida

---

## Instalación

```bash
# 1. Clonar el repositorio
git clone git@github.com:Oscarpeg/rag-corona.git
cd rag-corona

# 2. Crear entorno conda
conda create -n rag-env python=3.10 -y
conda activate rag-env

# 3. Instalar dependencias Python
pip install -r requirements.txt

# 4. Descargar modelos Ollama
ollama pull nomic-embed-text
ollama pull llama3          # CPU (~4.7 GB)
# ollama pull llama3.1      # GPU, mejor español (~4.7 GB)
```

---

## Estructura del proyecto

```
rag-corona/
├── data/
│   ├── raw/            # PDFs originales de las fichas de seguridad (17 archivos)
│   ├── processed/      # Markdown extraídos (17 archivos .md, 16 secciones H2 c/u)
│   └── chroma_db/      # Índice vectorial ChromaDB (generado por indexer.py)
├── src/
│   ├── pdf_extractor.py   # Fase 1: PDF → Markdown estructurado
│   ├── section_mapper.py  # Fase 1: Detección y normalización de secciones GHS
│   ├── chunker.py         # Fase 2: Segmentación semántica por sección GHS
│   ├── indexer.py         # Fase 2: Embeddings + carga a ChromaDB
│   ├── retriever.py       # Fase 2: Búsqueda semántica sobre ChromaDB
│   ├── rag_prompt.py      # Fase 2: Constructor de prompts RAG
│   ├── rag_pipeline.py    # Fase 2: Pipeline completo retrieve → prompt → LLM
│   ├── demo.py            # Demo CLI interactiva para la presentación
│   ├── run_pipeline.py    # Orquestador de extracción masiva
│   ├── evaluate.py        # Evaluación piloto (3 preguntas, salida JSON)
│   ├── verify_index.py    # Auditoría de relevancia semántica del índice
│   ├── test_env.py        # Verificación del entorno (librerías + Ollama)
│   └── startup.sh         # Script de arranque de Ollama
├── evaluation/
│   ├── ground_truth.csv         # 20 preguntas técnicas con respuesta esperada
│   ├── run_evaluation.py        # Evaluación automatizada con métricas
│   ├── analisis_evaluacion.md   # Análisis científico de resultados
│   └── resultados_rag.csv       # Generado tras correr run_evaluation.py
├── requirements.txt
└── README.md
```

---

## Uso

### 1. Extraer PDFs a Markdown (Fase 1)
```bash
# Procesar todos los PDFs en data/raw/
python src/run_pipeline.py
```

### 2. Indexar en ChromaDB (Fase 2)
```bash
# Primera vez (o para re-indexar todo)
python src/indexer.py --reset

# Re-indexar solo los archivos nuevos (checkpointing)
python src/indexer.py
```

### 3. Demo interactiva
```bash
python src/demo.py
```
Al iniciar se muestran 3 ejemplos automáticos y se abre el prompt:
```
Pregunta RAG > ¿Qué EPP se requiere para PINTURA TOTAL?
```
Comandos disponibles: `ayuda`, `productos`, `salir`.

### 4. Consulta programática
```python
from src.rag_pipeline import query_rag

resultado = query_rag("¿Cuál es el punto de inflamación del RECUBRIMIENTO ANTIGRAFFITI?")
print(resultado["respuesta"])
# → Fuente: Sección 9 — PROPIEDADES FISICOQUÍMICAS | Producto: FDS 31 ...
```

### 5. Evaluación automatizada
```bash
# Piloto — primeras 3 preguntas (~15 min CPU)
python evaluation/run_evaluation.py --limite 3

# Evaluación completa — 20 preguntas (~90 min CPU)
python evaluation/run_evaluation.py
```
Genera `evaluation/resultados_rag.csv` y `evaluation/metricas_rag.json`.

---

## Métricas del sistema (evaluación piloto)

| Métrica | Valor |
|---|---|
| % Fuente citada correctamente | 80 % |
| % Acierto exacto de sección | 40 % |
| Respuestas con fallback correcto | 20 % |
| Tiempo promedio por consulta (CPU) | ~334 seg |
| Score semántico promedio (top-1) | 0.74 |

---

## Notas de reproducibilidad

- Los PDFs originales deben colocarse en `data/raw/` antes de ejecutar `run_pipeline.py`.
- El directorio `data/chroma_db/` está excluido del repositorio (`.gitignore`); debe regenerarse localmente con `indexer.py`.
- Al reiniciar la instancia EC2/SageMaker es necesario re-iniciar el agente SSH: `eval "$(ssh-agent -s)" && ssh-add ~/.ssh/id_ed25519`.

---

## 📸 Screenshots

<img width="829" height="596" alt="image" src="https://github.com/user-attachments/assets/ac9f8a38-2fcc-4906-9b54-bf3a33d0eb47" />
<img width="786" height="791" alt="image" src="https://github.com/user-attachments/assets/ce039ee9-3648-40fe-b8ba-542e677bfac1" />
<img width="786" height="791" alt="image" src="https://github.com/user-attachments/assets/a75075ab-92f1-467f-9cd6-93dc3a2ef50e" />



