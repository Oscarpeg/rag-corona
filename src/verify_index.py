import chromadb
import ollama
import sys

# Documento de referencia para el aislamiento por metadatos
FUENTE_REFERENCIA = "FDS 29 - PINTURA PRIMERA MANO & ACABADO - CORONA .md"

# 1. Matriz de consultas — mejor query por sección según rondas de evaluación
MATRIZ_EVALUACION = {
    1:  "CORLANC S.A.S. datos del proveedor Carrera 48 Sabaneta Antioquia teléfono emergencias identificador SGA producto",
    2:  "Carc. 2 H351 Sens. Cut. 1 H317 clasificación SGA etiqueta indicaciones peligro consejos prudencia P280 P261 P302",
    3:  "primeros auxilios inhalación contacto con la piel contacto con los ojos ingestión síntomas intoxicación solicitar atención médica",
    4:  "Primeros auxilios sección 4 GHS medidas urgentes actuación inmediata",
    5:  "Medidas de lucha contra incendios medios de extinción",
    6:  "Medidas en caso de vertido accidental precauciones personales equipo protector contención limpieza",
    7:  "Manipulación y almacenamiento condiciones de almacenamiento seguro",
    8:  "EPP equipo protección personal TLV-TWA Dioxido titanio guantes protección química máscara autofiltrante protección respiratoria pantalla facial",
    9:  "Propiedades físicas y químicas estado físico 20°C Líquido densidad 1371 kg/m³ pH 8,5 9,5 temperatura ebullición 102°C blanco inodoro",
    10: "efectos CMR carcinogenicidad mutagenicidad IARC Dioxido titanio grupo 2B toxicidad reproducción STOT dermatitis alérgica",
    11: "Información toxicológica vías de exposición efectos sobre la salud DL50 oral cutánea Rata Conejo",
    12: "Información ecotoxicológica persistencia degradabilidad bioacumulación movilidad suelo PBT mPmB",
    13: "textos frases legislativas H351 susceptible provocar cáncer abreviaturas IMDG IATA DL50 CL50 Log POW BCF ICONTEC IARC bibliográficas",
    14: "Información relativa al transporte terrestre mercancías peligrosas clasificación número ONU ADR IATA",
    15: "Información sobre la reglamentación decreto 1496 legislación aplicable disposiciones específicas nacionales",
    16: "Otras informaciones fuentes bibliográficas metodología condiciones trabajo usuario ficha datos seguridad",
}

def auditar_sistema_rag():
    print("=" * 85)
    print(" AUDITORÍA DE RELEVANCIA SEMÁNTICA — AISLAMIENTO POR METADATOS")
    print(f" Documento de referencia: {FUENTE_REFERENCIA}")
    print("=" * 85)
    
    try:
        # Conectar al cliente persistente de ChromaDB
        client = chromadb.PersistentClient(path="data/chroma_db")
        col = client.get_collection("fichas_seguridad_corona")
    except Exception as e:
        print(f"ERROR CRÍTICO: No se pudo conectar a ChromaDB o la colección no existe.\nDetalle: {e}")
        sys.exit(1)
        
    correctos = 0
    errores = 0
    
    for sec_id, query in MATRIZ_EVALUACION.items():
        try:
            # 1. Generar embedding de la consulta en lenguaje natural
            resp = ollama.embeddings(model="nomic-embed-text", prompt=query)
            query_vector = resp["embedding"]
            
            # 2. Consultar ChromaDB con filtro estricto al documento de referencia
            res = col.query(
                query_embeddings=[query_vector],
                n_results=1,
                where={"fuente": FUENTE_REFERENCIA},
            )
            
            if not res["documents"] or not res["documents"][0]:
                print(f"[✗] Sección {sec_id:2d}: ERROR -> No se recuperó ningún contenido.")
                errores += 1
                continue
                
            texto_recuperado = res["documents"][0][0]
            texto_upper = texto_recuperado.upper()
            
            # 3. Validación Orgánica: Verificar que el chunk contenga su respectiva cabecera GHS
            # Cubre: "## SECCIÓN N:" (markdown) y "GHS: N " (header de enriquecimiento)
            patron_seccion = f"SECCIÓN {sec_id}"
            patron_ghs     = f"GHS: {sec_id} "   # header de enriquecimiento en chunker.py

            if patron_seccion in texto_upper or patron_ghs in texto_upper:
                print(f"[✓] Sección {sec_id:2d}: MATCH EXITOSO  -> Contexto orgánico verificado por Embedding.")
                correctos += 1
            else:
                print(f"[✗] Sección {sec_id:2d}: FALLO SEMÁNTICO -> El embedding recuperó la sección incorrecta.")
                errores += 1
                
        except Exception as e:
            print(f"[✗] Sección {sec_id:2d}: ERROR DE EJECUCIÓN -> {e}")
            errores += 1
            
    print("=" * 85)
    print(f" RESUMEN DE LA AUDITORÍA: {correctos:2d} Secciones OK | {errores:2d} Secciones Fallidas")
    print("=" * 85)

if __name__ == "__main__":
    auditar_sistema_rag()