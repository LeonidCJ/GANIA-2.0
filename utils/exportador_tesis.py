import json
import csv
import pandas as pd
from pathlib import Path

def exportar_evaluaciones_a_csv(ruta_jsonl="datos/resultados/estudio_creativo.jsonl", ruta_csv="datos/resultados/analisis_estadistico.csv"):
    """
    Convierte las evaluaciones del estudio creativo a formato CSV para análisis en SPSS/Excel.
    """
    ruta_in = Path(ruta_jsonl)
    if not ruta_in.exists():
        print(f"No se encontró el archivo de datos en {ruta_jsonl}")
        return

    datos_aplanados = []

    with open(ruta_in, "r", encoding="utf-8") as f:
        for linea in f:
            reg = json.loads(linea)
            # Aplanamos la estructura para que sea una fila de tabla
            fila = {
                "timestamp": reg.get("timestamp"),
                "participante_id": reg.get("participante_id"),
                "modelo": reg.get("modelo"),
                # Dimensiones de la Variable Dependiente
                "utilidad_percibida": reg["dimensiones_v_dependiente"].get("utilidad_percibida"),
                "fluidez_ideas": reg["dimensiones_v_dependiente"].get("fluidez_ideas"),
                "satisfaccion_estetica": reg["dimensiones_v_dependiente"].get("satisfaccion_estetica"),
                "feedback_cualitativo": reg.get("feedback_cualitativo"),
                "version_sistema": reg["metadatos_sistema"].get("version")
            }
            datos_aplanados.append(fila)

    df = pd.DataFrame(datos_aplanados)
    df.to_csv(ruta_csv, index=False, encoding="utf-8-sig")
    print(f"Datos exportados exitosamente a: {ruta_csv}")
    return ruta_csv

if __name__ == "__main__":
    # Script ejecutable para el tesista
    exportar_evaluaciones_a_csv()
