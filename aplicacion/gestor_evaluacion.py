import json
from datetime import datetime
from pathlib import Path

# Ruta donde se guardarán los datos para el análisis estadístico de la tesis
RUTA_EVALUACIONES = Path("datos") / "resultados" / "estudio_creativo.jsonl"

def guardar_evaluacion_estimulo(id_usuario, modelo_usado, scores, comentarios=""):
    """
    Guarda la evaluación de un estímulo visual generado.

    Args:
        id_usuario (str): Identificador anónimo del participante.
        modelo_usado (str): DCGAN, CycleGAN o Combinado.
        scores (dict): Diccionario con dimensiones de la variable dependiente:
            - utilidad_percibida (1-5)
            - fluidez_ideas (1-5)
            - satisfaccion_estetica (1-5)
        comentarios (str): Notas cualitativas sobre el estímulo.
    """
    RUTA_EVALUACIONES.parent.mkdir(parents=True, exist_ok=True)

    registro = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "participante_id": id_usuario,
        "modelo": modelo_usado,
        "dimensiones_v_dependiente": scores,
        "feedback_cualitativo": comentarios,
        "metadatos_sistema": {
            "version": "2.0",
            "tipo_estudio": "No experimental / Transversal"
        }
    }

    with RUTA_EVALUACIONES.open("a", encoding="utf-8") as archivo:
        archivo.write(json.dumps(registro, ensure_ascii=False) + "\n")

    return RUTA_EVALUACIONES
