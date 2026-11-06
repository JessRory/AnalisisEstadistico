import numpy as np
import pandas as pd

def analiza_bernoulli(columna:pd.Series) -> tuple[dict, list[str]]:
    """
    Analiza una columna de datos binarios (0 y 1) y calcula métricas relevantes para una distribución de Bernoulli.

    Args:
        columna (pd.Series): Columna de datos binarios (0 y 1).

    Returns:
        dict: Diccionario con métricas de la distribución de Bernoulli, incluyendo:
            - 'Variable': Nombre de la columna.
            - 'p': Probabilidad de éxito (1).
            - 'q': Probabilidad de fracaso (0).
            - 'ratio_desbalanceo': Ratio de desbalanceo (q/p).
            - 'media': Media de la distribución (no aplicable para Bernoulli).
            - 'Asimetria': Asimetría de la distribución (no aplicable
            - 'Porcentaje Outliers': Porcentaje de valores atípicos (no aplicable).
            - 'Nulos': Número de valores nulos en la columna.
            - 'Pct_Nulos': Porcentaje de valores nulos en la columna.
    Raises:
        ValueError: Si la columna contiene valores distintos de 0 y 1.

    """

    # Verificar que la columna contenga solo valores binarios (0 y 1)
    if not set(columna.unique()).issubset({0, 1}):
        raise ValueError("La columna debe contener solo valores binarios (0 y 1).")

    n = len(columna)
    conteos = columna.value_counts()
    p = conteos.get(1, 0) / n  # Probabilidad de éxito (1)
    q = conteos.get(0, 0) / n  # Probabilidad de fracaso (0)

    fila = {
        'Variable': columna.name,
        'p': p,
        'q': q,
        'ratio_desbalanceo': round(q / p, 2) if p > 0 else None,
        'media': None,        # no aplica
        'Asimetria': None,    # no aplica
        'Porcentaje Outliers': None # no aplica
    }

    ratio_txt = f"{round(q / p, 2)}" if p > 0 else "∞"

    mensajes = [
        f"Columna binaria detectada (p={p:.4f}, ratio {ratio_txt}:1). "
        "Métricas de distribución continua no aplicables."
    ]
    
    return fila, mensajes
