# análisis_estadístico.py

# Importamos las librerías necesarias:
import numpy as np
import pandas as pd
from scipy import stats

class AnalisisEstadistico:
    def __init__(self, null_threshold: float = 0.05, 
                 outliers_factor: float = 1.5,
                 rare_threshold: float = 0.01):

        # Configuración centralizada:
        self.null_threshold = null_threshold
        self.outliers_factor = outliers_factor
        self.rare_threshold = rare_threshold
        
        # Estado opcional para trazabilidad
        self._last_analysis = {}
        self._warnings = []
    
    def analizar_numericas(self, columnas):

        resultados = {}
        for i, columna in enumerate(columnas):

            self.columnas = columnas

            # Cálculamos la moda usando scipy.stats
            moda = stats.mode(np.array(columna))
            resultados[f'columna_{i}'] = {
                'media': np.mean(columna),
                'mediana': np.median(columna),
                'moda': moda[0],
                'desviacion_estandar': np.std(columna),
                'rango': np.ptp(columna),
                'rango_intercuartilico': np.percentile(columna, 75) - np.percentile(columna, 25),
                'coeficiente_variacion': np.std(columna) / np.mean(columna),
                'Curtosis': stats.kurtosis(columna),
                'Asimetria': stats.skew(columna)
            }

        df = pd.DataFrame(resultados)
        

        return df

    def analizar_categoricas(self, df, columns):
        # Lógica actual  
        pass
    
    def get_warnings(self):
        return self._warnings.copy()
    
    def reset_warnings(self):
        self._warnings.clear()

prueba = AnalisisEstadistico()

columna1 = [1, 2, 3, 4, 5, 5]
columna2 = [10, 20, 30, 40, 50, 50]
df = prueba.analizar_numericas([columna1, columna2])
print(df)
