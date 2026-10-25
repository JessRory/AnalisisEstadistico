# análisis_estadístico.py

# Importamos las librerías necesarias:
import numpy as np
import pandas as pd
from statistics import mode
from scipy import stats

class AnalisisEstadistico:

    """
    Clase para realizar análisis estadístico de columnas numéricas y categóricas 
    en un DataFrame de pandas.   
    """
    def __init__(self, null_threshold: float = 0.05, 
                 outliers_factor: float = 1.5,
                 rare_threshold: float = 0.01):

        """
        Constructor de la clase AnalisisEstadistico.
        Parámetros:
        - null_threshold: Umbral para considerar una columna con demasiados valores nulos.
        - outliers_factor: Factor para determinar los outliers en columnas numéricas.
        - rare_threshold: Umbral para considerar un nivel de categoría como raro en columnas categó
        
        """

        # Configuración centralizada:
        self.null_threshold = null_threshold
        self.outliers_factor = outliers_factor
        self.rare_threshold = rare_threshold
        
        # Estado opcional para trazabilidad
        self._last_analysis = {}
    
    def analizar_numericas(self, df, columnas):

        """
        Analiza las columnas numéricas de un DataFrame y devuelve un DataFrame con estadísticas descriptivas.
        Parámetros:
        - df: DataFrame de pandas que contiene los datos.
        - columnas: Lista de nombres de columnas numéricas a analizar.
        Retorna:
        - DataFrame con estadísticas descriptivas para cada columna numérica.
        Estadísticas calculadas:
        - media
        - mediana
        - desviación estándar
        - rango (max - min)
        - rango intercuartílico (Q3 - Q1)
        - coeficiente de variación (std / media)
        - curtosis
        - asimetría (skewness)
        Validaciones:
        - Todas las columnas deben existir en el DataFrame.
        - Las columnas deben ser de tipo numérico.
        - No deben contener valores nulos o infinitos.
        
        """
        resultados = {}
        
        # Validar que todas las columnas existan
        missing = set(columnas) - set(df.columns)
        if missing:
            raise ValueError(f'Columnas inexistentes: {missing}')
        
        for columna in columnas:
            datos = df[columna]
            
            # Validar existencia y longitud
            if datos is None or len(datos) == 0:
                raise ValueError(f'Columna {columna} está vacía o es None.')
            
            # Validar tipo numérico (rápido, en C)
            if not pd.api.types.is_numeric_dtype(datos):
                raise ValueError(f'Columna {columna} no es de tipo numérico.')
            
            # Detectar nulos e infinitos
            if datos.isna().sum() > 0:
                raise ValueError(f'Columna {columna} contiene valores nulos.')
            if np.isinf(datos).any():
                raise ValueError(f'Columna {columna} contiene valores infinitos.')
            if datos.dtype == bool:
                raise ValueError(f'Columna {columna} es booleana. Trátela como categórica.')
            
            # Calcular estadísticos
            media = np.mean(datos)
            std = np.std(datos)
            
            # Protección división por cero
            if abs(media) < 1e-8:
                coef_var = float('inf')
            else:
                coef_var = std / media

            if np.ptp(datos) == 0:
                skew = None
                curtosis = None
            else:
                skew = stats.skew(datos)
                curtosis = stats.kurtosis(datos)
            
            resultados[columna] = {
                'media': media,
                'mediana': np.median(datos),
                'desviacion_estandar': std,
                'rango': np.ptp(datos),
                'rango_intercuartilico': np.percentile(datos, 75) - np.percentile(datos, 25),
                'coeficiente_variacion': coef_var,
                'Curtosis': curtosis,
                'Asimetria': skew
            }
        
        df_resultado = pd.DataFrame(resultados).T
        return df_resultado

    def analizar_categoricas(self, df, columnas, rare_threshold=0.01):

        """
        Analiza las columnas categóricas de un DataFrame y devuelve un DataFrame con estadísticas descriptivas.
        Parámetros:
        - df: DataFrame de pandas que contiene los datos.
        - columnas: Lista de nombres de columnas categóricas a analizar.
        - rare_threshold: Umbral para considerar un nivel de categoría como raro.
        Retorna:
        - DataFrame con estadísticas descriptivas para cada columna categórica.
        Estadísticas calculadas:
        - Cardinalidad (número de categorías únicas)
        - Cuota mayoritaria (proporción de la categoría más frecuente)
        - Niveles raros (número de categorías con frecuencia menor al umbral)
        - Distribución de frecuencias (diccionario con categorías y sus proporciones)
        - Número de valores nulos
        - Porcentaje de valores nulos
        Validaciones:
        - Todas las columnas deben existir en el DataFrame.
        - Las columnas deben ser de tipo categórico (no numérico).
        - No deben contener valores infinitos.
        """

        filas = []
        
        # Validar que todas las columnas existan
        missing = set(columnas) - set(df.columns)
        if missing:
            raise ValueError(f'Columnas inexistentes: {missing}')

        for columna in columnas:

            datos = df[columna]
            
            # Validar existencia y longitud
            if datos is None or len(datos) == 0:
                raise ValueError(f'La columna {columna} está vacía o es None.')

            # Validar columna con una sola categoría:
            if datos.nunique(dropna=False) <= 1:
                raise ValueError(f'La columna {columna} tiene una sola categoría, no parece útil para análisis.')

            # Validar infinitos categóricos:
            if datos.isin([np.inf, -np.inf]).any():
                raise ValueError(f'La columna {columna} contiene valores infinitos, lo cual no es válido para categóricas.')

            # Validar tipo numérico (rápido, en C)
            if pd.api.types.is_numeric_dtype(datos):
                raise ValueError(f'La columna {columna} es de tipo numérico.')

            # ---------- CALCULAMOS ESTADÍSTICOS ----------

            # Detectar frecuencias:
            frecuencias = datos.value_counts(normalize=True, dropna=False)
            
            # Detectar nulos:
            nulos = datos.isnull().sum()
            pct_nulos = (nulos/len(datos)) * 100

            if pct_nulos > self.null_threshold * 100:
                self._warnings.append(f'La columna {columna} tiene un {pct_nulos:.2f}% de valores nulos, lo cual supera el umbral de {self.null_threshold*100:.2f}%.')
            
            fila = {
                'Variable': columna,
                'Cardinalidad': datos.nunique(dropna=False),
                'Cuota mayoritaria': frecuencias.iloc[0],
                'Niveles raros': (frecuencias < rare_threshold).sum(),
                'Distribución': frecuencias.to_dict(),
                'Nulos': nulos,
                'Porcentaje nulos': pct_nulos
            }
            filas.append(fila)

        return pd.DataFrame(filas)