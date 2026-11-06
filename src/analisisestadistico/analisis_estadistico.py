# Importamos las librerías necesarias:
import numpy as np
import pandas as pd
from statistics import mode
from scipy import stats

from .funciones import analiza_bernoulli

class AnalisisEstadistico:

    """
    Clase para realizar análisis estadístico de columnas numéricas y categóricas en un DataFrame.
    Proporciona métodos para calcular estadísticas descriptivas, detectar outliers y analizar la variable objetivo (target).
    Permite configurar umbrales para nulos, outliers y categorías raras.
    Mantiene un registro de advertencias generadas durante el análisis para su revisión posterior.
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

        # Lista para almacenar advertencias durante el análisis
        self._warnings = []
        
        # Estado opcional para trazabilidad
        self._last_analysis = {}
    
    def analizar_numericas(self, df: pd.DataFrame, columnas: list) -> pd.DataFrame:

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

            unicos = set(pd.unique(datos.dropna()))
            if len(unicos) == 2 and unicos.issubset({0, 1}):

                fila, mensajes = analiza_bernoulli(datos)
                self._warnings.extend(mensajes)
                resultados[columna] = fila
                continue
            
            # Calcular estadísticos
            media = np.mean(datos)
            std = np.std(datos)
            
            # Protección división por cero
            if abs(media) < 1e-8:
                coef_var = float('inf')
            else:
                coef_var = std / media

                if coef_var > 2.0:
                    self._warnings.append(f"Columna {columna}: Coeficiente de variación alto ({coef_var:.2f}). Revisar si es adecuado para el análisis.")

            if np.ptp(datos) == 0:
                skew = None
                curtosis = None
            else:
                skew = stats.skew(datos)
                if skew >=1.0 or skew <= -1.0:
                    self._warnings.append(f"Columna {columna}: Asimetría extrema ({skew:.2f}). Considerar transformación si hay modelo y este lo requiere.")
                curtosis = stats.kurtosis(datos)

            # Detectar outliers usando el método del rango intercuartílico (IQR)
            q1 = np.percentile(datos, 25)
            q3 = np.percentile(datos, 75)
            iqr = q3 - q1
            limite_inf = q1 - self.outliers_factor * iqr
            limite_sup = q3 + self.outliers_factor * iqr

            outliers_mask = (datos < limite_inf) | (datos > limite_sup)
            pct_outliers = outliers_mask.sum() / len(datos)

            if pct_outliers > 0:
                self._warnings.append(
                    f"Columna {columna}: {pct_outliers:.2%} de valores fuera de límites IQR "
                    f"([{limite_inf:.2f}, {limite_sup:.2f}]). Revisar origen.")
                
            resultados[columna] = {
                'media': media,
                'mediana': np.median(datos),
                'desviacion_estandar': std,
                'rango': np.ptp(datos),
                'rango_intercuartilico': iqr,
                'coeficiente_variacion': coef_var,
                'Curtosis': curtosis,
                'Asimetria': skew,
                'Limite Inferior': limite_inf,
                'Limite Superior': limite_sup,
                'Porcentaje Outliers': pct_outliers
            }
        
        df_resultado = pd.DataFrame(resultados).T
        return df_resultado

    def analizar_categoricas(self, df: pd.DataFrame, columnas: list) -> pd.DataFrame:

        """
        Analiza las columnas categóricas de un DataFrame y devuelve un DataFrame con estadísticas descriptivas.
        Parámetros:
        - df: DataFrame de pandas que contiene los datos.
        - columnas: Lista de nombres de columnas categóricas a analizar.
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
                'Niveles raros': (frecuencias < self.rare_threshold).sum(),
                'Distribución': frecuencias.to_dict(),
                'Nulos': nulos,
                'Porcentaje nulos': pct_nulos
            }
            filas.append(fila)

        return pd.DataFrame(filas)

    def analiza_target(self, df: pd.DataFrame, target: str)-> pd.DataFrame:
        """
        Analiza la variable objetivo (target). Soporta tanto numéricas discretas 
        como categóricas (incluyendo strings).

        Parámetros:

        - df: DataFrame de pandas que contiene los datos.
        - target: Nombre de la columna objetivo a analizar.
        Retorna:

        - DataFrame con estadísticas descriptivas del target.
        Estadísticas calculadas:

        - Tipo de dato.
        - Número de valores nulos.
        - Porcentaje de valores nulos.

        - Para categóricas:
            - Número de clases.
            - Cuota mayoritaria.
            - Distribución de frecuencias.
            - Ratio de desbalanceo (si hay más de una clase).

        - Para numéricas continuas:
            - Media.
            - Mediana.
            - Desviación estándar.
            - Rango (max - min).
            - Coeficiente de variación (std / media).
            
        Validaciones:
        - La columna objetivo debe existir en el DataFrame.
        """

        if target not in df.columns:
            raise ValueError(f'La columna objetivo {target} no existe en el DataFrame.')
        
        datos = df[target]
        
        if datos is None or len(datos) == 0:
            raise ValueError(f'La columna objetivo {target} está vacía.')
        
        if datos.isin([np.inf, -np.inf]).any():
            raise ValueError(f'La columna objetivo {target} contiene valores infinitos.')

        nulos = datos.isnull().sum()
        pct_nulos = (nulos / len(datos)) * 100
        
        # Advertencia de nulos (común a ambos tipos):
        if pct_nulos > self.null_threshold * 100:
            self._warnings.append(f'Target {target}: {pct_nulos:.2f}% de valores nulos (umbral: {self.null_threshold*100}%).')

        resultados = {
            'Tipo': datos.dtype,
            'Nulos': nulos,
            'Porcentaje nulos': pct_nulos
        }

        # Caso 1: Target Categórico (o numérico discreto con pocas clases):
        if datos.dtype == 'object' or str(datos.dtype) == 'category' or (pd.api.types.is_numeric_dtype(datos) and datos.nunique() < 20):
            frecuencias = datos.value_counts(normalize=True, dropna=False)
            conteo_clases = datos.value_counts(dropna=False)
            
            resultados['Clases'] = datos.nunique(dropna=False)
            resultados['Cuota mayoritaria'] = frecuencias.iloc[0]
            resultados['Distribución'] = frecuencias.to_dict()
            
            # Detectar categorías raras:
            categorias_raras = frecuencias[frecuencias < self.rare_threshold].index.tolist()
            if categorias_raras:
                self._warnings.append(f'Target {target} tiene categorías raras (<{self.rare_threshold*100}%): {categorias_raras}')

            # Ratio de desbalanceo (solo si hay más de 1 clase):
            if len(conteo_clases) > 1:
                ratio = round(conteo_clases.max() / conteo_clases.min(), 2)
                resultados['Ratio desbalanceo'] = ratio
            else:
                resultados['Ratio desbalanceo'] = None
                
        # Caso 2: Target numérico continuo:
        elif pd.api.types.is_numeric_dtype(datos):
            media = np.mean(datos)
            std = np.std(datos)
            resultados['Media'] = media
            resultados['Mediana'] = np.median(datos)
            resultados['Desviación estándar'] = std
            resultados['Rango'] = np.ptp(datos)
            
            if abs(media) > 1e-8:
                resultados['Coeficiente variación'] = std / media
            else:
                resultados['Coeficiente variación'] = float('inf')

        else:
            raise ValueError(f'El tipo de dato {datos.dtype} para {target} no está soportado actualmente.')

        return pd.DataFrame(resultados).T

    @property
    def warnings(self):
        """Devuelve una copia de las advertencias acumuladas."""
        return self._warnings.copy()