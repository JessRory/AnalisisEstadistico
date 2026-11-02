# Análisis Estadístico

Kit de análisis estadístico exploratorio para datasets de Machine Learning, diseñado para uso personal en proyectos de churn prediction.

## Características

Dos funciones principales para auditar tu dataset antes de modelar:

- **Variables numéricas**: media, mediana, desviación, coeficiente de variación, curtosis, asimetría
- **Variables categóricas**: cardinalidad, cuota mayoritaria, niveles raros, distribución completa

## Instalación

```bash
cd ~/Documentos/Programming/AnalisisEstadistico
uv sync
```

Activa el entorno virtual (opcional):
```bash
source .venv/bin/activate
```

## Uso Básico

### 1. Importar la clase

```python
from src.analisisestadistico.src import AnalisisEstadistico
```

### 2. Configurar umbrales (opcional)

```python
analizador = AnalisisEstadistico(
    null_threshold=0.05,      # 5% de nulos como umbral de advertencia
    outliers_factor=1.5,      # Factor IQR para outliers
    rare_threshold=0.01       # 1% como umbral para nivel raro
)
```

Si no especificas nada, se usan los valores por defecto.

### 3. Preparar tus datos

Carga tu dataset y separa las columnas por tipo:

```python
import pandas as pd

df = pd.read_csv("tu_dataset.csv")

columnas_numericas = ["edad", "ingresos", "tiempo_servicio"]
columnas_categoricas = ["genero", "region", "canal_adquisicion"]
```

### 4. Analizar variables numéricas

```python
resultado_num = analizador.analizar_numericas(df, columnas_numericas)
print(resultado_num)
```

**Salida:**
```
           media  mediana  desviacion_estandar  rango  ...  Asimetria
edad         35.2     34.0               12.5   55.0  ...       0.32
ingresos    45000   42000               15000   80000  ...      -0.15
...
```

### 5. Analizar variables categóricas

```python
resultado_cat = analizador.analizar_categoricas(df, columnas_categoricas)
print(resultado_cat)
```

**Salida:**
```
           Variable  Cardinalidad  Cuota mayoritaria  Niveles raros  Nulos  ...
genero       genero             2                0.62              0      0  ...
region      region            47                0.18             12      5  ...
...
```

La columna `Distribución` contiene un diccionario con las frecuencias relativas de cada nivel. Para inspeccionarla:

```python
distribucion_region = resultado_cat.loc[resultado_cat['Variable'] == 'region', 'Distribución'].iloc[0]
print(distribucion_region['Norte'])  # Muestra frecuencia de Norte
```

## Métricas Calculadas

### Numéricas

| Métrica | Descripción |
|---|---|
| media | Promedio aritmético |
| mediana | Valor central (50% percentile) |
| desviacion_estandar | Dispersión alrededor de la media |
| rango | Máximo - Mínimo |
| rango_intercuartilico | P75 - P25 (dispersión del 50% central) |
| coeficiente_variacion | σ/μ (dispersión relativa) |
| Curtosis | Forma de la cola de la distribución |
| Asimetria | Sesgo de la distribución |

**Protecciones automáticas:**
- Si media ≈ 0: coeficiente de variación = ∞
- Si varianza = 0: Curtosis y Asimetría = `None`

### Categóricas

| Métrica | Descripción |
|---|---|
| Cardinalidad | Número total de niveles (incluye nulos) |
| Cuota mayoritaria | Proporción del nivel más frecuente |
| Niveles raros | Cuenta de niveles por debajo del umbral configurado |
| Nulos | Cantidad absoluta de valores ausentes |
| Porcentaje nulos | Proporción de nulos sobre el total |
| Distribución | Diccionario con frecuencia relativa por nivel |

## Validaciones

### Numéricas

La función lanza `ValueError` si detecta:
- Columna inexistente
- Columna vacía o `None`
- dtype no numérico (texto, fecha, booleanos)
- Valores nulos o infinitos
- Booleanos explícitos (se tratan como categóricas)

### Categóricas

La función lanza `ValueError` si detecta:
- Columna inexistente
- Columna vacía o `None`
- Solo una categoría (incluso contando nulos)
- Valores infinitos
- dtype numérico (se rechaza)

## Advertencias

Si alguna columna supera los umbrales configurables, el analizador registra advertencias internas:

```python
analizador.analizar_numericas(df, ["edad", "ingresos"])
warnings = analizador.get_warnings()
for w in warnings:
    print(w)
```

**Ejemplo de aviso:**
```
Columna ingresos tiene 40.50% de valores nulos, lo cual supera el umbral de 5.00%.
```

## Ejemplo Completo

```python
import pandas as pd
from src.analisisestadistico.src import AnalisisEstadistico

# 1. Cargar datos
df = pd.read_csv("customer_churn.csv")

# 2. Definir variables
num = ["age", "income", "tenure", "monthly_charges"]
cat = ["gender", "contract_type", "payment_method"]

# 3. Configurar analizador (opcional)
analizador = AnalisisEstadistico(null_threshold=0.10, rare_threshold=0.05)

# 4. Ejecutar análisis
resultados_num = analizador.analizar_numericas(df, num)
resultados_cat = analizador.analizar_categoricas(df, cat)

# 5. Revisar advertencias
if analizador.get_warnings():
    print("\n⚠️  ADVERTENCIAS:")
    for w in analizador.get_warnings():
        print(f"  • {w}")

# 6. Inspeccionar distribución específica
distribucion = resultados_cat.loc[resultados_cat['Variable'] == 'contract_type', 'Distribución'].iloc[0]
for nivel, frecuencia in distribucion.items():
    print(f"{nivel}: {frecuencia:.2%}")
```

## Tests

Ejecutar la suite completa de pruebas:

```bash
uv run pytest -v
```

Ejecutar solo tests de numéricas:
```bash
uv run pytest -v Test/tests.py
```

Ejecutar solo tests de categóricas:
```bash
uv run pytest -v Test/test_categoricas.py
```

**Resultados actuales:** 87/87 tests passing

## Estructura del Proyecto

```
AnalisisEstadistico/
├── src/
│   └── analisisestadistico/
│       ├── __init__.py
│       └── src.py
├── Test/
│   ├── tests.py
│   └── test_categoricas.py
├── pyproject.toml
├── uv.lock
├── README.md
└── DECISIONES.md
```

## Notas de Uso Personal

- Diseñado para análisis exploratorio previo al modelado
- No incluye análisis de particiones (train/test split)
- No incluye recomendaciones de codificación automática
- Los warnings se acumulan entre llamadas; usar `reset_warnings()` si es necesario
- Ideal para identificar variables con problemas antes de entrenar modelos

## Versión

**Fecha:** 24 Oct 2026  
**Estado:** Prototipo funcional con 87 tests  
**Próxima iteración:** Integración con dataset de churn real

---

*Desarrollado por Josu con asistencia de Lumo.*