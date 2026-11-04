
### 1. Actualización de `README.md`

```markdown
# Análisis Estadístico

Kit de análisis estadístico exploratorio para datasets de Machine Learning, diseñado para uso personal en proyectos de churn prediction.

## Características

Tres funciones principales para auditar tu dataset antes de modelar:

- **Variables numéricas**: media, mediana, desviación, coeficiente de variación, curtosis, asimetría
- **Variables categóricas**: cardinalidad, cuota mayoritaria, niveles raros, distribución completa
- **Detección inteligente de Outliers**: Cálculo automático de límites IQR y alerta si existen valores atípicos.
- **Variable objetivo (Target)**: análisis determinístico de desbalanceo, detección de nulos y categorías raras

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
target_column = "churn"
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

### 6. Analizar la variable objetivo (Target)

Este método es crucial para detectar problemas de desbalanceo antes de entrenar.

```python
resultado_target = analizador.analiza_target(df, target_column)
print(resultado_target)
```

**Salida:**
```
            Tipo  Clases  Nulos  Porcentaje nulos  Ratio desbalanceo
churn    int64        2      0               0.0                2.77
```

**Características del análisis de target:**
- Detecta automáticamente si es numérico o categórico (strings, category).
- Calcula el ratio de desbalanceo (clase mayoritaria / minoritaria).
- Identifica categorías raras (< `rare_threshold`).
- Genera advertencias si hay nulos o clases dominantes.

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
| Limite Inferior | Umbral mínimo (Q1 - k·IQR) para detectar outliers inferiores. |
| Limite Superior | Umbral máximo (Q3 + k·IQR) para detectar outliers superiores. |
| Porcentaje Outliers | Proporción de valores que exceden los límites IQR. |

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

### Variable Objetivo

| Métrica | Descripción |
|---|---|
| Tipo | dtype de la columna (int, float, object, category) |
| Clases | Número de valores únicos |
| Ratio desbalanceo | Cociente entre la clase mayoritaria y la minoritaria |
| Categorías raras | Lista de niveles con frecuencia < umbral |

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

### Variable Objetivo

La función lanza `ValueError` si detecta:
- Columna inexistente
- Columna vacía o `None`
- Valores infinitos
- Categoría con frecuencia cero (vacía)

## Advertencias

El analizador genera advertencias automáticas cuando detecta patrones que requieren atención humana:

1. **Coeficiente de Variación alto** (> 2.0): Dispersión extrema respecto a la media.
2. **Asimetría extrema** (≥ 1.0 o ≤ -1.0): Distribución fuertemente sesgada.
3. **Presencia de Outliers**: Cualquier valor fuera de los límites IQR (revisa si son errores o casos naturales).

```python
# Ejemplo de salida de advertencias
warnings = analizador.warnings
for w in warnings:
    print(w)

# Salida posible:
# • Columna TotalCharges: 1.24% de valores fuera de límites IQR ([-150.00, 12500.00]). Revisar origen.
# • Columna Tenure: Asimetría extrema (-2.15). Considerar transformación si el modelo lo requiere.

## Ejemplo Completo

```python
import pandas as pd
from src.analisisestadistico.src import AnalisisEstadistico

# 1. Cargar datos
df = pd.read_csv("customer_churn.csv")

# 2. Definir variables
num = ["age", "income", "tenure", "monthly_charges"]
cat = ["gender", "contract_type", "payment_method"]
target = "churn"

# 3. Configurar analizador (opcional)
analizador = AnalisisEstadistico(null_threshold=0.10, rare_threshold=0.05)

# 4. Ejecutar análisis
resultados_num = analizador.analizar_numericas(df, num)
resultados_cat = analizador.analizar_categoricas(df, cat)
resultados_target = analizador.analiza_target(df, target)

# 5. Revisar advertencias
if analizador.warnings:
    print("\n⚠️  ADVERTENCIAS:")
    for w in analizador.warnings:
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

**Resultados actuales:** 87/87 tests passing (en desarrollo: tests para `analiza_target`)

## Estructura del Proyecto

```
AnalisisEstadistico/
├── src/
│   └── analisisestadistico/
│       ├── __init__.py
│       └── src.py          # Clase AnalisisEstadistico
├── partition_validator.py  # Próxima implementación: Validador de particiones
├── Test/
│   ├── tests.py
│   └── test_categoricas.py
├── pyproject.toml
├── uv.lock
├── README.md
└── DECISIONES.md
```

## Notas de Uso Personal

- Diseñado para análisis exploratorio previo al modelado.
- **Nuevo**: Análisis determinístico del target para detectar desbalanceo temprano.
- **Nuevo**: Los outliers se detectan con el método IQR (factor configurable). Cualquier desviación lanza una alerta para revisión manual.
- No incluye análisis de particiones (train/test split) aún (próxima clase: `ValidadorParticiones`).
- No incluye recomendaciones de codificación automática.
- Los warnings se acumulan entre llamadas; se recomienda instanciar un nuevo objeto para cada dataset.
- Ideal para identificar variables con problemas antes de entrenar modelos.

## Versión

**Fecha:** 03 Nov 2026  
**Estado:** Prototipo funcional con 87 tests + método `analiza_target`  
**Próxima iteración:** Implementación de clase `ValidadorParticiones`

---

*Desarrollado por Josu con asistencia de Lumo.*
```