¡Por supuesto, Josu! Tienes toda la razón: un LOG con "(ver versión anterior)" es contraproducente, porque justamente el LOG sirve para **no depender de versiones anteriores**. Aquí tienes el documento completo, con las secciones 3 y 4 restauradas desde tu LOG del 24 Oct (que me pasaste antes) y la nueva sección de outliers/asimetría integrada como sección 5, renumerando el resto:

```markdown
# Log de Decisiones — Análisis Estadístico

**Fecha:** 23 Oct 2026 (actualizado 04 Nov 2026)  
**Proyecto:** AnalisisEstadistico  
**Autor:** Josu (con apoyo de Lumo)

---

## 1. Infraestructura del Proyecto

| Decisión | Elección | Razón |
|---|---|---|
| Gestor de paquetes | uv | Moderno, rápido, consistente con proyectos actuales |
| Estructura de directorios | Layout `src/` | uv lo exige por defecto cuando hay sistema de construcción |
| Control de versiones | Git | Iniciado automáticamente por uv, seguimiento estándar |
| Tipo de proyecto | `--app` (empaquetado) | Permite futuras utilidades reutilizables, aunque sea uso personal |
| Instalación | Editable (`uv pip install -e`) | Los cambios en el repo se reflejan al instante en los entornos virtuales consumidores |

**Reflejo técnico:**  
```
AnalisisEstadistico/
├── src/<br>
│   └── analisisestadistico/<br>
│       ├── __init__.py<br>
│       └── src.py          # contiene la clase AnalisisEstadistico<br>
├── partition_validator.py  # Próxima implementación<br>
├── Test/<br>
│   ├── tests.py            # batería de pruebas para numéricas (38 tests)<br>
│   └── test_categoricas.py # batería de pruebas para categóricas (49 tests)<br>
├── pyproject.toml<br>
├── uv.lock<br>
├── README.md<br>
└── DECISIONES.md<br>
```

---

## 2. Diseño de la Clase

| Decisión | Elección | Razón |
|---|---|---|
| Patrón arquitectónico | Clase con métodos de instancia (self) | Flexibilidad futura para umbrales configurables y estado compartido |
| Flujo de datos | Inmutable (devuelve nuevos objetos) | Seguridad de datos originales, trazabilidad entre ejecuciones |
| Configuración | Umbrales inicializados en `__init__` | Cambiar criterios sin tocar el código de cálculo |
| Estado interno | `_warnings = []` en `__init__` | Registro de advertencias sin detener la ejecución |

**Constructor (implementado):**
```python
def __init__(self, null_threshold=0.05, outliers_factor=1.5, rare_threshold=0.01)
```

**Acceso a advertencias:**
- Se expone como propiedad `warnings` (devuelve copia de la lista) para seguridad.
- Se mantiene método `obtener_warnings_df()` para exportación estructurada.
- Se eliminó `get_warnings()` antiguo por ambigüedad de nombre.

---

## 3. Función `analizar_numericas()`

### 3.1. Contratos definidos

| Aspecto | Decisión |
|---|---|
| Parámetros | `(self, df, columnas)` — DataFrame explícito + lista de nombres |
| Salida | DataFrame pandas transpuesto (filas = variables, columnas = métricas) |
| Modificación de datos | Read-only: nunca altera el DataFrame original |

### 3.2. Validaciones implementadas

| Validación | Acción |
|---|---|
| Columnas inexistentes | `ValueError` con listado completo |
| Columna vacía o None | `ValueError` |
| dtype no numérico | `ValueError` |
| Valores nulos (NA) | `ValueError` |
| Valores infinitos (+/- inf) | `ValueError` |
| dtype booleano | `ValueError` — se trata como categórica |

### 3.3. Protecciones numéricas

| Caso degenerado | Tratamiento |
|---|---|
| Media ≈ 0 (abs(media) < 1e-8) | Coeficiente de variación = `float('inf')` |
| Varianza ≈ 0 (rango == 0) | Skew y Kurtosis = `None` (no se calculan) |

### 3.4. Métricas calculadas

- Media, Mediana, Desviación estándar
- Rango (max - min), Rango intercuartílico (P75 - P25)
- Coeficiente de variación (σ/μ, protegido)
- Curtosis, Asimetría (Skewness)
- Límite Inferior, Límite Superior, Porcentaje Outliers (nuevo, ver sección 5)

### 3.5. Tests unitarios

**38 pruebas** pasadas, cubriendo:
- Cálculos correctos con series conocidas
- Protección de división por cero
- Validación de entrada (tipos, nulos, infinitos, booleanos)
- Casos borde (constante, 1 observación, escala grande)
- Contrato de salida (forma, orden, inmutabilidad)
- Comportamiento documentado (booleanas rechazadas, skew/kurtosis None en constantes)

**Pendiente:** actualizar tests para cubrir las nuevas métricas de outliers y las nuevas reglas de warning.

---

## 4. Función `analizar_categoricas()`

### 4.1. Objetivos definidos (sesión 24 Oct)

Auditar estructura del espacio de estados, **no calcular momentos estadísticos**. Análisis exploratorio previo al modelado, no evaluación de impacto en el modelo.

### 4.2. Patrones de diseño adoptados

| Decisión | Elección | Razón |
|---|---|---|
| Patrón de acumulación | Lista de filas, no dict incremental | Tipos de datos consistentes, orden garantizado |
| Nulos en análisis | `value_counts(dropna=False)` | Nulos son una categoría más, coherencia total |
| Particiones | Sin soporte (`partitions` eliminado) | Análisis exploratorio previo al modelado, no cross-partición |
| Validación de monotipo | `nunique(dropna=False) <= 1` | Consistente con el resto de la función |

**Corrección histórica:** inicialmente la validación usaba `nunique()` sin `dropna`, lo que generaba inconsistencia con la distribución que sí incluía nulos. Se corrigió en sesión 24 Oct para coherencia total.

### 4.3. Métricas calculadas

| Categoría | Métricas |
|---|---|
| **Cardinalidad** | Total de niveles distintos (incluyendo nulos) |
| **Desbalance** | Quota de la mayoritaria (proporción del nivel top) |
| **Cola larga** | Número de niveles raros (< 1% configurable) |
| **Nulos** | Conteo absoluto y porcentaje |
| **Distribución** | Dict con frecuencia relativa por nivel |

### 4.4. Validaciones implementadas

- Columna vacía → `ValueError`
- Columna con 1 sola categoría (contando nulos) → `ValueError`
- Infinitos en categórico → `ValueError`
- Tipo numérico detectado → `ValueError`
- Mix de tipos dentro de la columna → Rechazado (objeto con inf mezclado)

### 4.5. Formato de salida

- DataFrame con métricas (filas = variables)
- Columna `Distribución` incrustada con JSON de frecuencias
- Columnas con tipos consistentes (`int`, `float`) gracias al patrón de lista

### 4.6. Tests unitarios

**49 pruebas** pasadas, cubriendo:
- Validación de entrada (columnas inexistentes, numéricas, monotipo, infinitos, strings puros)
- Frecuencias y cardinalidad (coherencia interna, suma de proporciones = 1)
- Nulos (conteo, porcentaje, como clave en distribución, umbral configurable)
- Niveles raros (umbral exacto, nulos como raros, cola larga)
- Contrato de salida (forma, orden, inmutabilidad, tipos limpios, serialización)
- Comportamiento documentado (booleanas como numéricas, números como texto aceptados)

---

## 5. Mejoras en `analizar_numericas()`: Outliers y Asimetría

**Fecha:** 04 Nov 2026  
**Motivación:** Transformar el análisis de "solo estadísticos" a "análisis orientado a la investigación". El objetivo no es solo calcular, sino señalar áreas que requieren revisión manual por parte del científico de datos.

### 5.1. Nuevas métricas calculadas

| Métrica | Fórmula / Cálculo | Propósito |
|---|---|---|
| **Limite Inferior** | `Q1 - k * IQR` | Umbral mínimo para considerar un valor como outlier (inferior). |
| **Limite Superior** | `Q3 + k * IQR` | Umbral máximo para considerar un valor como outlier (superior). |
| **Porcentaje Outliers** | `(valores fuera de límites) / N` | Proporción de datos que exceden los límites IQR. |

*Donde `k` es el parámetro `outliers_factor` configurado en el constructor (por defecto 1.5).*

### 5.2. Nuevas reglas de advertencia (Warnings)

| Condición | Umbral | Acción recomendada |
|---|---|---|
| **Coeficiente de Variación alto** | `CV > 2.0` | Evaluar si la variable tiene dispersión excesiva respecto a la media (común en datos monetarios). |
| **Asimetría extrema** | `skew ≥ 1.0` o `skew ≤ -1.0` | Investigar el sesgo. Plantear transformaciones (log, sqrt) si el modelo lo requiere. |
| **Existencia de outliers** | `Pct Outliers > 0` | **Investigación obligatoria:** determinar si son errores de entrada o valores naturales de cola pesada. |

### 5.3. Decisión de diseño: warning con outliers > 0%

Se decidió lanzar un warning **siempre que exista al menos un outlier**, aunque el porcentaje sea mínimo.
- **Razón:** Obliga al analista a localizar esos valores y estudiarlos expresamente. No se asume que son "ruido" sin preguntarse antes si son errores o casos legítimos del negocio.
- **Impacto:** Fomenta un conocimiento más profundo del dataset y evita limpiezas automáticas indiscriminadas.
- **Nota sobre el umbral de asimetría:** se eligió 1.0 (regla empírica estándar) para evitar saturar de avisos en datasets donde la asimetría moderada (0.5–1.0) es natural. Si resulta demasiado laxo en la práctica, se ajustará en una iteración futura.

### 5.4. Reutilización activa de `outliers_factor`

El parámetro `outliers_factor` del constructor estaba definido pero sin uso (código muerto). Con esta iteración pasa a utilizarse activamente en el cálculo de los límites IQR.

### 5.5. Corrección técnica

Se corrigió un `KeyError` en la construcción del diccionario `resultados`: se intentaban asignar claves parciales (`resultados[col]['key'] = ...`) antes de crear la entrada de la columna. La solución fue construir el diccionario completo en una sola asignación al final del bucle.

### 5.6. Estado

- ✅ Cálculo de límites IQR y porcentaje de outliers.
- ✅ Warnings para CV > 2.0, |skew| ≥ 1.0 y presencia de outliers.
- ✅ Columnas nuevas en el DataFrame de salida.
- ⚠️  Pendiente: tests para las nuevas métricas y reglas de warning.

---

## 6. Función `analiza_target()`

### 6.1. Objetivo
Realizar un análisis determinístico de la variable objetivo antes del modelado para detectar:
- Desbalanceo severo de clases.
- Presencia de nulos (crítico en target).
- Categorías raras o inesperadas.

### 6.2. Diseño Adoptado

| Aspecto | Decisión | Razón |
|---|---|---|
| Soporte de tipos | Híbrido (Numérico + Categórico) | Targets reales vienen como strings ('Yes'/'No') o category, no solo int/float |
| Detección de tipo | `dtype == 'object'` o `str(dtype) == 'category'` | Evita dependencia de `is_categorical_dtype()` obsoleta en pandas 2.1+ |
| Lógica de rareza | Aplicada solo a distribuciones discretas | No tiene sentido en variables continuas |
| Retorno | DataFrame transpuesto | Consistencia con otros métodos de la clase |
| Validación crítica | Rechazo de categoría con frecuencia 0 | Indica error en los datos o en la extracción |

### 6.3. Correcciones Históricas (03 Nov 2026)
- **Error inicial**: Uso de `pd.api.types.is_categorical_dtype()` que fallaba en pandas reciente.
  - **Solución**: Cambio a comprobación directa de `str(datos.dtype) == 'category'`.
- **Error lógico**: Cálculo de "categorías raras" en targets numéricos continuos.
  - **Solución**: Condicionar la lógica de rareza solo a casos discretos/categóricos.
- **Error de retorno**: Falta de `return` explícito.
  - **Solución**: Añadir `return pd.DataFrame(resultados).T`.

### 6.4. Métricas Calculadas

| Métrica | Descripción |
|---|---|
| Tipo | dtype original de la columna |
| Clases | Número de valores únicos |
| Nulos / % Nulos | Conteo y proporción de valores ausentes |
| Ratio desbalanceo | Máximo / Mínimo de conteos por clase |
| Categorías raras | Lista de niveles con frecuencia < `rare_threshold` |

### 6.5. Próximos Pasos
- Añadir tests unitarios específicos para `analiza_target` (casos: strings, category, numérico discreto, nulos, desbalanceo extremo).

---

## 7. Planificación: Clase `ValidadorParticiones`

Se ha decidido crear una nueva clase separada (`partition_validator.py`) para manejar validaciones de train/test split, ya que no es "análisis estadístico puro".

**Funcionalidades previstas:**
- Verificación de que `train` y `test` no compartan índices (evitar leakage).
- Comprobación de que la proporción de la clase positiva se mantiene tras `stratify`.
- Validación de reproducibilidad con `random_state`.
- Detección de filas duplicadas repartidas entre particiones.
- Integración con `AnalisisEstadistico` para reutilizar métricas de desbalanceo.

**Nombre provisional:** `ValidadorParticiones` o `PartitionValidator`.

---

## 8. Decisiones sobre Advertencias (Warnings)

| Tema | Situación actual | Decisión tomada |
|---|---|---|
| _warnings como estado | Implementado, se usa para avisos de umbral | **Mantener** para análisis iterativo |
| get_warnings() | Existía, nombre ambiguo | **Eliminado** |
| warnings (propiedad) | Nuevo acceso seguro | **Implementado** (devuelve copia) |
| obtener_warnings_df() | Método auxiliar | **Mantenido** para exportación |
| Warnings en numéricas | Antes vacío siempre (solo excepciones) | Ahora informa de CV, asimetría y outliers (sesión 04 Nov) |

**Nota:** Las advertencias se acumulan entre llamadas. Se recomienda instanciar un nuevo objeto `AnalisisEstadistico` para cada dataset nuevo para evitar contaminación cruzada de logs.

---

## 9. Próximos Hitos

| Paso | Acción | Prioridad |
|---|---|---|
| 1 | Consolidar proyecto (git commit) | ✅ Completado |
| 2 | Actualizar tests de numéricas (outliers, warnings) | Alta |
| 3 | Implementar tests para `analiza_target` | Alta |
| 4 | Integración con dataset real de churn | Media |
| 5 | Notebook de EDA que use las funciones | Media |
| 6 | Diseñar e implementar `ValidadorParticiones` | Alta |
| 7 | Publicación en GitHub (opcional) | Baja |

---

## 10. Referencias Técnicas

- Código: `~/Documentos/Programming/AnalisisEstadistico/`
- Tests numéricas: `Test/tests.py` (38 pruebas)
- Tests categóricas: `Test/test_categoricas.py` (49 pruebas)
- Documentación de uv: https://docs.astral.sh/uv/
- Pandas: https://pandas.pydata.org/docs/
- SciPy stats: https://docs.scipy.org/doc/scipy/reference/stats.html

---

*Última actualización: 04 Nov 2026*
```
