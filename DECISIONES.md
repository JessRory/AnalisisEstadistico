# Log de Decisiones — Análisis Estadístico

**Fecha:** 23 Oct 2026 (actualizado 24 Oct 2026)  
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

**Reflejo técnico:**  
```
AnalisisEstadistico/
├── src/
│   └── analisisestadistico/
│       ├── __init__.py
│       └── src.py          # contiene la clase AnalisisEstadistico
├── Test/
│   ├── tests.py            # batería de pruebas para numéricas (38 tests)
│   └── test_categoricas.py # batería de pruebas para categóricas (49 tests)
├── pyproject.toml
├── uv.lock
├── README.md
└── DECISIONES.md
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

**Nota:** `get_warnings()` y `reset_warnings()` existen pero no se usan activamente. Son candidatos a eliminación en futuras iteraciones si no aportan valor real en el flujo de trabajo.

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

### 3.5. Tests unitarios

**38 pruebas** pasadas, cubriendo:
- Cálculos correctos con series conocidas
- Protección de división por cero
- Validación de entrada (tipos, nulos, infinitos, booleanos)
- Casos borde (constante, 1 observación, escala grande)
- Contrato de salida (forma, orden, inmutabilidad)
- Comportamiento documentado (booleanas rechazadas, skew/kurtosis None en constantes)

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

## 5. Decisiones sobre Advertencias (Warnings)

| Tema | Situación actual | Decisión pendiente |
|---|---|---|
| _warnings como estado | Implementado, se usa para superar el umbral de nulos | ¿Eliminar o mantener solo para avisos globales? |
| get_warnings() | Existe, no usado activamente | Candidato a eliminación si no aporta valor |
| reset_warnings() | Existe, no usado activamente | Candidato a eliminación |

**Nota:** Las advertencias por columna están registradas en el objeto interno pero no se integran en el DataFrame de retorno. Si en el futuro se quiere integrar, se recomienda usar registros estructurados (columna, tipo, valor) en vez de frases formateadas.

---

## 6. Próximos Hitos

| Paso | Acción | Prioridad |
|---|---|---|
| 1 | Consolidar proyecto (git commit) | ✅ Completado |
| 2 | Integración con dataset real de churn | Media |
| 3 | Notebook de EDA que use las funciones | Media |
| 4 | Evaluar eliminación de `get_warnings()`/`reset_warnings()` | Baja |
| 5 | Publicación en GitHub (opcional) | Baja |

---

## 7. Notas de Diseño (no obligatorias)

| Tema | Situación actual | Decisión pendiente |
|---|---|---|
| Validación vs. cálculo | Se calcula en bucle único, aborta al primer error | ¿Validar todo antes de calcular? (UX mejora) |
| Columnas binarias int | Aceptadas como numéricas (0/1) | ¿Rechazar explícitamente? (más restrictivo) |
| Separación de responsabilidades | Función única por tipo de variable | ¿Crear sub-clases para sub-tareas? |
| Nombre de variable en salida | Numéricas: índice; Categóricas: columna | ¿Unificar el comportamiento? |

---

## 8. Referencias Técnicas

- Código: `~/Documentos/Programming/AnalisisEstadistico/`
- Tests numéricas: `Test/tests.py` (38 pruebas)
- Tests categóricas: `Test/test_categoricas.py` (49 pruebas)
- Documentación de uv: https://docs.astral.sh/uv/
- Pandas: https://pandas.pydata.org/docs/
- SciPy stats: https://docs.scipy.org/doc/scipy/reference/stats.html

---

*Última actualización: 24 Oct 2026*