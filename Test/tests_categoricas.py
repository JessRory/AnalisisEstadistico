"""
Tests unitarios para AnalisisEstadistico.analizar_categoricas()

Cómo ejecutarlos (desde la carpeta Test/):
    uv run pytest -v test_categoricas.py
    uv run pytest -v                      # corre también los de numéricas

Solo este archivo, solo una clase, o solo un test:
    uv run pytest -v test_categoricas.py::TestNulos
    uv run pytest -v -k "nulos" test_categoricas.py

Organización:
    - TestValidacionDeEntrada ....... qué debe rechazar y con qué mensaje
    - TestFrecuenciasYCardinalidad .. conteos, proporciones, coherencia interna
    - TestNulos ..................... nulos como categoría + warning por umbral
    - TestNivelesRaros .............. umbral de rareza, incluido el límite exacto
    - TestContratoDeSalida .......... forma, orden, inmutabilidad, idempotencia
    - TestComportamientoDocumentado . decisiones implícitas y tensiones de diseño

AJUSTA la línea de import a tu estructura real.
"""

import numpy as np
import pandas as pd
import pytest

from analisisestadistico.src import AnalisisEstadistico


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def tiene_clave_nula(diccionario):
    """True si el dict de distribución tiene NaN como clave.

    No se puede usar `np.nan in diccionario` porque nan != nan: la
    comparación de identidad falla siempre. Hay que recorrer las claves.
    """
    return any(pd.isna(clave) for clave in diccionario.keys())


def suma_de_frecuencias(diccionario):
    """Suma de las proporciones, ignorando las claves nulas."""
    return sum(v for k, v in diccionario.items() if not pd.isna(k))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def analizador():
    """Instancia nueva por test: los _warnings no se arrastran entre tests."""
    return AnalisisEstadistico()


@pytest.fixture
def df_cat():
    """Dataset categórico con casos verificables a mano."""
    return pd.DataFrame({
        # 3 niveles: Madrid 3/5 = 0.6, Lisboa 0.2, Oporto 0.2
        "ciudad": ["Madrid", "Lisboa", "Madrid", "Oporto", "Madrid"],
        # 2 niveles equilibrados: M 0.6, F 0.4
        "genero": ["M", "F", "M", "F", "M"],
        # numérica: debe rechazarse
        "edad": [10, 20, 30, 40, 50],
        # un solo nivel: debe rechazarse
        "monotipo": ["A", "A", "A", "A", "A"],
        # 2 niveles + 40% de nulos
        "con_nulos": ["X", "Y", None, "X", None],
        # booleana: debe rechazarse como numérica
        "flag": [True, False, True, True, False],
    })


# ---------------------------------------------------------------------------
# 1. Validación de entrada
# ---------------------------------------------------------------------------

class TestValidacionDeEntrada:

    def test_columna_inexistente(self, analizador, df_cat):
        with pytest.raises(ValueError, match="inexistentes"):
            analizador.analizar_categoricas(df_cat, ["no_existe"])

    def test_reporta_todas_las_columnas_faltantes_a_la_vez(self, analizador, df_cat):
        with pytest.raises(ValueError) as exc:
            analizador.analizar_categoricas(df_cat, ["fantasma", "espectro"])
        mensaje = str(exc.value)
        assert "fantasma" in mensaje and "espectro" in mensaje

    def test_columna_numerica(self, analizador, df_cat):
        with pytest.raises(ValueError, match="numérico"):
            analizador.analizar_categoricas(df_cat, ["edad"])

    def test_columna_con_un_solo_nivel(self, analizador, df_cat):
        with pytest.raises(ValueError, match="una sola categoría"):
            analizador.analizar_categoricas(df_cat, ["monotipo"])

    def test_columna_vacia(self, analizador):
        df = pd.DataFrame({"vacia": pd.Series([], dtype="object")})
        with pytest.raises(ValueError, match="vacía"):
            analizador.analizar_categoricas(df, ["vacia"])

    def test_dataframe_sin_columnas(self, analizador):
        with pytest.raises(ValueError, match="inexistentes"):
            analizador.analizar_categoricas(pd.DataFrame(), ["cualquiera"])

    def test_columna_de_strings_puros_no_explota(self, analizador):
        """REGRESIÓN: el bug de np.isinf() sobre dtype object.

        Con np.isinf(datos) esta llamada lanzaba
        TypeError: ufunc 'isinf' not supported for the input types.
        Con isin([np.inf, -np.inf]) funciona sin problema.
        """
        df = pd.DataFrame({"ciudad": ["Madrid", "Lisboa", "Oporto"]})
        res = analizador.analizar_categoricas(df, ["ciudad"])
        assert res.loc[0, "Cardinalidad"] == 3

    def test_infinito_mezclado_con_strings(self, analizador):
        """Object dtype con un inf dentro: debe cazarlo el chequeo de infinitos."""
        df = pd.DataFrame({"raro": [1.0, np.inf, "texto"]})
        with pytest.raises(ValueError, match="infinitos"):
            analizador.analizar_categoricas(df, ["raro"])

    def test_infinito_negativo(self, analizador):
        df = pd.DataFrame({"raro": [1.0, -np.inf, "texto"]})
        with pytest.raises(ValueError, match="infinitos"):
            analizador.analizar_categoricas(df, ["raro"])

    def test_una_columna_mala_aborta_todo(self, analizador, df_cat):
        """No debe devolver resultados parciales si algo falla."""
        with pytest.raises(ValueError):
            analizador.analizar_categoricas(df_cat, ["ciudad", "genero", "edad"])

    def test_lista_de_columnas_vacia(self, analizador, df_cat):
        res = analizador.analizar_categoricas(df_cat, [])
        assert res.empty


# ---------------------------------------------------------------------------
# 2. Frecuencias y cardinalidad
# ---------------------------------------------------------------------------

class TestFrecuenciasYCardinalidad:

    def test_cardinalidad(self, analizador, df_cat):
        res = analizador.analizar_categoricas(df_cat, ["ciudad", "genero"])
        assert res.loc[0, "Cardinalidad"] == 3
        assert res.loc[1, "Cardinalidad"] == 2

    def test_cuota_mayoritaria(self, analizador, df_cat):
        """Madrid aparece 3 de 5 veces -> 0.6."""
        res = analizador.analizar_categoricas(df_cat, ["ciudad"])
        assert res.loc[0, "Cuota mayoritaria"] == pytest.approx(0.6)

    def test_cuota_mayoritaria_es_el_maximo_de_la_distribucion(self, analizador, df_cat):
        res = analizador.analizar_categoricas(df_cat, ["ciudad", "genero"])
        for _, fila in res.iterrows():
            maximo = max(fila["Distribución"].values())
            assert fila["Cuota mayoritaria"] == pytest.approx(maximo)

    def test_distribucion_contiene_todos_los_niveles(self, analizador, df_cat):
        res = analizador.analizar_categoricas(df_cat, ["ciudad"])
        distribucion = res.loc[0, "Distribución"]
        assert set(distribucion.keys()) == {"Madrid", "Lisboa", "Oporto"}

    def test_proporciones_suman_uno(self, analizador, df_cat):
        """value_counts(normalize=True) ya normaliza: no hay que dividir otra vez."""
        res = analizador.analizar_categoricas(df_cat, ["ciudad", "genero"])
        for _, fila in res.iterrows():
            assert suma_de_frecuencias(fila["Distribución"]) == pytest.approx(1.0)

    def test_cardinalidad_coincide_con_numero_de_claves(self, analizador, df_cat):
        """Coherencia interna: nunique(dropna=False) == len(distribución).

        Este es el test que habría cazado la inconsistencia entre
        nunique() (excluye nulos) y value_counts(dropna=False) (los incluye).
        """
        res = analizador.analizar_categoricas(df_cat, ["ciudad", "genero", "con_nulos"])
        for _, fila in res.iterrows():
            assert fila["Cardinalidad"] == len(fila["Distribución"])

    def test_niveles_equilibrados(self, analizador):
        df = pd.DataFrame({"par": ["A", "B"] * 50})
        res = analizador.analizar_categoricas(df, ["par"])
        assert res.loc[0, "Cuota mayoritaria"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 3. Nulos
# ---------------------------------------------------------------------------

class TestNulos:

    def test_conteo_de_nulos(self, analizador, df_cat):
        """con_nulos tiene 2 None de 5 filas."""
        res = analizador.analizar_categoricas(df_cat, ["con_nulos"])
        assert res.loc[0, "Nulos"] == 2

    def test_porcentaje_de_nulos(self, analizador, df_cat):
        res = analizador.analizar_categoricas(df_cat, ["con_nulos"])
        assert res.loc[0, "Porcentaje nulos"] == pytest.approx(40.0)

    def test_nulos_aparecen_como_clave_en_la_distribucion(self, analizador, df_cat):
        res = analizador.analizar_categoricas(df_cat, ["con_nulos"])
        assert tiene_clave_nula(res.loc[0, "Distribución"])

    def test_sin_nulos_no_hay_clave_nula(self, analizador, df_cat):
        res = analizador.analizar_categoricas(df_cat, ["ciudad"])
        assert not tiene_clave_nula(res.loc[0, "Distribución"])

    def test_cardinalidad_incluye_nulos(self, analizador, df_cat):
        """X, Y y NaN -> 3 niveles."""
        res = analizador.analizar_categoricas(df_cat, ["con_nulos"])
        assert res.loc[0, "Cardinalidad"] == 3

    def test_warning_cuando_los_nulos_superan_el_umbral(self, analizador, df_cat):
        """40% de nulos > 5% por defecto: debe registrarse un aviso."""
        analizador.analizar_categoricas(df_cat, ["con_nulos"])
        assert len(analizador._warnings) == 1
        assert "con_nulos" in analizador._warnings[0]

    def test_sin_warning_cuando_los_nulos_estan_bajo_el_umbral(self, analizador, df_cat):
        analizador.analizar_categoricas(df_cat, ["ciudad"])
        assert analizador._warnings == []

    def test_umbral_configurable_en_el_constructor(self, df_cat):
        """Con umbral 0.50, un 40% de nulos ya no dispara el aviso."""
        permisivo = AnalisisEstadistico(null_threshold=0.50)
        permisivo.analizar_categoricas(df_cat, ["con_nulos"])
        assert permisivo._warnings == []

    def test_los_warnings_se_acumulan_entre_columnas(self, analizador):
        df = pd.DataFrame({
            "a": ["X", None, None, None, None],   # 80% nulos
            "b": ["Y", None, None, None, None],   # 80% nulos
        })
        analizador.analizar_categoricas(df, ["a", "b"])
        assert len(analizador._warnings) == 2

    def test_el_warning_no_detiene_el_analisis(self, analizador, df_cat):
        """Avisar no es abortar: la fila debe devolverse igualmente."""
        res = analizador.analizar_categoricas(df_cat, ["con_nulos"])
        assert res.shape[0] == 1


# ---------------------------------------------------------------------------
# 4. Niveles raros
# ---------------------------------------------------------------------------

class TestNivelesRaros:

    def test_sin_niveles_raros_con_umbral_por_defecto(self, analizador, df_cat):
        """Lisboa y Oporto están al 20%, muy por encima del 1%."""
        res = analizador.analizar_categoricas(df_cat, ["ciudad"])
        assert res.loc[0, "Niveles raros"] == 0

    def test_umbral_alto_marca_niveles_como_raros(self, analizador, df_cat):
        """Con umbral 0.25, Lisboa y Oporto (0.2) pasan a ser raros."""
        res = analizador.analizar_categoricas(df_cat, ["ciudad"], rare_threshold=0.25)
        assert res.loc[0, "Niveles raros"] == 2

    def test_limite_exacto_del_umbral(self, analizador):
        """Un nivel al 1% exacto NO es raro con umbral 0.01: la comparación es <."""
        df = pd.DataFrame({"x": ["A"] * 99 + ["B"]})
        res = analizador.analizar_categoricas(df, ["x"], rare_threshold=0.01)
        assert res.loc[0, "Niveles raros"] == 0

    def test_justo_por_debajo_del_umbral(self, analizador):
        """Con umbral 0.011, ese mismo nivel al 1% sí es raro."""
        df = pd.DataFrame({"x": ["A"] * 99 + ["B"]})
        res = analizador.analizar_categoricas(df, ["x"], rare_threshold=0.011)
        assert res.loc[0, "Niveles raros"] == 1

    def test_los_nulos_pueden_contar_como_nivel_raro(self, analizador):
        """Consistencia: si los nulos son una categoría, también pueden ser raros."""
        df = pd.DataFrame({"x": ["A"] * 980 + [None] * 20})
        res = analizador.analizar_categoricas(df, ["x"], rare_threshold=0.05)
        assert res.loc[0, "Niveles raros"] == 1

    def test_cola_larga(self, analizador):
        """Muchos niveles con frecuencia diminuta: escenario típico de churn."""
        valores = ["comun"] * 900 + [f"raro_{i}" for i in range(100)]
        df = pd.DataFrame({"canal": valores})
        res = analizador.analizar_categoricas(df, ["canal"])
        assert res.loc[0, "Cardinalidad"] == 101
        assert res.loc[0, "Niveles raros"] == 100


# ---------------------------------------------------------------------------
# 5. Contrato de salida
# ---------------------------------------------------------------------------

class TestContratoDeSalida:

    COLUMNAS_ESPERADAS = {
        "Variable",
        "Cardinalidad",
        "Cuota mayoritaria",
        "Niveles raros",
        "Distribución",
        "Nulos",
        "Porcentaje nulos",
    }

    def test_devuelve_un_dataframe(self, analizador, df_cat):
        res = analizador.analizar_categoricas(df_cat, ["ciudad"])
        assert isinstance(res, pd.DataFrame)

    def test_columnas_exactas(self, analizador, df_cat):
        res = analizador.analizar_categoricas(df_cat, ["ciudad"])
        assert set(res.columns) == self.COLUMNAS_ESPERADAS

    def test_una_fila_por_columna(self, analizador, df_cat):
        res = analizador.analizar_categoricas(df_cat, ["ciudad", "genero", "con_nulos"])
        assert res.shape[0] == 3

    def test_preserva_el_orden_de_entrada(self, analizador, df_cat):
        res = analizador.analizar_categoricas(df_cat, ["genero", "ciudad"])
        assert list(res["Variable"]) == ["genero", "ciudad"]

    def test_no_modifica_el_dataframe_original(self, analizador, df_cat):
        copia = df_cat.copy(deep=True)
        analizador.analizar_categoricas(df_cat, ["ciudad", "con_nulos"])
        pd.testing.assert_frame_equal(df_cat, copia)

    def test_dos_llamadas_dan_el_mismo_resultado(self, analizador, df_cat):
        primera = analizador.analizar_categoricas(df_cat, ["ciudad"])
        segunda = analizador.analizar_categoricas(df_cat, ["ciudad"])
        pd.testing.assert_frame_equal(primera, segunda)

    def test_tipos_de_datos_limpios(self, analizador, df_cat):
        """Ventaja del patrón de lista: las métricas conservan su tipo."""
        res = analizador.analizar_categoricas(df_cat, ["ciudad", "con_nulos"])
        assert pd.api.types.is_integer_dtype(res["Cardinalidad"])
        assert pd.api.types.is_integer_dtype(res["Nulos"])
        assert pd.api.types.is_float_dtype(res["Cuota mayoritaria"])
        assert pd.api.types.is_float_dtype(res["Porcentaje nulos"])

    def test_distribucion_es_un_dict_serializable(self, analizador, df_cat):
        """La columna Distribución debe poder volcarse a JSON sin sorpresas."""
        import json
        res = analizador.analizar_categoricas(df_cat, ["ciudad"])
        distribucion = res.loc[0, "Distribución"]
        assert isinstance(distribucion, dict)
        # Las claves numéricas y los valores float son serializables;
        # las claves NaN no lo son en JSON estricto, pero sí en pandas.to_json.
        assert json.dumps({str(k): float(v) for k, v in distribucion.items()})


# ---------------------------------------------------------------------------
# 6. Comportamiento documentado (decisiones implícitas)
# ---------------------------------------------------------------------------

class TestComportamientoDocumentado:
    """Fijan el comportamiento ACTUAL, aunque en algún caso sea discutible.

    Si cambias el diseño, estos tests te obligarán a hacerlo a propósito.
    """

    def test_booleana_se_rechaza_con_mensaje_de_numerica(self, analizador, df_cat):
        """pandas considera bool un subtipo de numérico, así que cae en esa
        validación. El mensaje dice 'numérico' aunque sea un flag: aceptable,
        pero conviene saberlo."""
        with pytest.raises(ValueError, match="numérico"):
            analizador.analizar_categoricas(df_cat, ["flag"])

    def test_numeros_como_texto_se_aceptan_como_categoricos(self, analizador):
        """Códigos guardados como string: dtype object, se tratan como
        categóricos. Es lo deseable para un identificador de zona o producto."""
        df = pd.DataFrame({"codigo_postal": ["28001", "08001", "28001"]})
        res = analizador.analizar_categoricas(df, ["codigo_postal"])
        assert res.loc[0, "Cardinalidad"] == 2

    def test_un_nivel_mas_nulos_se_acepta_con_dos_niveles(self, analizador):
        """Contrato actual: la validación de monotipo usa nunique(dropna=False),
        coherente con el resto de la función. Un valor real + nulos = DOS niveles,
        así que se analiza en lugar de rechazarse."""
        df = pd.DataFrame({"x": ["A", "A", None, "A"]})
        res = analizador.analizar_categoricas(df, ["x"])
        assert res.loc[0, "Cardinalidad"] == 2
        assert res.loc[0, "Nulos"] == 1
        assert tiene_clave_nula(res.loc[0, "Distribución"])

    def test_columna_solo_nulos_da_mensaje_enganoso(self, analizador):
        """nunique() de una columna toda nula es 0, así que cae en la validación
        de monotipo y el mensaje habla de 'una sola categoría'. El diagnóstico
        es correcto (la columna es inútil) pero el motivo no es el que dice."""
        df = pd.DataFrame({"x": [None, None, None]})
        with pytest.raises(ValueError, match="una sola categoría"):
            analizador.analizar_categoricas(df, ["x"])

    def test_el_orden_de_las_validaciones_importa(self, analizador):
        """Una columna numérica CON infinitos se rechaza por infinitos, no por
        numérica: el chequeo de inf va antes en el código."""
        df = pd.DataFrame({"x": [1.0, np.inf, 3.0]})
        with pytest.raises(ValueError, match="infinitos"):
            analizador.analizar_categoricas(df, ["x"])

    def test_el_parametro_del_metodo_tapa_al_de_la_instancia(self, analizador):
        """DUPLICACIÓN: rare_threshold existe en el constructor (self) y en la
        firma del método. El del método SIEMPRE gana, así que self.rare_threshold
        nunca se usa. O unificas (quitarlo del método) o documentas que el
        método manda. Aquí se fija el comportamiento actual."""
        estricto = AnalisisEstadistico(rare_threshold=0.5)
        df = pd.DataFrame({"x": ["A"] * 96 + ["B"] * 4})   # B al 4%
        # Sin pasar el parámetro, se usa el 0.01 por defecto del método:
        res = estricto.analizar_categoricas(df, ["x"])
        assert res.loc[0, "Niveles raros"] == 0            # 4% no es < 1%

    def test_la_variable_es_columna_no_indice(self, analizador, df_cat):
        """INCONSISTENCIA con analizar_numericas(): allí el nombre de la
        variable acaba como índice del DataFrame; aquí es una columna más.
        Ninguna opción es incorrecta, pero conviene unificarlas si vas a
        concatenar o comparar las dos salidas."""
        res = analizador.analizar_categoricas(df_cat, ["ciudad"])
        assert "Variable" in res.columns
        assert res.index.tolist() == [0]