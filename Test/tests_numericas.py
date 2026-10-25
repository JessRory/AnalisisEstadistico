"""
Tests unitarios para AnalisisEstadistico.analizar_numericas()

Cómo ejecutarlos:
    uv add --dev pytest
    uv run pytest -v

Organización:
    - TestCalculosCorrectos ......... ¿calcula bien lo que dice calcular?
    - TestProteccionDivisionPorCero . comportamiento cuando la media ~ 0
    - TestValidacionDeEntrada ....... ¿rechaza lo que debe rechazar?
    - TestCasosBorde ................ constantes, 1 observación, enteros
    - TestContratoDeSalida .......... forma, índice, orden, inmutabilidad
    - TestComportamientoDocumentado . trampas conocidas (bool, duplicados)

NOTA: ajusta la línea de import a tu estructura real. Si usas layout src/
con el paquete instalado en modo editable, la ruta de abajo funciona tal cual.
"""

import numpy as np
import pandas as pd
import pytest

# Ajusta esto a la ubicación real de tu clase:
from analisisestadistico.src import AnalisisEstadistico


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def analizador():
    """Instancia nueva por test: aislamiento total del estado interno.

    Si tu __init__ exige parámetros, pásalos aquí con sus valores por defecto.
    """
    return AnalisisEstadistico()


@pytest.fixture
def df_simple():
    """Dataset pequeño con valores verificables a mano."""
    return pd.DataFrame({
        "uno_al_cinco": [1, 2, 3, 4, 5],           # media 3, std sqrt(2)
        "negativos": [-10.0, -5.0, 0.0, 5.0, 10.0],  # media 0 -> dispara protección
        "constante": [7.0, 7.0, 7.0, 7.0, 7.0],      # std 0
        "categoria": ["a", "b", "a", "c", "b"],      # no numérica
    })


# ---------------------------------------------------------------------------
# 1. Cálculos correctos
# ---------------------------------------------------------------------------

class TestCalculosCorrectos:

    def test_estadisticos_de_serie_conocida(self, analizador, df_simple):
        """[1,2,3,4,5]: todos los valores son deducibles a mano."""
        res = analizador.analizar_numericas(df_simple, ["uno_al_cinco"])
        fila = res.loc["uno_al_cinco"]

        assert fila["media"] == pytest.approx(3.0)
        assert fila["mediana"] == pytest.approx(3.0)
        assert fila["desviacion_estandar"] == pytest.approx(np.sqrt(2.0))
        assert fila["rango"] == pytest.approx(4.0)
        assert fila["rango_intercuartilico"] == pytest.approx(2.0)
        assert fila["coeficiente_variacion"] == pytest.approx(np.sqrt(2.0) / 3.0)
        # Distribución simétrica: asimetría 0, curtosis en exceso -1.3
        assert fila["Asimetria"] == pytest.approx(0.0, abs=1e-12)
        assert fila["Curtosis"] == pytest.approx(-1.3, abs=1e-12)

    def test_concuerda_con_pandas_describe(self, analizador):
        """Contraste contra una implementación independiente (pandas).

        OJO al detalle: describe()["std"] usa ddof=1 (muestral) y np.std usa
        ddof=0 (poblacional). El factor de conversión es sqrt((n-1)/n).
        """
        rng = np.random.default_rng(seed=42)
        valores = rng.normal(loc=50.0, scale=10.0, size=500)
        df = pd.DataFrame({"normal": valores})

        res = analizador.analizar_numericas(df, ["normal"])
        desc = df["normal"].describe()
        n = len(valores)

        assert res.loc["normal", "media"] == pytest.approx(desc["mean"])
        assert res.loc["normal", "mediana"] == pytest.approx(desc["50%"])
        assert res.loc["normal", "desviacion_estandar"] == pytest.approx(
            desc["std"] * np.sqrt((n - 1) / n)
        )

    def test_dtype_entero(self, analizador):
        """Enteros puros: la media debe salir decimal, no truncada."""
        df = pd.DataFrame({"enteros": [1, 2, 3, 4]})
        res = analizador.analizar_numericas(df, ["enteros"])
        assert res.loc["enteros", "media"] == pytest.approx(2.5)

    def test_coeficiente_variacion_negativo_con_media_negativa(self, analizador):
        """Con pérdidas (media negativa) el CV sale negativo: es matemática
        correcta, pero te obliga a interpretarlo con cuidado."""
        df = pd.DataFrame({"perdidas": [-10.0, -20.0, -30.0]})
        res = analizador.analizar_numericas(df, ["perdidas"])
        assert res.loc["perdidas", "coeficiente_variacion"] < 0


# ---------------------------------------------------------------------------
# 2. Protección de la división por cero
# ---------------------------------------------------------------------------

class TestProteccionDivisionPorCero:

    def test_media_exactamente_cero_devuelve_infinito(self, analizador, df_simple):
        """[-10,-5,0,5,10] tiene media 0 y std 7.07: sin protección -> inf/nan."""
        res = analizador.analizar_numericas(df_simple, ["negativos"])
        cv = res.loc["negativos", "coeficiente_variacion"]
        assert np.isinf(cv)
        assert cv > 0

    def test_media_dentro_del_umbral_dispara_la_proteccion(self, analizador):
        """Media 1.5e-9 < 1e-8: debe activarse la protección."""
        df = pd.DataFrame({"casi_cero": [1e-9, 2e-9]})
        res = analizador.analizar_numericas(df, ["casi_cero"])
        assert np.isinf(res.loc["casi_cero", "coeficiente_variacion"])

    def test_media_fuera_del_umbral_no_dispara_la_proteccion(self, analizador):
        """Media 2e-6 > 1e-8: valor finito y correcto (CV = 0.5)."""
        df = pd.DataFrame({"pequena": [1e-6, 3e-6]})
        res = analizador.analizar_numericas(df, ["pequena"])
        cv = res.loc["pequena", "coeficiente_variacion"]
        assert np.isfinite(cv)
        assert cv == pytest.approx(0.5)

    def test_el_resto_de_estadisticos_sigue_siendo_valido(self, analizador, df_simple):
        """La protección no debe contaminar las otras métricas."""
        res = analizador.analizar_numericas(df_simple, ["negativos"])
        fila = res.loc["negativos"]
        assert fila["media"] == pytest.approx(0.0)
        assert fila["mediana"] == pytest.approx(0.0)
        assert fila["desviacion_estandar"] == pytest.approx(np.sqrt(50.0))


# ---------------------------------------------------------------------------
# 3. Validación de entrada
# ---------------------------------------------------------------------------

class TestValidacionDeEntrada:

    def test_columna_inexistente(self, analizador, df_simple):
        with pytest.raises(ValueError, match="inexistentes"):
            analizador.analizar_numericas(df_simple, ["no_existe"])

    def test_el_error_nombra_la_columna_que_falta(self, analizador, df_simple):
        with pytest.raises(ValueError) as exc:
            analizador.analizar_numericas(df_simple, ["uno_al_cinco", "fantasma"])
        assert "fantasma" in str(exc.value)

    def test_varias_columnas_faltantes_en_un_solo_error(self, analizador, df_simple):
        """Debe reportar todas de golpe, no fallar en la primera."""
        with pytest.raises(ValueError) as exc:
            analizador.analizar_numericas(df_simple, ["fantasma", "espectro"])
        mensaje = str(exc.value)
        assert "fantasma" in mensaje and "espectro" in mensaje

    def test_columna_de_texto(self, analizador, df_simple):
        with pytest.raises(ValueError, match="no es de tipo numérico"):
            analizador.analizar_numericas(df_simple, ["categoria"])

    def test_numeros_guardados_como_texto(self, analizador):
        """Trampa clásica: '1','2','3' parecen números pero son object."""
        df = pd.DataFrame({"codigo": ["1", "2", "3"]})
        with pytest.raises(ValueError, match="no es de tipo numérico"):
            analizador.analizar_numericas(df, ["codigo"])

    def test_columna_de_fechas(self, analizador):
        """datetime64 no es numérico para pandas: debe rechazarse."""
        df = pd.DataFrame({"fecha": pd.to_datetime(["2024-01-01", "2024-01-02"])})
        with pytest.raises(ValueError, match="no es de tipo numérico"):
            analizador.analizar_numericas(df, ["fecha"])

    def test_columna_con_nulos(self, analizador):
        df = pd.DataFrame({"con_nulos": [1.0, np.nan, 3.0]})
        with pytest.raises(ValueError, match="nulos"):
            analizador.analizar_numericas(df, ["con_nulos"])

    def test_columna_con_infinito_positivo(self, analizador):
        df = pd.DataFrame({"con_inf": [1.0, np.inf, 3.0]})
        with pytest.raises(ValueError, match="infinitos"):
            analizador.analizar_numericas(df, ["con_inf"])

    def test_columna_con_infinito_negativo(self, analizador):
        """El bug original: -np.isinf(...).any() no detectaba -inf."""
        df = pd.DataFrame({"con_inf": [1.0, -np.inf, 3.0]})
        with pytest.raises(ValueError, match="infinitos"):
            analizador.analizar_numericas(df, ["con_inf"])

    def test_columna_vacia(self, analizador):
        df = pd.DataFrame({"vacia": pd.Series([], dtype="float64")})
        with pytest.raises(ValueError, match="vacía"):
            analizador.analizar_numericas(df, ["vacia"])

    def test_dataframe_vacio(self, analizador):
        """Sin columnas, cualquier nombre pedido es inexistente."""
        with pytest.raises(ValueError, match="inexistentes"):
            analizador.analizar_numericas(pd.DataFrame(), ["cualquiera"])

    def test_la_validacion_ocurre_antes_de_calcular(self, analizador, df_simple):
        """Una columna inválida al final no debe dejar resultados a medias:
        al lanzar excepción, no hay retorno parcial."""
        with pytest.raises(ValueError):
            analizador.analizar_numericas(
                df_simple, ["uno_al_cinco", "constante", "categoria"]
            )


# ---------------------------------------------------------------------------
# 4. Casos borde
# ---------------------------------------------------------------------------

class TestCasosBorde:

    def test_columna_constante(self, analizador, df_simple):
        res = analizador.analizar_numericas(df_simple, ["constante"])
        fila = res.loc["constante"]

        assert fila["media"] == pytest.approx(7.0)
        assert fila["mediana"] == pytest.approx(7.0)
        assert fila["desviacion_estandar"] == pytest.approx(0.0)
        assert fila["rango"] == pytest.approx(0.0)
        assert fila["rango_intercuartilico"] == pytest.approx(0.0)
        assert fila["coeficiente_variacion"] == pytest.approx(0.0)

        # Asimetría y curtosis de una constante: scipy devuelve 0.0 o NaN
        # según la versión (división 0/0 internamente). Ejecuta el test,
        # mira qué te sale y ENDURECE esta aserción con el valor real.
        assert np.isnan(fila["Asimetria"]) or fila["Asimetria"] == 0
        assert np.isnan(fila["Curtosis"]) or fila["Curtosis"] == 0

    def test_una_sola_observacion(self, analizador):
        df = pd.DataFrame({"unico": [42.0]})
        res = analizador.analizar_numericas(df, ["unico"])
        assert res.loc["unico", "media"] == pytest.approx(42.0)
        assert res.loc["unico", "desviacion_estandar"] == pytest.approx(0.0)
        assert res.loc["unico", "rango"] == pytest.approx(0.0)

    def test_dos_observaciones(self, analizador):
        df = pd.DataFrame({"par": [10.0, 20.0]})
        res = analizador.analizar_numericas(df, ["par"])
        assert res.loc["par", "media"] == pytest.approx(15.0)
        assert res.loc["par", "mediana"] == pytest.approx(15.0)

    def test_valores_negativos_y_positivos(self, analizador):
        df = pd.DataFrame({"mixto": [-100.0, 50.0, 150.0]})
        res = analizador.analizar_numericas(df, ["mixto"])
        assert res.loc["mixto", "rango"] == pytest.approx(250.0)

    def test_escala_muy_grande(self, analizador):
        """Valores enormes: comprobar que no hay overflow silencioso."""
        df = pd.DataFrame({"grande": [1e15, 2e15, 3e15]})
        res = analizador.analizar_numericas(df, ["grande"])
        assert res.loc["grande", "media"] == pytest.approx(2e15)

    def test_lista_de_columnas_vacia(self, analizador, df_simple):
        """Sin columnas que analizar: DataFrame vacío, no excepción."""
        res = analizador.analizar_numericas(df_simple, [])
        assert res.empty


# ---------------------------------------------------------------------------
# 5. Contrato de salida
# ---------------------------------------------------------------------------

class TestContratoDeSalida:

    METRICAS_ESPERADAS = {
        "media",
        "mediana",
        "desviacion_estandar",
        "rango",
        "rango_intercuartilico",
        "coeficiente_variacion",
        "Curtosis",
        "Asimetria",
    }

    def test_una_fila_por_columna(self, analizador, df_simple):
        cols = ["uno_al_cinco", "negativos", "constante"]
        res = analizador.analizar_numericas(df_simple, cols)
        assert res.shape[0] == len(cols)
        assert list(res.index) == cols

    def test_metricas_exactas(self, analizador, df_simple):
        """Si renombras o añades métricas, este test te avisa."""
        res = analizador.analizar_numericas(df_simple, ["uno_al_cinco"])
        assert set(res.columns) == self.METRICAS_ESPERADAS

    def test_respeta_el_orden_solicitado(self, analizador, df_simple):
        """El orden de salida debe seguir el de entrada, no el del df."""
        res = analizador.analizar_numericas(df_simple, ["constante", "uno_al_cinco"])
        assert list(res.index) == ["constante", "uno_al_cinco"]

    def test_devuelve_un_dataframe(self, analizador, df_simple):
        res = analizador.analizar_numericas(df_simple, ["uno_al_cinco"])
        assert isinstance(res, pd.DataFrame)

    def test_no_modifica_el_dataframe_original(self, analizador, df_simple):
        """Read-only: el df de entrada debe quedar intacto."""
        copia = df_simple.copy(deep=True)
        analizador.analizar_numericas(df_simple, ["uno_al_cinco", "constante"])
        pd.testing.assert_frame_equal(df_simple, copia)

    def test_dos_llamadas_dan_el_mismo_resultado(self, analizador, df_simple):
        """Idempotencia: sin efectos colaterales entre llamadas."""
        primera = analizador.analizar_numericas(df_simple, ["uno_al_cinco"])
        segunda = analizador.analizar_numericas(df_simple, ["uno_al_cinco"])
        pd.testing.assert_frame_equal(primera, segunda)


# ---------------------------------------------------------------------------
# 6. Comportamiento documentado (trampas conocidas)
# ---------------------------------------------------------------------------

class TestComportamientoDocumentado:
    """Tests que fijan el comportamiento ACTUAL, aunque sea discutible.

    No son fallos: son decisiones implícitas de tu código. Si algún día
    cambias el comportamiento, estos tests te obligarán a hacerlo a propósito.
    """

    def test_columna_booleana_lanza_excepcion(self, analizador):
        df = pd.DataFrame({"flag": [True, False, True, True]})
        with pytest.raises(ValueError, match="booleana"):
            analizador.analizar_numericas(df, ["flag"])

    def test_columna_repetida_se_colapsa_en_una_fila(self, analizador, df_simple):
        """El diccionario interno usa el nombre como clave: pedir la misma
        columna dos veces produce UNA fila, no dos."""
        res = analizador.analizar_numericas(
            df_simple, ["uno_al_cinco", "uno_al_cinco"]
        )
        assert res.shape[0] == 1

    def test_df_con_nombres_de_columna_duplicados(self, analizador):
        """Punto ciego conocido: si el propio df tiene nombres repetidos,
        df[nombre] devuelve un DataFrame en vez de una Serie y la función
        falla con un error poco descriptivo. Arréglalo validando la unicidad
        de df.columns al inicio, no aquí."""
        df = pd.DataFrame([[1, 2], [3, 4]], columns=["x", "x"])
        with pytest.raises(Exception):
            analizador.analizar_numericas(df, ["x"])

    def test_el_infinito_del_cv_contamina_agregaciones_posteriores(self, analizador, df_simple):
        """Consecuencia de usar float('inf') en vez de None: cualquier
        agregación posterior sobre esa columna se vuelve infinita."""
        res = analizador.analizar_numericas(df_simple, ["negativos", "uno_al_cinco"])
        assert np.isinf(res["coeficiente_variacion"]).any()
        assert np.isinf(res["coeficiente_variacion"].mean())

    def test_skew_kurtosis_son_None_en_columna_constante(self, analizador, df_simple):
        res = analizador.analizar_numericas(df_simple, ["constante"])
        assert pd.isna(res.loc["constante", "Curtosis"]) or res.loc["constante", "Curtosis"] is None
        assert pd.isna(res.loc["constante", "Asimetria"]) or res.loc["constante", "Asimetria"] is None

    def test_skew_kurtosis_son_validas_en_columna_variable(self, analizador, df_simple):
        res = analizador.analizar_numericas(df_simple, ["uno_al_cinco"])
        assert np.isfinite(res.loc["uno_al_cinco", "Curtosis"])
        assert np.isfinite(res.loc["uno_al_cinco", "Asimetria"])