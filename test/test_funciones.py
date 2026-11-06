import pandas as pd


from analisisestadistico.funciones import analiza_bernoulli

def test_p_correcta():
    serie = pd.Series([0, 0, 0, 1])          # p = 0.25
    fila, mensajes = analiza_bernoulli(serie)
    assert fila['p'] == 0.25
    assert fila['ratio_desbalanceo'] == 3.0

def test_devuelve_mensaje():
    serie = pd.Series([0, 1])
    _, mensajes = analiza_bernoulli(serie)
    assert len(mensajes) == 1
    assert "binaria" in mensajes[0]