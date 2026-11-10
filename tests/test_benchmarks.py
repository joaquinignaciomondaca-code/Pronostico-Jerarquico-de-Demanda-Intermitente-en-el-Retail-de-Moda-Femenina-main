"""Tests de las fórmulas puras de los benchmarks (datos sintéticos, sin el parquet).

Verifican la corrección matemática de los primitivos usados por los scripts de
`01_modelos_base/`: WAPE/RMSE y los métodos de Croston y SBA (Syntetos-Boylan).
"""
import importlib.util
import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(rel_path, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cro = load('01_modelos_base/f)_croston_topdown.py', 'croston')
sba = load('01_modelos_base/g)_sba_topdown.py', 'sba')
# Las funciones calc_wape/calc_rmse son idénticas entre scripts; se importan del
# módulo ligero (solo numpy/pandas) para no arrastrar statsmodels/torch al CI.


# --- WAPE / RMSE ---
def test_wape_prediccion_perfecta_es_cero():
    r = np.array([10, 0, 50, 30])
    assert cro.calc_wape(r, r.copy()) == 0.0


def test_wape_todo_error():
    r = np.array([10, 20])
    assert cro.calc_wape(r, np.zeros(2)) == 1.0


def test_wape_demanda_cero_no_divide():
    # suma real = 0 -> NaN por convención, nunca Inf/error
    r = np.array([0, 0])
    assert np.isnan(cro.calc_wape(r, np.array([1, 1])))


def test_rmse_basico():
    assert cro.calc_rmse(np.array([2.0, 4.0]), np.array([0.0, 0.0])) == pytest.approx(np.sqrt((4 + 16) / 2))


# --- Croston (1972) ---
def test_croston_serie_sin_demanda_es_cero():
    assert cro.croston_method(np.zeros(36)) == 0.0


def test_croston_serie_constante_recupera_la_constante():
    pred = cro.croston_method(np.full(36, 5.0))
    assert pred == pytest.approx(5.0)


def test_croston_nunca_negativo():
    rng = np.random.default_rng(42)
    ts = (rng.random(48) < 0.3) * rng.integers(1, 20, 48)
    assert cro.croston_method(ts.astype(float)) >= 0.0


# --- SBA (Syntetos-Boylan, 2005) ---
def test_sba_aplica_deflactor_1_menos_alpha_mitad():
    rng = np.random.default_rng(7)
    ts = (rng.random(60) < 0.35) * rng.integers(1, 30, 60)
    ts = ts.astype(float)
    assert sba.sba_method(ts) == pytest.approx((1 - 0.10 / 2) * cro.croston_method(ts))


def test_sba_inter_croston_implícito():
    # con demanda intermitente, SBA debe ser <= Croston (deflactor < 1)
    ts = np.array([0, 4, 0, 0, 8, 0, 2, 0, 0, 0, 6, 0] * 4, dtype=float)
    assert sba.sba_method(ts) < cro.croston_method(ts)


def test_sba_serie_sin_demanda_es_cero():
    assert sba.sba_method(np.zeros(24)) == 0.0
