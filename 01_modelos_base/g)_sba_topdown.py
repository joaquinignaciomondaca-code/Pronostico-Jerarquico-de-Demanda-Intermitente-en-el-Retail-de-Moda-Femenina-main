# -*- coding: utf-8 -*-
"""
=============================================================================
SEMINARIO DE TÍTULO - INGENIERÍA COMERCIAL (UTFSM, 2026)
Autor: Joaquín Ignacio Mondaca Parada | Profesor Guía: Marcelo Villena Chamorro
Modelo Benchmark: SBA (Syntetos-Boylan Approximation, 2005)
=============================================================================

¿QUÉ HACE ESTE SCRIPT EN SIMPLE?
1. Corrige el sesgo de sobreestimación del método Croston con un factor deflactor (1 - alpha/2).
2. Genera pronósticos insesgados para demanda intermitente a nivel Familia y SKU.
3. Evalúa el desempeño en el período de prueba oficial (febrero a agosto de 2025).
=============================================================================
"""

# =============================================================================
# ---------- ⚙️ PARÁMETROS CONFIGURABLES ----------
# =============================================================================
CUT_DATE = '2025-01-01'        # Fecha de corte de entrenamiento
TEST_START = '2025-02-01'      # Inicio de prueba oficial
TEST_END = '2025-08-01'        # Fin de prueba oficial
SKU_COL = 'SKU_Personalizado'  # Identificador de producto
ALPHA_SBA = 0.10               # Factor de suavizamiento estándar


# =============================================================================
# ---------- [1/5] Librerías y Rutas de Trabajo ----------
# =============================================================================
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
INPUT_PATH = os.path.join(ROOT, 'inputs', 'base_datos.parquet')
RES_DIR = os.path.join(ROOT, 'resultados')
os.makedirs(RES_DIR, exist_ok=True)


# =============================================================================
# ---------- [2/5] Funciones de Carga y Algoritmo SBA ----------
# =============================================================================
def load_and_prepare_data(input_path):
    df = pd.read_parquet(input_path)
    year_col = [c for c in df.columns if 'a' in c.lower() and 'o' in c.lower()][0]
    df['Periodo'] = pd.to_datetime(df[year_col].astype(str) + '-' + df['Mes'].astype(str).str.zfill(2) + '-01')
    return df

def calc_wape(y_true, y_pred):
    sum_true = np.sum(y_true)
    return np.sum(np.abs(y_true - y_pred)) / sum_true if sum_true > 0 else np.nan

def calc_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def sba_method(series, alpha=0.10):
    ts = np.array(series, dtype=float)
    first_non_zero = np.where(ts > 0)[0]
    if len(first_non_zero) == 0:
        return 0.0
    
    z = ts[first_non_zero[0]]
    p = 1.0
    q = 1.0
    
    for t in range(first_non_zero[0] + 1, len(ts)):
        if ts[t] > 0:
            z = z + alpha * (ts[t] - z)
            p = p + alpha * (q - p)
            q = 1.0
        else:
            q += 1.0
            
    return (1.0 - alpha / 2.0) * (z / p) if p > 0 else 0.0


# =============================================================================
# ---------- [3/5] Inferencia y Evaluación del Modelo ----------
# =============================================================================
def main():
    print("=" * 75)
    print("  EJECUTANDO BENCHMARK: SBA (Syntetos-Boylan Approximation, 2005)")
    print("=" * 75)
    t0 = time.time()
    
    print("[1/4] Cargando dataset transaccional y seleccionando muestra Pareto...")
    df = load_and_prepare_data(INPUT_PATH)
    top100 = df[df['Periodo'] < pd.to_datetime(CUT_DATE)].groupby(SKU_COL)['Unidades'].sum().nlargest(100).index
    test_dates = pd.date_range(TEST_START, TEST_END, freq='MS')
    
    print(f"      -> {df['Familia'].nunique()} Familias y 100 SKUs evaluados en {len(test_dates)} meses de test.")

    print("[2/4] Calculando estimaciones SBA a nivel Familia con ventana rodante...")
    fam_ts = df.groupby(['Familia', 'Periodo'])['Unidades'].sum().unstack(fill_value=0)
    records_fam = []
    for fam in fam_ts.index:
        s = fam_ts.loc[fam]
        for t in test_dates:
            history = s[s.index < t]
            pred = sba_method(history, alpha=ALPHA_SBA)
            actual = s[s.index == t].values[0] if t in s.index else 0.0
            records_fam.append({'Familia': fam, 'Periodo': t, 'Real': actual, 'Pred': float(pred)})
            
    df_fam_res = pd.DataFrame(records_fam)
    wape_fam = calc_wape(df_fam_res['Real'].values, df_fam_res['Pred'].values)
    rmse_fam = calc_rmse(df_fam_res['Real'].values, df_fam_res['Pred'].values)

    print("[3/4] Calculando estimaciones SBA a nivel SKU con ventana rodante...")
    df_top = df[df[SKU_COL].isin(top100)].copy()
    sku_ts = df_top.groupby([SKU_COL, 'Periodo'])['Unidades'].sum().unstack(fill_value=0)
    records_sku = []
    for sku in sku_ts.index:
        s = sku_ts.loc[sku]
        for t in test_dates:
            history = s[s.index < t]
            pred = sba_method(history, alpha=ALPHA_SBA)
            actual = s[s.index == t].values[0] if t in s.index else 0.0
            records_sku.append({'SKU': sku, 'Periodo': t, 'Real': actual, 'Pred': float(pred)})
            
    df_sku_res = pd.DataFrame(records_sku)
    wape_sku = calc_wape(df_sku_res['Real'].values, df_sku_res['Pred'].values)
    rmse_sku = calc_rmse(df_sku_res['Real'].values, df_sku_res['Pred'].values)

    print(f"\n[4/4] Resultados Finales de SBA (2005):")
    print(f"      • WAPE Familia : {wape_fam*100:.2f}% | RMSE Familia : {rmse_fam:.2f}")
    print(f"      • WAPE SKU     : {wape_sku*100:.2f}% | RMSE SKU     : {rmse_sku:.2f}")
    print(f"      • Tiempo total : {time.time() - t0:.2f}s")
    print("=" * 75)

if __name__ == '__main__':
    main()
