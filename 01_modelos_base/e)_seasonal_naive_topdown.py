# -*- coding: utf-8 -*-
"""
=============================================================================
SEMINARIO DE TÍTULO - INGENIERÍA COMERCIAL (UTFSM, 2026)
Autor: Joaquín Ignacio Mondaca Parada | Profesor Guía: Marcelo Villena Chamorro
Modelo Benchmark: Seasonal Naive (Heurística Estacional pura, m = 12)
=============================================================================

¿QUÉ HACE ESTE SCRIPT EN SIMPLE?
1. Toma las ventas de hace exactamente 12 meses (mismo mes del año anterior) 
   para cada Familia y cada SKU y las proyecta como pronóstico del mes actual.
2. Evalúa el desempeño en el período de prueba oficial (febrero a agosto de 2025).
3. Sirve como la línea base mínima que cualquier modelo de Machine Learning debe superar.
=============================================================================
"""

# =============================================================================
# ---------- ⚙️ PARÁMETROS CONFIGURABLES ----------
# =============================================================================
CUT_DATE = '2025-01-01'        # Fecha de corte histórico
TEST_START = '2025-02-01'      # Inicio del conjunto de prueba oficial
TEST_END = '2025-08-01'        # Fin del conjunto de prueba oficial
SKU_COL = 'SKU_Personalizado'  # Identificador de producto
SEASONAL_LAG = 12              # Ciclo estacional anual (12 meses)


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
# ---------- [2/5] Funciones de Carga y Métricas ----------
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


# =============================================================================
# ---------- [3/5] Inferencia y Evaluación del Modelo ----------
# =============================================================================
def main():
    print("=" * 75)
    print("  EJECUTANDO BENCHMARK: Seasonal Naive (Heurística Estacional m=12)")
    print("=" * 75)
    t0 = time.time()
    
    print("[1/4] Cargando dataset transaccional y seleccionando muestra Pareto...")
    df = load_and_prepare_data(INPUT_PATH)
    
    # 100 SKUs top-tier del período histórico
    top100 = df[df['Periodo'] < pd.to_datetime(CUT_DATE)].groupby(SKU_COL)['Unidades'].sum().nlargest(100).index
    test_dates = pd.date_range(TEST_START, TEST_END, freq='MS')
    
    print(f"      -> {df['Familia'].nunique()} Familias y 100 SKUs evaluados en {len(test_dates)} meses de test.")

    # Inferencia Familia
    print("[2/4] Calculando pronósticos a nivel Familia con rezago estacional t-12...")
    fam_ts = df.groupby(['Familia', 'Periodo'])['Unidades'].sum().unstack(fill_value=0)
    records_fam = []
    for fam in fam_ts.index:
        s = fam_ts.loc[fam]
        for t in test_dates:
            t_lag = t - pd.DateOffset(months=SEASONAL_LAG)
            pred = s[s.index == t_lag].values[0] if t_lag in s.index else 0.0
            actual = s[s.index == t].values[0] if t in s.index else 0.0
            records_fam.append({'Familia': fam, 'Periodo': t, 'Real': actual, 'Pred': float(pred)})
            
    df_fam_res = pd.DataFrame(records_fam)
    wape_fam = calc_wape(df_fam_res['Real'].values, df_fam_res['Pred'].values)
    rmse_fam = calc_rmse(df_fam_res['Real'].values, df_fam_res['Pred'].values)

    # Inferencia SKU
    print("[3/4] Calculando pronósticos a nivel SKU con rezago estacional t-12...")
    df_top = df[df[SKU_COL].isin(top100)].copy()
    sku_ts = df_top.groupby([SKU_COL, 'Periodo'])['Unidades'].sum().unstack(fill_value=0)
    records_sku = []
    for sku in sku_ts.index:
        s = sku_ts.loc[sku]
        for t in test_dates:
            t_lag = t - pd.DateOffset(months=SEASONAL_LAG)
            pred = s[s.index == t_lag].values[0] if t_lag in s.index else 0.0
            actual = s[s.index == t].values[0] if t in s.index else 0.0
            records_sku.append({'SKU': sku, 'Periodo': t, 'Real': actual, 'Pred': float(pred)})
            
    df_sku_res = pd.DataFrame(records_sku)
    wape_sku = calc_wape(df_sku_res['Real'].values, df_sku_res['Pred'].values)
    rmse_sku = calc_rmse(df_sku_res['Real'].values, df_sku_res['Pred'].values)

    # Resumen
    print(f"\n[4/4] Resultados Finales de Seasonal Naive:")
    print(f"      • WAPE Familia : {wape_fam*100:.2f}% | RMSE Familia : {rmse_fam:.2f}")
    print(f"      • WAPE SKU     : {wape_sku*100:.2f}% | RMSE SKU     : {rmse_sku:.2f}")
    print(f"      • Tiempo total : {time.time() - t0:.2f}s")
    print("=" * 75)

if __name__ == '__main__':
    main()
