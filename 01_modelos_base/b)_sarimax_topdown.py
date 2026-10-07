# -*- coding: utf-8 -*-
"""
=============================================================================
SEMINARIO DE TÍTULO - INGENIERÍA COMERCIAL (UTFSM, 2026)
Autor: Joaquín Ignacio Mondaca Parada | Profesor Guía: Marcelo Villena Chamorro
Modelo Benchmark: SARIMAX con Regresores Exógenos (Precio y Descuento)
=============================================================================

¿QUÉ HACE ESTE SCRIPT EN SIMPLE?
1. Incorpora señales comerciales ex-ante (Precio planificado y Descuento del ERP).
2. Ajusta un modelo SARIMAX para cuantificar cuánto aporta la señal de precios
   bajo una estructura lineal clásica frente al SARIMA univariado.
3. Desagrega el pronóstico agregado hacia cada SKU usando proporciones históricas.
=============================================================================
"""

# =============================================================================
# ---------- ⚙️ PARÁMETROS CONFIGURABLES ----------
# =============================================================================
CUT_DATE = '2025-01-01'        # Fecha de corte de entrenamiento
SKU_COL = 'SKU_Personalizado'  # Identificador de producto
SARIMA_ORDER = (1, 1, 1)       # Parámetros ARIMA
SARIMA_SEASONAL = (1, 1, 1, 12)# Componente estacional de 12 meses


# =============================================================================
# ---------- [1/5] Librerías y Rutas de Trabajo ----------
# =============================================================================
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
INPUT_PATH = os.path.join(ROOT, 'inputs', 'base_datos.parquet')
RES_DIR = os.path.join(ROOT, 'resultados')
os.makedirs(RES_DIR, exist_ok=True)


# =============================================================================
# ---------- [2/5] Funciones de Carga y Ajuste ----------
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

def fit_predict_sarimax(train_y, train_x, test_x):
    y_log = np.log1p(train_y.values)
    try:
        model = SARIMAX(
            y_log,
            exog=train_x.values,
            order=SARIMA_ORDER,
            seasonal_order=SARIMA_SEASONAL,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        res = model.fit(disp=False, maxiter=50)
        pred_log = res.forecast(steps=1, exog=test_x.values.reshape(1, -1))[0]
        return max(0.0, np.expm1(pred_log))
    except Exception:
        return max(0.0, float(train_y.iloc[-1]))


# =============================================================================
# ---------- [3/5] Inferencia Temporal y Desagregación ----------
# =============================================================================
def main():
    print("=" * 75)
    print("  EJECUTANDO BENCHMARK: SARIMAX con Regresores Exógenos (Precio Ex-Ante)")
    print("=" * 75)
    t0 = time.time()
    
    print("[1/4] Cargando dataset transaccional unificado...")
    df = load_and_prepare_data(INPUT_PATH)
    
    # Agregación a nivel Familia
    desc_col = [c for c in df.columns if 'descuento' in c.lower()][0]
    fam_df = df.groupby(['Familia', 'Periodo']).agg({
        'Unidades': 'sum',
        'Precio_Estimado': 'mean',
        desc_col: 'mean'
    }).reset_index()
    fam_df.rename(columns={desc_col: 'Descuento_promedio'}, inplace=True)
    
    test_dates = sorted(fam_df[fam_df['Periodo'] >= pd.to_datetime(CUT_DATE)]['Periodo'].unique())
    fam_list = fam_df['Familia'].unique()
    
    print(f"      -> {len(fam_list)} Familias detectadas. Periodo de prueba: {len(test_dates)} meses.")

    print("[2/4] Ajustando SARIMAX con variables comerciales ex-ante...")
    records_fam = []
    for fam in fam_list:
        sub = fam_df[fam_df['Familia'] == fam].sort_values('Periodo').set_index('Periodo')
        for t in test_dates:
            history = sub[sub.index < t]
            current = sub[sub.index == t]
            if len(history) < 12 or len(current) == 0:
                continue
            
            train_y = history['Unidades']
            train_x = history[['Precio_Estimado', 'Descuento_promedio']]
            test_x = current[['Precio_Estimado', 'Descuento_promedio']].iloc[0]
            
            pred = fit_predict_sarimax(train_y, train_x, test_x)
            actual = current['Unidades'].values[0]
            records_fam.append({'Familia': fam, 'Periodo': t, 'Real': actual, 'Pred': float(pred)})
            
    df_fam_res = pd.DataFrame(records_fam)
    wape_fam = calc_wape(df_fam_res['Real'].values, df_fam_res['Pred'].values)
    rmse_fam = calc_rmse(df_fam_res['Real'].values, df_fam_res['Pred'].values)

    print("[3/4] Desagregando a nivel SKU con participaciones históricas...")
    df_train = df[df['Periodo'] < pd.to_datetime(CUT_DATE)]
    fam_tot = df_train.groupby('Familia')['Unidades'].sum()
    sku_tot = df_train.groupby(['Familia', SKU_COL])['Unidades'].sum()
    props = (sku_tot / fam_tot).fillna(0).reset_index()
    props.rename(columns={'Unidades': 'Prop'}, inplace=True)
    
    df_sku_res = df[df['Periodo'] >= pd.to_datetime(CUT_DATE)][['Familia', SKU_COL, 'Periodo', 'Unidades']].copy()
    df_sku_res.rename(columns={'Unidades': 'Real'}, inplace=True)
    df_sku_res = df_sku_res.merge(df_fam_res[['Familia', 'Periodo', 'Pred']], on=['Familia', 'Periodo'], how='left')
    df_sku_res = df_sku_res.merge(props, on=['Familia', SKU_COL], how='left')
    df_sku_res['Prop'] = df_sku_res['Prop'].fillna(0)
    df_sku_res['Pred_SKU'] = df_sku_res['Pred'] * df_sku_res['Prop']
    
    wape_sku = calc_wape(df_sku_res['Real'].values, df_sku_res['Pred_SKU'].values)
    rmse_sku = calc_rmse(df_sku_res['Real'].values, df_sku_res['Pred_SKU'].values)

    print(f"\n[4/4] Resultados Finales de SARIMAX:")
    print(f"      • WAPE Familia : {wape_fam*100:.2f}% | RMSE Familia : {rmse_fam:.2f}")
    print(f"      • WAPE SKU     : {wape_sku*100:.2f}% | RMSE SKU     : {rmse_sku:.2f}")
    print(f"      • Tiempo total : {time.time() - t0:.2f}s")
    print("=" * 75)

if __name__ == '__main__':
    main()
