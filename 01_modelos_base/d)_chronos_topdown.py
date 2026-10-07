# -*- coding: utf-8 -*-
"""
=============================================================================
SEMINARIO DE TÍTULO - INGENIERÍA COMERCIAL (UTFSM, 2026)
Autor: Joaquín Ignacio Mondaca Parada | Profesor Guía: Marcelo Villena Chamorro
Modelo Benchmark: Amazon Chronos-Bolt Small (Foundation Model Zero-Shot)
=============================================================================

¿QUÉ HACE ESTE SCRIPT EN SIMPLE?
1. Utiliza un modelo fundacional pre-entrenado de Amazon (Chronos-Bolt) basado en Transformers.
2. Realiza inferencia 'Zero-Shot' (sin reentrenamiento local) sobre el historial de cada Familia.
3. Extrae la mediana probabilística (cuantil 0.5) para horizonte h=1 mensual.
4. Desagrega el pronóstico de Familia hacia cada SKU usando la mezcla histórica de catálogo.
=============================================================================
"""

# =============================================================================
# ---------- ⚙️ PARÁMETROS CONFIGURABLES ----------
# =============================================================================
CUT_DATE = '2025-01-01'                    # Fecha de corte de prueba
SKU_COL = 'SKU_Personalizado'              # Identificador de producto
CHRONOS_MODEL = 'amazon/chronos-bolt-small'# Modelo fundacional de HuggingFace


# =============================================================================
# ---------- [1/5] Librerías y Configuración Inicial ----------
# =============================================================================
import os
import sys
import time
import warnings
import torch
import numpy as np
import pandas as pd
from chronos import ChronosBoltPipeline

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
# ---------- [3/5] Inferencia Zero-Shot y Desagregación ----------
# =============================================================================
def main():
    print("=" * 75)
    print("  EJECUTANDO BENCHMARK: Amazon Chronos-Bolt (Foundation Model Zero-Shot)")
    print("=" * 75)
    t0 = time.time()
    
    print("[1/4] Cargando dataset transaccional unificado...")
    df = load_and_prepare_data(INPUT_PATH)
    
    fam_ts = df.groupby(['Familia', 'Periodo'])['Unidades'].sum().unstack(fill_value=0)
    test_dates = [c for c in fam_ts.columns if c >= pd.to_datetime(CUT_DATE)]
    
    print(f"      -> {len(fam_ts)} Familias detectadas. Periodo de prueba: {len(test_dates)} meses.")

    print(f"[2/4] Inicializando pipeline de Chronos-Bolt ({CHRONOS_MODEL})...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = ChronosBoltPipeline.from_pretrained(
        CHRONOS_MODEL,
        device_map=device,
        torch_dtype=torch.float32
    )

    print("[3/4] Generando inferencia probabilística Familia por Familia...")
    records_fam = []
    for fam in fam_ts.index:
        s = fam_ts.loc[fam]
        for t in test_dates:
            history = s[s.index < t].values
            if len(history) < 6:
                pred = float(history[-1]) if len(history) > 0 else 0.0
            else:
                context = torch.tensor(history, dtype=torch.float32).unsqueeze(0)
                forecast = pipeline.predict(context, prediction_length=1)
                pred = float(np.median(forecast[0, :, 0].numpy()))
                
            pred = max(0.0, pred)
            actual = float(s.loc[t])
            records_fam.append({'Familia': fam, 'Periodo': t, 'Real': actual, 'Pred': pred})
            
    df_fam_res = pd.DataFrame(records_fam)
    wape_fam = calc_wape(df_fam_res['Real'].values, df_fam_res['Pred'].values)
    rmse_fam = calc_rmse(df_fam_res['Real'].values, df_fam_res['Pred'].values)

    print("[4/4] Desagregando a nivel SKU con participaciones históricas...")
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

    print(f"\n[OK] Resultados Finales de Amazon Chronos-Bolt:")
    print(f"      • WAPE Familia : {wape_fam*100:.2f}% | RMSE Familia : {rmse_fam:.2f}")
    print(f"      • WAPE SKU     : {wape_sku*100:.2f}% | RMSE SKU     : {rmse_sku:.2f}")
    print(f"      • Tiempo total : {time.time() - t0:.2f}s")
    print("=" * 75)

if __name__ == '__main__':
    main()
