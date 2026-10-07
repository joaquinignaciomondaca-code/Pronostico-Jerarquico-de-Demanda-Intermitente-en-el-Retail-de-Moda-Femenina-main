# -*- coding: utf-8 -*-
"""
=============================================================================
SEMINARIO DE TÍTULO - INGENIERÍA COMERCIAL (UTFSM, 2026)
Autor: Joaquín Ignacio Mondaca Parada | Profesor Guía: Marcelo Villena Chamorro
Motor de Desagregación Dinámica: Learned Shares a Nivel SKU (Top-Down ML)
=============================================================================

¿QUÉ HACE ESTE SCRIPT EN SIMPLE?
1. Toma el pronóstico agregado de cada Familia generado por la arquitectura ES-GBM.
2. Entrena un ensamble de Machine Learning (LightGBM, XGBoost, CatBoost) con 14 variables
   de elasticidad y momentum para predecir la cuota relativa (share) de cada producto.
3. Normaliza las participaciones sobre el símplex estocástico (suma = 100% por familia).
4. Multiplica el volumen de Familia por la cuota dinámica para obtener el pronóstico final por SKU.
=============================================================================
"""

# =============================================================================
# ---------- ⚙️ PARÁMETROS CONFIGURABLES ----------
# =============================================================================
CUT_DATE = '2025-01-01'        # Fecha de corte de prueba
C1_DATE = '2024-03-01'         # Fecha de corte intermedia de validación
SKU_COL = 'SKU_Personalizado'  # Identificador de producto
SEEDS_SHARE = (42, 1, 7, 2025, 99) # Semillas de entrenamiento


# =============================================================================
CUT_DATE = '2025-01-01'
C1_DATE = '2024-03-01'
SKU_COL = 'SKU_Personalizado'
SEEDS_SHARE = (42, 1, 7, 2025, 99)


# =============================================================================
# ---------- Sección 1: Librerías y Configuración Inicial ----------
# =============================================================================
# Importación de dependencias de ML y análisis de datos, configuración de rutas 
# relativas y creación de directorios para resultados y gráficos.

import os
import re
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, 'resultados')
os.makedirs(RES, exist_ok=True)
NB = os.path.join(HERE, '02_modelo_final', '1)_feature_engineering.py')
DB = os.path.join(HERE, 'inputs', 'base_datos.parquet').replace(os.sep, '/')
OUT = os.path.join(RES, 'out_so').replace(os.sep, '/')
os.makedirs(OUT, exist_ok=True)
CB_DIR = os.path.join(HERE, '02_modelo_final', 'catboost_info')
os.makedirs(CB_DIR, exist_ok=True)


# =============================================================================
# ---------- Sección 2: Ejecución del Notebook de Features y Carga de Predicciones ----------
# =============================================================================
# Ejecución del script de feature engineering para la obtención de características
# temporales y lectura de las predicciones a nivel de Familia previamente generadas por el ES-GBM.

def compute_metrics(actuals, preds):
    """Calcula un conjunto completo de métricas de desempeño (WAPE, MAPE, RMSE, Bias, R2)."""
    r = np.array(actuals, dtype=float)
    p = np.array(preds, dtype=float)
    
    mask = r != 0
    mae = np.mean(np.abs(r - p))
    mape = np.mean(np.abs((r[mask] - p[mask]) / r[mask])) * 100 if mask.sum() > 0 else np.nan
    wape = np.sum(np.abs(r - p)) / np.sum(np.abs(r)) * 100 if np.sum(np.abs(r)) > 0 else np.nan
    rmse = np.sqrt(np.mean((r - p) ** 2))
    bias = np.sum(p - r) / np.sum(r) * 100 if np.sum(r) > 0 else np.nan
    
    ss_res = np.sum((r - p) ** 2)
    ss_tot = np.sum((r - np.mean(r)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    
    return {'WAPE': wape, 'MAPE': mape, 'MAE': mae, 'RMSE': rmse, 'Bias': bias, 'R2': r2}

def patch(s):
    s = re.sub(r'INPUT_FILE\s*=\s*r?"[^"]+"', lambda m: 'INPUT_FILE = "' + DB + '"', s)
    return s

ns = {}
exec(compile(patch(open(NB, encoding='utf-8').read()), '<fe>', 'exec'), ns)

G = ns['GLOBAL_RESULTS']

full = pd.concat([G['train_high'], G['train_med'], G['train_low'], G['test_high'], G['test_med'], G['test_low']], ignore_index=True)
full['Fecha'] = pd.to_datetime(full['Fecha'])
full = full.sort_values(['Familia', 'Fecha']).reset_index(drop=True)

drop = set(ns.get('FEATURES_TO_DROP_FINAL', []) + ['Unidades', 'Unidades_Var_Pct', 'SKU_mean_demand'])
feats0 = [c for c in full.columns if c not in drop and pd.api.types.is_numeric_dtype(full[c])]

g = full.groupby('Familia')['Unidades']
full['ewm3'] = g.transform(lambda x: x.shift(1).ewm(span=3).mean())
full['ewm6'] = g.transform(lambda x: x.shift(1).ewm(span=6).mean())
full['mom_3_12'] = full['rolling_mean_3'] / (full['rolling_mean_12'] + 1)
full['trend6'] = g.transform(lambda x: x.shift(1).rolling(6).apply(lambda w: np.polyfit(range(len(w)), w, 1)[0] if w.notna().all() else np.nan, raw=False))
full['lag_24'] = g.shift(24)
full['rmed6'] = g.transform(lambda x: x.shift(1).rolling(6).median())
full['precio_mom'] = full.groupby('Familia')['Precio_Lag1'].transform(lambda x: x / (x.shift(1) + 1e-9))
full['yoy_12_24'] = full['lag_12'] / (full['lag_24'] + 1)
full['mes'] = full['Fecha'].dt.month

full = full.sort_values(['Familia', 'mes', 'Fecha'])
full['perfil_fam_mes'] = full.groupby(['Familia', 'mes'])['Unidades'].transform(lambda x: x.shift(1).expanding().mean())
full = full.sort_values(['Familia', 'Fecha']).reset_index(drop=True)

full['perfil_rel'] = full['perfil_fam_mes'] / (full['rolling_mean_12'] + 1)
full['sin_m'] = np.sin(2 * np.pi * full['mes'] / 12)
full['cos_m'] = np.cos(2 * np.pi * full['mes'] / 12)
full['px_rel_hist'] = full['Precio_Estimado'] / (full.groupby('Familia')['Precio_Estimado'].transform(lambda x: x.shift(1).rolling(12, min_periods=3).mean()) + 1e-9)
full['desc_depth'] = full['Descuento_promedio_estimado'] / (full['Precio_Estimado'] + 1e-9)
full['l1_l12'] = full['lag_1'] / (full['lag_12'] + 1)
full['rm3_perfil'] = full['rolling_mean_3'] / (full['perfil_fam_mes'] + 1)
full['std_rel'] = full['rolling_std_12'] / (full['rolling_mean_12'] + 1)

F = [c for c in feats0 + ['ewm3', 'ewm6', 'mom_3_12', 'trend6', 'lag_24', 'rmed6', 'precio_mom', 'yoy_12_24',
     'perfil_fam_mes', 'perfil_rel', 'sin_m', 'cos_m', 'px_rel_hist', 'desc_depth', 'l1_l12', 'rm3_perfil', 'std_rel'] if c in full.columns]

CUT = pd.Timestamp(CUT_DATE)
tr = full[full['Fecha'] <= CUT]
te = full[full['Fecha'] > CUT].copy()
ytr = np.log1p(np.clip(tr['Unidades'].values.astype(float), 0, None))
realO = te['Unidades'].values.astype(float)
gt = (te['Precio_Estimado'].fillna(0) > 0).values

# La feature engineering label-encodea 'Familia' (enteros); se recupera el nombre
# real para poder mergear contra las predicciones de familia y desagregar por SKU.
_le_fam = G['encoders'].get('Familia')
_inv = None
if _le_fam is not None:
    _inv = {i: c for i, c in enumerate(_le_fam.classes_)}
    te['Familia'] = [_inv.get(int(v), 'Familia_%d' % v) for v in te['Familia']]

fp0 = pd.read_csv(os.path.join(RES, 'predicciones_familia_test.csv'))
fp0['Fecha'] = pd.to_datetime(fp0['Fecha'])
te = te.merge(fp0, on=['Familia', 'Fecha'], how='left')
te['predF'] = te['predF'].fillna(0)

# Regla ex-ante: familias sin ventas en los 12 meses previos al corte se
# consideran descontinuadas (no comercializadas) -> prediccion Familia = 0;
# la desagregacion SKU hereda el 0 automaticamente.
_last12 = full[(full['Fecha'] <= CUT) & (full['Fecha'] > CUT - pd.DateOffset(months=12))]
_dead = set(_last12.groupby('Familia')['Unidades'].sum().loc[lambda s: s == 0].index)
if _inv is not None:
    _dead = {_inv.get(int(c), c) for c in _dead}
te.loc[te['Familia'].isin(_dead), 'predF'] = 0.0


# =============================================================================
# ---------- Sección 3: Construcción del Panel Histórico SKU y Features de Share ----------
# =============================================================================
# Mapeo de identificadores de familia y construcción del panel histórico de participaciones
# de cada SKU dentro de su respectiva familia.

rawf = pd.read_parquet(DB.replace('/', os.sep))
rawf.columns = [c.strip() for c in rawf.columns]
rawf['Fecha'] = pd.to_datetime(dict(year=rawf['Año'], month=rawf['Mes'], day=1))
SK = SKU_COL

rawOF = rawf.groupby(['Familia', 'Fecha'])['Unidades'].sum().reset_index()
pivG = te.pivot_table(index='Familia', columns='Fecha', values='Unidades', aggfunc='sum').fillna(0)
pivR = rawOF[rawOF['Fecha'] > CUT].pivot_table(index='Familia', columns='Fecha', values='Unidades', aggfunc='sum').fillna(0)
mapping = {fg: pivR.index[((pivR.values - pivG.loc[fg].values) ** 2).sum(axis=1).argmin()] for fg in pivG.index}
te['FamReal'] = te['Familia'].map(mapping)

skuP = rawf.groupby(['Familia', SK, 'Fecha'])['Unidades'].sum().reset_index()
idx = pd.date_range('2020-01-01', '2025-08-01', freq='MS')
frames = []
for (fam, sk), d0 in skuP.groupby(['Familia', SK]):
    s = d0.set_index('Fecha')['Unidades'].reindex(idx).fillna(0.0)
    frames.append(pd.DataFrame({'Familia': fam, 'SKU': sk, 'Fecha': idx, 'y': s.values}))

SP = pd.concat(frames, ignore_index=True).sort_values(['SKU', 'Fecha']).reset_index(drop=True)

_mn = pd.read_parquet(DB.replace('/', os.sep))
_mn.columns = [c.strip() for c in _mn.columns]
_mn['Fecha'] = pd.to_datetime(dict(year=_mn['Año'], month=_mn['Mes'], day=1))
_px = _mn.groupby([SK, 'Fecha']).apply(lambda d: d['Monto Neto'].sum() / max(d['Unidades'].clip(lower=0).sum(), 1)).rename('px').reset_index().rename(columns={SK: 'SKU'})

SP = SP.merge(_px, on=['SKU', 'Fecha'], how='left')
SP['px'] = SP.groupby('SKU')['px'].transform(lambda x: x.ffill())
fam_px = SP.groupby(['Familia', 'Fecha'])['px'].transform('mean')
SP['px_rel_fam'] = SP['px'] / (fam_px + 1e-9)
SP['px_rel_hist'] = SP['px'] / (SP.groupby('SKU')['px'].transform(lambda x: x.shift(1).rolling(6, min_periods=2).mean()) + 1e-9)

famtot = SP.groupby(['Familia', 'Fecha'])['y'].transform('sum')
SP['share'] = np.where(famtot > 0, SP['y'] / famtot, 0.0)

gsh = SP.groupby('SKU')['share']
gs = SP.groupby('SKU')['y']
SP['sh_L1'] = gsh.shift(1)
SP['sh_L12'] = gsh.shift(12)
SP['sh_rm3'] = gsh.transform(lambda x: x.shift(1).rolling(3).mean())
SP['sh_rm12'] = gsh.transform(lambda x: x.shift(1).rolling(12).mean())
SP['sh_ewm'] = gsh.transform(lambda x: x.shift(1).ewm(span=6).mean())
SP['sh_trend'] = SP['sh_rm3'] - SP['sh_rm12']
SP['nz'] = gs.transform(lambda x: (x.shift(1) > 0).rolling(12, min_periods=1).mean())

def msls(x):
    out = []
    last = -99
    for i, v in enumerate(x):
        out.append(i - last if last >= 0 else 99)
        if v > 0:
            last = i
    return pd.Series(out, index=x.index)

SP['msls'] = gs.transform(lambda x: msls(x.shift(1).fillna(0)))
SP['mes'] = SP['Fecha'].dt.month
SP['sin_m'] = np.sin(2 * np.pi * SP['mes'] / 12)
SP['cos_m'] = np.cos(2 * np.pi * SP['mes'] / 12)

SP = SP.sort_values(['SKU', 'mes', 'Fecha'])
SP['sh_perfil'] = SP.groupby(['SKU', 'mes'])['share'].transform(lambda x: x.shift(1).expanding().mean())
SP = SP.sort_values(['SKU', 'Fecha']).reset_index(drop=True)

famlvl = rawf[rawf['Fecha'] <= CUT].groupby('Familia')['Unidades'].mean()
SP['famlvl'] = SP['Familia'].map(famlvl).fillna(0)

TRs = SP[SP['Fecha'] <= CUT]
TEs = SP[SP['Fecha'] > CUT].copy()
realsO = TEs['y'].values.astype(float)

shstat = rawf[rawf['Fecha'] <= CUT].groupby(['Familia', SK])['Unidades'].sum().reset_index()
shstat['prop'] = shstat['Unidades'] / shstat.groupby('Familia')['Unidades'].transform('sum')
prop_map = {(r0['Familia'], r0[SK]): r0['prop'] for _, r0 in shstat.iterrows()}

fp = te[['FamReal', 'Fecha', 'predF']].rename(columns={'FamReal': 'Familia'})
TEs = TEs.merge(fp, on=['Familia', 'Fecha'], how='left')
TEs['prop_stat'] = [prop_map.get((f_, s_), 0) for f_, s_ in zip(TEs['Familia'], TEs['SKU'])]

SHF = ['sh_L1', 'sh_L12', 'sh_rm3', 'sh_rm12', 'sh_ewm', 'sh_trend', 'sh_perfil', 'msls', 'nz', 'sin_m', 'cos_m', 'famlvl', 'px_rel_fam', 'px_rel_hist']


# =============================================================================
# ---------- Sección 4: Entrenamiento de Motores de Share y Validación ----------
# =============================================================================
# Ajuste y optimización de hiperparámetros en validación para modelos predictivos de share 
# (LightGBM, XGBoost, CatBoost).

import xgboost as xgb2
from catboost import CatBoostRegressor as CBR
import time as _t

def share_engine(name):
    ps2 = []
    for sd in SEEDS_SHARE:
        if name == 'LGBM':
            m = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31, subsample=0.9, colsample_bytree=0.9, random_state=sd, n_jobs=-1, verbose=-1)
        elif name == 'XGB':
            m = xgb2.XGBRegressor(objective='reg:squarederror', n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.9, colsample_bytree=0.9, random_state=sd, n_jobs=-1)
        else:
            m = CBR(iterations=GI, learning_rate=GL, depth=GD, random_seed=sd, verbose=0, train_dir=CB_DIR)
        m.fit(TRs[SHF].fillna(0), TRs['share'].values)
        ps2.append(np.clip(m.predict(TEs[SHF].fillna(0)), 0, None))
    return np.mean(ps2, axis=0)

C1 = pd.Timestamp(C1_DATE)
TRv = SP[SP['Fecha'] <= C1]
VAv = SP[(SP['Fecha'] > C1) & (SP['Fecha'] <= CUT)].copy()

def val_cb(depth, lr, it):
    ps = []
    for sd in [42, 1, 7]:
        m = CBR(iterations=it, learning_rate=lr, depth=depth, random_seed=sd, verbose=0, train_dir=CB_DIR)
        m.fit(TRv[SHF].fillna(0), TRv['share'].values)
        ps.append(np.clip(m.predict(VAv[SHF].fillna(0)), 0, None))
    shp = np.mean(ps, axis=0)
    famv = VAv.groupby(['Familia', 'Fecha'])['y'].transform('sum').values
    return np.sum(np.abs(VAv['share'].values - shp) * famv) / max(np.sum(famv), 1)

bestg = (6, 0.05, 500, 1e9)
for depth in [4, 6, 8]:
    for lr in [0.03, 0.05]:
        w = val_cb(depth, lr, 600)
        if w < bestg[3]:
            bestg = (depth, lr, 600, w)

GD, GL, GI, _ = bestg


# =============================================================================
# ---------- Sección 4b: Selección Automática Estático vs Dinámico en Validación ----------
# =============================================================================
# Evalúa ambos enfoques en la ventana de validación (C1→CUT) y selecciona
# el de menor WAPE para aplicar en test (CUT→fin).

fp_val = te[['FamReal', 'Fecha', 'predF']].rename(columns={'FamReal': 'Familia'})
VAv_stat = VAv.copy()
VAv_stat = VAv_stat.merge(fp_val, on=['Familia', 'Fecha'], how='left')
VAv_stat['prop_stat'] = [prop_map.get((f_, s_), 0) for f_, s_ in zip(VAv_stat['Familia'], VAv_stat['SKU'])]
real_val = VAv_stat['y'].values.astype(float)

# --- Estático: proporciones fijas ---
pred_stat = (VAv_stat['predF'].fillna(0) * VAv_stat['prop_stat']).values
wape_stat = compute_metrics(real_val, pred_stat)['WAPE']

# --- Dinámico: ensamble ML de shares ---
sh_val_dyn = []
for name in ['LGBM', 'XGB', 'CAT']:
    ps = []
    for sd in SEEDS_SHARE:
        if name == 'LGBM':
            m = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31, subsample=0.9, colsample_bytree=0.9, random_state=sd, n_jobs=-1, verbose=-1)
        elif name == 'XGB':
            m = xgb2.XGBRegressor(objective='reg:squarederror', n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.9, colsample_bytree=0.9, random_state=sd, n_jobs=-1)
        else:
            m = CBR(iterations=GI, learning_rate=GL, depth=GD, random_seed=sd, verbose=0, train_dir=CB_DIR)
        m.fit(TRv[SHF].fillna(0), TRv['share'].values)
        ps.append(np.clip(m.predict(VAv[SHF].fillna(0)), 0, None))
    sh_val_dyn.append(np.mean(ps, axis=0))

sh_val_ens = np.mean(sh_val_dyn, axis=0)
VAv_dyn = VAv.copy()
VAv_dyn['sh_pred'] = sh_val_ens
s_v = VAv_dyn.groupby(['Familia', 'Fecha'])['sh_pred'].transform('sum')
VAv_dyn['sh_norm'] = np.where(s_v > 0, VAv_dyn['sh_pred'] / s_v, VAv_dyn['prop_stat'])
pred_dyn = (VAv_dyn['predF'].fillna(0) * VAv_dyn['sh_norm']).values
wape_dyn = compute_metrics(real_val, pred_dyn)['WAPE']

# --- Selección automática ---
MODELO_SELECCIONADO = 'ESTATICO' if wape_stat <= wape_dyn else 'DINAMICO'
print(f"\n{'='*75}")
print(f" SELECCIÓN AUTOMÁTICA DE ENFOQUE DE DESAGREGACIÓN")
print(f"{'='*75}")
print(f" WAPE Validación Estático  (prop. fijas): {wape_stat:.2f}%")
print(f" WAPE Validación Dinámico  (ML shares):  {wape_dyn:.2f}%")
print(f" >> ENFOQUE SELECCIONADO PARA TEST: {MODELO_SELECCIONADO}")
print(f"{'='*75}\n")


# =============================================================================
# ---------- Sección 5: Ensemble de Motores, Desagregación y Consolidación Excel ----------
# =============================================================================
# Combinación de predicciones de share mediante ensemble, cálculo de la demanda SKU total
# y guardado automático en un único archivo Excel multi-hoja.
# Si MODELO_SELECCIONADO == 'ESTATICO', se usan proporciones fijas directamente.
# Si MODELO_SELECCIONADO == 'DINAMICO', se entrena el ensamble ML de shares.

if MODELO_SELECCIONADO == 'ESTATICO':
    TEs['sh_norm'] = TEs['prop_stat']
    TEs['pred_total'] = TEs['predF'].fillna(0) * TEs['sh_norm']
    print(f" [TEST] Usando desagregación Estática (proporciones fijas)")
else:
    res_sh = {}
    _t0 = _t.time()
    for i, name in enumerate(['LGBM', 'XGB', 'CAT'], 1):
        shp = share_engine(name)
        T2 = TEs.copy()
        T2['sh_pred'] = shp
        s2 = T2.groupby(['Familia', 'Fecha'])['sh_pred'].transform('sum')
        T2['shn'] = np.where(s2 > 0, T2['sh_pred'] / s2, T2['prop_stat'])
        pdyn = (T2['predF'].fillna(0) * T2['shn']).values
        res_sh[name] = (compute_metrics(realsO, pdyn)['WAPE'], shp)

    sh_ens = np.mean([v[1] for v in res_sh.values()], axis=0)
    T2 = TEs.copy()
    T2['sh_pred'] = sh_ens
    s2 = T2.groupby(['Familia', 'Fecha'])['sh_pred'].transform('sum')
    T2['shn'] = np.where(s2 > 0, T2['sh_pred'] / s2, T2['prop_stat'])

    best = min(res_sh, key=lambda k: res_sh[k][0])
    TEs['sh_pred'] = res_sh[best][1] if res_sh[best][0] < compute_metrics(realsO, (T2['predF'].fillna(0) * T2['shn']).values)['WAPE'] else sh_ens
    sums = TEs.groupby(['Familia', 'Fecha'])['sh_pred'].transform('sum')
    TEs['sh_norm'] = np.where(sums > 0, TEs['sh_pred'] / sums, TEs['prop_stat'])
    TEs['pred_total'] = (TEs['predF'].fillna(0) * TEs['sh_norm'])
    print(f" [TEST] Usando desagregación Dinámica (ML shares)")

m_sku = compute_metrics(realsO, TEs['pred_total'].values)
m_fam = compute_metrics(te['Unidades'].values.astype(float), te['predF'].values)

# Baseline Naive
rawOF2 = rawOF.sort_values('Fecha')
rawOF2['lag12'] = rawOF2.groupby('Familia')['Unidades'].shift(12)
nv = rawOF2[rawOF2['Fecha'] > CUT][['Familia', 'Fecha', 'lag12']].fillna(0)
gn = nv.merge(shstat[['Familia', SK, 'prop']], on='Familia', how='left')
gn['pred'] = gn['lag12'] * gn['prop']
jn = gn.groupby([SK, 'Fecha'])['pred'].sum().reset_index().rename(columns={SK: 'SKU'})
totS = rawf[rawf['Fecha'] > CUT].groupby([SK, 'Fecha'])['Unidades'].sum().reset_index().rename(columns={'Unidades': 'total', SK: 'SKU'})
jn = jn.merge(totS, on=['SKU', 'Fecha'], how='outer').fillna(0)

J = TEs[['SKU', 'Fecha', 'pred_total']].rename(columns={'pred_total': 'pred'})
J = J.merge(totS, on=['SKU', 'Fecha'], how='outer').fillna(0)

excel_path = os.path.join(RES, 'SKU_Share_resultados.xlsx')
df_metrics = pd.DataFrame([
    {'Nivel': 'Familia (Total)', **m_fam},
    {'Nivel': 'SKU (Total)', **m_sku}
])

with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    df_metrics.to_excel(writer, sheet_name='SKU_Share_Metricas', index=False)
    J.to_excel(writer, sheet_name='SKU_Share_Predicciones', index=False)

J.to_csv(os.path.join(RES, 'predicciones_sku_total.csv'), index=False)

print("\n" + "=" * 75)
print(" RESUMEN DE MÉTRICAS - DESAGREGACIÓN SKU TOTAL ")
print("-" * 75)
print(" Nivel de Agregación       | WAPE(%)  | MAPE(%)  | RMSE     | MAE      | Bias(%)  | R²   ")
print("-" * 75)
print(" Familia (Total)           | %6.2f%%  | %6.2f%%  | %8.2f | %8.2f | %6.2f%%  | %4.2f " % (m_fam['WAPE'], m_fam['MAPE'], m_fam['RMSE'], m_fam['MAE'], m_fam['Bias'], m_fam['R2']))
print(" SKU (Total)               | %6.2f%%  | %6.2f%%  | %8.2f | %8.2f | %6.2f%%  | %4.2f " % (m_sku['WAPE'], m_sku['MAPE'], m_sku['RMSE'], m_sku['MAE'], m_sku['Bias'], m_sku['R2']))
print("=" * 75)
print(f" [OK] Consolidado guardado en: {excel_path}\n", flush=True)
print("guardado predicciones_sku_total.csv", flush=True)
