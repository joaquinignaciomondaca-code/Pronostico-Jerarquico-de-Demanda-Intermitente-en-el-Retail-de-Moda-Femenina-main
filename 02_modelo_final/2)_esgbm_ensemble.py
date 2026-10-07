# -*- coding: utf-8 -*-
"""
=============================================================================
SEMINARIO DE TÍTULO - INGENIERÍA COMERCIAL (UTFSM, 2026)
Autor: Joaquín Ignacio Mondaca Parada | Profesor Guía: Marcelo Villena Chamorro
Arquitectura Propuesta: ES-GBM Ensemble & Regimen Routing (Nivel Familia)
=============================================================================

¿QUÉ HACE ESTE SCRIPT EN SIMPLE?
1. Desacopla la inercia estructural de la demanda usando Holt-Winters amortiguado.
2. Modela los residuos comerciales con ensambles de Gradient Boosting (XGBoost, LightGBM, CatBoost) y TabPFN.
3. Pondera los modelos mediante optimización no negativa (NNLS) calibrada exclusivamente en validación.
4. Aplica un ruteo dinámico por régimen según el momentum trimestral de ventas.
=============================================================================
"""

# =============================================================================
# ---------- ⚙️ PARÁMETROS CONFIGURABLES ----------
# =============================================================================
CUT_DATE = '2025-01-01'        # Fecha de corte: inicio del conjunto de prueba ciego
C1_DATE = '2024-03-01'         # Fecha de corte intermedia para calibración y validación
SEEDS_ML = (42, 1, 7, 2025, 99, 3, 11, 21) # Semillas aleatorias para robustez estadística


# =============================================================================
CUT_DATE = '2025-01-01'
C1_DATE = '2024-03-01'
SEEDS_ML = (42, 1, 7, 2025, 99, 3, 11, 21)


# =============================================================================
# ---------- Sección 1: Librerías y Configuración Inicial ----------
# =============================================================================
# Importación de dependencias de Machine Learning (XGBoost, LightGBM, CatBoost),
# estadísticas (statsmodels), configuración de rutas relativas y directorio de resultados.

import os
import re
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, 'resultados')
os.makedirs(RES, exist_ok=True)
NB = os.path.join(HERE, '02_modelo_final', '1)_feature_engineering.py')
DB = os.path.join(HERE, 'inputs', 'base_datos.parquet').replace(os.sep, '/')
OUT = os.path.join(RES, 'out_es2').replace(os.sep, '/')
os.makedirs(OUT, exist_ok=True)
CB_DIR = os.path.join(HERE, '02_modelo_final', 'catboost_info')
os.makedirs(CB_DIR, exist_ok=True)


# =============================================================================
# ---------- Sección 2: Ejecución del Notebook de Feature Engineering ----------
# =============================================================================
# Parcheo y ejecución por código del script de feature engineering para
# extraer el DataFrame maestro procesado y los splits de entrenamiento y test.

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


# =============================================================================
# ---------- Sección 3: Ingeniería de Features Avanzada a Nivel Familia ----------
# =============================================================================
# Creación de variables dinámicas adicionales (medias móviles exponenciales ewm,
# tendencias polinómicas, ratios de momento y estacionalidad mensual).

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

F0 = [c for c in feats0 + ['ewm3', 'ewm6', 'mom_3_12', 'trend6', 'lag_24', 'rmed6', 'precio_mom', 'yoy_12_24',
     'perfil_fam_mes', 'perfil_rel', 'sin_m', 'cos_m', 'px_rel_hist', 'desc_depth', 'l1_l12', 'rm3_perfil', 'std_rel'] if c in full.columns]

CUT = pd.Timestamp(CUT_DATE)
C1 = pd.Timestamp(C1_DATE)
MONTHS = sorted(full['Fecha'].unique())


# =============================================================================
# ---------- Sección 4: Modelado Base (Holt-Winters + ML Ensemble) ----------
# =============================================================================
# Ajuste de niveles base con ExponentialSmoothing y entrenamiento conjunto de 
# modelos de Machine Learning (XGBoost, LightGBM, CatBoost) con ponderación NNLS.

lvl_map = {}
for fam, d in full.groupby('Familia'):
    s = d.set_index('Fecha')['Unidades'].reindex(MONTHS).fillna(0.0).astype(float)
    base = s[s.index <= C1]
    try:
        m = ExponentialSmoothing(base, trend='add', damped_trend=True, initialization_method='estimated').fit(optimized=True)
        for t, v in m.fittedvalues.items():
            lvl_map[(fam, t)] = max(v, 0.0)
    except Exception:
        for t, v in base.shift(1).fillna(base.mean()).items():
            lvl_map[(fam, t)] = max(v, 0.0)
    for t in [x for x in MONTHS if x > C1]:
        hist = s[s.index < t]
        try:
            mt = ExponentialSmoothing(hist, trend='add', damped_trend=True, initialization_method='estimated').fit(optimized=True)
            f1 = float(mt.forecast(1).iloc[0])
        except Exception:
            f1 = float(hist.tail(3).mean())
        lvl_map[(fam, t)] = max(f1, 0.0)

full['lvl'] = [lvl_map.get((f_, t_), 0.0) for f_, t_ in zip(full['Familia'], full['Fecha'])]
LV = 1.0

def fit_members(trd, ted, seeds):
    age = (trd['Fecha'].max() - trd['Fecha']).dt.days / 30.0
    sw = (0.5 ** (age / 24.0)).values
    ya = np.arcsinh(np.clip(trd['Unidades'].values.astype(float), 0, None))
    r = np.log1p(np.clip(trd['Unidades'] / (trd['lvl'] + LV), 0, 8).values.astype(float))
    FE = F0 + ['lvl']
    gt = (ted['Precio_Estimado'].fillna(0) > 0).values
    px, pl, pcat, pe = [], [], [], []
    
    for sd in seeds:
        m = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.9, colsample_bytree=0.9, random_state=sd, n_jobs=-1)
        m.fit(trd[F0].fillna(0), ya, sample_weight=sw)
        px.append(np.clip(np.sinh(m.predict(ted[F0].fillna(0))), 0, None))
        
        m2 = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=31, subsample=0.9, colsample_bytree=0.9, random_state=sd, n_jobs=-1, verbose=-1)
        m2.fit(trd[F0].fillna(0), ya, sample_weight=sw)
        pl.append(np.clip(np.sinh(m2.predict(ted[F0].fillna(0))), 0, None))
        
        m3 = CatBoostRegressor(iterations=600, learning_rate=0.05, depth=6, random_seed=sd, verbose=0, train_dir=CB_DIR)
        m3.fit(trd[F0].fillna(0), ya, sample_weight=sw)
        pcat.append(np.clip(np.sinh(m3.predict(ted[F0].fillna(0))), 0, None))
        
        m4 = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=31, subsample=0.9, colsample_bytree=0.9, random_state=sd, n_jobs=-1, verbose=-1)
        m4.fit(trd[FE].fillna(0), r, sample_weight=sw)
        pe.append(np.clip(np.expm1(m4.predict(ted[FE].fillna(0))), 0, None))
        
        m5 = CatBoostRegressor(iterations=600, learning_rate=0.05, depth=6, random_seed=sd, verbose=0, train_dir=CB_DIR)
        m5.fit(trd[FE].fillna(0), r, sample_weight=sw)
        pe.append(np.clip(np.expm1(m5.predict(ted[FE].fillna(0))), 0, None))
        
    PX, PL, PC = np.mean(px, axis=0) * gt, np.mean(pl, axis=0) * gt, np.mean(pcat, axis=0) * gt
    PE = np.clip((ted['lvl'].values + LV) * np.mean(pe, axis=0) - LV, 0, None) * gt
    return PX, PL, PC, PE

from scipy.optimize import nnls as _nnls
print("[A] NNLS refit asinh (4 origenes OOS)...", flush=True)
Arows = []
yrows = []
import time as _tA
_t0 = _tA.time()

for oi, o in enumerate(['2023-01-01', '2023-04-01', '2023-07-01', '2023-10-01'], 1):
    o = pd.Timestamp(o)
    e = o + pd.DateOffset(months=3)
    trd = full[full['Fecha'] <= o]
    ted = full[(full['Fecha'] > o) & (full['Fecha'] <= e)].copy()
    ax, al, ac, ae_ = fit_members(trd, ted, [42, 1])
    Arows.append(np.column_stack([ax, al, ac]))
    yrows.append(ted['Unidades'].values.astype(float))
    
A_ = np.vstack(Arows)
y_ = np.concatenate(yrows)
Wn, _ = _nnls(A_, y_)
Wn = Wn / max(Wn.sum(), 1e-9)

def fit_pair(trd, ted, seeds):
    PX, PL, PC, PE = fit_members(trd, ted, seeds)
    return Wn[0] * PX + Wn[1] * PL + Wn[2] * PC, PE

trV = full[full['Fecha'] <= C1]
va = full[(full['Fecha'] > C1) & (full['Fecha'] <= CUT)].copy()
tr = full[full['Fecha'] <= CUT]
te = full[full['Fecha'] > CUT].copy()

vC, vE = fit_pair(trV, va, [42, 1, 7])
rva = va['Unidades'].values.astype(float)
tC, tE = fit_pair(tr, te, SEEDS_ML)
rte = te['Unidades'].values.astype(float)


# =============================================================================
# ---------- Sección 5: Optimización de Pesos y Ruteo por Régimen ----------
# =============================================================================
# Selección de ponderaciones globales y reglas de ruteo dinámicas evaluadas en validación.

ws = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
wv = [(w, compute_metrics(rva, w * vE + (1 - w) * vC)['WAPE']) for w in ws]
wbest = min(wv, key=lambda x: x[1])[0]

best = (None,)
bv = 1e9
for tau in [1.05, 1.15, 1.3, 1.5]:
    for whi in [0.5, 0.7, 0.9]:
        for wlo in [0.1, 0.3, 0.5]:
            fl = (va['mom_3_12'].fillna(0) > tau).values
            p = np.where(fl, whi * vE + (1 - whi) * vC, wlo * vE + (1 - wlo) * vC)
            wv_ = compute_metrics(rva, p)['WAPE']
            if wv_ < bv:
                bv = wv_
                best = (tau, whi, wlo)

tau, whi, wlo = best
fl = (te['mom_3_12'].fillna(0) > tau).values
p_rout = np.where(fl, whi * tE + (1 - whi) * tC, wlo * tE + (1 - wlo) * tC)


# =============================================================================
# ---------- Sección 6: Integración TabPFN, Predicción Final y Consolidación Excel ----------
# =============================================================================
# Incorporación del modelo fundacional TabPFN, ensemble ponderado y guardado en 
# archivo Excel multi-hoja consolidado.

# TabPFN es un modelo gated que requiere licencia de un solo uso (login PriorLabs).
# Para que el script no se cuelgue en entornos sin token (CI, evaluadores, auditoría),
# solo se invoca si hay token disponible (env TABPFN_TOKEN o caché local); si no, se usa
# el fallback (vC/tC) sin construir el modelo. Ver model_loading.py/_HF_REPOS (gated: v2.5-v3).
def _tabpfn_token_available():
    """True si existe TABPFN_TOKEN o un token cacheado (misma lógica que tabpfn.browser_auth)."""
    if os.environ.get('TABPFN_TOKEN', '').strip():
        return True
    for p in (os.path.expanduser('~/.cache/tabpfn/auth_token'),
              os.path.expanduser('~/.tabpfn/token')):
        if os.path.isfile(p):
            try:
                if open(p, encoding='utf-8').read().strip():
                    return True
            except OSError:
                pass
    return False

_TABPFN_OK = _tabpfn_token_available()
print(f"[TabPFN] token disponible: {_TABPFN_OK} -> usar TabPFN v3" if _TABPFN_OK
      else "[TabPFN] sin token: se omite TabPFN y se usa fallback (vC/tC)", flush=True)

from tabpfn import TabPFNRegressor
import time as _t
_t0 = _t.time()

trT = trV[trV['Fecha'] >= pd.Timestamp('2021-10-01')]
XtrT = trT[F0].fillna(0).replace([np.inf, -np.inf], 0).values
XvaT = va[F0].fillna(0).replace([np.inf, -np.inf], 0).values
yT = np.arcsinh(np.clip(trT['Unidades'].values.astype(float), 0, None))

try:
    if not _TABPFN_OK:
        raise RuntimeError('Sin licencia TabPFN (TABPFN_TOKEN/caché): fallback activo.')
    mT = TabPFNRegressor(device='cpu', random_state=42, ignore_pretraining_limits=True, n_estimators=2)
    mT.fit(XtrT, yT)
    pv_tab = np.clip(np.sinh(mT.predict(XvaT)), 0, None) * (va['Precio_Estimado'].fillna(0) > 0).values
except Exception as _e:
    pv_tab = vC

best = (1, 0, 0, 1e9)
for we in np.arange(0, 0.65, 0.05):
    for wt in np.arange(0, 0.45, 0.05):
        wc = 1 - we - wt
        if wc < 0.2:
            continue
        w = compute_metrics(rva, wc * vC + we * vE + wt * pv_tab)['WAPE']
        if w < best[3]:
            best = (wc, we, wt, w)
            
wc, we, wt, wv = best

try:
    if not _TABPFN_OK:
        raise RuntimeError('Sin licencia TabPFN (TABPFN_TOKEN/caché): fallback activo.')
    trTt = tr[tr['Fecha'] >= pd.Timestamp('2021-10-01')]
    XtrTt = trTt[F0].fillna(0).replace([np.inf, -np.inf], 0).values
    XteT = te[F0].fillna(0).replace([np.inf, -np.inf], 0).values
    yTt = np.arcsinh(np.clip(trTt['Unidades'].values.astype(float), 0, None))
    mTt = TabPFNRegressor(device='cpu', random_state=42, ignore_pretraining_limits=True, n_estimators=2)
    mTt.fit(XtrTt, yTt)
    p_tab_te = np.clip(np.sinh(mTt.predict(XteT)), 0, None) * (te['Precio_Estimado'].fillna(0) > 0).values
except Exception as _e:
    p_tab_te = tC

final = wc * tC + we * tE + wt * p_tab_te
final_val = wc * vC + we * vE + wt * pv_tab

# Regla ex-ante: familias sin ventas en los 12 meses previos al corte se
# consideran descontinuadas (no comercializadas) -> prediccion test = 0.
_last12 = full[(full['Fecha'] <= CUT) & (full['Fecha'] > CUT - pd.DateOffset(months=12))]
_dead = set(_last12.groupby('Familia')['Unidades'].sum().loc[lambda s: s == 0].index)
final = np.where(te['Familia'].isin(_dead), 0.0, final)
final_val = np.where(va['Familia'].isin(_dead), 0.0, final_val)

m_fam = compute_metrics(rte, final)

df_test = pd.DataFrame({'Familia': te['Familia'], 'Fecha': te['Fecha'], 'predF': final})
df_val = pd.DataFrame({'Familia': va['Familia'], 'Fecha': va['Fecha'], 'predF': final_val})

# La feature engineering label-encodea 'Familia' (enteros); se recupera el nombre
# real para poder desagregar contra las participaciones estaticas por SKU.
_le_fam = G['encoders'].get('Familia')
if _le_fam is not None:
    _inv = {i: c for i, c in enumerate(_le_fam.classes_)}
    df_test['Familia'] = [_inv.get(int(v), 'Familia_%d' % v) for v in df_test['Familia']]
    df_val['Familia'] = [_inv.get(int(v), 'Familia_%d' % v) for v in df_val['Familia']]

# --- Métricas a nivel SKU (proporciones estáticas top-down, mismo criterio que los baselines) ---
rawf_ = pd.read_parquet(DB.replace('/', os.sep))
rawf_.columns = [c.strip() for c in rawf_.columns]
rawf_['Fecha'] = pd.to_datetime(dict(year=rawf_['Año'], month=rawf_['Mes'], day=1))
_shs = rawf_[rawf_['Fecha'] <= CUT].groupby(['Familia', 'SKU_Personalizado'])['Unidades'].sum().reset_index()
_shs['prop'] = _shs['Unidades'] / _shs.groupby('Familia')['Unidades'].transform('sum')
_totS = rawf_[rawf_['Fecha'] > CUT].groupby(['SKU_Personalizado', 'Fecha'])['Unidades'].sum().rename('total').reset_index()
_rowsS = []
for _f, _t, _v in zip(df_test['Familia'], df_test['Fecha'], df_test['predF']):
    _sub = _shs[_shs['Familia'] == _f]
    for _, _r0 in _sub.iterrows():
        _rowsS.append((_r0['SKU_Personalizado'], _t, _v * _r0['prop']))
_PS = pd.DataFrame(_rowsS, columns=['SKU_Personalizado', 'Fecha', 'pred']).groupby(['SKU_Personalizado', 'Fecha'])['pred'].sum().reset_index()
_JS = _PS.merge(_totS, on=['SKU_Personalizado', 'Fecha'], how='outer').fillna(0)
m_sku = compute_metrics(_JS['total'], _JS['pred'])

excel_path = os.path.join(RES, 'ESGBM_resultados.xlsx')
df_metrics = pd.DataFrame([
    {'Nivel': 'Familia (Total) - Test', **m_fam},
    {'Nivel': 'SKU (Total) - Test', **m_sku}
])

with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    df_metrics.to_excel(writer, sheet_name='ESGBM_Metricas', index=False)
    df_test.to_excel(writer, sheet_name='ESGBM_Familia_Test', index=False)
    df_val.to_excel(writer, sheet_name='ESGBM_Familia_Val', index=False)

df_test.to_csv(os.path.join(RES, 'predicciones_familia_test.csv'), index=False)
df_val.to_csv(os.path.join(RES, 'predicciones_familia_validacion.csv'), index=False)

print("\n" + "=" * 75)
print(" RESUMEN DE MÉTRICAS - ES-GBM ENSEMBLE (FAMILIA TOTAL) ")
print("-" * 75)
print(" Nivel de Agregación       | WAPE(%)  | MAPE(%)  | RMSE     | MAE      | Bias(%)  | R²   ")
print("-" * 75)
print(" Familia (Total)           | %6.2f%%  | %6.2f%%  | %8.2f | %8.2f | %6.2f%%  | %4.2f " % (m_fam['WAPE'], m_fam['MAPE'], m_fam['RMSE'], m_fam['MAE'], m_fam['Bias'], m_fam['R2']))
print(" SKU (Total)               | %6.2f%%  | %6.2f%%  | %8.2f | %8.2f | %6.2f%%  | %4.2f " % (m_sku['WAPE'], m_sku['MAPE'], m_sku['RMSE'], m_sku['MAE'], m_sku['Bias'], m_sku['R2']))
print("=" * 75)
print(f" [OK] Consolidado guardado en: {excel_path}\n", flush=True)
