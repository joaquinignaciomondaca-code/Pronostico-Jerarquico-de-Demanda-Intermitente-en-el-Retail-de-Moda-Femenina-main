# =============================================================================
# FEATURE ENGINEERING PARA LOS MODELOS FINALES (ES-GBM y Desagregación SKU)
# -----------------------------------------------------------------------------
# Esta script se convirtió desde el notebook 1)_notebook_feature_engineering
# y contiene SOLO lo que el pipeline utiliza:
#   Sección 1: Configuración global
#   Sección 2: Carga, agregación e ingeniería de features base
#   Sección 3: Lags seguros para variables macro (anti-leakage t-1)
#   Sección 4: Preprocesamiento, encoding y segmentación ABC
#   Sección 5: Baseline naive (media móvil 3M) y cuantificación segmento LOW
#
# Quedan eliminados: la predicción instrumental de Monto Neto
# (Monto_Neto_Pred_Log / Prediccion_Monto_Neto_t), las rutas locales de
# despliegue del dataset original y las celdas de entrenamiento XGBoost que
# 2)_esgbm_ensemble.py y 3)_desagregacion_sku_share.py replican por su cuenta.
#
# 2)_esgbm_ensemble.py y 3)_desagregacion_sku_share.py ejecutan este archivo
# por `exec`, en cuyo caso INPUT_FILE se sobrescribe automáticamente.
# Si se ejecuta en forma independiente, INPUT_FILE apunta al parquet del repo.
# =============================================================================

import os
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

pd.set_option('display.max_columns', None)
np.random.seed(42)
warnings.filterwarnings('ignore')
xgb.set_config(verbosity=0)

# --- VARIABLES GLOBALES ---
# Base orgánica (demanda total excluyendo B2B) con SOLO las columnas que el
# pipeline usa. En el pipeline (2)/3)) este valor se sobrescribe con la ruta
# absoluta del parquet (regex sobre el literal). Al ejecutarse en forma
# independiente se resuelve en forma portable, relativa a este script.
INPUT_FILE = "inputs/base_datos.parquet"
if not os.path.exists(INPUT_FILE) and '__file__' in globals():
    _alt = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'inputs', 'base_datos.parquet')
    if os.path.exists(_alt):
        INPUT_FILE = _alt
TARGET_VAR = 'Unidades'
TARGET_TRANSFORMED = 'Unidades'
AGG_LEVEL_COL = 'Familia'
GROUP_COLS_TS = [AGG_LEVEL_COL]

# --- SPLIT (se calcula dinámicamente en Sección 2) ---
SPLIT_RATIO = 0.90
N_ROLLING_FOLDS = 5

GLOBAL_RESULTS = {}
encoders = {}

# Mapeo de columnas
COL_NAME_MAP = {
    'Año': 'A_o', 'Día de la madre': 'D_a_de_la_madre', 'Navidad_Año_Nuevo': 'Navidad_A_o_Nuevo',
    'BlackFrida_CyberMonday': 'BlackFrida_CyberMonday',
    'BackToSchool:RetornoTrabajo_DíaDeLaMujer': 'BackToSchool_RetornoTrabajo_D_aDeLaMujer',
    'Día del amor/Vacaciones': 'DiaDelAmor_Vacaciones', 'Desdempleo (%)': 'Desempleo_porcentaje',
    'InflacionMes': 'InlfacionMes'
}
CAT_VARS_ORIGINAL = ['Empresa', 'Sucursal', 'Superfamilia', 'Familia', 'Subfamilia', 'Temporada']
LAGS_BASE = [1, 2, 3, 6, 12]
WINDOW_SIZES = [3, 6, 12]
FEATURES_TO_DROP_FINAL = ['Empresa', 'A_o', 'Mes', 'tratamiento_ceros', 'Unidades', 'Fecha', 'Familia_mean_demand']

EXPERIMENTS = {
    'TWEEDIE': {
        'granularidad': 'monthly', 'lags': LAGS_BASE, 'scorer': 'WAPE',
        'objective': 'reg:tweedie', 'tweedie_variance_power': 1.5,
        'description': 'Objetivo Tweedie (Base)'
    },
    'LOG_NORMAL': {
        'granularidad': 'monthly', 'lags': LAGS_BASE, 'scorer': 'WAPE',
        'objective': 'reg:squarederror',
        'description': 'Log1p Transform + Rolling Max + Full History'
    },
}

CURRENT_EXP_ID = 'LOG_NORMAL'
CONFIG = EXPERIMENTS[CURRENT_EXP_ID]

print("=" * 60)
print("INICIANDO: Feature Engineering (Modelos Finales)")
print("=" * 60)
print(f"Configuración. Experimento: {CURRENT_EXP_ID} | Objetivo: {CONFIG['description']}")
print(f"Split deseado: {SPLIT_RATIO * 100:.0f}/{100 - SPLIT_RATIO * 100:.0f}")
GLOBAL_RESULTS['N_ROLLING_FOLDS'] = N_ROLLING_FOLDS


# =============================================================================
# SECCIÓN 2: CARGA, AGREGACIÓN E INGENIERÍA DE FEATURES BASE
# =============================================================================
print("\n" + "=" * 50)
print("SECCIÓN 2: Carga y Feature Engineering")
print("=" * 50)

# --- 1. CARGA Y AGREGACIÓN ---
try:
    df_raw = pd.read_parquet(INPUT_FILE)
    df_raw = df_raw.rename(columns=COL_NAME_MAP)
    df_raw['A_o'] = pd.to_numeric(df_raw['A_o'], errors='coerce').astype('Int64').fillna(df_raw['A_o'].median())
    df_raw['Mes'] = pd.to_numeric(df_raw['Mes'], errors='coerce').astype('Int64').fillna(df_raw['Mes'].median())
    date_df = df_raw[['A_o', 'Mes']].assign(Day=1).rename(columns={'A_o': 'year', 'Mes': 'month'})
    df_raw['Fecha'] = pd.to_datetime(date_df, errors='coerce')
    df_raw = df_raw.sort_values(by=['SKU_Personalizado', 'Fecha'])

    GLOBAL_RESULTS['df_master_unaggregated'] = df_raw.copy()

    print(f"Agregando datos a nivel de '{AGG_LEVEL_COL}'...")

    # Definir funciones de agregación
    agg_funcs_deseadas = {
        TARGET_VAR: 'sum',
        'Precio_Estimado': 'mean', 'Descuento_promedio_estimado': 'mean',
        'IMACEC_Comercio': 'mean', 'IPC': 'mean', 'Cambio USD/CLP (promedio mes)': 'mean',
        'Desempleo_porcentaje': 'mean', 'TPMporcentaje': 'mean', 'IVCM-INE': 'mean', 'CNC': 'mean',
        'IndiceDeConfianzaDelConsumidor': 'mean', 'IVDCMmensual': 'mean', 'VarIVDCMmensual': 'mean',
        'InlfacionMes': 'mean', 'D_a_de_la_madre': 'max', 'Navidad_A_o_Nuevo': 'max',
        'BlackFrida_CyberMonday': 'max', 'BackToSchool_RetornoTrabajo_D_aDeLaMujer': 'max',
        'DiaDelAmor_Vacaciones': 'max', 'Vacaciones': 'max', 'Bin_OtoñoInvierno': 'max',
        'Bin_PrimaveraVerano': 'max', 'Bin_Reutilizados': 'max'
    }
    agg_funcs = {k: v for k, v in agg_funcs_deseadas.items() if k in df_raw.columns}

    def safe_mode(x):
        m = x.mode()
        return m[0] if not m.empty else np.nan

    for col in CAT_VARS_ORIGINAL:
        if col in df_raw.columns and col not in agg_funcs and col != AGG_LEVEL_COL:
            agg_funcs[col] = safe_mode

    df_agg = df_raw.groupby([AGG_LEVEL_COL, 'Fecha']).agg(agg_funcs).reset_index()
    df_agg.sort_values(by=[AGG_LEVEL_COL, 'Fecha'], inplace=True)
    df = df_agg.copy()

    # Limpiezas básicas
    if TARGET_VAR in df.columns:
        df[TARGET_VAR] = df[TARGET_VAR].clip(lower=0)

    print(f"Datos AGREGADOS cargados. Shape: {df.shape}")

except Exception as e:
    print(f"ERROR al cargar datos: {e}")
    raise

# --- 2. CÁLCULO DE SPLIT TEMPORAL ---
unique_dates = np.sort(df['Fecha'].unique())
split_idx = int(len(unique_dates) * SPLIT_RATIO)
TRAIN_END_DATE = pd.to_datetime(unique_dates[split_idx - 1])
TEST_START_DATE = pd.to_datetime(unique_dates[split_idx])
print(f"Split {SPLIT_RATIO * 100:.0f}/{100 - SPLIT_RATIO * 100:.0f} -> Train hasta: {TRAIN_END_DATE.date()} | Test desde: {TEST_START_DATE.date()}")

# --- 3. FUNCIONES DE MÉTRICAS ---
def wape(y_true, y_pred, return_percentage=True):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    sum_abs_err = np.sum(np.abs(y_true - y_pred))
    sum_true = np.sum(np.abs(y_true))
    res = sum_abs_err / sum_true if sum_true != 0 else 0
    return res * 100 if return_percentage else res

def smape(y_true, y_pred, return_percentage=True):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    num = np.abs(y_true - y_pred)
    den = np.abs(y_true) + np.abs(y_pred)
    with np.errstate(divide='ignore', invalid='ignore'):
        res = 2 * num / den
        res[den == 0] = 0
    val = np.mean(res)
    return val * 100 if return_percentage else val

# --- 4. FEATURE ENGINEERING ---
def apply_feature_engineering(df_input, lags_list):
    print("\nAplicando Feature Engineering...")

    df_eng = df_input.copy().sort_values([AGG_LEVEL_COL, 'Fecha'])

    # Target para lags
    df_eng[TARGET_TRANSFORMED] = df_eng[TARGET_VAR]

    # 1. Lags (sobre todo el dataset, para que el Test vea el final del Train)
    for lag in lags_list:
        df_eng[f'lag_{lag}'] = df_eng.groupby(GROUP_COLS_TS)[TARGET_TRANSFORMED].shift(lag)

    # 2. Rolling Stats (Mean, Std y MAX) agrupados por familia (anti-leakage)
    #    El shift(1) ya está agrupado; el rolling debe aplicarse dentro de cada
    #    familia para no cruzar fronteras entre series (Bug de la versión anterior).
    for w in WINDOW_SIZES:
        g = df_eng.groupby(GROUP_COLS_TS)[TARGET_TRANSFORMED].shift(1)
        grp = g.groupby(df_eng[AGG_LEVEL_COL])
        df_eng[f'rolling_mean_{w}'] = grp.rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
        df_eng[f'rolling_std_{w}'] = grp.rolling(w, min_periods=1).std().reset_index(level=0, drop=True)
        df_eng[f'rolling_max_{w}'] = grp.rolling(w, min_periods=1).max().reset_index(level=0, drop=True)

    for w in WINDOW_SIZES:
        for stat in ('mean', 'std', 'max'):
            df_eng[f'rolling_{stat}_{w}'] = df_eng[f'rolling_{stat}_{w}'].fillna(0)

    # 3. Features de Precio
    if 'Precio_Estimado' in df_eng.columns:
        df_eng['Precio_Lag1'] = df_eng.groupby(GROUP_COLS_TS)['Precio_Estimado'].shift(1).fillna(0)

    # 4. Tendencias — la variación % en t contiene el target; solo se conserva la
    #    versión retrasada t-1 (el drop en 2)/3) la excluye, y además no se deja
    #    la columna intermedia para no arriesgar leakage).
    _pct = df_eng.groupby(GROUP_COLS_TS)[TARGET_VAR].pct_change().fillna(0).replace([np.inf, -np.inf], 0)
    df_eng['Unidades_Var_Pct_Lag1'] = _pct.groupby(df_eng[AGG_LEVEL_COL]).shift(1).fillna(0)

    print(f"Ingeniería completada sobre dataset completo. Shape: {df_eng.shape}")
    return df_eng

print("SECCIÓN 2 COMPLETADA.")


# =============================================================================
# SECCIÓN 3: LAGS SEGUROS PARA VARIABLES MACRO (Anti-Leakage)
# =============================================================================
print("\n" + "=" * 50)
print("SECCIÓN 3: Creación de Lags Seguros para Modelo Unidades (t-1)")
print("=" * 50)

df_lags = df.copy()

OPERATIONAL_VARS_LAG2 = []

MACRO_VARS_LAG1 = [
    'IMACEC_Comercio', 'Cambio USD/CLP (promedio mes)', 'IPC', 'Desempleo_porcentaje',
    'IndiceDeConfianzaDelConsumidor', 'IVDCMmensual', 'VarIVDCMmensual',
    'InlfacionMes', 'TPMporcentaje', 'IVCM-INE', 'CNC',
    'Ticket_promedio'
]
print("Aplicando lag t-1 (estándar) a variables macro y operacionales...")

for var in MACRO_VARS_LAG1:
    if var in df_lags.columns:
        var_cleaned = var.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
        df[f'{var_cleaned}_Lag1_Seguro'] = df_lags.groupby(GROUP_COLS_TS)[var].shift(1)
        if var in df.columns:
            df = df.drop(columns=[var], errors='ignore')

print(f"Lags seguros creados. {len(OPERATIONAL_VARS_LAG2)} variables con t-2, {len(MACRO_VARS_LAG1)} con t-1.")

MACRO_LAGGED_FEATURES = [
    f'{var.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")}_Lag2_Seguro'
    for var in OPERATIONAL_VARS_LAG2
    if f'{var.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")}_Lag2_Seguro' in df.columns
]
MACRO_LAGGED_FEATURES.extend([
    f'{var.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")}_Lag1_Seguro'
    for var in MACRO_VARS_LAG1
    if f'{var.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")}_Lag1_Seguro' in df.columns
])

print("SECCIÓN 3 COMPLETADA.")


# =============================================================================
# SECCIÓN 4: PREPROCESAMIENTO, ENCODING Y SEGMENTACIÓN ABC
# =============================================================================
print("\n" + "=" * 50)
print(f"SECCIÓN 4: Preprocesamiento para {CURRENT_EXP_ID}")
print("=" * 50)

# 1. Aplicar Ingeniería
df_exp = df.copy()
df_exp = apply_feature_engineering(df_exp, CONFIG['lags'])

if df_exp['Fecha'].dtype != '<M8[ns]':
    df_exp['Fecha'] = pd.to_datetime(df_exp['Fecha'])

# 2. Split Train/Test
train_master = df_exp[df_exp['Fecha'] <= TRAIN_END_DATE].copy()
test_master = df_exp[df_exp['Fecha'] >= TEST_START_DATE].copy().reset_index(drop=True)

print(f"Train Rows: {len(train_master)} | Rango: {train_master['Fecha'].min().date()} - {train_master['Fecha'].max().date()}")
print(f"Test Rows:  {len(test_master)}  | Rango: {test_master['Fecha'].min().date()} - {test_master['Fecha'].max().date()}")

# 3. Demanda Media para Segmentación (calculada solo con Train)
sku_demand = train_master.groupby(AGG_LEVEL_COL)[TARGET_VAR].mean().reset_index(name='SKU_mean_demand')
train_master = train_master.merge(sku_demand, on=AGG_LEVEL_COL, how='left')
test_master = test_master.merge(sku_demand, on=AGG_LEVEL_COL, how='left').fillna({'SKU_mean_demand': 0})

# 4. Encoders (híbrido: OHE para pocas categorías, LabelEncoder para muchas)
print("\n--- Aplicando Encoders (Anti-Leakage) ---")
OHE_VARS = ['Sucursal', 'Superfamilia']
cat_cols = [c for c in CAT_VARS_ORIGINAL if c in train_master.columns]

# OHE Manual
for col in OHE_VARS:
    if col in train_master.columns:
        dummies = pd.get_dummies(train_master[col], prefix=f"Bin_{col}", dummy_na=False)
        train_master = pd.concat([train_master, dummies], axis=1)
        dummies_test = pd.get_dummies(test_master[col], prefix=f"Bin_{col}", dummy_na=False)
        dummies_test = dummies_test.reindex(columns=dummies.columns, fill_value=0)
        test_master = pd.concat([test_master, dummies_test], axis=1)

# Label Encoding para el resto
encoders = {}
LE_VARS = [c for c in cat_cols if c not in OHE_VARS]
for col in LE_VARS:
    le = LabelEncoder()
    train_master[col] = train_master[col].astype(str)
    test_master[col] = test_master[col].astype(str)

    le.fit(train_master[col])
    encoders[col] = le

    train_master[col] = le.transform(train_master[col])
    # Para test, map y fillna -1 para desconocidos
    test_master[col] = test_master[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

# 5. Segmentación (High/Med/Low)
print("\n--- Segmentación ABC ---")
stats = train_master.groupby(AGG_LEVEL_COL)['SKU_mean_demand'].first()
Q1 = stats.quantile(0.25)
Q3 = stats.quantile(0.75)

GLOBAL_RESULTS.update({
    'train_high': train_master[train_master['SKU_mean_demand'] > Q3].copy(),
    'train_med': train_master[(train_master['SKU_mean_demand'] > Q1) & (train_master['SKU_mean_demand'] <= Q3)].copy(),
    'train_low': train_master[train_master['SKU_mean_demand'] <= Q1].copy(),
    'test_high': test_master[test_master['SKU_mean_demand'] > Q3].copy(),
    'test_med': test_master[(test_master['SKU_mean_demand'] > Q1) & (test_master['SKU_mean_demand'] <= Q3)].copy(),
    'test_low': test_master[test_master['SKU_mean_demand'] <= Q1].copy(),
    'train_master': train_master,
    'test_master': test_master,
    'encoders': encoders
})

print(f"Segmentos definidos. High: >{Q3:.1f}, Med: {Q1:.1f}-{Q3:.1f}, Low: <{Q1:.1f}")
print("SECCIÓN 4 COMPLETADA.")


# =============================================================================
# SECCIÓN 5: CUANTIFICACIÓN SEGMENTO LOW Y BASELINE NAIVE
# =============================================================================
print("\n" + "=" * 50)
print("SECCIÓN 5: Análisis del Baseline y Segmento LOW")
print("=" * 50)

# --- TAREA 1: CUANTIFICAR SEGMENTO LOW ---
print("\n--- TAREA 1: Cuantificación de Impacto del Segmento LOW ---")

try:
    train_high = GLOBAL_RESULTS.get('train_high', pd.DataFrame({'Unidades': [0]}))
    train_med = GLOBAL_RESULTS.get('train_med', pd.DataFrame({'Unidades': [0]}))
    train_low = GLOBAL_RESULTS.get('train_low', pd.DataFrame({'Unidades': [0]}))

    df_master_unaggregated = GLOBAL_RESULTS.get('df_master_unaggregated')
    if df_master_unaggregated is None:
        raise ValueError("'df_master_unaggregated' no encontrado. Revisar Sección 2.")

    vol_high = train_high['Unidades'].sum()
    vol_med = train_med['Unidades'].sum()
    vol_low = train_low['Unidades'].sum()
    vol_total = vol_high + vol_med + vol_low

    perc_vol_low = (vol_low / vol_total) * 100 if vol_total > 0 else 0.0

    num_skus_low = len(train_low[AGG_LEVEL_COL].unique()) if AGG_LEVEL_COL in train_low.columns else 0
    num_skus_total = len(df_master_unaggregated[AGG_LEVEL_COL].unique()) if AGG_LEVEL_COL in df_master_unaggregated.columns else 0

    print(f"Volumen total (Train): {vol_total:,.0f} unidades")
    print(f"Volumen Segmento LOW: {vol_low:,.0f} unidades ({perc_vol_low:.1f}%)")
    print(f"# {AGG_LEVEL_COL} en Segmento LOW: {num_skus_low} ({num_skus_low / num_skus_total * 100:.1f}% del total)" if num_skus_total > 0 else f"# {AGG_LEVEL_COL} en Segmento LOW: {num_skus_low}")

    GLOBAL_RESULTS['perc_vol_low'] = perc_vol_low
    GLOBAL_RESULTS['num_skus_low'] = num_skus_low

    if perc_vol_low < 5:
        print("Veredicto: Volumen bajo (<5%). Justificación de exclusión viable.")
    elif perc_vol_low > 15:
        print("Veredicto: Volumen alto (>15%). Segmento LOW relevante en volumen.")
    else:
        print("Veredicto: Volumen moderado.")

    print("Nota: la segmentación ABC es diagnóstica. El ES-GBM entrena un modelo único")
    print("sobre todas las familias; el segmento LOW no se excluye ni se modela por")
    print("separado con Poisson. El peso de muestras por antigüedad (decay exponencial)")
    print("y el ruteo por régimen de crecimiento se encargan del comportamiento de baja")
    print("y alta rotación.")
except Exception as e:
    print(f"ERROR en Tarea 1: Fallo al calcular estadísticas de volumen: {e}")

# --- TAREA 2: BASELINES DE REFERENCIA (h=1 con actualización, anti-leakage) ---
# Baselines justos bajo el protocolo h=1 rolling: al predecir el mes t, cada
# baseline usa información disponible hasta t-1 (igual que los modelos, que
# reciben lag_1 con el valor real del mes anterior).

print("\n--- TAREA 2: Baselines de referencia (rolling h=1, naive y snaive) ---")

def _baselines(df_input):
    d = df_input.sort_values([AGG_LEVEL_COL, 'Fecha']).copy()
    g = d.groupby(AGG_LEVEL_COL)['Unidades']
    d['shift1'] = g.shift(1)
    d['ma3'] = d['shift1'].groupby(d[AGG_LEVEL_COL]).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    d['naive'] = d['shift1']
    d['snaive'] = g.shift(12)
    return d

try:
    bl = _baselines(df)
    test_bl = bl[bl['Fecha'] >= TEST_START_DATE].copy()

    y_true_bl = test_bl['Unidades'].values
    y_ma3 = test_bl['ma3'].fillna(0).values
    y_naive = test_bl['naive'].fillna(0).values
    y_snaive = test_bl['snaive'].fillna(0).values

    if len(y_true_bl) > 1 and np.sum(y_true_bl) > 0:
        wape_baseline = wape(y_true_bl, y_ma3, return_percentage=True)
        wape_naive = wape(y_true_bl, y_naive, return_percentage=True)
        wape_snaive = wape(y_true_bl, y_snaive, return_percentage=True)

        # MASE: MAE(baseline)/MAE(snaive in-sample) — escala estándar escalada
        tr_bl = bl[bl['Fecha'] <= TRAIN_END_DATE]
        mae_snaive_insample = np.mean(np.abs(tr_bl['Unidades'] - tr_bl['snaive']))
        mae_baseline_test = np.mean(np.abs(y_true_bl - y_ma3))
        mase_baseline = (mae_baseline_test / mae_snaive_insample) if mae_snaive_insample > 0 else np.inf
    else:
        wape_baseline = wape_naive = wape_snaive = float('inf')
        mase_baseline = float('inf')

    if wape_baseline != float('inf'):
        print(f"WAPE Baseline Rolling MA3 (h=1, con actuales): {wape_baseline:.2f}%")
        print(f"WAPE Baseline Naive (y_t = y_(t-1)):            {wape_naive:.2f}%")
        print(f"WAPE Baseline SNaive (y_t = y_(t-12)):          {wape_snaive:.2f}%")
        print(f"MASE Baseline Rolling MA3 (vs SNaive in-sample): {mase_baseline:.3f}")
        GLOBAL_RESULTS['wape_baseline'] = wape_baseline
        GLOBAL_RESULTS['wape_naive'] = wape_naive
        GLOBAL_RESULTS['wape_snaive'] = wape_snaive
        GLOBAL_RESULTS['mase_baseline'] = mase_baseline
    else:
        GLOBAL_RESULTS['wape_baseline'] = float('inf')
        print("Resultado: Baselines no calculados (datos insuficientes o error).")

except Exception as e:
    print(f"ERROR CRÍTICO en Tarea 2: Falló el cálculo de baselines. {e}")
    GLOBAL_RESULTS['wape_baseline'] = float('inf')

print("\nSECCIÓN 5 COMPLETADA.")
print("=" * 60)
print("FEATURE ENGINEERING COMPLETADO.")
print("=" * 60)