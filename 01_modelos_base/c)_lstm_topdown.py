# -*- coding: utf-8 -*-
"""
Modelo LSTM Top-Down (PyTorch) - Protocolo Parejo (Demanda Total)
Implementa redes recurrentes LSTM profundas (56 -> 32 celdas con Dropout) para la
predicción de demanda a nivel Familia, con entrenamiento multi-semilla y early stopping,
desagregando posteriormente a nivel SKU mediante proporciones estáticas.
"""

# =============================================================================
# ---------- PARÁMETROS CONFIGURABLES (Fácil Edición) ----------
# =============================================================================
# Puedes modificar los siguientes parámetros según tus necesidades de experimento:
# - CUT_DATE: Fecha de corte entre entrenamiento/validación y el conjunto de test.
# - SKU_COL: Nombre de la columna que identifica de forma única a cada SKU.
# - SEQ_LEN: Longitud de la ventana temporal (lags) para la red LSTM.
# - SEEDS: Semillas aleatorias para robustecer el entrenamiento de red neuronal.
# =============================================================================
CUT_DATE = '2025-01-01'
SKU_COL = 'SKU_Personalizado'
SEQ_LEN = 12
SEEDS = (42, 1, 7, 2025, 99)


# =============================================================================
# ---------- Sección 1: Librerías y Configuración Inicial ----------
# =============================================================================
# Importación de librerías para aprendizaje profundo (PyTorch), manipulación de datos,
# configuración de rutas relativas y directorio de resultados.

import os
import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, 'resultados')
os.makedirs(RES, exist_ok=True)
I = os.path.join(HERE, 'inputs')


# =============================================================================
# ---------- Sección 2: Carga, Normalización y Construcción de Tensores ----------
# =============================================================================
# Lectura de la base consolidada, escalado min-max por familia, generación de ventanas
# temporales deslizantes (SEQ) y estructuración de tensores PyTorch para Train/Val/Test.

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

CUT = pd.Timestamp(CUT_DATE)
SK = SKU_COL
SEQ = SEQ_LEN

rawT = pd.read_parquet(os.path.join(I, 'base_datos.parquet'))
rawT.columns = [c.strip() for c in rawT.columns]
rawT['Fecha'] = pd.to_datetime(dict(year=rawT['Año'], month=rawT['Mes'], day=1))

idx = pd.date_range('2020-01-01', '2025-08-01', freq='MS')
piv = rawT.groupby(['Familia', 'Fecha'])['Unidades'].sum().unstack(0).reindex(idx).fillna(0)

pxcol = [c for c in rawT.columns if 'recio' in c][0]
px = rawT.groupby(['Familia', 'Fecha'])[pxcol].mean().unstack(0).reindex(idx).ffill().fillna(0)

FAMS = list(piv.columns)
TESTM = [m for m in idx if m > CUT]
VALM = [m for m in idx if pd.Timestamp('2024-04-01') <= m <= CUT]  # Split temporal para Early Stopping

sin_m = np.sin(2 * np.pi * idx.month / 12)
cos_m = np.cos(2 * np.pi * idx.month / 12)

# Escalado por familia utilizando únicamente datos de entrenamiento
sc = {}
for f in FAMS:
    h = piv[f][piv[f].index <= CUT]
    hp = px[f][px[f].index <= CUT]
    sc[f] = (h.min(), max(h.max() - h.min(), 1e-6), hp.mean(), max(hp.std(), 1e-6))


# =============================================================================
# ---------- Sección 3: Definición de la Arquitectura de Red Neuronal (LSTM) ----------
# =============================================================================
# Arquitectura secuencial: LSTM(56) -> Dropout(0.4) -> LSTM(32) -> Dropout(0.4) 
# -> Dense(16, ReLU) -> Dense(1, Linear).

class Net(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.l1 = nn.LSTM(nf, 56, batch_first=True)
        self.l2 = nn.LSTM(56, 32, batch_first=True)
        self.d = nn.Dropout(0.4)
        self.f1 = nn.Linear(32, 16)
        self.f2 = nn.Linear(16, 1)

    def forward(self, x):
        o, _ = self.l1(x)
        o = self.d(o)
        o, _ = self.l2(o)
        o = self.d(o[:, -1, :])
        return self.f2(torch.relu(self.f1(o))).squeeze(-1)


def build(con_precio):
    Xtr, ytr, Xva, yva, Xte, meta = [], [], [], [], [], []
    for f in FAMS:
        mn, rg, pm, ps = sc[f]
        ys = ((piv[f] - mn) / rg).values
        ps_ = ((px[f] - pm) / ps).values
        for i in range(SEQ, len(idx)):
            t = idx[i]
            feats = [ys[i - SEQ:i], sin_m[i - SEQ:i], cos_m[i - SEQ:i]]
            if con_precio:
                feats.append(ps_[i - SEQ + 1:i + 1])
            w = np.stack(feats, axis=1).astype(np.float32)
            if t <= CUT:
                if t in VALM:
                    Xva.append(w)
                    yva.append(ys[i])
                else:
                    Xtr.append(w)
                    ytr.append(ys[i])
            elif t in TESTM:
                Xte.append(w)
                meta.append((f, t))
    T = lambda a: torch.tensor(np.array(a, dtype=np.float32))
    return T(Xtr), T(ytr), T(Xva), T(yva), T(Xte), meta


# =============================================================================
# ---------- Sección 4: Entrenamiento Multi-Semilla y Predicción ----------
# =============================================================================
# Entrenamiento por descenso de gradiente (Adam 5e-4, MSELoss) con semillas configuradas
# y Early Stopping basado en el conjunto de validación.

def run(con_precio, tag):
    Xtr, ytr, Xva, yva, Xte, meta = build(con_precio)
    print("  [%s] train %d val %d test %d seq" % (tag, len(Xtr), len(Xva), len(Xte)), flush=True)
    preds = []
    t0 = time.time()
    
    for si, sd in enumerate(SEEDS, 1):
        torch.manual_seed(sd)
        np.random.seed(sd)
        net = Net(Xtr.shape[2])
        opt = torch.optim.Adam(net.parameters(), lr=5e-4)
        lo = nn.MSELoss()
        best = 1e9
        bstate = None
        bad = 0
        
        for ep in range(300):
            net.train()
            perm = torch.randperm(len(Xtr))
            for b in range(0, len(perm), 64):
                ix = perm[b:b + 64]
                opt.zero_grad()
                l = lo(net(Xtr[ix]), ytr[ix])
                l.backward()
                opt.step()
                
            net.eval()
            with torch.no_grad():
                vl = lo(net(Xva), yva).item()
            if vl < best - 1e-5:
                best = vl
                bstate = {k: v.clone() for k, v in net.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= 15:
                    break
                    
        net.load_state_dict(bstate)
        net.eval()
        with torch.no_grad():
            preds.append(net(Xte).numpy())
        el = time.time() - t0
        print("  [%d/%d] semillas | ep=%d val=%.5f | %ds ETA ~%ds" % (si, len(SEEDS), ep + 1, best, el, el / si * (len(SEEDS) - si)), flush=True)
        
    pm = np.mean(preds, axis=0)
    out = {}
    for (f, t), v in zip(meta, pm):
        mn, rg, _, _ = sc[f]
        out[(f, t)] = max(v * rg + mn, 0.0)
    return out

print("[1/3] LSTM parejo sin precio...", flush=True)
o1 = run(False, 'sin_precio')
print("[2/3] LSTM parejo con precio ex-ante...", flush=True)
o2 = run(True, 'con_precio')

# Regla ex-ante: familias sin ventas en los 12 meses previos al corte se
# consideran descontinuadas (no comercializadas) -> prediccion test = 0.
_last12 = piv[piv.index <= CUT].iloc[-12:]
for _f in FAMS:
    if _last12[_f].sum() == 0:
        for _t in TESTM:
            o1[(_f, _t)] = 0.0
            o2[(_f, _t)] = 0.0


# =============================================================================
# ---------- Sección 5: Desagregación Top-Down a SKU y Consolidación en Excel ----------
# =============================================================================
# Conversión de las predicciones de Familia a SKU mediante proporciones estáticas
# y guardado automático en un único archivo Excel multi-hoja.

print("[3/3] a SKU total (prop estaticas)...", flush=True)
shs = rawT[rawT['Fecha'] <= CUT].groupby(['Familia', SK])['Unidades'].sum().reset_index()
shs['prop'] = shs['Unidades'] / shs.groupby('Familia')['Unidades'].transform('sum')

totS = rawT[rawT['Fecha'] > CUT].groupby([SK, 'Fecha'])['Unidades'].sum().rename('total').reset_index()

metrics_table = []
excel_path = os.path.join(RES, 'LSTM_resultados.xlsx')
excel_data = {}

for tag, preds in (('sin_precio', o1), ('con_precio', o2)):
    rows = []
    for (f, t), v in preds.items():
        sub = shs[shs['Familia'] == f]
        for _, r0 in sub.iterrows():
            rows.append((r0[SK], t, v * r0['prop']))
            
    Pq = pd.DataFrame(rows, columns=[SK, 'Fecha', 'pred']).groupby([SK, 'Fecha'])['pred'].sum().reset_index()
    J = Pq.merge(totS, on=[SK, 'Fecha'], how='outer').fillna(0)
    
    rowsF = [(f, t, v) for (f, t), v in preds.items()]
    PF = pd.DataFrame(rowsF, columns=['Familia', 'Fecha', 'pred'])
    realF = rawT[rawT['Fecha'] > CUT].groupby(['Familia', 'Fecha'])['Unidades'].sum().rename('real').reset_index()
    JF = PF.merge(realF, on=['Familia', 'Fecha'], how='outer').fillna(0)
    
    m_fam = compute_metrics(JF['real'], JF['pred'])
    m_sku = compute_metrics(J['total'], J['pred'])
    
    metrics_table.append({'Variante': tag, 'Nivel': 'Familia (Total)', **m_fam})
    metrics_table.append({'Variante': tag, 'Nivel': 'SKU (Total)', **m_sku})
    
    excel_data[f'LSTM_{tag}_Familia'] = JF
    excel_data[f'LSTM_{tag}_SKU'] = J

df_metrics = pd.DataFrame(metrics_table)
excel_data['LSTM_Metricas'] = df_metrics

with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    for sheet_name, df_s in excel_data.items():
        df_s.to_excel(writer, sheet_name=sheet_name, index=False)

print("\n" + "=" * 78)
print(" RESUMEN DE MÉTRICAS - LSTM DEEP LEARNING (TOTAL) ")
print("-" * 78)
print(" Variante         | Nivel    | WAPE(%)  | MAPE(%)  | RMSE     | MAE      | Bias(%)  | R²   ")
print("-" * 78)
for row in metrics_table:
    print(" %-16s | %-8s | %6.2f%%  | %6.2f%%  | %8.2f | %8.2f | %6.2f%%  | %4.2f " % (
        row['Variante'], row['Nivel'], row['WAPE'], row['MAPE'], row['RMSE'], row['MAE'], row['Bias'], row['R2']
    ))
print("=" * 78)
print(f" [OK] Consolidado guardado en: {excel_path}\n", flush=True)
