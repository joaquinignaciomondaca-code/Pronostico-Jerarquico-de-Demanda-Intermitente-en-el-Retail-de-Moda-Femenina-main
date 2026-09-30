# Pronóstico Jerárquico de Demanda Intermitente en Retail de Moda Femenina

Repositorio acompañante de la memoria de título "Pronóstico Jerárquico de Demanda Intermitente en el Retail de Moda Femenina", presentada para optar al título profesional de Ingeniero Comercial en la Universidad Técnica Federico Santa María.

## Descripción

El pipeline implementa una arquitectura híbrida (ES-GBM) que combina suavizamiento exponencial Holt amortiguado con ensambles de árboles potenciados (XGBoost, LightGBM, CatBoost) y un modelo fundacional tabular (TabPFN), con desagregación jerárquica Top-Down hacia nivel SKU. La evaluación se realizó sobre 7 meses fuera de muestra (febrero–agosto 2025) en 100 SKUs de alto volumen y 58 Familias.

## Estructura del repositorio

```
├── 01_modelos_base/          # Scripts de benchmarks (SARIMA, SARIMAX, LSTM, Chronos, Croston, SBA, SN)
├── 02_modelo_final/          # Pipeline ES-GBM (feature engineering, ensemble, desagregación)
├── inputs/                   # Base de datos de entrada (parquet)
├── outputs/                  # Resultados y artefactos generados
└── modelos/                  # Modelos entrenados
```

## Scripts principales

| Script | Descripción |
|--------|-------------|
| `01_modelos_base/a)_sarima_topdown.py` | SARIMA univariado con desagregación Top-Down |
| `01_modelos_base/b)_sarimax_topdown.py` | SARIMAX con precio y descuento como regresores |
| `01_modelos_base/c)_lstm_topdown.py` | Red LSTM con y sin precio ex-ante |
| `01_modelos_base/d)_chronos_topdown.py` | Amazon Chronos-Bolt (zero-shot) |
| `01_modelos_base/e)_seasonal_naive_topdown.py` | Seasonal Naïive |
| `01_modelos_base/f)_croston_topdown.py` | Croston clásico |
| `01_modelos_base/g)_sba_topdown.py` | Syntetos-Boylan Approximation |
| `02_modelo_final/1)_feature_engineering.py` | Construcción de features de panel |
| `02_modelo_final/2)_esgbm_ensemble.py` | Calibración del ensamble ES-GBM con ponderaciones NNLS |
| `02_modelo_final/3)_desagregacion_sku_share.py` | Desagregación jerárquica con selección automática de modo estático/dinámico |

## Métricas principales (test, Feb–Ago 2025)

| Modelo | WAPE SKU | WAPE Familia |
|--------|----------|--------------|
| ES-GBM (propuesto) | 16,10% | 12,89% |
| ES-GBM-Learned Shares | 24,35% | — |
| LSTM (con precio) | 45,75% | — |
| Chronos-Bolt | — | 36,35% |
| Seasonal Naïive | 69,89% | 59,78% |
| SARIMAX | 73,24% | 59,64% |
| SARIMA | 76,74% | 64,84% |
| SBA | 83,42% | — |
| Croston | 84,87% | — |

## Requisitos

- Python 3.9+
- pandas, numpy, scikit-learn
- xgboost, lightgbm, catboost
- tabpfn
- statsmodels
- torch (para LSTM)
- chronos-forecasting (Amazon Chronos-Bolt)
- tectonic (compilación LaTeX, solo para la tesis)

## Datos

La base de datos (`inputs/base_datos.parquet`) contiene series de demanda mensual anonymizadas y escaladas linealmente por un factor proporcional común para preservar confidencialidad comercial. Las variaciones porcentuales, elasticidades y métricas relativas se mantienen intactas.

## Nota

Este repositorio contiene exclusivamente los scripts del pipeline y los datos de entrada/salida. La versión LaTeX de la tesis se agregará tras la defensa.
