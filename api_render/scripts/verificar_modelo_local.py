"""Reproduce las inferencias de referencia con el modelo canónico local.

Calcula las 4 probabilidades de referencia usadas por pruebas_produccion.py
para comparar producción contra local. Debe ejecutarse dentro del proyecto
canónico (codigo/ instalable o en PYTHONPATH) con los artefactos de
data/PROCESADA/ia_v1/ presentes.

Uso:
    python verificar_modelo_local.py --artefactos ../data/PROCESADA/ia_v1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

FEATURES = [
    "precip_mm", "temp_media_c", "vpd_kpa", "viento_ms",
    "humedad_suelo_m3m3", "fwi_medio", "dias_fwi_alto",
    "precip_lag1", "fwi_lag1", "humedad_suelo_lag1",
    "fuego_lag1", "fuego_12m", "fuego_mes_anio_anterior",
    "pct_arboles", "pct_pastizal", "pct_cultivo",
]
REFERENCIAS = [
    {"lat": -24.05, "lon": -54.25, "mes": "2025-06", "p": 0.357805},
    {"lat": -25.35, "lon": -57.55, "mes": "2024-08", "p": 0.619886},
    {"lat": -22.45, "lon": -61.85, "mes": "2025-09", "p": 0.360970},
    {"lat": -23.15, "lon": -58.15, "mes": "2026-04", "p": 0.308019},
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artefactos", default="data/PROCESADA/ia_v1")
    args = ap.parse_args()
    dir_ia = Path(args.artefactos)

    import pandas as pd

    modelo = joblib.load(dir_ia / "modelo_random_forest.joblib")
    df = pd.read_parquet(dir_ia / "dataset_riesgo_celda_mes.parquet")

    ok = True
    salidas = []
    for ref in REFERENCIAS:
        celda = f"{ref['lon']}_{ref['lat']}"
        r = df[(df["celda"] == celda) & (df["mes"] == ref["mes"])]
        if len(r) != 1:
            print(f"FALTA fila para {celda} {ref['mes']}")
            ok = False
            continue
        p = float(modelo.predict_proba(r[FEATURES].to_numpy())[:, 1][0])
        coincide = abs(p - ref["p"]) <= 1e-6
        ok = ok and coincide
        salidas.append({**ref, "calculado": round(p, 6), "coincide": coincide})
        print(f"{celda} {ref['mes']}: calculado {p:.6f} | referencia {ref['p']:.6f} | {'OK' if coincide else 'DIFIERE'}")

    print(json.dumps({"verificado": ok, "referencias": salidas}, ensure_ascii=False, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
