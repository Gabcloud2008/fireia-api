"""Pruebas del servicio de IA FIREIA (contrato de la API).

Se ejecutan offline y sin el modelo real de 193 MB: se entrena en el momento
un RF minúsculo (fixture declarado como SINTÉTICO) y se arman artefactos
temporales con el mismo esquema del dataset canónico. Esto valida el CONTRATO
del servicio (endpoints, validación, errores, ausencia de invención de datos).

La identidad del modelo real en producción se garantiza por otra vía: el
build verifica los SHA-256 del manifiesto canónico y
scripts/pruebas_produccion.py compara inferencias de producción contra las
referencias locales (data/PROCESADA/ia_v1).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app as servicio

FEATS = servicio.FEATURES


@pytest.fixture(scope="module")
def cliente(tmp_path_factory):
    """Servicio de prueba con artefactos sintéticos mínimos (declarados)."""
    rng = np.random.default_rng(20261113)
    dir_datos = tmp_path_factory.mktemp("datos_ia")

    # dataset sintético: 2 celdas × 3 meses
    filas = []
    for celda in ("-54.25_-24.05", "-57.55_-25.35"):
        for mes in ("2025-01", "2025-02", "2025-03"):
            fila = {"celda": celda, "mes": mes}
            for f in FEATS:
                fila[f] = float(rng.random() * 10)
            filas.append(fila)
    df = pd.DataFrame(filas)
    df.to_parquet(dir_datos / "dataset_riesgo_celda_mes.parquet", index=False)

    # RF minúsculo entrenado en datos sintéticos (solo para el contrato)
    X = df[FEATS].to_numpy()
    y = (rng.random(len(df)) > 0.5).astype(int)
    rf = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)
    joblib.dump(rf, dir_datos / "modelo_random_forest.joblib")
    joblib.dump(rf, dir_datos / "modelo_gradient_boosting.joblib")

    metricas = {
        "version": "ia_riesgo_fuego_v1",
        "semilla": 20261113,
        "sklearn": "test",
        "mejor_modelo_por_validacion": "random_forest",
        "particiones": {"temporal": {"entrenamiento": ["2020-01", "2023-12"], "validacion": ["2024-01", "2024-12"], "evaluacion": ["2025-01", "2026-04"], "filas": [1, 1, 1]}},
        "dataset": {"definicion_etiqueta": "test", "filas": 6, "celdas": 2, "positivos": 3, "tasa_positivos": 0.5, "filas_excluidas": {"clima_o_etiqueta_incompletos": 0}, "regla_antifuga": "test: features solo de t y t-1"},
        "modelos": {m: {"validacion": {"roc_auc": 0.5}, "evaluacion": {"roc_auc": 0.5, "pr_auc": 0.5, "brier": 0.25}} for m in ("baseline_climatologia", "random_forest", "gradient_boosting")},
    }
    (dir_datos / "metricas_ia.json").write_text(json.dumps(metricas), encoding="utf-8")

    # SHA-256 falsos coherentes con los artefactos sintéticos
    for nombre in ("metricas_ia.json", "dataset_riesgo_celda_mes.parquet", "modelo_random_forest.joblib", "modelo_gradient_boosting.joblib"):
        h = __import__("hashlib").sha256((dir_datos / nombre).read_bytes()).hexdigest()
        servicio.ARTEFACTOS_ESPERADOS[nombre] = h

    # estado limpio + carga síncrona
    servicio.ESTADO.update(modelos={}, sha256={}, grilla=None, grilla_ids=None, feats_por_clave=None,
                           meses_disponibles=None, metricas=None, listo=False, error_carga=None, sklearn=None)
    servicio.DATA_DIR = dir_datos
    servicio.cargar_modelos()
    assert servicio.ESTADO["listo"]

    with TestClient(servicio.app, raise_server_exceptions=False) as c:
        yield c


# ----------------------------------------------------------------- contrato
def test_raiz_y_health(cliente):
    r = cliente.get("/")
    assert r.status_code == 200
    d = r.json()
    assert d["api_version"] == servicio.API_VERSION
    assert d["modelo_version"] == "ia_riesgo_fuego_v1"
    assert set(d["endpoints"]) == {"GET /", "GET /health", "GET /model/info", "POST /predict", "GET /metrics"}

    r = cliente.get("/health")
    assert r.status_code == 200
    d = r.json()
    assert d["modelo_cargado"] is True
    assert d["status"] == "ok"


def test_model_info_declara_tarea_y_limitaciones(cliente):
    d = cliente.get("/model/info").json()
    assert d["tarea"] == "clasificación binaria por celda 0,1° × mes"
    assert "VIIRS" in d["definicion_objetivo"]
    assert len(d["features"]) == 16
    assert any("NO es probabilidad de incendio confirmado" in x for x in d["que_no_es"])
    assert len(d["limitaciones"]) >= 5
    assert "sha256" in d["modelos_disponibles"][0]


def test_metrics_incluye_procedencia(cliente):
    d = cliente.get("/metrics").json()
    assert "procedencia" in d and "sha256" in d["procedencia"]
    assert d["version"] == "ia_riesgo_fuego_v1"


def test_predict_valido_con_alcance(cliente):
    r = cliente.post("/predict", json={"lat": -24.05, "lon": -54.25, "mes": "2025-02"})
    assert r.status_code == 200
    d = r.json()
    assert 0.0 <= d["probabilidad_deteccion_viirs_prox_mes"] <= 1.0
    assert d["celda"]["id"] == "-54.25_-24.05"
    assert d["mes_objetivo"] == "2025-03"
    assert "detección satelital no equivale a un incendio confirmado" in d["aviso_cientifico"]
    assert set(d["variables_usadas"]) == set(FEATS)
    assert d["modelo"]["sha256"]


def test_determinismo_de_la_misma_entrada(cliente):
    a = cliente.post("/predict", json={"lat": -25.35, "lon": -57.55, "mes": "2025-01"}).json()
    b = cliente.post("/predict", json={"lat": -25.35, "lon": -57.55, "mes": "2025-01"}).json()
    assert a["probabilidad_deteccion_viirs_prox_mes"] == b["probabilidad_deteccion_viirs_prox_mes"]


def test_fuera_de_grilla_rechazado(cliente):
    r = cliente.post("/predict", json={"lat": 10.0, "lon": 10.0, "mes": "2025-02"})
    assert r.status_code == 422
    d = r.json()
    assert d["error"]["codigo"] == "fuera_de_grilla"


def test_mes_sin_caracteristicas_no_se_inventa(cliente):
    r = cliente.post("/predict", json={"lat": -24.05, "lon": -54.25, "mes": "2026-12"})
    assert r.status_code == 422
    d = r.json()
    assert d["error"]["codigo"] == "sin_caracteristicas_validas"
    assert "no inventa" in d["error"]["mensaje"]


def test_entrada_invalida_rechazada(cliente):
    for cuerpo in (
        {"lat": "x", "lon": -54.25, "mes": "2025-02"},       # lat no numérica
        {"lon": -54.25, "mes": "2025-02"},                    # falta lat
        {"lat": -24.05, "lon": -54.25, "mes": "2025-13"},     # mes imposible
        {"lat": -24.05, "lon": -54.25, "mes": "junio"},       # formato incorrecto
        {"lat": -24.05, "lon": -54.25, "mes": "2025-02", "modelo": "gpt"},  # modelo inexistente
        {"lat": -24.05, "lon": -54.25, "mes": "2025-02", "umbral": 1.5},    # umbral fuera de rango
    ):
        r = cliente.post("/predict", json=cuerpo)
        assert r.status_code == 422, cuerpo


def test_modelo_no_cargado_se_declara_sin_simular(cliente):
    """Solo se sirve el modelo principal cargado (limitación de RAM declarada).

    En instancias free de Render (512 MB) cargar además el RF de 193 MB
    produce OOM; el servicio carga únicamente FIREIA_MODELO y, si se pide
    otro, responde 422 explicativo en lugar de simular la inferencia.
    """
    alt = "gradient_boosting" if servicio.MODELO_PRINCIPAL == "random_forest" else "random_forest"
    r = cliente.post("/predict", json={"lat": -24.05, "lon": -54.25, "mes": "2025-02", "modelo": alt})
    if alt == servicio.MODELO_PRINCIPAL:  # pragma: no cover (fixture: principal fijo)
        assert r.status_code == 200
        assert r.json()["modelo"]["nombre"] == alt
        return
    assert r.status_code == 422
    d = r.json()
    assert d["error"]["codigo"] == "modelo_no_cargado"
    assert "no está cargado" in d["error"]["mensaje"]
    assert alt not in d["error"]["detalle"]["modelos_cargados"]

    # el principal sí responde
    r2 = cliente.post("/predict", json={"lat": -24.05, "lon": -54.25, "mes": "2025-02",
                                        "modelo": servicio.MODELO_PRINCIPAL})
    assert r2.status_code == 200
    assert r2.json()["modelo"]["nombre"] == servicio.MODELO_PRINCIPAL
