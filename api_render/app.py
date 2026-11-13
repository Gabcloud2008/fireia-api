"""FIREIA — Servicio de IA para producción (Render).

API REST FastAPI que expone el modelo predictivo REAL `ia_riesgo_fuego_v1`
(Random Forest validado en la Misión E sobre la base real M014: 2.691.999
observaciones FIRMS; dataset etiquetado de 284.012 filas celda×mes, 3.737
celdas, semilla 20261113).

Tarea del modelo (definición exacta, sin adornos):
    P(la celda registra >= 1 detección VIIRS en el mes siguiente, t+1)
    dadas las características climáticas, de historia de fuego y de cobertura
    de la celda en t y t-1.

Reglas científicas obligatorias (Misión 015-K):
    - Una detección satelital NO equivale a un incendio confirmado.
    - La probabilidad devuelta es de DETECCIÓN VIIRS, no de incendio confirmado.
    - Una anomalía no es una predicción (el módulo de anomalías es sintético y
      declarado; NO se expone aquí).
    - No se inventan datos faltantes: si no hay fila de características válida
      para la celda/mes solicitados, se devuelve un error explicativo (422).
    - Un fallo de adquisición no se interpreta como cero detecciones.
    - Cada respuesta incluye alcance, versión, SHA-256 del modelo y limitaciones.

Constantes fuente: codigo/src/hotspots/analysis/prediccion.py (canónico);
la consistencia se verifica con scripts/verificar_consistencia_api.py.

Variables de entorno:
    PORT                  puerto de escucha (Render lo inyecta; defecto 10000)
    FIREIA_DATA_DIR       carpeta con los artefactos IA (defecto ./datos_ia)
    FIREIA_MODELO         modelo principal: random_forest | gradient_boosting
                          (defecto random_forest; el GBM es el plan B documentado
                          para instancias con < 512 MB de RAM)
    FIREIA_HF_REPO        origen de los artefactos (p400-full/...)
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

API_VERSION = "015k-api-1.0.0"
IA_VERSION = "ia_riesgo_fuego_v1"
SEMILLA = 20261113
SKLEARN_ESPERADO = "1.5.2"

# Constantes canónicas (hotspots/analysis/prediccion.py)
FEATURES = [
    "precip_mm", "temp_media_c", "vpd_kpa", "viento_ms",
    "humedad_suelo_m3m3", "fwi_medio", "dias_fwi_alto",
    "precip_lag1", "fwi_lag1", "humedad_suelo_lag1",
    "fuego_lag1", "fuego_12m", "fuego_mes_anio_anterior",
    "pct_arboles", "pct_pastizal", "pct_cultivo",
]
DESCRIPCION_FEATURES = {
    "precip_mm": "precipitación mensual CHIRPS del mes t (mm)",
    "temp_media_c": "temperatura media mensual ERA5-Land del mes t (°C)",
    "vpd_kpa": "déficit de presión de vapor ERA5-Land del mes t (kPa)",
    "viento_ms": "viento medio mensual ERA5-Land del mes t (m/s)",
    "humedad_suelo_m3m3": "humedad de suelo ERA5-Land del mes t (m3/m3)",
    "fwi_medio": "índice FWI GFWED medio mensual del mes t",
    "dias_fwi_alto": "días del mes t con FWI alto",
    "precip_lag1": "precipitación del mes t-1 (mm)",
    "fwi_lag1": "FWI del mes t-1",
    "humedad_suelo_lag1": "humedad de suelo del mes t-1 (m3/m3)",
    "fuego_lag1": "la celda tuvo >=1 detección VIIRS en t-1 (0/1)",
    "fuego_12m": "número de meses con detección VIIRS en los 12 meses previos a t",
    "fuego_mes_anio_anterior": "la celda tuvo fuego en el mismo mes-calendario del año previo (0/1)",
    "pct_arboles": "cobertura arbórea ESA WorldCover de la celda (%)",
    "pct_pastizal": "cobertura de pastizal ESA WorldCover de la celda (%)",
    "pct_cultivo": "cobertura de cultivo ESA WorldCover de la celda (%)",
}

PERIODO_EFECTIVO = {"inicio": "2020-01", "fin": "2026-04"}  # filas etiquetadas presentes en el dataset
LIMITACIONES = [
    ("La etiqueta es detección VIIRS por celda-mes: agrupación inferida de detecciones, "
     "no un incendio confirmado ni superficie quemada."),
    "Horizonte de 1 mes; no hay validación de horizontes mayores.",
    ("La evaluación (ROC-AUC 0,7967; PR-AUC 0,6247) corresponde al periodo 2025-01..2026-04 "
     "con partición temporal estricta; no garantiza desempeño en meses futuros."),
    ("La calibración es imperfecta (ECE 0,1028 del RF): las probabilidades ordenan el riesgo "
     "pero no deben leerse como frecuencias exactas."),
    ("Los falsos negativos se concentran en fuegos nuevos (celdas sin historia reciente); "
     "los falsos positivos, en transiciones seca-húmeda."),
    "El módulo de anomalías (exp_001) es SINTÉTICO y declarado; no forma parte de este servicio.",
]
AVISO_CIENTIFICO = (
    "Probabilidad de que la celda registre >= 1 detección satelital VIIRS (VNP14IMGML NRT, "
    "NOAA-20/NOAA-21) durante el mes siguiente (t+1). Una detección satelital no equivale a "
    "un incendio confirmado. Esta cifra NO es probabilidad de incendio confirmado ni de "
    "superficie quemada. Una anomalía no es una predicción."
)

DATA_DIR = Path(os.environ.get("FIREIA_DATA_DIR", "datos_ia"))
MODELO_PRINCIPAL = os.environ.get("FIREIA_MODELO", "random_forest")
HF_REPO = os.environ.get("FIREIA_HF_REPO", "p400-full/modis-viirs-hotspots-paraguay")

ARTEFACTOS_ESPERADOS = {  # del manifiesto_ia_v1.json (verificado en build)
    "metricas_ia.json": "ce2e5c940123650c524dbe85463f8e6f595579ea0528445db07b70e78a514c6c",
    "dataset_riesgo_celda_mes.parquet": "37c4c012328407b22550f341c6032ac8e8a751199a24b5b82bf2299a67c3a745",
    "modelo_random_forest.joblib": "2361c743bd51c72bc4b09c13bbcc2a83e3e70bc905bdb6f68efa69881e93c1c2",
    "modelo_gradient_boosting.joblib": "630cc704ed15ab1c7c2932e5b6fb2221361cd2e6a543c5402a9ed35cd4749b76",
}

logging.basicConfig(
    level=logging.INFO,
    format='{"ts":"%(asctime)s","nivel":"%(levelname)s","modulo":"%(name)s","msj":%(message)s}',
    force=True,
)
logger = logging.getLogger("fireia.api")

ESTADO: dict[str, Any] = {
    "modelos": {},           # nombre -> modelo sklearn cargado
    "sha256": {},            # artefacto -> sha256 calculado
    "grilla": None,          # np.array [N,2] de (lat, lon) de las celdas
    "grilla_ids": None,      # list de ids de celda (formato canonico lon_lat)
    "feats_por_clave": None, # dict (celda, mes) -> np.array(16) de features
    "meses_disponibles": None,
    "metricas": None,        # dict de metricas_ia.json
    "listo": False,
    "error_carga": None,
    "inicio": time.time(),
    "sklearn": None,
}


def _sha256_archivo(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def _mes_siguiente(mes: str) -> str:
    anio, m = int(mes[:4]), int(mes[5:7])
    return f"{anio + (m == 12)}-{(m % 12) + 1:02d}"


def cargar_modelos() -> None:
    """Carga artefactos y modelo(s). Se ejecuta en un hilo al arrancar."""
    try:
        import joblib
        import sklearn

        t0 = time.time()
        # 1) verificación de integridad de los artefactos (cadena de custodia)
        for nombre, esperado in ARTEFACTOS_ESPERADOS.items():
            ruta = DATA_DIR / nombre
            if not ruta.exists():
                raise FileNotFoundError(f"artefacto ausente: {ruta}")
            real = _sha256_archivo(ruta)
            ESTADO["sha256"][nombre] = real
            if real != esperado:
                raise ValueError(
                    f"SHA-256 de {nombre} no coincide (esperado {esperado[:12]}..., real {real[:12]}...)"
                )
        logger.info('{"evento":"artefactos_verificados","n":4}')

        # 2) métricas y dataset
        ESTADO["metricas"] = json.loads((DATA_DIR / "metricas_ia.json").read_text(encoding="utf-8"))
        tabla = pq.read_table(DATA_DIR / "dataset_riesgo_celda_mes.parquet", columns=["celda", "mes", *FEATURES])
        df = tabla.to_pandas()

        # 3) grilla nacional + índice de características (vectorizado)
        partes = df["celda"].str.split("_", expand=True)
        df["lon"] = partes[0].astype(float)
        df["lat"] = partes[1].astype(float)
        celdas = df[["celda", "lat", "lon"]].drop_duplicates().sort_values("celda")
        ESTADO["grilla_ids"] = celdas["celda"].tolist()
        ESTADO["grilla"] = celdas[["lat", "lon"]].to_numpy(dtype=float)

        claves = list(zip(df["celda"].tolist(), df["mes"].tolist(), strict=True))
        matriz = df[FEATURES].to_numpy(dtype=float)
        ESTADO["feats_por_clave"] = dict(zip(claves, matriz, strict=True))
        ESTADO["meses_disponibles"] = sorted({m for _, m in claves})
        ESTADO["sklearn"] = sklearn.__version__
        del df, partes

        # 4) modelos
        for nombre in ("random_forest", "gradient_boosting"):
            ESTADO["modelos"][nombre] = joblib.load(DATA_DIR / f"modelo_{nombre}.joblib")
        if MODELO_PRINCIPAL not in ESTADO["modelos"]:
            raise ValueError(f"FIREIA_MODELO invalido: {MODELO_PRINCIPAL}")

        ESTADO["listo"] = True
        logger.info(f'{{"evento":"modelo_listo","principal":"{MODELO_PRINCIPAL}","segundos":{time.time() - t0:.1f}}}')
    except Exception as e:
        ESTADO["error_carga"] = f"{type(e).__name__}: {e}"
        logger.exception('{"evento":"error_carga_modelo"}')


@asynccontextmanager
async def lifespan(app: FastAPI):
    hilo = threading.Thread(target=cargar_modelos, daemon=True, name="carga-modelo")
    hilo.start()
    yield


app = FastAPI(
    title="FIREIA — Servicio de IA",
    version=API_VERSION,
    description=(
        "API de inferencia del modelo predictivo de actividad de fuego de FIREIA "
        "(Paraguay Fire Intelligence, NASA Space Apps 2026). "
        "Modelo: " + IA_VERSION + ". "
        "La probabilidad devuelta es de DETECCIÓN VIIRS en el mes siguiente por celda 0,1°, "
        "no de incendio confirmado."
    ),
    lifespan=lifespan,
)


class EntradaPredict(BaseModel):
    lat: float = Field(..., ge=-90, le=90, description="latitud decimal (Paraguay: aprox -27.65..-19.35)")
    lon: float = Field(..., ge=-180, le=180, description="longitud decimal (Paraguay: aprox -62.65..-54.25)")
    mes: str = Field(..., pattern=r"^\d{4}-(0[1-9]|1[0-2])$", description='mes de referencia t, formato "YYYY-MM"')
    modelo: str = Field(MODELO_PRINCIPAL, description="random_forest (por defecto y validado) o gradient_boosting")
    umbral: float = Field(0.5, gt=0.0, lt=1.0, description="umbral de decisión para la clasificación binaria")

    @field_validator("modelo")
    @classmethod
    def _validar_modelo(cls, v: str) -> str:
        if v not in ("random_forest", "gradient_boosting"):
            raise ValueError("modelo debe ser 'random_forest' o 'gradient_boosting'")
        return v


def _error(codigo: str, mensaje: str, detalle: dict | None = None, estado_http: int = 422):
    return JSONResponse(
        status_code=estado_http,
        content={"error": {"codigo": codigo, "mensaje": mensaje, "detalle": detalle or {}}},
    )


def _resolver_celda(lat: float, lon: float) -> tuple[str, float]:
    """Celda 0,1° más cercana de la grilla nacional; rechaza puntos alejados."""
    grilla = ESTADO["grilla"]
    dist = np.sqrt((grilla[:, 0] - lat) ** 2 + (grilla[:, 1] - lon) ** 2)
    i = int(np.argmin(dist))
    d = float(dist[i])
    if d > 0.075:  # media celda (0,0707°) con margen mínimo; fuera de la grilla nacional
        raise HTTPException(status_code=422, detail={
            "codigo": "fuera_de_grilla",
            "mensaje": "La coordenada está fuera de la grilla nacional 0,1° de FIREIA "
                       "(no hay celda dentro de la tolerancia de media celda).",
            "detalle": {"celda_mas_cercana": ESTADO["grilla_ids"][i], "distancia_grados": round(d, 4), "tolerancia_grados": 0.075},
        })
    return ESTADO["grilla_ids"][i], d


@app.get("/")
def raiz():
    return {
        "servicio": "FIREIA — Servicio de IA",
        "proyecto": "Paraguay Fire Intelligence — Harmonization of MODIS and VIIRS Hot Spots (NASA Space Apps 2026)",
        "api_version": API_VERSION,
        "modelo_version": IA_VERSION,
        "estado": "listo" if ESTADO["listo"] else ("error" if ESTADO["error_carga"] else "cargando"),
        "endpoints": ["GET /", "GET /health", "GET /model/info", "POST /predict", "GET /metrics"],
        "documentacion": "/docs (OpenAPI)",
        "generado_en_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


@app.get("/health")
def health():
    return {
        "status": "ok" if ESTADO["listo"] else ("error" if ESTADO["error_carga"] else "cargando"),
        "modelo_cargado": ESTADO["listo"],
        "modelo_principal": MODELO_PRINCIPAL,
        "modelo_version": IA_VERSION,
        "api_version": API_VERSION,
        "sklearn": ESTADO["sklearn"],
        "error_carga": ESTADO["error_carga"],
        "uptime_s": round(time.time() - ESTADO["inicio"], 1),
        "fuente_datos": "base real M014 (2.691.999 observaciones FIRMS) + contexto ambiental M013/D",
        "repositorio_modelo": f"https://huggingface.co/datasets/{HF_REPO}/tree/main/data/PROCESADA/ia_v1",
    }


@app.get("/model/info")
def model_info():
    if ESTADO["metricas"] is None:
        return _error("cargando", "El modelo todavía se está cargando; reintente en unos segundos.", estado_http=503)
    m = ESTADO["metricas"]
    return {
        "nombre": "FIREIA — modelo predictivo de actividad de fuego",
        "version": IA_VERSION,
        "api_version": API_VERSION,
        "tarea": "clasificación binaria por celda 0,1° × mes",
        "definicion_objetivo": (
            "P(celda registra >= 1 detección VIIRS en el mes siguiente, t+1 | clima de t y t-1, "
            "historia de fuego VIIRS de la celda, cobertura WorldCover)."
        ),
        "definicion_etiqueta": m["dataset"]["definicion_etiqueta"],
        "que_no_es": [
            "NO es probabilidad de incendio confirmado",
            "NO es superficie quemada",
            "NO es detección en tiempo real (eso es FIRMS)",
            "NO es el módulo de anomalías (exp_001, sintético y declarado)",
        ],
        "features": [{"nombre": f, "descripcion": DESCRIPCION_FEATURES[f]} for f in FEATURES],
        "regla_antifuga": m["dataset"].get("regla_antifuga", "(no declarado en este artefacto)"),
        "dataset": {
            "filas": m["dataset"]["filas"],
            "celdas": m["dataset"]["celdas"],
            "positivos": m["dataset"]["positivos"],
            "tasa_positivos": m["dataset"]["tasa_positivos"],
            "particiones": m["particiones"]["temporal"],
            "filas_excluidas": m["dataset"]["filas_excluidas"],
            "periodo_efectivo_cobertura_api": PERIODO_EFECTIVO,
        },
        "modelos_disponibles": [
            {"nombre": "random_forest", "rol": "principal (mejor por validación)", "sha256": ESTADO["sha256"].get("modelo_random_forest.joblib")},
            {"nombre": "gradient_boosting", "rol": "alternativa documentada", "sha256": ESTADO["sha256"].get("modelo_gradient_boosting.joblib")},
        ],
        "modelo_por_defecto": MODELO_PRINCIPAL,
        "semilla": SEMILLA,
        "sklearn": ESTADO["sklearn"],
        "metricas_resumen": {
            modelo: {
                "roc_auc_evaluacion": m["modelos"][modelo]["evaluacion"]["roc_auc"],
                "pr_auc_evaluacion": m["modelos"][modelo]["evaluacion"]["pr_auc"],
                "brier_evaluacion": m["modelos"][modelo]["evaluacion"]["brier"],
            }
            for modelo in ("baseline_climatologia", "random_forest", "gradient_boosting")
        },
        "mejor_modelo_por_validacion": m["mejor_modelo_por_validacion"],
        "procedencia": {
            "repositorio": f"https://huggingface.co/datasets/{HF_REPO}",
            "artefactos": "data/PROCESADA/ia_v1/ (manifiesto con SHA-256)",
            "sha256_verificados_en_arranque": ESTADO["sha256"],
        },
        "limitaciones": LIMITACIONES,
        "aviso_cientifico": AVISO_CIENTIFICO,
    }


@app.get("/metrics")
def metrics():
    if ESTADO["metricas"] is None:
        return _error("cargando", "El modelo todavía se está cargando; reintente en unos segundos.", estado_http=503)
    m = ESTADO["metricas"]
    return {
        "version": IA_VERSION,
        "semilla": m["semilla"],
        "sklearn": m["sklearn"],
        "procedencia": {
            "fuente": f"{HF_REPO}: data/PROCESADA/ia_v1/metricas_ia.json",
            "sha256": ESTADO["sha256"].get("metricas_ia.json"),
            "generado_en_utc": "2026-11-13T17:24:09Z (manifiesto ia_v1)",
            "evaluacion": "partición temporal 2025-01..2026-04 (efectivo; FWI GFWED sin cobertura 2026-05..09)",
            "informe": "mision_finalizacion/E_informe_ia.md (Misión E, repositorio HF)",
        },
        "particiones": m["particiones"],
        "importancia_variables_random_forest": m.get("importancia_variables_random_forest"),
        "validacion_geografica_gbm": m.get("validacion_geografica_gbm"),
        "mejor_modelo_por_validacion": m["mejor_modelo_por_validacion"],
        "modelos": m["modelos"],
        "lectura_honesta": (
            "RF supera al baseline de climatología por +0,029 ROC-AUC y +0,059 PR-AUC (+10,5% relativo). "
            "La ventaja es modesta y así se declara. La probabilidad ordena riesgo; la calibración es imperfecta."
        ),
    }


@app.post("/predict")
def predict(entrada: EntradaPredict):
    if not ESTADO["listo"]:
        if ESTADO["error_carga"]:
            return _error("error_carga_modelo",
                          "El modelo no pudo cargarse; el servicio no puede inferir. Contacte al operador.",
                          {"error_carga": ESTADO["error_carga"]}, estado_http=503)
        return _error("cargando", "El modelo todavía se está cargando; reintente en unos segundos.", estado_http=503)

    t0 = time.time()
    # celda más cercana de la grilla nacional
    try:
        celda, dist = _resolver_celda(entrada.lat, entrada.lon)
    except HTTPException as e:
        det = e.detail if isinstance(e.detail, dict) else {}
        return _error(det.get("codigo", "fuera_de_grilla"), det.get("mensaje", "coordenada fuera de grilla"), det.get("detalle"), 422)

    # fila de características: sin invención de datos
    clave = (celda, entrada.mes)
    fila = ESTADO["feats_por_clave"].get(clave)
    if fila is None:
        return _error(
            "sin_caracteristicas_validas",
            "No existe una fila de características válida para esa celda y mes. "
            "El servicio no inventa datos: sin clima completo (p. ej. FWI GFWED sin cobertura 2026-05..2026-09) "
            "o fuera del periodo etiquetado 2020-01..2026-04, no hay inferencia.",
            {
                "celda": celda,
                "mes_solicitado": entrada.mes,
                "periodo_efectivo": PERIODO_EFECTIVO,
                "meses_con_datos": [ESTADO["meses_disponibles"][0], ESTADO["meses_disponibles"][-1]],
            },
            422,
        )

    modelo = ESTADO["modelos"][entrada.modelo]
    prob = float(modelo.predict_proba(fila.reshape(1, -1))[:, 1][0])
    binaria = int(prob >= entrada.umbral)
    lat_c, lon_c = celda.split("_")
    resultado = {
        "probabilidad_deteccion_viirs_prox_mes": round(prob, 6),
        "clasificacion_umbral": {
            "umbral": entrada.umbral,
            "prediccion_binaria": binaria,
            "etiqueta": "actividad detectable esperada" if binaria else "sin actividad detectable esperada",
            "nota": "la elección del umbral es una decisión operativa; 0,5 es el umbral de evaluación del informe",
        },
        "celda": {
            "id": celda,
            "lat": float(lat_c),
            "lon": float(lon_c),
            "resolucion_grados": 0.1,
            "distancia_solicitada_grados": round(dist, 4),
        },
        "mes_referencia": entrada.mes,
        "mes_objetivo": _mes_siguiente(entrada.mes),
        "modelo": {
            "nombre": entrada.modelo,
            "version": IA_VERSION,
            "sha256": ESTADO["sha256"].get(f"modelo_{entrada.modelo}.joblib"),
            "sklearn": ESTADO["sklearn"],
            "semilla": SEMILLA,
        },
        "variables_usadas": {f: round(float(v), 6) for f, v in zip(FEATURES, fila, strict=True)},
        "alcance": {
            "unidad": "celda 0,1° (grilla ambiental nacional)",
            "horizonte": "1 mes (t+1)",
            "objetivo": ">= 1 detección VIIRS en la celda en t+1 (agrupación inferida de detecciones)",
        },
        "aviso_cientifico": AVISO_CIENTIFICO,
        "limitaciones": LIMITACIONES,
        "generado_en_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "latencia_ms": round((time.time() - t0) * 1000, 1),
    }
    logger.info(f'{{"evento":"predict","celda":"{celda}","mes":"{entrada.mes}","modelo":"{entrada.modelo}","p":{prob:.6f}}}')
    return resultado


@app.exception_handler(Exception)
async def handler_generico(request, exc):  # pragma: no cover — red de seguridad
    logger.exception('{"evento":"error_interno"}')
    return JSONResponse(
        status_code=500,
        content={"error": {"codigo": "error_interno",
                           "mensaje": "Error interno del servicio; el incidente quedó registrado en los logs.",
                           "detalle": {}}},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "10000")), log_level="info")
