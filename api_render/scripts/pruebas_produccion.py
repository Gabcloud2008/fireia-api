"""Batería de pruebas de producción del servicio de IA FIREIA (Misión 015-K, Fase 5).

Ejecuta contra --url (staging local o la URL pública de Render):
  1.  GET /health                    → estado y modelo cargado
  2.  GET /                          → info y versión
  3.  GET /model/info                → tarea, variables, limitaciones, SHA-256
  4.  GET /metrics                   → métricas con procedencia
  5.  POST /predict válidas          → comparación contra inferencias de
      referencia calculadas LOCALMENTE con el modelo canónico (valores fijos
      registrados en REFERENCIAS; reproducibles con
      scripts/verificar_modelo_local.py)
  6.  Entradas inválidas             → 422 con error explicativo
  7.  Datos incompletos / sin cobertura → 422 sin invención de datos
  8.  Formato JSON y esquema de respuesta
  9.  Tiempos de respuesta
  10. Identidad del modelo (SHA-256 vs manifiesto canónico)

Salida: resumen en consola + evidencia JSON (--salida).
Códigos de salida: 0 = todas las pruebas pasan; 1 = hay fallos.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from urllib import error as ue
from urllib import request as ur

# SHA-256 canónicos (manifiesto_ia_v1.json)
SHA_RF = "2361c743bd51c72bc4b09c13bbcc2a83e3e70bc905bdb6f68efa69881e93c1c2"
SHA_GBM = "630cc704ed15ab1c7c2932e5b6fb2221361cd2e6a543c5402a9ed35cd4749b76"
SHA_METRICAS = "ce2e5c940123650c524dbe85463f8e6f595579ea0528445db07b70e78a514c6c"

# Inferencias de referencia calculadas LOCALMENTE con modelo_random_forest.joblib
# (sha 2361c743...) sobre dataset_riesgo_celda_mes.parquet (sha 37c4c012...);
# reproducibles con scripts/verificar_modelo_local.py. Tolerancia 1e-6.
REFERENCIAS = [
    {"lat": -24.05, "lon": -54.25, "mes": "2025-06", "p": 0.357805},
    {"lat": -25.35, "lon": -57.55, "mes": "2024-08", "p": 0.619886},
    {"lat": -22.45, "lon": -61.85, "mes": "2025-09", "p": 0.360970},
    {"lat": -23.15, "lon": -58.15, "mes": "2026-04", "p": 0.308019},
]

INVALIDAS = [
    {"cuerpo": {"lat": 10.0, "lon": 10.0, "mes": "2025-06"}, "esperado": "fuera_de_grilla", "rotulo": "coordenada fuera de Paraguay"},
    {"cuerpo": {"lat": -24.05, "lon": -54.25, "mes": "2027-01"}, "esperado": "sin_caracteristicas_validas", "rotulo": "mes sin cobertura (futuro)"},
    {"cuerpo": {"lat": -24.05, "lon": -54.25, "mes": "2019-06"}, "esperado": "sin_caracteristicas_validas", "rotulo": "mes anterior al dataset"},
    {"cuerpo": {"lat": -24.05, "lon": -54.25, "mes": "2026-08"}, "esperado": "sin_caracteristicas_validas", "rotulo": "mes sin FWI (brecha documentada)"},
]

MALFORMADAS = [
    {"cuerpo": {"lon": -54.25, "mes": "2025-06"}, "rotulo": "falta lat"},
    {"cuerpo": {"lat": -24.05, "mes": "2025-06"}, "rotulo": "falta lon"},
    {"cuerpo": {"lat": -24.05, "lon": -54.25}, "rotulo": "falta mes"},
    {"cuerpo": {"lat": -24.05, "lon": -54.25, "mes": "junio-2025"}, "rotulo": "formato de mes incorrecto"},
    {"cuerpo": {"lat": "veinte", "lon": -54.25, "mes": "2025-06"}, "rotulo": "lat no numérica"},
    {"cuerpo": {"lat": -24.05, "lon": -54.25, "mes": "2025-06", "modelo": "chatbot"}, "rotulo": "modelo inexistente"},
    {"cuerpo": {"lat": -24.05, "lon": -54.25, "mes": "2025-06", "umbral": 2.0}, "rotulo": "umbral fuera de rango"},
]

PREDICT_CLAVES = {
    "probabilidad_deteccion_viirs_prox_mes", "clasificacion_umbral", "celda", "mes_referencia",
    "mes_objetivo", "modelo", "variables_usadas", "alcance", "aviso_cientifico",
    "limitaciones", "generado_en_utc",
}


class Pruebas:
    def __init__(self, base: str):
        self.base = base.rstrip("/")
        self.resultados: list[dict] = []
        self.latencias: dict[str, list[float]] = {}

    def _peticion(self, metodo: str, ruta: str, cuerpo: dict | None = None, timeout: int = 60):
        url = self.base + ruta
        datos = json.dumps(cuerpo).encode() if cuerpo is not None else None
        req = ur.Request(url, data=datos, method=metodo,
                         headers={"Content-Type": "application/json"} if datos else {})
        t0 = time.time()
        try:
            with ur.urlopen(req, timeout=timeout) as r:
                crudo = r.read()
                ms = (time.time() - t0) * 1000
                self.latencias.setdefault(f"{metodo} {ruta}", []).append(ms)
                try:
                    return r.status, json.loads(crudo), ms
                except json.JSONDecodeError:
                    return r.status, {"_no_json": True}, ms
        except ue.HTTPError as e:
            ms = (time.time() - t0) * 1000
            self.latencias.setdefault(f"{metodo} {ruta}", []).append(ms)
            try:
                return e.code, json.loads(e.read()), ms
            except Exception:  # noqa: BLE001
                return e.code, {}, ms

    def registrar(self, nombre: str, ok: bool, detalle: str = ""):
        self.resultados.append({"prueba": nombre, "ok": bool(ok), "detalle": detalle})
        simbolo = "PASS" if ok else "FALLA"
        print(f"  [{simbolo}] {nombre}" + (f" — {detalle}" if detalle else ""))
        return ok

    # ---------------------------------------------------------------- pruebas
    def probar_health(self):
        st, d, ms = self._peticion("GET", "/health")
        ok = st == 200 and d.get("modelo_cargado") is True and d.get("status") == "ok"
        self.registrar("1. /health responde con modelo cargado", ok,
                       f"status={d.get('status')} modelo={d.get('modelo_principal')} {ms:.0f}ms")
        return ok

    def probar_raiz(self):
        st, d, _ = self._peticion("GET", "/")
        ok = st == 200 and d.get("modelo_version") == "ia_riesgo_fuego_v1" and len(d.get("endpoints", [])) == 5
        return self.registrar("2. / informa versión y endpoints", ok, f"api={d.get('api_version')}")

    def probar_model_info(self):
        st, d, _ = self._peticion("GET", "/model/info")
        ok = (
            st == 200
            and d.get("tarea") == "clasificación binaria por celda 0,1° × mes"
            and len(d.get("features", [])) == 16
            and d.get("dataset", {}).get("celdas") == 3737
            and any("NO es probabilidad de incendio confirmado" in q for q in d.get("que_no_es", []))
            and len(d.get("limitaciones", [])) >= 5
        )
        return self.registrar("3. /model/info declara tarea, 16 variables, 3.737 celdas y limitaciones", ok)

    def probar_metrics(self):
        st, d, _ = self._peticion("GET", "/metrics")
        rf = d.get("modelos", {}).get("random_forest", {}).get("evaluacion", {})
        ok = (
            st == 200
            and d.get("procedencia", {}).get("sha256") == SHA_METRICAS
            and rf.get("roc_auc") == 0.7967
            and rf.get("pr_auc") == 0.6247
            and "baseline_climatologia" in d.get("modelos", {})
        )
        return self.registrar("4. /metrics con ROC-AUC 0,7967 y procedencia verificable", ok)

    def probar_predict_validas(self):
        todo_ok = True
        for ref in REFERENCIAS:
            st, d, ms = self._peticion("POST", "/predict", ref | {})
            if st != 200:
                todo_ok = self.registrar(f"5. predict válida {ref['lat']},{ref['lon']} {ref['mes']}", False, f"HTTP {st}")
                continue
            diff = abs(d["probabilidad_deteccion_viirs_prox_mes"] - ref["p"])
            ok = diff <= 1e-6 and PREDICT_CLAVES <= set(d)
            todo_ok = self.registrar(
                f"5. predict {ref['lat']},{ref['lon']} {ref['mes']} == referencia local",
                ok, f"p={d['probabilidad_deteccion_viirs_prox_mes']} (ref {ref['p']}, Δ{diff:.2e}) {ms:.0f}ms",
            )
        return todo_ok

    def probar_invalidas(self):
        todo_ok = True
        for caso in INVALIDAS:
            st, d, _ = self._peticion("POST", "/predict", caso["cuerpo"])
            ok = st == 422 and d.get("error", {}).get("codigo") == caso["esperado"] and d["error"]["mensaje"]
            todo_ok = self.registrar(f"6. rechaza {caso['rotulo']} sin inventar datos", ok,
                                     f"codigo={d.get('error', {}).get('codigo')}")
        return todo_ok

    def probar_malformadas(self):
        todo_ok = True
        for caso in MALFORMADAS:
            st, _d, _ = self._peticion("POST", "/predict", caso["cuerpo"])
            ok = st == 422
            todo_ok = self.registrar(f"7. rechaza entrada malformada ({caso['rotulo']})", ok, f"HTTP {st}")
        return todo_ok

    def probar_modelo_identidad(self):
        st, d, _ = self._peticion("POST", "/predict", {"lat": -24.05, "lon": -54.25, "mes": "2025-06"})
        sha = d.get("modelo", {}).get("sha256")
        ok = st == 200 and sha == SHA_RF
        return self.registrar("8. identidad del modelo: SHA-256 == manifiesto canónico", ok, f"sha={str(sha)[:16]}...")

    def probar_tiempos(self):
        _st, _, ms = self._peticion("POST", "/predict", {"lat": -24.05, "lon": -54.25, "mes": "2025-06"})
        return self.registrar("9. tiempo de respuesta de /predict < 5 s", ms < 5000, f"{ms:.0f}ms")

    def latencias_resumen(self) -> dict:
        return {
            ruta: {"n": len(v), "media_ms": round(statistics.mean(v), 1), "max_ms": round(max(v), 1)}
            for ruta, v in sorted(self.latencias.items())
        }

    def ejecutar(self) -> bool:
        print(f"== Batería de pruebas FIREIA contra {self.base} ==\n")
        ok = True
        for paso in (
            self.probar_health, self.probar_raiz, self.probar_model_info, self.probar_metrics,
            self.probar_predict_validas, self.probar_invalidas, self.probar_malformadas,
            self.probar_modelo_identidad, self.probar_tiempos,
        ):
            ok = paso() and ok
        n_ok = sum(1 for r in self.resultados if r["ok"])
        print(f"\n== Resultado: {n_ok}/{len(self.resultados)} pruebas OK ==")
        print(json.dumps({"latencias": self.latencias_resumen()}, ensure_ascii=False, indent=1))
        return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="URL base del servicio (p. ej. https://fireia-api.onrender.com)")
    ap.add_argument("--salida", default=None, help="ruta del archivo JSON de evidencia")
    args = ap.parse_args()

    p = Pruebas(args.url)
    ok = p.ejecutar()
    if args.salida:
        evidencia = {
            "url": args.url,
            "fecha_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total": len(p.resultados),
            "ok": sum(1 for r in p.resultados if r["ok"]),
            "pruebas": p.resultados,
            "latencias": p.latencias_resumen(),
        }
        with open(args.salida, "w", encoding="utf-8") as fh:
            json.dump(evidencia, fh, ensure_ascii=False, indent=1)
        print(f"evidencia -> {args.salida}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
