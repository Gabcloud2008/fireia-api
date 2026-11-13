"""Verifica la consistencia entre la API autónoma y el módulo canónico.

El servicio de Render (api_render/) es autocontenido para poder vivir en un
repositorio Git público pequeño. Este script —que corre en el proyecto
canónico— garantiza que las constantes replicadas (FEATURES, versiones,
SHA-256) coinciden con la fuente única de verdad
(codigo/src/hotspots/analysis/prediccion.py + manifiesto_ia_v1.json).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]          # fireia_canonico/
sys.path.insert(0, str(RAIZ / "codigo" / "src"))
sys.path.insert(0, str(RAIZ / "api_render"))

from hotspots.analysis.prediccion import FEATURES as F_CANON
from hotspots.analysis.prediccion import IA_VERSION as V_CANON
from hotspots.analysis.prediccion import SEMILLA as S_CANON

import app as api


def main() -> int:
    fallos = []
    if list(api.FEATURES) != list(F_CANON):
        fallos.append("FEATURES difieren entre api_render/app.py y hotspots/analysis/prediccion.py")
    if api.IA_VERSION != V_CANON:
        fallos.append(f"IA_VERSION difiere: api={api.IA_VERSION} canonico={V_CANON}")
    if api.SEMILLA != S_CANON:
        fallos.append(f"SEMILLA difiere: api={api.SEMILLA} canonico={S_CANON}")

    man = json.loads((RAIZ / "data/PROCESADA/ia_v1/manifiesto_ia_v1.json").read_text(encoding="utf-8"))
    for nombre, info in man["artefactos"].items():
        esperado = api.ARTEFACTOS_ESPERADOS.get(nombre)
        if esperado and esperado != info["sha256"]:
            fallos.append(f"SHA-256 de {nombre} difiere del manifiesto canónico")
    faltan = set(man["artefactos"]) - set(api.ARTEFACTOS_ESPERADOS)
    if faltan:
        fallos.append(f"la API no verifica artefactos del manifiesto: {faltan}")

    if fallos:
        print("INCONSISTENCIAS:")
        for f in fallos:
            print(" -", f)
        return 1
    print("Consistencia verificada: FEATURES/IA_VERSION/SEMILLA/SHA-256 idénticos al canónico.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
