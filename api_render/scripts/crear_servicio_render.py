"""Crea y despliega el servicio de IA FIREIA en Render (un solo comando).

Requiere:
  - RENDER_API_KEY en el entorno (clave de la cuenta Render del propietario).
  - --repo URL de un repositorio Git PÚBLICO (GitHub/GitLab/Bitbucket) que
    contenga este paquete `api_render/` en su raíz (o ajustar --root-dir).
    Render despliega desde repos públicos sin conectar la cuenta Git
    (verificado con la API de Render el 14-nov-2026).

Qué hace:
  1. POST /v1/services (runtime python, plan FREE, región oregon, rootDir).
  2. Espera el primer deploy hasta estado `live` (consultando /deploys).
  3. Verifica la URL pública (GET /health) e imprime el resultado.

No sube secretos: el servicio no necesita credenciales (los artefactos son
públicos en Hugging Face y se verifican por SHA-256).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from urllib import error as ue
from urllib import request as ur

API = "https://api.render.com/v1"
OWNER_ID = os.environ.get("RENDER_OWNER_ID", "tea-dbrk3ioae00c73fsum5g")


def api_call(metodo: str, ruta: str, clave: str, cuerpo: dict | None = None, timeout: int = 60):
    req = ur.Request(
        API + ruta,
        data=json.dumps(cuerpo).encode() if cuerpo is not None else None,
        method=metodo,
        headers={"Authorization": f"Bearer {clave}", "Content-Type": "application/json",
                 "Accept": "application/json"},
    )
    try:
        with ur.urlopen(req, timeout=timeout) as r:
            crudo = r.read().decode()
            return r.status, (json.loads(crudo) if crudo else {})
    except ue.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="URL del repo Git público con api_render/ en la raíz")
    ap.add_argument("--branch", default="main")
    ap.add_argument("--nombre", default="fireia-api")
    ap.add_argument("--root-dir", default="api_render")
    ap.add_argument("--region", default="oregon")
    ap.add_argument("--plan", default="free")
    ap.add_argument("--esperar-min", type=int, default=25, help="minutos máximos de espera del deploy")
    args = ap.parse_args()

    clave = os.environ.get("RENDER_API_KEY")
    if not clave:
        print("ERROR: exporta RENDER_API_KEY antes de ejecutar (no se acepta la clave por argumento).")
        return 2

    cuerpo = {
        "type": "web_service",
        "name": args.nombre,
        "ownerId": OWNER_ID,
        "repo": args.repo,
        "branch": args.branch,
        "rootDir": args.root_dir,
        "autoDeploy": "yes",
        "envVars": [
            {"key": "PYTHON_VERSION", "value": "3.12.5"},
            {"key": "FIREIA_DATA_DIR", "value": "datos_ia"},
            {"key": "FIREIA_MODELO", "value": "random_forest"},
            {"key": "FIREIA_HF_REPO", "value": "p400-full/modis-viirs-hotspots-paraguay"},
        ],
        "serviceDetails": {
            "runtime": "python",
            "plan": args.plan,
            "region": args.region,
            "healthCheckPath": "/health",
            "envSpecificDetails": {
                "buildCommand": "pip install -r requirements.txt && python descargar_modelo.py --data-dir datos_ia",
                "startCommand": "uvicorn app:app --host 0.0.0.0 --port $PORT --workers 1",
            },
        },
    }

    st, resp = api_call("POST", "/services", clave, cuerpo)
    if st not in (200, 201):
        print(f"ERROR creando servicio (HTTP {st}): {json.dumps(resp, ensure_ascii=False)[:600]}")
        return 1
    servicio = resp.get("service", resp)
    sid = servicio.get("id")
    print(f"Servicio creado: {sid} ({servicio.get('name')})")
    print(f"URL esperada: https://{args.nombre}.onrender.com")

    # esperar el primer deploy
    print("Esperando el primer deploy (build: pip install + descarga y verificación del modelo)...")
    limite = time.time() + args.esperar_min * 60
    ultimo = None
    while time.time() < limite:
        st, deps = api_call("GET", f"/services/{sid}/deploys?limit=1", clave)
        if st == 200 and isinstance(deps, list) and deps:
            d = deps[0].get("deploy", deps[0])
            estado = d.get("status")
            if estado != ultimo:
                print(f"  deploy {d.get('id','?')[:8]}: {estado} ({time.strftime('%H:%M:%S')})")
                ultimo = estado
            if estado == "live":
                break
            if estado in ("deactivated", "build_failed", "canceled"):
                print(f"El deploy terminó en estado {estado}. Revisar logs en el panel de Render.")
                return 1
        time.sleep(20)
    else:
        print(f"Tiempo de espera agotado ({args.esperar_min} min). El deploy puede seguir en curso en el panel.")
        return 1

    url = f"https://{args.nombre}.onrender.com"
    print(f"\nDeploy LIVE. Verificando {url}/health ...")
    try:
        with ur.urlopen(url + "/health", timeout=90) as r:
            print(json.dumps(json.loads(r.read()), ensure_ascii=False, indent=1))
        print(f"\nOK — servicio de IA FIREIA desplegado y vivo en {url}")
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"El deploy está live pero /health no respondió todavía ({e}); "
              f"el modelo puede estar cargando. Reintentar en 1-2 min.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
