"""Descarga los artefactos IA de FIREIA desde Hugging Face (build de Render).

Se ejecuta como parte del buildCommand (ver render.yaml). Los artefactos viven
en el repositorio público del proyecto; no se requieren tokens. La integridad
se verifica contra los SHA-256 del manifiesto canónico `manifiesto_ia_v1.json`
(pubilcado en el mismo repositorio): si algo no coincide, el build FALLA.

Origen por defecto: el espejo completo `p400-full/...` (contiene data/ completa).
Uso:  python descargar_modelo.py [--data-dir datos_ia] [--repo p400-full/...]
"""
from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/datasets/{repo}/resolve/main/data/PROCESADA/ia_v1/{nombre}"

# SHA-256 canónicos (manifiesto_ia_v1.json, Misión E; también embebidos en app.py)
SHA256 = {
    "metricas_ia.json": "ce2e5c940123650c524dbe85463f8e6f595579ea0528445db07b70e78a514c6c",
    "dataset_riesgo_celda_mes.parquet": "37c4c012328407b22550f341c6032ac8e8a751199a24b5b82bf2299a67c3a745",
    "modelo_random_forest.joblib": "2361c743bd51c72bc4b09c13bbcc2a83e3e70bc905bdb6f68efa69881e93c1c2",
    "modelo_gradient_boosting.joblib": "630cc704ed15ab1c7c2932e5b6fb2221361cd2e6a543c5402a9ed35cd4749b76",
}


def sha256_archivo(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def descargar(repo: str, nombre: str, destino: Path, reintentos: int = 4) -> None:
    url = BASE.format(repo=repo, nombre=nombre)
    for intento in range(1, reintentos + 1):
        try:
            print(f"[descargar_modelo] {nombre} (intento {intento}/{reintentos}) <- {url}", flush=True)
            with urllib.request.urlopen(url, timeout=120) as r, destino.open("wb") as out:
                while True:
                    bloque = r.read(1 << 20)
                    if not bloque:
                        break
                    out.write(bloque)
            return
        except Exception as e:
            if intento == reintentos:
                raise SystemExit(f"[descargar_modelo] FALLO descargando {nombre}: {e}") from e
            time_espera = 5 * intento
            print(f"[descargar_modelo] error ({e}); reintento en {time_espera}s", flush=True)
            import time

            time.sleep(time_espera)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="datos_ia")
    ap.add_argument("--repo", default="p400-full/modis-viirs-hotspots-paraguay")
    args = ap.parse_args()

    destino_dir = Path(args.data_dir)
    destino_dir.mkdir(parents=True, exist_ok=True)

    for nombre, esperado in SHA256.items():
        destino = destino_dir / nombre
        if destino.exists() and sha256_archivo(destino) == esperado:
            print(f"[descargar_modelo] {nombre}: ya presente y verificado", flush=True)
            continue
        descargar(args.repo, nombre, destino)
        real = sha256_archivo(destino)
        if real != esperado:
            raise SystemExit(
                f"[descargar_modelo] INTEGRIDAD FALLIDA para {nombre}: "
                f"esperado {esperado}, real {real}. Build abortado."
            )
        print(f"[descargar_modelo] {nombre}: SHA-256 verificado ({real[:12]}...)", flush=True)

    # guarda el manifiesto tal cual para trazabilidad en producción
    man = destino_dir / "manifiesto_ia_v1.json"
    if not man.exists():
        descargar(args.repo, "manifiesto_ia_v1.json", man)
    print("[descargar_modelo] todos los artefactos están verificados", flush=True)


if __name__ == "__main__":
    main()
