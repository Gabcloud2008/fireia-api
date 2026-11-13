# FIREIA — Servicio de IA (API REST) · Paquete de despliegue en Render

Este repositorio contiene el paquete `api_render/`: el servicio REST (FastAPI)
que expone el modelo predictivo real `ia_riesgo_fuego_v1` del proyecto FIREIA
(Paraguay Fire Intelligence — NASA Space Apps Challenge 2026, reto 1.9).

**Tarea del modelo (definición exacta):** estimar la probabilidad de que una
celda registre **al menos una detección VIIRS en el mes siguiente (t+1)**,
dadas sus características climáticas, de historia de fuego y de cobertura en
`t` y `t-1`. Una detección satelital **no** equivale a un incendio confirmado:
la probabilidad devuelta es de **detección VIIRS**, no de incendio confirmado.

## Endpoints

| Método | Ruta | Función |
| --- | --- | --- |
| GET | `/health` | Estado del servicio (si el modelo está cargado) |
| GET | `/` | Información del servicio y versión |
| GET | `/model/info` | Identidad del modelo: versión, SHA-256, métricas, alcance y limitaciones |
| POST | `/predict` | Inferencia real con validación de entrada (probabilidad de detección VIIRS t+1) |
| GET | `/metrics` | Métricas de uso del servicio (no del modelo) |

## Despliegue en Render (plan free)

Opción Blueprint: Dashboard → New → Blueprint (usa `api_render/render.yaml`).

Opción de un comando (requiere `RENDER_API_KEY` en el entorno y este repo público):

```bash
python api_render/scripts/crear_servicio_render.py --repo https://github.com/Gabcloud2008/fireia-api.git
```

Pruebas de aceptación contra la URL pública:

```bash
python api_render/scripts/pruebas_produccion.py --url https://<servicio>.onrender.com --salida evidencia.json
```

## Origen e integridad de los artefactos

El modelo y sus artefactos **no viven en este repositorio** (límite de 100 MB
de GitHub). Se descargan durante el build desde el repositorio público de
Hugging Face del proyecto (`p400-full/modis-viirs-hotspots-paraguay`, ruta
`data/PROCESADA/ia_v1/`) y se verifican contra los SHA-256 del manifiesto
canónico `manifiesto_ia_v1.json`. Si la integridad no coincide, el build falla.

- Repositorio de datos canónico: https://huggingface.co/datasets/p400/modis-viirs-hotspots-paraguay
- Espejo completo: https://huggingface.co/datasets/p400-full/modis-viirs-hotspots-paraguay
- El servicio no requiere secretos ni tokens.

## Estructura

```
api_render/
├── app.py                    # Servicio FastAPI (carga, verificación e inferencia)
├── render.yaml               # Blueprint de Render (plan free, región oregon)
├── requirements.txt          # Dependencias fijadas (scikit-learn 1.5.2 del manifiesto)
├── descargar_modelo.py       # Descarga + verificación SHA-256 en el build
├── .env.example              # Variables de entorno documentadas (sin secretos)
├── scripts/                  # Despliegue, pruebas de producción y consistencia
└── tests/                    # Pruebas unitarias del servicio
```

## Licencia

MIT — Copyright (c) 2026 Equipo P400 (ver `LICENSE` en el repositorio canónico
del proyecto y en los repositorios de datos de Hugging Face).
