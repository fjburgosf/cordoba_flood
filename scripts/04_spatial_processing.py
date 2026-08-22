#!/usr/bin/env python
"""
04_spatial_processing.py — Construcción de la máscara de la cuenca del río Sinú.

Fase 1 / paso (c). Usa HydroBASINS (HydroSHEDS) para delinear el dominio primario
del estudio (D-07). Método: sembrar un punto sobre el cauce del Sinú (Montería),
leer su MAIN_BAS y agregar todos los sub-basins con el mismo MAIN_BAS (= toda la
red de drenaje del Sinú), disolver y guardar como GeoPackage + mapa de control.

No descarga datasets climáticos. Solo el vector HydroBASINS Sudamérica (~cientos MB
comprimido) si no está presente en data/raw.

Salidas:
  data/raw/hybas_sa_lev01-12_v1c.zip        (descarga única)
  data/processed/sinu_basin.gpkg            (polígono disuelto de la cuenca)
  data/processed/sinu_basin_subbasins.gpkg  (sub-basins nivel 12 que la componen)
  figures/sinu_basin_mask.png               (mapa de control)
  data/processed/sinu_basin_meta.json       (metadatos: área, bbox, n sub-basins)
"""
from __future__ import annotations
import json
import sys
import zipfile
from pathlib import Path

# Consola Windows (cp1252) no soporta UTF-8: forzar salida UTF-8
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely.geometry import Point

# ---------------------------------------------------------------- config
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
FIG = ROOT / "figures"
for d in (RAW, PROC, FIG):
    d.mkdir(parents=True, exist_ok=True)

HYBAS_URL = "https://data.hydrosheds.org/file/hydrobasins/standard/hybas_sa_lev01-12_v1c.zip"
HYBAS_ZIP = RAW / "hybas_sa_lev01-12_v1c.zip"
LEVEL = 12                      # sub-basins finos para delinear bien la cuenca
SEED_LON, SEED_LAT = -75.8919, 8.7527   # Montería, sobre el río Sinú
# bbox de prefiltro (Caribe colombiano) para no leer toda Sudamérica
BBOX = (-77.0, 7.0, -74.0, 10.0)        # minx, miny, maxx, maxy (lon/lat WGS84)
SINU_AREA_REF_KM2 = 13700               # referencia bibliográfica aprox. para sanity-check


def download_hybas() -> None:
    if HYBAS_ZIP.exists() and HYBAS_ZIP.stat().st_size > 1_000_000:
        print(f"[ok] HydroBASINS ya presente: {HYBAS_ZIP} "
              f"({HYBAS_ZIP.stat().st_size/1e6:.1f} MB)")
        return
    import requests
    print(f"[..] Descargando HydroBASINS Sudamérica desde {HYBAS_URL}")
    with requests.get(HYBAS_URL, stream=True, timeout=600) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(HYBAS_ZIP, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r     {done/1e6:6.1f}/{total/1e6:.1f} MB", end="")
    print(f"\n[ok] Descargado: {HYBAS_ZIP} ({HYBAS_ZIP.stat().st_size/1e6:.1f} MB)")


def shp_path_in_zip() -> str:
    with zipfile.ZipFile(HYBAS_ZIP) as z:
        names = [n for n in z.namelist()
                 if n.lower().endswith(".shp") and f"lev{LEVEL:02d}" in n.lower()]
    if not names:
        sys.exit(f"[error] No se encontró shapefile nivel {LEVEL} en {HYBAS_ZIP}")
    return names[0]


def main() -> None:
    download_hybas()
    shp = shp_path_in_zip()
    vsi = f"zip://{HYBAS_ZIP.as_posix()}!/{shp}"
    print(f"[..] Leyendo (bbox prefiltro {BBOX}): {shp}")
    gdf = gpd.read_file(vsi, bbox=BBOX)
    print(f"[ok] {len(gdf)} sub-basins nivel {LEVEL} en el bbox. Campos: {list(gdf.columns)}")

    # localizar sub-basin que contiene el punto semilla (Montería)
    seed = Point(SEED_LON, SEED_LAT)
    hit = gdf[gdf.contains(seed)]
    if hit.empty:
        hit = gdf.iloc[[gdf.distance(seed).idxmin()]]
        print("[warn] semilla no contenida; usando sub-basin más cercano")
    main_bas = int(hit.iloc[0]["MAIN_BAS"])
    print(f"[ok] MAIN_BAS del Sinú (desde Montería) = {main_bas}")

    # toda la red de drenaje del Sinú = mismo MAIN_BAS
    sinu = gdf[gdf["MAIN_BAS"] == main_bas].copy()
    print(f"[ok] {len(sinu)} sub-basins comparten MAIN_BAS (cuenca del Sinú)")

    # disolver a un polígono limpio (evitar colisión con campos HydroBASINS en GPKG)
    dissolved_geom = sinu.dissolve().geometry.iloc[0]
    basin = gpd.GeoDataFrame(
        {
            "name": ["Sinu River Basin"],
            "main_bas": [main_bas],
            "source": ["HydroBASINS v1c lev12 (HydroSHEDS)"],
        },
        geometry=[dissolved_geom],
        crs=sinu.crs,
    )

    # área en km2 (proyección equiárea global EASE-Grid 2.0, EPSG:6933)
    area_km2 = float(basin.to_crs(6933).area.iloc[0] / 1e6)
    minx, miny, maxx, maxy = [float(v) for v in basin.total_bounds]
    print(f"[ok] Área cuenca ≈ {area_km2:,.0f} km²  (ref. ~{SINU_AREA_REF_KM2:,} km²)")
    print(f"[ok] BBox cuenca: lon[{minx:.3f},{maxx:.3f}] lat[{miny:.3f},{maxy:.3f}]")

    # guardar
    out_basin = PROC / "sinu_basin.gpkg"
    out_sub = PROC / "sinu_basin_subbasins.gpkg"
    basin.to_file(out_basin, driver="GPKG")
    sinu.to_file(out_sub, driver="GPKG")
    meta = {
        "name": "Sinu River Basin",
        "source": "HydroBASINS v1c level 12 (HydroSHEDS)",
        "main_bas": main_bas,
        "n_subbasins": int(len(sinu)),
        "area_km2": round(area_km2, 1),
        "area_ref_km2": SINU_AREA_REF_KM2,
        "bbox_lonlat": [minx, miny, maxx, maxy],
        "seed_point_lonlat": [SEED_LON, SEED_LAT],
        "crs": str(basin.crs),
    }
    (PROC / "sinu_basin_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[ok] Guardado: {out_basin}, {out_sub}, sinu_basin_meta.json")

    # mapa de control
    fig, ax = plt.subplots(figsize=(7, 8))
    gdf.boundary.plot(ax=ax, color="0.85", linewidth=0.3)
    sinu.plot(ax=ax, color="#4a90d9", edgecolor="white", linewidth=0.2, alpha=0.7)
    basin.boundary.plot(ax=ax, color="#08306b", linewidth=1.5)
    ax.plot(SEED_LON, SEED_LAT, "r*", ms=14, label="Montería (seed)")
    ax.set_xlim(BBOX[0], BBOX[2]); ax.set_ylim(BBOX[1], BBOX[3])
    ax.set_xlabel("Lon"); ax.set_ylabel("Lat")
    ax.set_title(f"Sinú River basin — HydroBASINS lev{LEVEL}\n"
                 f"{len(sinu)} sub-basins · ≈{area_km2:,.0f} km²")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "sinu_basin_mask.png", dpi=300)
    print(f"[ok] Figura: {FIG / 'sinu_basin_mask.png'}")

    # sanity-check de área
    if not (0.4 * SINU_AREA_REF_KM2 < area_km2 < 2.0 * SINU_AREA_REF_KM2):
        print(f"[WARN] Área fuera del rango esperado — revisar semilla/nivel/MAIN_BAS.")
    else:
        print("[ok] Área dentro del rango esperado. Máscara construida.")


if __name__ == "__main__":
    main()
