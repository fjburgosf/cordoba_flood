#!/usr/bin/env python
"""11_spatial_ems.py — paso E (D-17): superposición espacial inundación EMS vs extremo hidrometeorológico.

Rasteriza la extensión observada (observedEventA) sobre las grillas de CHIRPS y GLDAS y mide
qué fracción del área inundada cae en celdas con precipitación/escorrentía/humedad extremas
durante el evento (1–6 feb 2026). Para centralidad Remote Sensing (D-17).

Uso: .venv\\Scripts\\python.exe scripts/11_spatial_ems.py
"""
from __future__ import annotations
import sys, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.enums import MergeAlg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
EMS_SHP = ROOT / "data" / "raw" / "copernicus_ems" / "EMSR865_AOI01_DEL_PRODUCT_v2" / "EMSR865_AOI01_DEL_PRODUCT_observedEventA_v2.shp"
RES = ROOT / "results"
FIG = ROOT / "figures"
for d in (RES, FIG):
    d.mkdir(parents=True, exist_ok=True)

EVENT = ("2026-02-01", "2026-02-06")


def load_flood():
    g = gpd.read_file(EMS_SHP)  # EPSG:4326
    g_utm = g.to_crs(epsg=32618)  # UTM 18N (áreas en m²)
    g["area_km2"] = g_utm.geometry.area / 1e6
    return g


def sample_centroids(field, lat, lon, g):
    """field: (lat asc, lon asc); g: GeoDataFrame 4326. Valor del campo en cada polígono (vecino más cercano)."""
    cents = g.geometry.centroid
    clon = cents.x.values; clat = cents.y.values
    ilat = np.argmin(np.abs(lat[:, None] - clat[None, :]), axis=0)
    ilon = np.argmin(np.abs(lon[:, None] - clon[None, :]), axis=0)
    return field[ilat, ilon]


def weighted_median(values, weights):
    order = np.argsort(values)
    v = values[order]; w = weights[order]
    cdf = np.cumsum(w) / w.sum()
    return float(v[np.searchsorted(cdf, 0.5)])


def event_map(ds, var, agg="sum"):
    v = ds[var].sel(time=slice(EVENT[0], EVENT[1]))
    return (v.sum("time") if agg == "sum" else v.max("time")).values  # (lat asc, lon)


def overlap_metrics_poly(sampled, areas, field):
    """% del área inundada cuyo valor del evento supera cuantiles del mapa completo (por centroide)."""
    tot = float(areas.sum())
    fflat = np.asarray(field, float).ravel()
    fflat = fflat[np.isfinite(fflat)]  # ignora fill values (p.ej. -9999/NaN de GLDAS)
    out = {"total_flood_km2": round(tot, 1),
           "basin_median_field": round(float(np.median(fflat)), 2) if len(fflat) else None}
    if tot > 0:
        out["weighted_median_field_flood"] = round(weighted_median(sampled, areas), 2)
    for q in (0.5, 0.9, 0.95, 0.99):
        thr = float(np.quantile(fflat, q))
        frac = float(areas[sampled > thr].sum() / tot) if tot > 0 else np.nan
        out[f"pct_flood_above_p{int(q*100)}"] = round(100.0 * frac, 1)
    return out


def main():
    g = load_flood()
    areas = g["area_km2"].values
    report = {"generated": pd.Timestamp.now().isoformat(timespec="seconds"),
              "n_polygons": int(len(g)), "total_area_km2": round(float(areas.sum()), 1)}

    # ---- CHIRPS (precip evento)
    c = xr.open_dataset(PROC / "chirps_sinu_daily_1981_2026.nc")
    p = event_map(c, "precip", "sum")  # (lat asc, lon)
    lat_c, lon_c = c["lat"].values, c["lon"].values
    report["CHIRPS_precip_event"] = overlap_metrics_poly(sample_centroids(p, lat_c, lon_c, g), areas, p)

    # ---- GLDAS (escorrentía y humedad evento)
    gd = xr.open_dataset(PROC / "gldas_sinu_daily_2000_2026.nc")
    lat_g, lon_g = gd["lat"].values, gd["lon"].values
    qs = event_map(gd, "Qs_acc", "sum")
    sm = event_map(gd, "SoilMoi0_10cm_inst", "max")
    report["GLDAS_Qs_event"] = overlap_metrics_poly(sample_centroids(qs, lat_g, lon_g, g), areas, qs)
    report["GLDAS_SM_event"] = overlap_metrics_poly(sample_centroids(sm, lat_g, lon_g, g), areas, sm)

    fig_map(c, p, g, FIG / "ems_flood_overlay_precip.png")

    (RES / "spatial_ems.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_md(report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("[ok] 11_spatial_ems.py terminado")


def fig_map(c, p, g, path):
    lat, lon = c["lat"].values, c["lon"].values
    fig, ax = plt.subplots(figsize=(9, 10))
    pc = ax.pcolormesh(lon, lat, p[::-1, :], cmap="YlGnBu", shading="auto")
    g.plot(ax=ax, facecolor="none", edgecolor="red", linewidth=0.3, alpha=0.8)
    ax.set_title("Precipitación del evento (1–6 feb 2026, CHIRPS) y extensión inundada (EMS)")
    ax.set_xlabel("Longitud"); ax.set_ylabel("Latitud")
    fig.colorbar(pc, ax=ax, label="Precip acumulada (mm)")
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def write_md(r):
    L = ["# Overlay espacial — inundación EMS vs extremo hidrometeorológico (Fase 6, paso E)", "",
         f"Generado: {r['generated']} · {r['n_polygons']} polígonos · área inundada {r['total_area_km2']} km²", "",
         "## Fracción del área inundada en celdas extremas (muestreo por centroide, ponderado por área)",
         "| Variable | % inundado >p50 | >p90 | >p95 | >p99 | mediana inundado / mediana cuenca |",
         "|---|---|---|---|---|---|"]
    for key, title in [("CHIRPS_precip_event", "CHIRPS precip evento (mm)"),
                       ("GLDAS_Qs_event", "GLDAS Qs evento (mm)"),
                       ("GLDAS_SM_event", "GLDAS SM evento (kg/m²)")]:
        d = r[key]
        L.append(f"| {title} | {d['pct_flood_above_p50']}% | {d['pct_flood_above_p90']}% | "
                 f"{d['pct_flood_above_p95']}% | {d['pct_flood_above_p99']}% | "
                 f"{d['weighted_median_field_flood']} / {d['basin_median_field']} |")
    L += ["", "- `% inundado >p90` = % del área anegada (ponderada por área) cuyo valor del evento supera el percentil 90 del mapa.",
          "- `mediana inundado / cuenca` = mediana (ponderada por área) del campo en celdas inundadas vs mediana de toda la cuenca.",
          "", "## Figuras", "- `figures/ems_flood_overlay_precip.png`."]
    (RES / "spatial_ems_report.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()

