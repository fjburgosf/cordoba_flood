#!/usr/bin/env python
"""12_era5land_crosscheck.py — ERA5-Land humedad de suelo mensual 1950→ (cross-check GLDAS, D-17 Gate 6).

Descarga volumetric_soil_water_layer_1 (0-7cm) mensual vía CDS, media-en-cuenca, y posiciona
feb-2026 dentro del registro 1950→2026 (76 años) vs los 26 de GLDAS.

Uso: .venv\\Scripts\\python.exe scripts/12_era5land_crosscheck.py
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
from shapely.geometry import Point
import cdsapi, zipfile

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "interim" / "era5land"
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
for d in (OUT, RES):
    d.mkdir(parents=True, exist_ok=True)

BBOX = [9.6, -76.7, 6.9, -75.2]  # N, W, S, E
CHUNKS = [(1950, 1969), (1970, 1989), (1990, 2009), (2010, 2026)]


def extract(fn):
    """El CDS devuelve un zip que contiene el .nc; lo extrae a un archivo _raw.nc."""
    if fn.read_bytes()[:2] != b'PK':
        return fn
    out = fn.with_name(fn.stem + '_raw.nc')
    if out.exists():
        return out
    with zipfile.ZipFile(fn) as z:
        nc = [n for n in z.namelist() if n.endswith('.nc')][0]
        out.write_bytes(z.read(nc))
    return out


def download(y0, y1):
    fn = OUT / f"swvl1_{y0}_{y1}.nc"
    if fn.exists():
        print(f"[cache] {fn.name}", flush=True)
        return extract(fn)
    c = cdsapi.Client(retry_max=5, sleep_max=30, timeout=60)
    c.retrieve(
        'reanalysis-era5-land-monthly-means',
        {'product_type': 'monthly_averaged_reanalysis',
         'variable': 'volumetric_soil_water_layer_1',
         'year': [str(y) for y in range(y0, y1 + 1)],
         'month': [f'{m:02d}' for m in range(1, 13)],
         'time': '00:00',
         'area': BBOX,
         'format': 'netcdf'},
        str(fn))
    print(f"[ok] {fn.name}", flush=True)
    return extract(fn)


def basin_mean(da):
    gdf = gpd.read_file(PROC / "sinu_basin.gpkg")
    poly = gdf.geometry.unary_union
    lat = da.lat.values
    lon = da.lon.values
    lon2d, lat2d = np.meshgrid(lon, lat)
    inside = np.array([poly.contains(Point(x, y))
                       for x, y in zip(lon2d.ravel(), lat2d.ravel())]).reshape(lon2d.shape)
    w = np.cos(np.deg2rad(lat2d)) * inside
    if w.sum() == 0:
        w = np.cos(np.deg2rad(lat2d))  # fallback: todo el bbox
    w = w / w.sum()
    return (da * w[None, :, :]).sum(dim=("lat", "lon"))


def main():
    files = [download(y0, y1) for y0, y1 in CHUNKS]
    das = []
    for f in files:
        ds = xr.open_dataset(f)
        var = [v for v in ds.data_vars if v not in ("expver",)][0]
        das.append(ds[var].rename({'valid_time': 'time', 'latitude': 'lat', 'longitude': 'lon'}))
    da = xr.concat(das, dim="time").sortby("time")
    s = basin_mean(da).to_series()
    s.index = pd.to_datetime(s.index)
    s.name = "swvl1_m3m3"
    s.to_csv(PROC / "basin_monthly_ERA5LAND_swvl1.csv")

    # cross-check: feb-2026 vs registro
    base = s[s.index.year < 2026]
    feb2026 = float(s[(s.index.year == 2026) & (s.index.month == 2)].iloc[0])
    feb_hist = base[base.index.month == 2]
    pct_feb = round(float(100 * (feb_hist < feb2026).mean()), 2)
    pct_all = round(float(100 * (base < feb2026).mean()), 2)
    report = {
        "generated": pd.Timestamp.now().isoformat(timespec="seconds"),
        "n_months": int(len(s)), "first": str(s.index.min().date()), "last": str(s.index.max().date()),
        "unit": "m3/m3 (0-7cm)", "feb2026_swvl1": round(feb2026, 4),
        "feb_hist_mean": round(float(feb_hist.mean()), 4),
        "feb2026_percentile_vs_feb": pct_feb,
        "feb2026_percentile_vs_all": pct_all,
        "ratio_vs_feb_mean": round(feb2026 / float(feb_hist.mean()), 2),
    }
    (RES / "era5land_crosscheck.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("[ok] 12_era5land_crosscheck.py terminado")


if __name__ == "__main__":
    main()
