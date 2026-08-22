#!/usr/bin/env python
"""
07_chirps_acquire.py — CHIRPS v2.0 diario (precipitación) 1981→hoy, cuenca del Sinú.

Fuente: ClimateSERV (NASA SERVIR) API — subsetting servidor-lado por bbox+tiempo con
operación "NetCDF" (operationtype=7). Devuelve un ZIP con un .nc del subset (~1–2 MB/año).
Misma filosofía que el 05/06: solo se baja el bbox, nunca el granule global.

Diseño:
  - Bucle por AÑO → checkpoint data/interim/historical/chirps/daily_YYYY.nc (reanudable).
  - Solo persiste el DIARIO (CHIRPS ya es diario nativo). Sin credenciales.
  - --combine → netCDF diario final + serie media-en-cuenca + QC + percentil del evento.
  - --cleanup → borra checkpoints anuales tras el combine.

Uso:
  python scripts/07_chirps_acquire.py --years 2025          # smoke 1 año
  python scripts/07_chirps_acquire.py                       # completo 1981→hoy
  python scripts/07_chirps_acquire.py --combine --cleanup
"""
from __future__ import annotations
import sys, json, warnings, importlib.util, time, io, zipfile
import datetime as dt
from pathlib import Path
warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

import numpy as np, pandas as pd, xarray as xr, requests
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

# helpers validados del 05 (BBOX, BASIN, basin_weighted_mean)
_P05 = Path(__file__).resolve().parent / "05_pilot_acquire_qc.py"
_spec = importlib.util.spec_from_file_location("pilot05", _P05)
p = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(p)

ROOT = p.ROOT; PROC = p.PROC; BASIN = p.BASIN; BBOX = p.BBOX
HIST = ROOT / "data" / "interim" / "historical"
RES = ROOT / "results"; FIG = ROOT / "figures"
for d in (HIST, RES, FIG): d.mkdir(parents=True, exist_ok=True)

API = "https://climateserv.servirglobal.net/api/"
DATATYPE = 0        # CHIRPS 2.0
OPERATION = 7       # NetCDF

START = "1981-01-01"
TODAY = dt.date.today()
EVENT = ("2026-02-01", "2026-02-06")


def geometry_json():
    minlon, minlat, maxlon, maxlat = BBOX
    g = {"type": "Polygon", "coordinates": [[
        [minlon, minlat], [maxlon, minlat], [maxlon, maxlat], [minlon, maxlat], [minlon, minlat]
    ]], "properties": {}}
    return json.dumps(g).replace(" ", "")


def fetch_year(y, retries=4):
    s = dt.date(y, 1, 1)
    e = min(dt.date(y, 12, 31), TODAY)
    params = {
        "datatype": DATATYPE,
        "seasonal_ensemble": "",
        "seasonal_variable": "",
        "begintime": s.strftime("%m/%d/%Y"),
        "endtime": e.strftime("%m/%d/%Y"),
        "intervaltype": 0,
        "operationtype": OPERATION,
        "dateType_Category": "default",
        "isZip_CurrentDataType": False,
        "geometry": geometry_json(),
    }
    for attempt in range(retries):
        try:
            r = requests.post(API + "submitDataRequest/", params=params, timeout=120)
            r.raise_for_status()
            rid = json.loads(r.text)[0]
            for _ in range(300):
                pr = requests.get(API + "getDataRequestProgress/", params={"id": rid}, timeout=60)
                pr.raise_for_status()
                val = json.loads(pr.text)[0]
                if val == 100:
                    break
                if val == -1:
                    raise RuntimeError("ClimateSERV devolvió -1 (error) en el request")
                time.sleep(3)
            else:
                raise TimeoutError("timeout esperando el request ClimateSERV")
            fr = requests.get(API + f"getFileForJobID/?id={rid}", timeout=300)
            fr.raise_for_status()
            z = zipfile.ZipFile(io.BytesIO(fr.content))
            name = [n for n in z.namelist() if n.endswith(".nc")][0]
            ds = xr.open_dataset(io.BytesIO(z.read(name)))
            da = ds["precipitation_amount"].rename({"latitude": "lat", "longitude": "lon"})
            da.name = "precip"
            da.attrs = {"units": "mm/day", "long_name": "CHIRPS v2.0 daily precipitation",
                        "source": "CHIRPS 2.0 (UCSB/CHC, Funk et al. 2015) vía ClimateSERV (NASA SERVIR)"}
            return da
        except Exception as ex:
            if attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
            else:
                raise ex


def acquire(years, force=False):
    for y in years:
        out = HIST / "chirps" / f"daily_{y}.nc"
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists() and not force:
            print(f"[skip] CHIRPS {y} (checkpoint)", flush=True); continue
        t0 = time.time()
        da = fetch_year(y)
        da.to_netcdf(out)
        print(f"[ok] CHIRPS {y}: {da.sizes['time']} días  "
              f"(~{out.stat().st_size/1e6:.2f} MB, {time.time()-t0:.0f}s)", flush=True)


def combine():
    files = sorted((HIST / "chirps").glob("daily_*.nc"))
    if not files:
        print("[!] sin checkpoints chirps"); return None
    ds = xr.concat([xr.open_dataset(f) for f in files], dim="time").sortby("time")
    out = PROC / "chirps_sinu_daily_1981_2026.nc"
    ds.to_netcdf(out)
    print(f"[ok] combine → {out.name} ({out.stat().st_size/1e6:.2f} MB)")
    return out


def qc_and_report(path):
    ds = xr.open_dataset(path)
    s = p.basin_weighted_mean(ds["precip"]).to_series()
    peak = s.loc[EVENT[0]:EVENT[1]].max()
    pct = round(100 * (s < peak).mean(), 2)
    qc = {
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "bbox": BBOX,
        "window": [str(s.index.min().date()), str(s.index.max().date())],
        "days": int(s.size), "nan_days": int(s.isna().sum()),
        "min_mm_day": round(float(s.min()), 2),
        "mean_mm_day": round(float(s.mean()), 2),
        "max_mm_day": round(float(s.max()), 2),
        "event_peak_mm_day": round(float(peak), 2),
        "event_peak_percentile": pct,
    }
    (RES / "chirps_qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    s.to_csv(PROC / "basin_daily_CHIRPS.csv", header=["precip_mm_day"])
    lines = [
        "# CHIRPS QC — precipitación diaria media-en-cuenca (Sinú)",
        "",
        f"Generado: {qc['generated']}  ·  Ventana: {qc['window'][0]} → {qc['window'][1]}  ·  BBox: {BBOX}",
        "",
        "| Métrica | Valor |", "|---|---|",
        f"| Días | {qc['days']} |",
        f"| NaN días | {qc['nan_days']} |",
        f"| min (mm/d) | {qc['min_mm_day']} |",
        f"| mean (mm/d) | {qc['mean_mm_day']} |",
        f"| max (mm/d) | {qc['max_mm_day']} |",
        f"| Pico evento (1–6 feb 2026, mm/d) | {qc['event_peak_mm_day']} |",
        f"| **Percentil del evento** | **{qc['event_peak_percentile']}%** |",
        "",
        "- `event_peak_percentile` = % de días históricos por debajo del pico diario del evento.",
        "- 100% ⇒ el pico del evento supera TODO el registro diario histórico.",
    ]
    (RES / "chirps_qc_report.md").write_text("\n".join(lines), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(s.index, s.values, lw=0.4, color="#08519c")
    ax.axvspan(pd.Timestamp(EVENT[0]), pd.Timestamp(EVENT[1]), color="yellow", alpha=0.3)
    ax.set_ylabel("Precip (mm/d)"); ax.set_title("CHIRPS — basin mean, Sinú (1981–)")
    fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(FIG / "chirps_basin_timeseries.png", dpi=300)
    print("[ok] QC + reporte + figura escritos")
    print(json.dumps(qc, indent=2))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default=None, help="años, p.ej. 2025 o 2000,2001")
    ap.add_argument("--force", action="store_true", help="re-descargar años con checkpoint")
    ap.add_argument("--combine", action="store_true")
    ap.add_argument("--cleanup", action="store_true")
    args = ap.parse_args()

    if args.combine:
        out = combine()
        if out is not None:
            qc_and_report(out)
        if args.cleanup:
            for f in (HIST / "chirps").glob("daily_*.nc"):
                f.unlink()
            print("[cleanup] chirps: checkpoints borrados")
    else:
        years = [int(y) for y in args.years.split(",")] if args.years \
            else list(range(1981, TODAY.year + 1))
        acquire(years, args.force)


if __name__ == "__main__":
    main()
