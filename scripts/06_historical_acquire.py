#!/usr/bin/env python
"""
06_historical_acquire.py — Adquisición HISTÓRICA diaria del núcleo (IMERG Late + GLDAS-2.1 + MERRA-2)
recortada a la cuenca del Sinú, agregada a diario, con checkpoints anuales reanudables.

Reutiliza (vía importlib) los helpers ya validados en 05_pilot_acquire_qc.py:
  session() Bearer+IPv4, fetch_subset_parallel (DAP2, cap 4 MB, warm-up),
  _dap2_ce/_idx/_host_url/_cloud_opendap_url, GRID, basin_weighted_mean, open_granule,
  GLDAS_INST/GLDAS_ACC/MERRA_VARS, BBOX/BASIN de la cuenca.

Diseño (D-12 / plan Fase 5 histórico):
  - Bucle por AÑO → checkpoint data/interim/historical/<prod>/daily_YYYY.nc (reanudable).
  - El crudo 3-horario (GLDAS) / horario (MERRA-2) NUNCA se escribe en disco: se agrega
    en memoria a diario y se descarta. Solo persiste el DIARIO.
  - IMERG es diario nativo; se guarda el subset bbox diario (streaming, nada persiste).
  - TODO dentro de data/ del proyecto. Tras el combine final se borran los checkpoints (--cleanup).

Uso:
  python scripts/06_historical_acquire.py --products gldas --years 2025   # smoke 1 año
  python scripts/06_historical_acquire.py --products imerg merra           # baratos, rango completo
  python scripts/06_historical_acquire.py --products gldas                 # completo, reanudable
  python scripts/06_historical_acquire.py --combine --cleanup --products all
"""
from __future__ import annotations
import sys, json, warnings, time, importlib.util
import datetime as dt
from pathlib import Path
warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

import numpy as np, pandas as pd, xarray as xr
import earthaccess
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

# --- cargar helpers de 05 (sin duplicar código validado) ---
_P05 = Path(__file__).resolve().parent / "05_pilot_acquire_qc.py"
_spec = importlib.util.spec_from_file_location("pilot05", _P05)
p = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p)

ROOT = p.ROOT
PROC = p.PROC
BASIN = p.BASIN
BBOX = p.BBOX
HIST = ROOT / "data" / "interim" / "historical"
RES = ROOT / "results"; FIG = ROOT / "figures"
for d in (HIST, PROC, RES, FIG): d.mkdir(parents=True, exist_ok=True)

# --- reintento con backoff para 5xx transitorios (p.ej. 503 de hydro1 bajo carga).
# Monkey-patch de p.fetch_subset sin tocar 05 (validado): fetch_subset_parallel usa el global.
_orig_fetch = p.fetch_subset
def _fetch_retry(url, retries=4, backoff=2.0):
    last = None
    for attempt in range(retries):
        try:
            return _orig_fetch(url)
        except ValueError:
            raise  # error de guardia de volumen: no reintentar
        except Exception as e:
            last = e
            if attempt < retries - 1:
                time.sleep(backoff * (attempt + 1))
    raise last
p.fetch_subset = _fetch_retry

# Rango temporal por producto (fin de datos verificado, D-08)
RANGES = {
    "gldas": ("2000-01-01", "2026-05-31"),
    "merra": ("1980-01-01", "2026-07-01"),
    "imerg": ("2000-06-01", "2026-08-20"),
}
EVENT = ("2026-02-01", "2026-02-06")


def year_bounds(y, start, end):
    s = max(f"{y:04d}-01-01", start)
    e = min(f"{y:04d}-12-31", end)
    return s, e


def years_in(start, end):
    return list(range(int(start[:4]), int(end[:4]) + 1))


# ----------------------------------------------------------- GLDAS 2.1 (3-horario -> diario)
def do_gldas_year(y, s, e, limit, force=False):
    d = HIST / "gldas"; d.mkdir(parents=True, exist_ok=True)
    ckpt = d / f"daily_{y}.nc"
    if ckpt.exists() and not force:
        print(f"[cache] GLDAS {y}"); return
    r = earthaccess.search_data(short_name="GLDAS_NOAH025_3H", version="2.1",
                                temporal=(s, e), bounding_box=BBOX)
    if limit: r = r[:limit]
    print(f"[GLDAS] {y}: {len(r)} granules (3-horario)", flush=True)
    if not r: return
    gr = p.GRID["GLDAS"]
    ilat = p._idx(gr["lat0"], gr["dlat"], BBOX[1], BBOX[3], gr["nlat"])
    ilon = p._idx(gr["lon0"], gr["dlon"], BBOX[0], BBOX[2], gr["nlon"])
    ce = p._dap2_ce(p.GLDAS_INST + p.GLDAS_ACC, "[0:1:0]", ilat, ilon)
    def url_builder(g):
        return p._host_url(g.data_links(access="external")[0], gr["host"]) + ".nc4?" + ce
    recs = p.fetch_subset_parallel(r, url_builder, f"GLDAS {y}")
    full = xr.concat(recs, dim="time").sortby("time")
    daily = xr.merge([full[p.GLDAS_INST].resample(time="1D").mean(),
                      full[p.GLDAS_ACC].resample(time="1D").sum()])
    daily.to_netcdf(ckpt)
    print(f"[GLDAS] {y}: checkpoint {len(recs)}/{len(r)} granules -> {ckpt.name}", flush=True)


# ----------------------------------------------------------- MERRA-2 (horario -> diario)
def do_merra_year(y, s, e, limit, force=False):
    d = HIST / "merra"; d.mkdir(parents=True, exist_ok=True)
    ckpt = d / f"daily_{y}.nc"
    if ckpt.exists() and not force:
        print(f"[cache] MERRA {y}"); return
    r = earthaccess.search_data(short_name="M2T1NXSLV", version="5.12.4",
                                temporal=(s, e), bounding_box=BBOX)
    if limit: r = r[:limit]
    print(f"[MERRA] {y}: {len(r)} granules (horario)", flush=True)
    if not r: return
    gr = p.GRID["MERRA"]
    ilat = p._idx(gr["lat0"], gr["dlat"], BBOX[1], BBOX[3], gr["nlat"])
    ilon = p._idx(gr["lon0"], gr["dlon"], BBOX[0], BBOX[2], gr["nlon"])
    ce = p._dap2_ce(p.MERRA_VARS, "[0:1:23]", ilat, ilon)
    def url_builder(g):
        return p._cloud_opendap_url(g) + ".nc4?" + ce
    recs = p.fetch_subset_parallel(r, url_builder, f"MERRA {y}")
    full = xr.concat(recs, dim="time").sortby("time")
    daily = full.resample(time="1D").mean()
    daily.to_netcdf(ckpt)
    print(f"[MERRA] {y}: checkpoint {len(recs)}/{len(r)} granules -> {ckpt.name}", flush=True)


# ----------------------------------------------------------- IMERG Late (diario nativo)
def _imerg_one(g):
    ds = p.open_granule(g)
    pp = ds["precipitation"].sel(lon=slice(BBOX[0], BBOX[2]), lat=slice(BBOX[1], BBOX[3]))
    return pp.load()


def do_imerg_year(y, s, e, limit, force=False):
    d = HIST / "imerg"; d.mkdir(parents=True, exist_ok=True)
    ckpt = d / f"daily_{y}.nc"
    if ckpt.exists() and not force:
        print(f"[cache] IMERG {y}"); return
    r = earthaccess.search_data(short_name="GPM_3IMERGDL", version="07",
                                temporal=(s, e), bounding_box=BBOX)
    if limit: r = r[:limit]
    print(f"[IMERG] {y}: {len(r)} granules (diario)", flush=True)
    if not r: return
    out = [None] * len(r)
    try:
        out[0] = _imerg_one(r[0])   # warm-up síncrono (auth/redirect)
    except Exception as e:
        print(f"   [skip] IMERG {y} #0: {e}")
    def job(i, g):
        return i, _imerg_one(g)
    with ThreadPoolExecutor(max_workers=p.NWORKERS) as ex:
        futs = {ex.submit(job, i, g): i for i, g in enumerate(r[1:], start=1)}
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                _, pp = fut.result(); out[i] = pp
            except Exception as e:
                print(f"   [skip] IMERG {y} #{i}: {e}")
    days = [x for x in out if x is not None]
    da = xr.concat(days, dim="time").sortby("time")
    da.attrs["units"] = "mm/day"
    da.to_netcdf(ckpt)
    print(f"[IMERG] {y}: checkpoint {len(days)}/{len(r)} granules -> {ckpt.name}", flush=True)


ACQUIRERS = {"gldas": do_gldas_year, "merra": do_merra_year, "imerg": do_imerg_year}


def acquire(products, years, limit, force=False):
    t0 = time.time()
    for prod in products:
        start, end = RANGES[prod]
        for y in years:
            s, e = year_bounds(y, start, end)
            if s > e:
                continue
            ACQUIRERS[prod](y, s, e, limit, force)
    print(f"[ok] adquisición terminada en {(time.time()-t0)/60:.1f} min", flush=True)


# ----------------------------------------------------------- combine + QC + posicionamiento evento
def ev_pct(s: pd.Series):
    """Percentil del pico del evento (1-6 feb 2026) dentro del registro diario histórico."""
    ev = s.loc[EVENT[0]:EVENT[1]]
    base = s.dropna()
    if ev.empty or base.empty:
        return None
    peak = float(ev.max())
    return round(float(100 * (base < peak).mean()), 1)


def qc_entry(s: pd.Series, unit):
    n = len(s); nan = int(s.isna().sum())
    return {"unit": unit, "days": n, "nan_days": nan,
            "coverage_pct": round(100 * (1 - nan / n), 1) if n else None,
            "min": round(float(np.nanmin(s)), 3), "max": round(float(np.nanmax(s)), 3),
            "mean": round(float(np.nanmean(s)), 3),
            "first": str(s.index.min().date()), "last": str(s.index.max().date()),
            "event_peak_percentile": ev_pct(s)}


def combine(product):
    d = HIST / product
    files = sorted(d.glob("daily_*.nc"))
    if not files:
        print(f"[combine] {product}: sin checkpoints"); return None
    rng = RANGES[product]
    out = PROC / f"{product}_sinu_daily_{rng[0][:4]}_{rng[1][:4]}.nc"
    if product == "imerg":
        da = xr.concat([xr.open_dataarray(f) for f in files], dim="time").sortby("time")
        da.attrs.setdefault("units", "mm/day")
        da.to_netcdf(out)
    else:
        ds = xr.concat([xr.open_dataset(f) for f in files], dim="time").sortby("time")
        ds.to_netcdf(out)
    print(f"[combine] {product}: {len(files)} años -> {out.name}")
    return out


def qc_and_basin(product, path, qc, series):
    """Serie media-en-cuenca diaria + QC por variable."""
    if product == "imerg":
        da = xr.open_dataarray(path)
        s = p.basin_weighted_mean(da).to_series()
        s.to_csv(PROC / "basin_daily_IMERG.csv", header=["precip_mm_day"])
        qc["products"]["IMERG_precip_mm_day"] = qc_entry(s, "mm/day")
        series["IMERG_precip_mm_day"] = s
    elif product == "gldas":
        ds = xr.open_dataset(path)
        m = [("SoilMoi0_10cm_inst", "GLDAS_SoilMoi0_10cm_kg_m2", "kg/m2"),
             ("SoilMoi10_40cm_inst", "GLDAS_SoilMoi10_40cm_kg_m2", "kg/m2"),
             ("Qs_acc", "GLDAS_Qs_runoff_mm_day", "mm/day(=kg/m2/d)"),
             ("Qsb_acc", "GLDAS_Qsb_runoff_mm_day", "mm/day(=kg/m2/d)")]
        cols = {}
        for var, name, unit in m:
            s = p.basin_weighted_mean(ds[var]).to_series()
            cols[name] = s; qc["products"][name] = qc_entry(s, unit); series[name] = s
        pd.DataFrame(cols).to_csv(PROC / "basin_daily_GLDAS.csv")
    elif product == "merra":
        ds = xr.open_dataset(path)
        st2 = (p.basin_weighted_mean(ds["T2M"]) - 273.15).to_series()
        stqv = p.basin_weighted_mean(ds["TQV"]).to_series()
        pd.DataFrame({"T2M_degC": st2, "TQV_kg_m2": stqv}).to_csv(PROC / "basin_daily_MERRA2.csv")
        qc["products"]["MERRA2_T2M_degC"] = qc_entry(st2, "degC"); series["MERRA2_T2M_degC"] = st2
        qc["products"]["MERRA2_TQV_kg_m2"] = qc_entry(stqv, "kg/m2"); series["MERRA2_TQV_kg_m2"] = stqv
    return series


def write_report(qc, series):
    (RES / "historical_qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    lines = ["# Historical QC Report — núcleo (IMERG/GLDAS/MERRA-2) diario",
             "", f"Generado: {qc['generated']}  ·  BBox: {qc['bbox']}", "",
             "| Variable | Unidad | Días | Cobertura | NaN | min | mean | max | 1er | último | Pct evento |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in qc["products"].items():
        pct = v.get("event_peak_percentile")
        lines.append(f"| {k} | {v['unit']} | {v['days']} | {v['coverage_pct']}% | {v['nan_days']} "
                     f"| {v['min']} | {v['mean']} | {v['max']} | {v['first']} | {v['last']} | {pct} |")
    lines += ["", "## Posicionamiento del evento (1–6 feb 2026)",
              "- `event_peak_percentile` = % de días históricos por debajo del pico diario del evento.",
              "- 100% ⇒ el pico del evento supera TODO el registro diario histórico."]
    (RES / "historical_qc_report.md").write_text("\n".join(lines), encoding="utf-8")

    keys = [k for k in ("IMERG_precip_mm_day", "GLDAS_Qs_runoff_mm_day",
                        "GLDAS_SoilMoi0_10cm_kg_m2", "MERRA2_T2M_degC") if k in series]
    if keys:
        fig, ax = plt.subplots(len(keys), 1, figsize=(11, 2.6 * len(keys)), sharex=True)
        if len(keys) == 1: ax = [ax]
        for a, k in zip(ax, keys):
            a.plot(series[k].index, series[k].values, lw=0.4, color="#08519c")
            a.set_ylabel(k, fontsize=7)
            a.axvspan(pd.Timestamp(EVENT[0]), pd.Timestamp(EVENT[1]), color="yellow", alpha=0.25)
        fig.suptitle("Histórico diario media-en-cuenca (Sinú) — evento 1–6 feb 2026 sombreado")
        fig.autofmt_xdate(); fig.tight_layout()
        fig.savefig(FIG / "historical_basin_timeseries.png", dpi=300)
    print("[ok] QC + reporte + figura escritos")
    print(json.dumps(qc["products"], indent=2))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--products", default="all", help="gldas,merra,imerg o all (coma-separados)")
    ap.add_argument("--years", default=None, help="años a procesar, p.ej. 2025 o 2000,2001")
    ap.add_argument("--limit", type=int, default=None, help="granules/año (smoke)")
    ap.add_argument("--workers", type=int, default=10, help="concurrencia")
    ap.add_argument("--force", action="store_true", help="re-descargar años con checkpoint existente")
    ap.add_argument("--combine", action="store_true", help="combinar checkpoints + QC + figura")
    ap.add_argument("--cleanup", action="store_true", help="borrar checkpoints anuales tras --combine")
    args = ap.parse_args()

    prods = ["gldas", "merra", "imerg"] if args.products == "all" \
        else [x.strip() for x in args.products.split(",") if x.strip()]
    p.NWORKERS = args.workers

    print("[..] login Earthdata"); earthaccess.login(strategy="netrc")

    if args.combine:
        qc = {"generated": dt.datetime.now().isoformat(timespec="seconds"),
              "bbox": BBOX, "products": {}}
        series = {}
        for pr in prods:
            out = combine(pr)
            if out:
                qc_and_basin(pr, out, qc, series)
        if qc["products"]:
            write_report(qc, series)
        if args.cleanup:
            for pr in prods:
                d = HIST / pr
                for f in d.glob("daily_*.nc"):
                    f.unlink()
                print(f"[cleanup] {pr}: checkpoints borrados")
    else:
        years = [int(y) for y in args.years.split(",")] if args.years \
            else sorted(set(y for pr in prods for y in years_in(*RANGES[pr])))
        acquire(prods, years, args.limit, args.force)


if __name__ == "__main__":
    main()
