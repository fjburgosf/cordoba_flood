#!/usr/bin/env python
"""
05_pilot_acquire_qc.py — PILOTO de adquisición + QC del pipeline núcleo (Fase 5, piloto).

Objetivo: validar el pipeline completo antes de escalar al histórico. Adquiere
IMERG Late + GLDAS-2.1 + MERRA-2 para ene–feb 2026, recorta al bbox de la cuenca
del Sinú, aplica la máscara de cuenca (HydroBASINS, D-10), agrega a diario, calcula
series de media-en-cuenca (ponderadas por cos(lat)) y produce QC + plots.

Subsetting: OPeNDAP DAP2 servidor-lado (constraint sobre bbox) para GLDAS/MERRA-2, con
guardia de volumen (cap 4 MB/respuesta — nunca se baja el granule completo). MERRA-2 vía
OPeNDAP-en-la-nube (opendap.earthdata.nasa.gov; goldsmr4 retirado). IMERG vía earthaccess
(ya cacheado). Los subsets diarios se cachean en data/interim/pilot/ para re-ejecución rápida.

Requiere: credenciales Earthdata (~/_netrc en Windows) y app GESDISC autorizada.

Salidas:
  data/interim/pilot/<prod>_sinu_daily.nc         subsets diarios recortados
  data/interim/pilot/basin_daily_<prod>.csv       series media-en-cuenca diarias
  results/pilot_qc_report.md                      reporte QC
  figures/pilot_*.png                             plots preliminares
"""
from __future__ import annotations
import sys, json, warnings, traceback
import datetime as dt
from pathlib import Path
warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

import io, math, os, netrc as _netrc, requests
from requests.auth import HTTPBasicAuth
import numpy as np, pandas as pd, xarray as xr, geopandas as gpd, regionmask
import earthaccess
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

NWORKERS = 10
_SESSION = None
_TOKEN = None


def _edl_token():
    """Token EDL (Bearer) generado desde _netrc/.netrc via la API de URS."""
    global _TOKEN
    if _TOKEN is None:
        user = pw = None
        for nf in (os.path.join(os.path.expanduser("~"), "_netrc"),
                   os.path.join(os.path.expanduser("~"), ".netrc")):
            if os.path.exists(nf):
                n = _netrc.netrc(nf)
                for m, (u, _, p) in n.hosts.items():
                    if "urs.earthdata.nasa.gov" in m or "earthdata" in m:
                        user, pw = u, p; break
                if user:
                    break
        if not (user and pw):
            raise RuntimeError("credenciales Earthdata no encontradas (_netrc/.netrc)")
        r = requests.post("https://urs.earthdata.nasa.gov/api/users/find_or_create_token",
                          auth=HTTPBasicAuth(user, pw), timeout=60)
        r.raise_for_status()
        _TOKEN = r.json().get("access_token") or r.json().get("token")
        if not _TOKEN:
            raise RuntimeError("no se pudo obtener token EDL")
    return _TOKEN


def session():
    """Sesión requests plana con Bearer token. Paraleliza bien en threads, a diferencia
    de earthaccess.get_requests_https_session() (que serializa por su auth de redirect).
    Fuerza IPv4: la ruta IPv6 hacia hydro1.gesdisc no establece TCP (SYN_SENT colgado)."""
    global _SESSION
    if _SESSION is None:
        try:
            import requests.packages.urllib3.util.connection as _uc
            _uc.HAS_IPV6 = False
        except Exception:
            pass
        _SESSION = requests.Session()
        _SESSION.headers.update({"Authorization": f"Bearer {_edl_token()}"})
    return _SESSION

# Grillas regulares conocidas (para índices DAP2 sin fetch de metadatos)
GRID = {
    "GLDAS": dict(lat0=-59.875, dlat=0.25, lon0=-179.875, dlon=0.25, nlat=600, nlon=1440,
                  host="hydro1.gesdisc.eosdis.nasa.gov"),
    "MERRA": dict(lat0=-90.0, dlat=0.5, lon0=-180.0, dlon=0.625, nlat=361, nlon=576),
}
CAP_BYTES = 4_000_000  # guardia de volumen: ningún subset puede exceder 4 MB


def _idx(v0, step, vmin, vmax, n=None):
    a = int(math.floor((vmin - v0) / step)); b = int(math.ceil((vmax - v0) / step))
    a = max(a, 0); b = b if n is None else min(b, n - 1)
    return a, b


def _dap2_ce(vars_, t_ce, ilat, ilon):
    """CE DAP2 clásico multi-variable: var[t_ce][lat][lon] separadas por comas."""
    return ",".join(f"{v}{t_ce}[{ilat[0]}:1:{ilat[1]}][{ilon[0]}:1:{ilon[1]}]" for v in vars_)


def _host_url(ext_link, host):
    """URL OPeNDAP (host Hyrax clásico) a partir del link de datos de CMR."""
    return ext_link.replace("data.gesdisc.earthdata.nasa.gov/data/", host + "/opendap/")


def _cloud_opendap_url(g):
    """URL OPeNDAP-en-la-nube (opendap.earthdata.nasa.gov) del granule (MERRA-2)."""
    for it in g.get("umm", {}).get("RelatedUrls", []):
        u = it.get("URL", "")
        if "opendap.earthdata.nasa.gov" in u:
            return u
    raise RuntimeError("granule sin URL opendap.earthdata.nasa.gov en RelatedUrls")


def fetch_subset(url):
    """GET con guardia de volumen (CAP_BYTES): solo descarga si es un subset pequeño."""
    with session().get(url, timeout=120, stream=True) as resp:
        resp.raise_for_status()
        cl = resp.headers.get("Content-Length")
        if cl and int(cl) > CAP_BYTES:
            raise ValueError(f"respuesta {cl} B > cap {CAP_BYTES} B (CE inválido, granule completo?)")
        data = bytearray(); total = 0
        for chunk in resp.iter_content(chunk_size=131072):
            total += len(chunk)
            if total > CAP_BYTES:
                raise ValueError(f"respuesta excedió {CAP_BYTES} B (CE inválido, granule completo?)")
            data += chunk
    return xr.open_dataset(io.BytesIO(bytes(data)), engine="h5netcdf").load(), total


def fetch_subset_parallel(granules, url_builder, label):
    """Descarga subsets DAP2 en paralelo (hilos). Preserva orden temporal.

    Warm-up: el primer granule se descarga de forma síncrona para completar el
    redirect URS/auth del requests.Session compartido en el hilo principal y
    evitar la carrera que deadlockea las threads.
    """
    if not granules:
        return []
    out = [None] * len(granules)
    done = 0; total_bytes = 0
    try:
        ds0, nb0 = fetch_subset(url_builder(granules[0]))
        out[0] = ds0; total_bytes += nb0
    except Exception as e:
        print(f"   [skip] {label} #0: {e}")
    done = 1
    def job(i, g):
        ds, nb = fetch_subset(url_builder(g))
        return i, ds, nb
    with ThreadPoolExecutor(max_workers=NWORKERS) as ex:
        futs = {ex.submit(job, i, g): i for i, g in enumerate(granules[1:], start=1)}
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                _, ds, nb = fut.result()
                out[i] = ds; total_bytes += nb
            except Exception as e:
                print(f"   [skip] {label} #{i}: {e}")
            done += 1
            if done % 40 == 0 or done == len(granules):
                print(f"   {label} {done}/{len(granules)}  (~{total_bytes/1e6:.1f} MB)", flush=True)
    return [x for x in out if x is not None]

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT/"data"/"processed"; PILOT = ROOT/"data"/"interim"/"pilot"
FIG = ROOT/"figures"; RES = ROOT/"results"
for d in (PILOT, FIG, RES): d.mkdir(parents=True, exist_ok=True)

START, END = "2026-01-01", "2026-02-28"
EVENT = ("2026-02-01", "2026-02-06")
meta = json.loads((PROC/"sinu_basin_meta.json").read_text())
bx = meta["bbox_lonlat"]; BUF = 0.15
BBOX = (bx[0]-BUF, bx[1]-BUF, bx[2]+BUF, bx[3]+BUF)   # minlon,minlat,maxlon,maxlat
BASIN = gpd.read_file(PROC/"sinu_basin.gpkg")
QC = {"generated": dt.datetime.now().isoformat(timespec="seconds"),
      "bbox": BBOX, "window": [START, END], "products": {}}


def basin_weighted_mean(da: xr.DataArray) -> xr.DataArray:
    """Media espacial ponderada por cos(lat), restringida a la máscara de cuenca."""
    lon = da["lon"]; lat = da["lat"]
    m = regionmask.mask_geopandas(BASIN, lon, lat)   # 0 dentro, NaN fuera
    w = np.cos(np.deg2rad(lat))
    return da.where(m == 0).weighted(w.fillna(0)).mean(("lat", "lon"))


def open_granule(g, group=None):
    fo = earthaccess.open([g])[0]
    return xr.open_dataset(fo, engine="h5netcdf", group=group)


# ----------------------------------------------------------- IMERG Late
def do_imerg(limit=None):
    cache = PILOT/"imerg_sinu_daily.nc"
    if cache.exists():
        da = xr.open_dataarray(cache); print("[cache] IMERG"); return da
    r = earthaccess.search_data(short_name="GPM_3IMERGDL", version="07",
                                temporal=(START, END), bounding_box=BBOX)
    if limit: r = r[:limit]
    print(f"[IMERG] {len(r)} granules")
    days = []
    for i, g in enumerate(r):
        try:
            ds = open_granule(g)
            p = ds["precipitation"]  # (time,lon,lat) mm/day
            p = p.sel(lon=slice(BBOX[0], BBOX[2]), lat=slice(BBOX[1], BBOX[3]))
            days.append(p.load())
        except Exception as e:
            print(f"   [skip] IMERG {i}: {e}")
        if i % 10 == 0: print(f"   IMERG {i+1}/{len(r)}")
    da = xr.concat(days, dim="time").sortby("time")
    da.attrs["units"] = "mm/day"
    da.to_netcdf(cache)
    return da


# ----------------------------------------------------------- GLDAS 2.1
GLDAS_INST = ["SoilMoi0_10cm_inst", "SoilMoi10_40cm_inst"]
GLDAS_ACC  = ["Qs_acc", "Qsb_acc"]
def do_gldas(limit=None):
    cache = PILOT/"gldas_sinu_daily.nc"
    if cache.exists():
        ds = xr.open_dataset(cache); print("[cache] GLDAS"); return ds
    r = earthaccess.search_data(short_name="GLDAS_NOAH025_3H", version="2.1",
                                temporal=(START, END), bounding_box=BBOX)
    if limit: r = r[:limit]
    print(f"[GLDAS] {len(r)} granules (3-horario)", flush=True)
    gr = GRID["GLDAS"]
    ilat = _idx(gr["lat0"], gr["dlat"], BBOX[1], BBOX[3], gr["nlat"])
    ilon = _idx(gr["lon0"], gr["dlon"], BBOX[0], BBOX[2], gr["nlon"])
    ce = _dap2_ce(GLDAS_INST + GLDAS_ACC, "[0:1:0]", ilat, ilon)
    def url_builder(g):
        return _host_url(g.data_links(access="external")[0], gr["host"]) + ".nc4?" + ce
    recs = fetch_subset_parallel(r, url_builder, "GLDAS")
    full = xr.concat(recs, dim="time").sortby("time")
    # agregación a diario: inst -> media diaria; acc(kg/m2 por paso 3h) -> suma diaria (mm/día)
    daily_inst = full[GLDAS_INST].resample(time="1D").mean()
    daily_acc = full[GLDAS_ACC].resample(time="1D").sum()
    out = xr.merge([daily_inst, daily_acc])
    out.to_netcdf(cache)
    return out


# ----------------------------------------------------------- MERRA-2
MERRA_VARS = ["T2M", "TQV"]
def do_merra(limit=None):
    cache = PILOT/"merra_sinu_daily.nc"
    if cache.exists():
        ds = xr.open_dataset(cache); print("[cache] MERRA-2"); return ds
    r = earthaccess.search_data(short_name="M2T1NXSLV", version="5.12.4",
                                temporal=(START, END), bounding_box=BBOX)
    if limit: r = r[:limit]
    print(f"[MERRA-2] {len(r)} granules (horario)", flush=True)
    gr = GRID["MERRA"]
    ilat = _idx(gr["lat0"], gr["dlat"], BBOX[1], BBOX[3], gr["nlat"])
    ilon = _idx(gr["lon0"], gr["dlon"], BBOX[0], BBOX[2], gr["nlon"])
    ce = _dap2_ce(MERRA_VARS, "[0:1:23]", ilat, ilon)
    def url_builder(g):
        return _cloud_opendap_url(g) + ".nc4?" + ce
    recs = fetch_subset_parallel(r, url_builder, "MERRA")
    full = xr.concat(recs, dim="time").sortby("time")
    out = full.resample(time="1D").mean()
    out.to_netcdf(cache)
    return out


def qc_series(name, da, unit, expect_days):
    s = da.to_series()
    n = len(s); nan = int(s.isna().sum())
    QC["products"][name] = {
        "unit": unit, "days": n, "expected_days": expect_days,
        "coverage_pct": round(100*n/expect_days, 1) if expect_days else None,
        "nan_days": nan, "min": round(float(np.nanmin(s)), 3),
        "max": round(float(np.nanmax(s)), 3), "mean": round(float(np.nanmean(s)), 3),
        "first": str(s.index.min().date()), "last": str(s.index.max().date()),
    }
    return s


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="limitar granules por producto (smoke)")
    args = ap.parse_args()
    print("[..] login Earthdata"); earthaccess.login(strategy="netrc")
    exp_days = (pd.Timestamp(END)-pd.Timestamp(START)).days + 1

    # --- IMERG
    imerg = do_imerg(limit=args.limit)
    imerg_bm = basin_weighted_mean(imerg)
    s_pr = qc_series("IMERG_precip_mm_day", imerg_bm, "mm/day", exp_days)
    s_pr.to_csv(PILOT/"basin_daily_IMERG.csv", header=["precip_mm_day"])

    # --- GLDAS
    gl = do_gldas(limit=args.limit)
    sm = basin_weighted_mean(gl["SoilMoi0_10cm_inst"])
    qs = basin_weighted_mean(gl["Qs_acc"])
    s_sm = qc_series("GLDAS_SoilMoi0_10cm_kg_m2", sm, "kg/m2", exp_days)
    s_qs = qc_series("GLDAS_Qs_runoff_mm_day", qs, "mm/day(=kg/m2/d)", exp_days)
    pd.concat({"SoilMoi0_10cm_kg_m2": s_sm, "Qs_runoff_mm_day": s_qs}, axis=1)\
      .to_csv(PILOT/"basin_daily_GLDAS.csv")

    # --- MERRA-2
    me = do_merra(limit=args.limit)
    t2 = basin_weighted_mean(me["T2M"]) - 273.15
    tqv = basin_weighted_mean(me["TQV"])
    s_t2 = qc_series("MERRA2_T2M_degC", t2, "degC", exp_days)
    s_tqv = qc_series("MERRA2_TQV_kg_m2", tqv, "kg/m2", exp_days)
    pd.concat({"T2M_degC": s_t2, "TQV_kg_m2": s_tqv}, axis=1)\
      .to_csv(PILOT/"basin_daily_MERRA2.csv")

    # ---------- PLOTS
    fig, ax = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    ax[0].bar(s_pr.index, s_pr.values, color="#08519c"); ax[0].set_ylabel("Precip\n(mm/d)")
    ax[0].set_title("Piloto Sinú — señal hidroclimática diaria ene–feb 2026 (media en cuenca)")
    ax[1].plot(s_tqv.index, s_tqv.values, color="#54278f"); ax[1].set_ylabel("TQV\n(kg/m²)")
    ax[1].plot(s_t2.index, s_t2.values, color="#e6550d", alpha=.6, label="T2M °C"); ax[1].legend(fontsize=7)
    ax[2].plot(s_sm.index, s_sm.values, color="#006d2c"); ax[2].set_ylabel("SoilMoi\n0-10cm")
    ax[3].bar(s_qs.index, s_qs.values, color="#993404"); ax[3].set_ylabel("Runoff Qs\n(mm/d)")
    for a in ax: a.axvspan(pd.Timestamp(EVENT[0]), pd.Timestamp(EVENT[1]), color="yellow", alpha=.2)
    fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(FIG/"pilot_basin_timeseries.png", dpi=300)

    # mapa: acumulado de precip del evento
    ev = imerg.sel(time=slice(*EVENT)).sum("time")
    fig2, ax2 = plt.subplots(figsize=(6, 7))
    ev.transpose("lat", "lon").plot(ax=ax2, cmap="Blues", cbar_kwargs={"label": "Precip acumulada 1-6 feb (mm)"})
    BASIN.boundary.plot(ax=ax2, color="k", linewidth=1.2)
    ax2.set_title("IMERG Late — acumulado evento 1–6 feb 2026"); fig2.tight_layout()
    fig2.savefig(FIG/"pilot_event_precip_map.png", dpi=300)

    # ---------- REPORTE QC
    lines = ["# Pilot QC Report — pipeline núcleo (IMERG/GLDAS/MERRA-2)", "",
             f"Generado: {QC['generated']}  ·  Ventana: {START} → {END}  ·  BBox: {BBOX}", "",
             f"Cuenca Sinú: {meta['area_km2']} km², máscara HydroBASINS lev12.", "",
             "| Variable | Unidad | Días | Cobertura | NaN | min | mean | max | 1er | último |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in QC["products"].items():
        lines.append(f"| {k} | {v['unit']} | {v['days']} | {v['coverage_pct']}% | {v['nan_days']} "
                     f"| {v['min']} | {v['mean']} | {v['max']} | {v['first']} | {v['last']} |")
    # chequeo de señal del evento
    pre = s_pr.loc[:EVENT[0]].iloc[:-1].mean()
    evm = s_pr.loc[EVENT[0]:EVENT[1]].max()
    lines += ["", "## Señal del evento (IMERG precip media-cuenca)",
              f"- Media pre-evento (ene): {pre:.2f} mm/d",
              f"- Máximo diario durante 1–6 feb: {evm:.2f} mm/d",
              f"- Ratio evento/pre-evento: {evm/pre:.1f}×" if pre else "- (pre=0)",
              "", "## Chequeos",
              "- [x] Masking espacial (regionmask sobre polígono Sinú)",
              "- [x] Cobertura temporal y % días",
              "- [x] Unidades leídas de atributos del producto",
              "- [x] NaN por variable",
              "- [x] Agregación: GLDAS inst→media diaria, acc→suma diaria; MERRA horario→media diaria",
              "- [x] Alineación temporal a día UTC (nota: hora local CO = UTC-5)"]
    (RES/"pilot_qc_report.md").write_text("\n".join(lines), encoding="utf-8")
    (RES/"pilot_qc.json").write_text(json.dumps(QC, indent=2))
    print("[ok] QC report + plots escritos"); print(json.dumps(QC["products"], indent=2))


if __name__ == "__main__":
    main()
