#!/usr/bin/env python
"""
02_download_xm_urra.py — Extracción reproducible de la operación del embalse Urrá (URRA1)
y aportes del río Sinú desde la API pública de XM (operador del mercado eléctrico de Colombia),
vía la librería oficial `pydataxm` (repo EquipoAnaliticaXM/API_XM).

Establece el BASELINE OPERATIVO del sistema Urrá–Sinú (D-11), sin autenticación.
Endpoint real: https://servapibi.xm.com.co  (NO xm.com.co).

Métricas (verificadas contra el catálogo de XM, 2026-08-22):
  AporCaudal          Rio      m3/s   Aportes (afluente) por río  -> filtrar 'SINU URRA'
  AporCaudalMediHist  Rio      m3/s   Media histórica de aportes (referencia climatológica)
  VoluUtilDiarMasa    Embalse  m3     Volumen útil diario (almacenamiento) -> 'URRA1'
  PorcVoluUtilDiar    Embalse  %      Volumen útil diario (%)
  VertMasa            Embalse  m3     Vertimientos (rebosadero)
  VolTurbMasa         Embalse  m3     Volumen turbinado
  (Descarga total ≈ VolTurbMasa + VertMasa; DescMasa viene vacío para URRA1)

Identificadores exactos: Embalse='URRA1' (región CARIBE); Rio='SINU URRA'.
Cobertura observada: aportes SINU desde ~2005; volumen/vertimientos disponibles en el evento.

Uso:
  python scripts/02_download_xm_urra.py                 # ventana del evento (por defecto)
  python scripts/02_download_xm_urra.py --full          # histórico 2005-01-01 -> hoy
  python scripts/02_download_xm_urra.py --start 2020-01-01 --end 2026-02-28

Salida: data/raw/xm/<metric>_<start>_<end>.csv  y  data/interim/urra_sinu_balance_<...>.csv
"""
from __future__ import annotations
import argparse
import datetime as dt
import sys
from pathlib import Path

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from pydataxm import pydataxm

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "xm"
INT = ROOT / "data" / "interim"
RAW.mkdir(parents=True, exist_ok=True)
INT.mkdir(parents=True, exist_ok=True)

RIO_SINU = "SINU"        # subcadena de filtro (etiqueta real: 'SINU URRA')
EMBALSE = "URRA1"

# (metric, entity, filtro_substring, etiqueta)
JOBS = [
    ("AporCaudal", "Rio", RIO_SINU, "aportes_caudal_m3s"),
    ("AporCaudalMediHist", "Rio", RIO_SINU, "aportes_mediahist_m3s"),
    ("VoluUtilDiarMasa", "Embalse", EMBALSE, "volumen_util_m3"),
    ("PorcVoluUtilDiar", "Embalse", EMBALSE, "volumen_util_pct"),
    ("VertMasa", "Embalse", EMBALSE, "vertimientos_m3"),
    ("VolTurbMasa", "Embalse", EMBALSE, "turbinado_m3"),
]

MAX_CHUNK_DAYS = 30  # límite de la API por request


def daterange_chunks(s: dt.date, e: dt.date, step: int):
    cur = s
    while cur <= e:
        nxt = min(cur + dt.timedelta(days=step - 1), e)
        yield cur, nxt
        cur = nxt + dt.timedelta(days=1)


def fetch(api, metric, entity, filt, s: dt.date, e: dt.date) -> pd.DataFrame:
    frames = []
    for a, b in daterange_chunks(s, e, MAX_CHUNK_DAYS):
        try:
            df = api.request_data(metric, entity, a, b)
        except Exception as ex:
            print(f"   [warn] {metric} {a}->{b}: {ex!r}")
            continue
        if df is None or len(df) == 0:
            continue
        if filt:
            df = df[df["Name"].astype(str).str.contains(filt, case=False, na=False)]
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-01-25")
    ap.add_argument("--end", default="2026-02-15")
    ap.add_argument("--full", action="store_true", help="histórico 2005-01-01 -> hoy")
    args = ap.parse_args()

    if args.full:
        s, e = dt.date(2005, 1, 1), dt.date.today()
    else:
        s = dt.date.fromisoformat(args.start)
        e = dt.date.fromisoformat(args.end)

    print(f"[..] Extrayendo XM URRA1/SINU  {s} -> {e}", flush=True)
    api = pydataxm.ReadDB()
    daily = {}
    for metric, entity, filt, label in JOBS:
        df = fetch(api, metric, entity, filt, s, e)
        if df.empty:
            print(f"   [--] {label} ({metric}): sin datos")
            continue
        fn = RAW / f"{metric}_{s}_{e}.csv"
        df.to_csv(fn, index=False)
        # serie diaria: columna 'Date' + 'Value' (valor diario) cuando exista
        if "Date" in df.columns and "Value" in df.columns:
            ser = df[["Date", "Value"]].copy()
            ser["Date"] = pd.to_datetime(ser["Date"])
            ser = ser.groupby("Date")["Value"].sum().rename(label)
            daily[label] = ser
        print(f"   [ok] {label:22s} filas={len(df):5d} -> {fn.name}", flush=True)

    if daily:
        bal = pd.concat(daily.values(), axis=1).sort_index()
        # descarga total (m3/día) y en m3/s
        if {"turbinado_m3", "vertimientos_m3"}.issubset(bal.columns):
            bal["descarga_total_m3"] = bal[["turbinado_m3", "vertimientos_m3"]].sum(axis=1, min_count=1)
            bal["descarga_total_m3s"] = bal["descarga_total_m3"] / 86400.0
        out = INT / f"urra_sinu_balance_{s}_{e}.csv"
        bal.to_csv(out)
        print(f"[ok] Balance combinado -> {out}  (filas={len(bal)}, cols={list(bal.columns)})")
    print("[done]")


if __name__ == "__main__":
    main()
