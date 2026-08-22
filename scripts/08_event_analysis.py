#!/usr/bin/env python
"""08_event_analysis.py — Fase 6 (análisis del evento Córdoba 2026).

Combina las series media-en-cuenca (CHIRPS/GLDAS/MERRA-2/IMERG evento/Urrá) y posiciona
el evento (1–6 feb 2026) en el registro histórico:
  1. Serie combinada -> data/processed/event_analysis_combined.csv
  2. Percentiles del evento (multi-variable)  [RQ2 compound]
  3. EVT: GEV (máximos anuales) + GPD-POT, periodo de retorno con IC bootstrap  [RQ1]
  4. Lluvia antecedente API (1/3/5/7/15/30 d)  [RQ3]
  5. Estacionalidad (percentil condicional a febrero)  [RQ4]
  6. Validación cruzada IMERG vs CHIRPS  [RQ6]
  7. Hidrograma del evento  [RQ5]

Uso:  .venv\\Scripts\\python.exe scripts/08_event_analysis.py
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
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
INTERIM = ROOT / "data" / "interim"
RES = ROOT / "results"
FIG = ROOT / "figures"
for d in (RES, FIG):
    d.mkdir(parents=True, exist_ok=True)

EVENT = ("2026-02-01", "2026-02-06")
BASELINE_EXCL = ("2026-01-15", "2026-02-28")
PLOT_WINDOW = ("2026-01-25", "2026-02-15")
RX5_WINDOW = ("2026-01-28", "2026-02-10")
SEED = 12345
NBOOT = 1000
RNG = np.random.default_rng(SEED)


def load_series():
    out = {}
    out["chirps"] = pd.read_csv(PROC / "basin_daily_CHIRPS.csv",
                                parse_dates=["time"], index_col="time")["precip_mm_day"]
    out["gldas"] = pd.read_csv(PROC / "basin_daily_GLDAS.csv", parse_dates=["time"], index_col="time")
    out["merra"] = pd.read_csv(PROC / "basin_daily_MERRA2.csv", parse_dates=["time"], index_col="time")
    out["imerg"] = pd.read_csv(INTERIM / "pilot" / "basin_daily_IMERG.csv",
                               parse_dates=["time"], index_col="time")["precip_mm_day"]
    out["urra"] = pd.read_csv(INTERIM / "urra_sinu_balance_2005-01-01_2026-08-22.csv",
                              parse_dates=["Date"], index_col="Date")
    return out


def historical_baseline(s):
    """Serie sin la ventana del evento (evita auto-inflación de percentiles)."""
    return s.dropna().loc[
        ~((s.index >= pd.Timestamp(BASELINE_EXCL[0])) &
          (s.index <= pd.Timestamp(BASELINE_EXCL[1])))
    ]


def pct_below(base, value):
    return float(100.0 * (base < value).mean())


def ev_peak_pct(s):
    base = historical_baseline(s)
    ev = s.loc[EVENT[0]:EVENT[1]].dropna()
    if ev.empty or base.empty:
        return {"value": None, "percentile": None}
    peak = float(ev.max())
    return {"value": round(peak, 3), "percentile": round(pct_below(base, peak), 2)}


def annual_maxima(s):
    """Máximos anuales solo de años completos (>=360 días válidos)."""
    s = s.dropna()
    am = {}
    for y in sorted(set(s.index.year)):
        yy = s[s.index.year == y]
        if len(yy) >= 360:
            am[y] = float(yy.max())
    return pd.Series(am, dtype=float)


def gev_return_period(am, obs):
    """GEV (máximos) + periodo de retorno del valor `obs`, con IC bootstrap."""
    x = am.values.astype(float)
    try:
        shape, loc, scale = stats.genextreme.fit(x)
    except Exception:
        return {"error": "GEV fit failed"}

    def rp(params, val):
        sh, lo, sc = params
        p = float(stats.genextreme.cdf(val, sh, loc=lo, scale=sc))
        return float("inf") if p >= 1.0 else float(1.0 / (1.0 - p))

    rp_obs = rp((shape, loc, scale), obs)
    rank = 1 + int(np.sum(x >= obs))
    rp_emp = (len(x) + 1.0) / rank
    boots = []
    for _ in range(NBOOT):
        xb = RNG.choice(x, size=len(x), replace=True)
        try:
            sb, lob, scb = stats.genextreme.fit(xb)
            boots.append(rp((sb, lob, scb), obs))
        except Exception:
            pass
    ci = (np.percentile(boots, 2.5), np.percentile(boots, 97.5)) if boots else (np.nan, np.nan)
    return {"n_years": len(x), "shape": round(float(shape), 4),
            "loc": round(float(loc), 3), "scale": round(float(scale), 3),
            "return_period_yr": round(rp_obs, 2), "rp_empirical_yr": round(rp_emp, 2),
            "rp_ci_yr": (round(float(ci[0]), 2), round(float(ci[1]), 2)),
            "obs": round(float(obs), 3)}


def gpd_return_period(s, obs, thr_q=0.95):
    """GPD-POT: umbral = cuantil `thr_q` de días húmedos (>0); RP del evento con IC."""
    s = s.dropna()
    wet = s[s > 0]
    u = float(np.quantile(wet, thr_q))
    exc = (wet[wet > u] - u).values
    n_days = len(s)
    years = (s.index[-1] - s.index[0]).days / 365.25
    lambda_u = len(exc) / years

    def rp(exc_vals, val):
        c, loc, sc = stats.genpareto.fit(exc_vals, floc=0)
        pu = len(exc_vals) / n_days
        tail = 1.0 - float(stats.genpareto.cdf(val - u, c, loc=loc, scale=sc))
        lam_x = 365.25 * pu * tail
        return float("inf") if lam_x <= 0 else 1.0 / lam_x

    rp_obs = rp(exc, obs)
    boots = []
    for _ in range(NBOOT):
        eb = RNG.choice(exc, size=len(exc), replace=True)
        try:
            boots.append(rp(eb, obs))
        except Exception:
            pass
    ci = (np.percentile(boots, 2.5), np.percentile(boots, 97.5)) if boots else (np.nan, np.nan)
    return {"threshold": round(u, 3), "n_exceedances": len(exc),
            "lambda_u_per_yr": round(lambda_u, 3), "return_period_yr": round(rp_obs, 2),
            "rp_ci_yr": (round(float(ci[0]), 2), round(float(ci[1]), 2))}


def antecedent_precip(precip, peak_date, windows=(1, 3, 5, 7, 15, 30)):
    """Suma de lluvia en los `w` días previos al pico + percentil histórico (sin evento)."""
    base_full = historical_baseline(precip)
    out = {}
    for w in windows:
        end = peak_date - pd.Timedelta(days=1)
        start = peak_date - pd.Timedelta(days=w)
        val = float(precip.loc[start:end].sum(min_count=w))
        roll = base_full.rolling(w, min_periods=w).sum().dropna()
        out[str(w)] = {"mm": round(val, 2), "percentile": round(pct_below(roll, val), 2)}
    return out


def seasonal_percentile(s):
    """Percentil del pico del evento condicional al MES (febrero) — estacionalidad [RQ4]."""
    base = historical_baseline(s)
    feb = base[base.index.month == 2]
    ev = s.loc[EVENT[0]:EVENT[1]].dropna()
    peak = float(ev.max())
    return {"feb_peak_percentile": round(pct_below(feb, peak), 2),
            "n_feb_days": int(len(feb))}


def cross_validate(imerg, chirps):
    df = pd.concat([imerg.rename("IMERG"), chirps.rename("CHIRPS")], axis=1).dropna()
    ev = df.loc[EVENT[0]:EVENT[1]]
    r = stats.pearsonr(df["IMERG"], df["CHIRPS"]).statistic
    rho = stats.spearmanr(df["IMERG"], df["CHIRPS"]).statistic
    err = df["IMERG"] - df["CHIRPS"]
    return {"n_days": int(len(df)), "pearson_r": round(float(r), 3),
            "spearman_rho": round(float(rho), 3), "bias_mm": round(float(err.mean()), 3),
            "rmse_mm": round(float(np.sqrt((err ** 2).mean())), 3),
            "mae_mm": round(float(err.abs().mean()), 3),
            "mean_imerg_mm": round(float(df["IMERG"].mean()), 2),
            "mean_chirps_mm": round(float(df["CHIRPS"].mean()), 2),
            "event_mean_imerg_mm": round(float(ev["IMERG"].mean()), 2),
            "event_mean_chirps_mm": round(float(ev["CHIRPS"].mean()), 2)}


def fig_hydrograph(chirps, imerg, gldas, merra, urra, path):
    sl = slice(PLOT_WINDOW[0], PLOT_WINDOW[1])
    fig, axs = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    axs[0].bar(chirps.loc[sl].index, chirps.loc[sl].values, color="#4daf4a", alpha=0.85, label="CHIRPS")
    axs[0].plot(imerg.loc[sl].index, imerg.loc[sl].values, color="black", lw=1.4, label="IMERG")
    axs[0].set_ylabel("Precip (mm/d)"); axs[0].legend(fontsize=8, ncol=2)
    axs[1].plot(gldas.loc[sl].index, gldas["GLDAS_Qs_runoff_mm_day"].loc[sl], color="#08519c", lw=1.4)
    axs[1].set_ylabel("GLDAS Qs (mm/d)")
    axs[2].plot(urra.loc[sl].index, urra["aportes_caudal_m3s"].loc[sl], color="#d7301f", lw=1.4, label="inflow")
    axs[2].plot(urra.loc[sl].index, urra["descarga_total_m3s"].loc[sl], color="#636363", lw=1.4, ls="--", label="discharge")
    axs[2].set_ylabel("Urrá (m³/s)"); axs[2].legend(fontsize=8)
    axs[3].plot(merra.loc[sl].index, merra["TQV_kg_m2"].loc[sl], color="#6a51a3", lw=1.4)
    axs[3].set_ylabel("TQV (kg/m²)")
    for a in axs:
        a.axvspan(pd.Timestamp(EVENT[0]), pd.Timestamp(EVENT[1]), color="yellow", alpha=0.22)
    fig.suptitle("Córdoba 2026 event — hydrograph (25 Jan–15 Feb)", fontsize=12)
    fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(path, dpi=300); plt.close(fig)


def fig_return_level(am, gev, obs, path):
    x = np.sort(am.values)
    n = len(x)
    T_emp = (n + 1) / (n + 1 - np.arange(1, n + 1))
    T = np.logspace(0, 2.3, 200)
    c, loc, sc = gev["shape"], gev["loc"], gev["scale"]
    if abs(c) < 1e-8:
        y = loc - sc * np.log(-np.log(1 - 1 / T))
    else:
        y = loc + sc / c * (1 - (-np.log(1 - 1 / T)) ** c)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.semilogx(T_emp, x, "o", ms=5, color="#08519c", label="Annual maxima (empirical)")
    ax.semilogx(T, y, "-", color="#d7301f", lw=2, label="GEV fitted")
    ax.axhline(obs, color="black", ls="--", lw=1, label=f"Event Rx1day={obs:.1f} mm")
    ax.set_xlabel("Return period (years)"); ax.set_ylabel("Precipitation (mm/d)")
    ax.set_title("Return levels — CHIRPS basin mean (Sinú)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def fig_scatter(imerg, chirps, path):
    df = pd.concat([imerg.rename("IMERG"), chirps.rename("CHIRPS")], axis=1).dropna()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df["CHIRPS"], df["IMERG"], s=14, alpha=0.7, color="#08519c")
    lo, hi = min(df.min().min(), 0), df.max().max()
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("CHIRPS (mm/d)"); ax.set_ylabel("IMERG (mm/d)")
    ax.set_title("IMERG vs CHIRPS — event window (Jan–Feb 2026)")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def main():
    d = load_series()
    chirps, gldas, merra, imerg, urra = (d["chirps"], d["gldas"], d["merra"],
                                         d["imerg"], d["urra"])
    report = {"generated": pd.Timestamp.now().isoformat(timespec="seconds")}

    # 1. serie combinada
    combined = pd.concat({
        "CHIRPS_precip_mm_day": chirps,
        "GLDAS_SoilMoi0_10cm_kg_m2": gldas["GLDAS_SoilMoi0_10cm_kg_m2"],
        "GLDAS_SoilMoi10_40cm_kg_m2": gldas["GLDAS_SoilMoi10_40cm_kg_m2"],
        "GLDAS_Qs_runoff_mm_day": gldas["GLDAS_Qs_runoff_mm_day"],
        "GLDAS_Qsb_runoff_mm_day": gldas["GLDAS_Qsb_runoff_mm_day"],
        "MERRA2_T2M_degC": merra["T2M_degC"],
        "MERRA2_TQV_kg_m2": merra["TQV_kg_m2"],
        "IMERG_precip_mm_day": imerg,
        "urra_aportes_m3s": urra["aportes_caudal_m3s"],
        "urra_descarga_total_m3s": urra["descarga_total_m3s"],
    }, axis=1).sort_index()
    combined.to_csv(PROC / "event_analysis_combined.csv")

    # 2. percentiles del evento (compound, RQ2)
    pct = {}
    for name, s in [
        ("CHIRPS_precip_mm_day", chirps),
        ("GLDAS_Qs_runoff_mm_day", gldas["GLDAS_Qs_runoff_mm_day"]),
        ("GLDAS_Qsb_runoff_mm_day", gldas["GLDAS_Qsb_runoff_mm_day"]),
        ("GLDAS_SoilMoi0_10cm_kg_m2", gldas["GLDAS_SoilMoi0_10cm_kg_m2"]),
        ("GLDAS_SoilMoi10_40cm_kg_m2", gldas["GLDAS_SoilMoi10_40cm_kg_m2"]),
        ("MERRA2_TQV_kg_m2", merra["TQV_kg_m2"]),
        ("MERRA2_T2M_degC", merra["T2M_degC"]),
    ]:
        pct[name] = ev_peak_pct(s)
    report["event_percentiles"] = pct

    # 3. EVT CHIRPS (Rx1day / Rx5day) y GLDAS Qs
    peak_date = chirps.loc[EVENT[0]:EVENT[1]].idxmax()
    rx1_event = float(chirps.loc[EVENT[0]:EVENT[1]].max())
    rx5 = chirps.rolling(5, min_periods=5).sum()
    rx5_event = float(rx5.loc[RX5_WINDOW[0]:RX5_WINDOW[1]].max())

    am_rx1 = annual_maxima(chirps)
    am_rx5 = annual_maxima(rx5)
    gev_rx1 = gev_return_period(am_rx1, rx1_event)
    gev_rx5 = gev_return_period(am_rx5, rx5_event)
    gpd_rx1 = gpd_return_period(chirps, rx1_event)
    report["EVT_CHIRPS"] = {"peak_date": str(peak_date.date()),
                            "rx1day_mm": round(rx1_event, 2),
                            "rx5day_mm": round(rx5_event, 2),
                            "GEV_rx1day": gev_rx1, "GEV_rx5day": gev_rx5,
                            "GPD_rx1day": gpd_rx1}

    qs = gldas["GLDAS_Qs_runoff_mm_day"]
    qs_event = float(qs.loc[EVENT[0]:EVENT[1]].max())
    report["EVT_GLDAS_Qs"] = {"event_peak_mm_d": round(qs_event, 2),
                              "GEV": gev_return_period(annual_maxima(qs), qs_event)}

    # 4-6. antecedente, estacionalidad, cross-validation
    report["antecedent_precip"] = antecedent_precip(chirps, peak_date)
    report["seasonal_CHIRPS"] = seasonal_percentile(chirps)
    report["seasonal_GLDAS_Qs"] = seasonal_percentile(qs)
    report["crossval_IMERG_vs_CHIRPS"] = cross_validate(imerg, chirps)

    # figuras
    fig_hydrograph(chirps, imerg, gldas, merra, urra, FIG / "event_hydrograph.png")
    fig_return_level(am_rx1, gev_rx1, rx1_event, FIG / "chirps_return_level.png")
    fig_scatter(imerg, chirps, FIG / "imerg_vs_chirps.png")

    (RES / "event_analysis.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_md(report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("[ok] 08_event_analysis.py terminado")


def write_md(r):
    L = ["# Event Analysis — evento Córdoba 2026 (Fase 6)", "",
         f"Generado: {r['generated']}", "",
         "## 1. Percentil del pico del evento (1–6 feb 2026) vs histórico sin el evento [RQ2]",
         "| Variable | Pico evento | Percentil histórico |", "|---|---|---|"]
    for k, v in r["event_percentiles"].items():
        L.append(f"| {k} | {v['value']} | {v['percentile']}% |")

    e = r["EVT_CHIRPS"]
    L += ["", "## 2. Periodo de retorno CHIRPS (media-en-cuenca) [RQ1]",
          f"- Rx1day del evento: **{e['rx1day_mm']} mm/d** ({e['peak_date']}); Rx5day: **{e['rx5day_mm']} mm**.",
          f"- GEV Rx1day ({e['GEV_rx1day']['n_years']} años): RP **{e['GEV_rx1day']['return_period_yr']} a** (IC95% {e['GEV_rx1day']['rp_ci_yr']}); empírico {e['GEV_rx1day']['rp_empirical_yr']} a.",
          f"- GEV Rx5day: RP **{e['GEV_rx5day']['return_period_yr']} a** (IC95% {e['GEV_rx5day']['rp_ci_yr']}).",
          f"- GPD-POT Rx1day (umbral {e['GPD_rx1day']['threshold']} mm): RP **{e['GPD_rx1day']['return_period_yr']} a** (IC95% {e['GPD_rx1day']['rp_ci_yr']})."]

    q = r["EVT_GLDAS_Qs"]["GEV"]
    L += ["", "## 3. Escorrentía GLDAS (registro corto)",
          f"- Pico evento Qs = **{r['EVT_GLDAS_Qs']['event_peak_mm_d']} mm/d**; GEV RP **{q['return_period_yr']} a** (IC95% {q['rp_ci_yr']}); empírico {q['rp_empirical_yr']} a.",
          "", "## 4. Lluvia antecedente (API) hasta el pico [RQ3]",
          "| Ventana (d) | Suma (mm) | Percentil histórico |", "|---|---|---|"]
    for w, v in r["antecedent_precip"].items():
        L.append(f"| {w} | {v['mm']} | {v['percentile']}% |")

    L += ["", "## 5. Estacionalidad (pico condicional a febrero) [RQ4]",
          f"- CHIRPS febrero: **{r['seasonal_CHIRPS']['feb_peak_percentile']}%** (n={r['seasonal_CHIRPS']['n_feb_days']}).",
          f"- GLDAS Qs febrero: **{r['seasonal_GLDAS_Qs']['feb_peak_percentile']}%**.",
          "", "## 6. Validación IMERG vs CHIRPS [RQ6]"]
    c = r["crossval_IMERG_vs_CHIRPS"]
    L += [f"- n={c['n_days']} d; Pearson r={c['pearson_r']}; Spearman ρ={c['spearman_rho']}.",
          f"- Sesgo (IMERG−CHIRPS)={c['bias_mm']} mm; RMSE={c['rmse_mm']} mm; MAE={c['mae_mm']} mm.",
          f"- Media evento: IMERG {c['event_mean_imerg_mm']} vs CHIRPS {c['event_mean_chirps_mm']} mm/d.",
          "", "## Figuras",
          "- `figures/event_hydrograph.png`, `chirps_return_level.png`, `imerg_vs_chirps.png`."]
    (RES / "event_analysis_report.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
