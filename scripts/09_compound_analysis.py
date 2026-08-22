#!/usr/bin/env python
"""09_compound_analysis.py — Fase 6: análisis compound (roadmap D-17: A, B, D + estacionalidad + XM).

Núcleo EMPÍRICO (sin supuestos de cópula); las cópulas con selección por AIC van aparte (paso C).
  A. Catálogo histórico de días compound (P, SM, Qs simultáneamente extremos) + rank de 2026.
  B. Probabilidad conjunta P(P>p90, SM>sm90, Qs>q90) con IC bootstrap.
  D. Análisis condicional P(Qs>q | P>p) vs P(Qs>q | P>p & SM>s) con IC bootstrap.
  Estacionalidad: GEV sobre máximos de FEBRERO (periodo de retorno condicional a la estación seca).
  Dependencia: Kendall tau (no paramétrica) para pares (P,SM),(P,Qs),(SM,Qs).
  XM: definición precisa del '19×' (media global vs climatología de febrero vs mediahist del día).

Uso: .venv\\Scripts\\python.exe scripts/09_compound_analysis.py
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
EXCL = ("2026-01-15", "2026-02-28")
SEED = 12345
NBOOT = 1000
RNG = np.random.default_rng(SEED)
Q = 0.90


def load():
    chirps = pd.read_csv(PROC / "basin_daily_CHIRPS.csv",
                         parse_dates=["time"], index_col="time")["precip_mm_day"]
    gldas = pd.read_csv(PROC / "basin_daily_GLDAS.csv", parse_dates=["time"], index_col="time")
    urra = pd.read_csv(INTERIM / "urra_sinu_balance_2005-01-01_2026-08-22.csv",
                       parse_dates=["Date"], index_col="Date")
    return chirps, gldas, urra


def common_frame(chirps, gldas):
    df = pd.DataFrame({
        "P": chirps,
        "SM": gldas["GLDAS_SoilMoi0_10cm_kg_m2"],
        "Qs": gldas["GLDAS_Qs_runoff_mm_day"],
    }).dropna()
    base = df.loc[~((df.index >= pd.Timestamp(EXCL[0])) &
                    (df.index <= pd.Timestamp(EXCL[1])))]
    return df, base


def boot_ci(df, stat_fn, n_boot=NBOOT):
    idx = np.arange(len(df))
    vals = []
    for _ in range(n_boot):
        b = RNG.choice(idx, size=len(idx), replace=True)
        try:
            vals.append(stat_fn(df.iloc[b]))
        except Exception:
            pass
    if not vals:
        return (np.nan, np.nan)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def kendall_pairs(base):
    pairs = [("P", "SM"), ("P", "Qs"), ("SM", "Qs")]
    out = {}
    for a, b in pairs:
        tau = stats.kendalltau(base[a], base[b]).statistic
        out[f"{a}_{b}"] = round(float(tau), 3)
    return out


def thresholds(base, q=Q):
    return {v: float(np.quantile(base[v], q)) for v in ("P", "SM", "Qs")}


def cdf_pct(base_s, val):
    """Fracción de días del baseline con valor < val (CDF empírica, en [0,1])."""
    return float((base_s < val).mean())


def compound_catalog(df, base, q=Q):
    """Días compound (P,SM,Qs simultáneamente > p_q) + severidad conjunta + rank de 2026."""
    th = thresholds(base, q)
    mask = (base["P"] > th["P"]) & (base["SM"] > th["SM"]) & (base["Qs"] > th["Qs"])
    n_days = int(mask.sum())
    n_total = len(base)

    # severidad conjunta sobre el BASELINE: promedio de CDFs empíricas (en [0,1])
    severity = pd.Series(
        (base["P"].rank(pct=True) + base["SM"].rank(pct=True) + base["Qs"].rank(pct=True)) / 3.0,
        index=base.index)

    hist_max = float(severity[mask].max())
    hist_max_date = str(severity[mask].idxmax().date())

    # severidad del EVENTO usando la MISMA CDF del baseline (no rank intra-evento)
    ev = df.loc[EVENT[0]:EVENT[1]]
    sevs = []
    for _, row in ev.iterrows():
        s = (cdf_pct(base["P"], row["P"]) + cdf_pct(base["SM"], row["SM"]) + cdf_pct(base["Qs"], row["Qs"])) / 3.0
        sevs.append(s)
    ev_sev = float(max(sevs))

    # rank del evento: fracción de días del baseline con severidad < ev_sev
    rank_pct = float(100.0 * (severity < ev_sev).mean())

    return {"thresholds": {k: round(v, 2) for k, v in th.items()},
            "n_compound_days": n_days, "n_total_days": n_total,
            "compound_day_freq_pct": round(100.0 * n_days / n_total, 2),
            "hist_max_severity": round(hist_max, 4), "hist_max_date": hist_max_date,
            "event_severity": round(ev_sev, 4),
            "event_rank_pct": round(rank_pct, 2),
            "event_peak_date": str(ev["P"].idxmax().date())}


def joint_exceedance(base, q=Q):
    th = thresholds(base, q)
    stat = lambda d: float(np.mean((d["P"] > th["P"]) & (d["SM"] > th["SM"]) & (d["Qs"] > th["Qs"])))
    p = stat(base)
    ci = boot_ci(base, stat)
    # margen univariado
    p_p = float(np.mean(base["P"] > th["P"]))
    p_sm = float(np.mean(base["SM"] > th["SM"]))
    p_qs = float(np.mean(base["Qs"] > th["Qs"]))
    return {"p_joint": round(p, 5), "ci": (round(ci[0], 5), round(ci[1], 5)),
            "p_P": round(p_p, 4), "p_SM": round(p_sm, 4), "p_Qs": round(p_qs, 4),
            "p_independent_if_no_dependence": round(p_p * p_sm * p_qs, 5),
            "amplification_ratio": round(p / (p_p * p_sm * p_qs), 2) if p_p * p_sm * p_qs > 0 else None}


def conditional(base, q=Q):
    th = thresholds(base, q)
    s_p = lambda d: float(np.mean((d["Qs"] > th["Qs"]) & (d["P"] > th["P"])) / np.mean(d["P"] > th["P"]))
    s_psm = lambda d: float(np.mean((d["Qs"] > th["Qs"]) & (d["P"] > th["P"]) & (d["SM"] > th["SM"]))
                            / np.mean((d["P"] > th["P"]) & (d["SM"] > th["SM"])))
    p_p = s_p(base); p_psm = s_psm(base)
    ci_p = boot_ci(base, s_p)
    ci_psm = boot_ci(base, s_psm)
    return {"P_Qs_extreme_given_P_extreme": round(p_p, 4),
            "ci_P": (round(ci_p[0], 4), round(ci_p[1], 4)),
            "P_Qs_extreme_given_P_and_SM_extreme": round(p_psm, 4),
            "ci_PSM": (round(ci_psm[0], 4), round(ci_psm[1], 4)),
            "delta": round(p_psm - p_p, 4)}


def feb_gev(df, var, obs):
    """GEV sobre máximos de FEBRERO (años<2026) → periodo de retorno de `obs` condicional a feb."""
    feb = df[(df.index.month == 2) & (df.index.year < 2026)]
    am = feb.groupby(feb.index.year)[var].max()
    am = am[am.notna()]
    x = am.values.astype(float)
    shape, loc, scale = stats.genextreme.fit(x)
    p = float(stats.genextreme.cdf(obs, shape, loc=loc, scale=scale))
    rp = float("inf") if p >= 1.0 else float(1.0 / (1.0 - p))
    rank = 1 + int(np.sum(x >= obs))
    rp_emp = (len(x) + 1.0) / rank
    return {"n_feb_years": len(x), "shape": round(float(shape), 3),
            "return_period_yr": round(rp, 2), "rp_empirical_yr": round(rp_emp, 2),
            "obs": round(float(obs), 3)}


def urra_definition(urra):
    a = urra["aportes_caudal_m3s"].dropna()
    ev = a.loc[EVENT[0]:EVENT[1]]
    peak_day = ev.idxmax()
    peak = float(ev.max())
    hist = a[a.index.year < 2026]
    full_mean = float(hist.mean())
    feb_mean = float(hist[hist.index.month == 2].mean())
    mediahist_peak = float(urra.loc[peak_day, "aportes_mediahist_m3s"]) if "aportes_mediahist_m3s" in urra else np.nan
    return {"peak_m3s": round(peak, 1), "peak_date": str(peak_day.date()),
            "ratio_vs_full_mean": round(peak / full_mean, 2),
            "ratio_vs_feb_mean": round(peak / feb_mean, 2),
            "ratio_vs_mediahist_same_day": round(peak / mediahist_peak, 2) if not np.isnan(mediahist_peak) else None,
            "full_mean_m3s": round(full_mean, 1), "feb_mean_m3s": round(feb_mean, 1),
            "mediahist_same_day_m3s": round(mediahist_peak, 1) if not np.isnan(mediahist_peak) else None,
            "feb2026_mean_m3s": round(float(ev.mean()), 1),
            "feb2026_vs_feb_mean": round(float(ev.mean()) / feb_mean, 2)}


def fig_severity(base, df, path):
    F = (base["P"].rank(pct=True) + base["SM"].rank(pct=True) + base["Qs"].rank(pct=True)) / 3.0
    ev = df.loc[EVENT[0]:EVENT[1]]
    sevs = []
    for _, row in ev.iterrows():
        sevs.append((cdf_pct(base["P"], row["P"]) + cdf_pct(base["SM"], row["SM"]) + cdf_pct(base["Qs"], row["Qs"])) / 3.0)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(F, bins=60, color="#9ecae1", alpha=0.8, label="Baseline (2000–2025)")
    for s in sevs:
        ax.axvline(s, color="#d7301f", lw=1.2, alpha=0.7)
    ax.axvline(max(sevs), color="#d7301f", lw=2, label="Event 2026 (severity)")
    ax.set_xlabel("Joint severity (mean of P, SM, Qs CDFs)")
    ax.set_ylabel("Days")
    ax.set_title("Historical compound severity — 2026 event at right")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def fig_conditional(report, path):
    a = report["conditional_p90"]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    vals = [a["P_Qs_extreme_given_P_extreme"], a["P_Qs_extreme_given_P_and_SM_extreme"]]
    lo = [a["P_Qs_extreme_given_P_extreme"] - a["ci_P"][0],
          a["P_Qs_extreme_given_P_and_SM_extreme"] - a["ci_PSM"][0]]
    hi = [a["ci_P"][1] - a["P_Qs_extreme_given_P_extreme"],
          a["ci_PSM"][1] - a["P_Qs_extreme_given_P_and_SM_extreme"]]
    ax.bar([0, 1], vals, 0.5, yerr=[lo, hi], capsize=5, color=["#08519c", "#d7301f"])
    ax.set_xticks([0, 1]); ax.set_xticklabels(["P(Qs>q | P>p)", "P(Qs>q | P>p & SM>s)"])
    ax.set_ylabel("Probability"); ax.set_title("Conditional effect of soil moisture (90% threshold)")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def fig_urra(urra, path):
    win = urra.loc["2026-01-25":"2026-02-15"]
    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    ax[0].plot(win.index, win["aportes_caudal_m3s"], color="#08519c", lw=1.2, label="inflow")
    ax[0].plot(win.index, win["aportes_mediahist_m3s"], color="#bdbdbd", ls="--", label="hist. same-day mean")
    ax[0].set_ylabel("Inflow (m\u00b3/s)"); ax[0].legend(fontsize=8)
    ax[1].plot(win.index, win["descarga_total_m3s"], color="#993404", lw=1.2, label="total discharge")
    ax[1].set_ylabel("Discharge (m\u00b3/s)"); ax[1].legend(fontsize=8)
    ax[2].plot(win.index, win["volumen_util_pct"] * 100, color="#006d2c", lw=1.2, label="useful storage %")
    ax[2].set_ylabel("Storage (% useful)"); ax[2].legend(fontsize=8)
    fig.suptitle("Urr\u00e1 reservoir \u2014 event balance (25 Jan\u201315 Feb 2026)")
    fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(path, dpi=300); plt.close(fig)


def main():
    chirps, gldas, urra = load()
    df, base = common_frame(chirps, gldas)
    report = {"generated": pd.Timestamp.now().isoformat(timespec="seconds"),
              "n_baseline_days": int(len(base)),
              "period": f"{base.index.min().date()} → {base.index.max().date()}"}

    report["kendall_tau"] = kendall_pairs(base)
    report["catalog"] = compound_catalog(df, base)
    report["joint_exceedance_p90"] = joint_exceedance(base)

    ev = df.loc[EVENT[0]:EVENT[1]]
    report["event_peaks"] = {"P_mm": round(float(ev["P"].max()), 2),
                             "Qs_mm": round(float(ev["Qs"].max()), 2),
                             "SM_kgm2": round(float(ev["SM"].max()), 2)}

    report["conditional_p90"] = conditional(base, q=0.90)
    report["conditional_p95"] = conditional(base, q=0.95)
    report["feb_GEV_P"] = feb_gev(df, "P", float(ev["P"].max()))
    report["feb_GEV_Qs"] = feb_gev(df, "Qs", float(ev["Qs"].max()))
    report["urra_definition"] = urra_definition(urra)

    fig_severity(base, df, FIG / "compound_severity.png")
    fig_conditional(report, FIG / "conditional_probability.png")
    fig_urra(urra, FIG / "urra_event_balance.png")

    (RES / "compound_analysis.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_md(report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("[ok] 09_compound_analysis.py terminado")


def write_md(r):
    L = ["# Compound Analysis — evento Córdoba 2026 (Fase 6)", "",
         f"Generado: {r['generated']} · baseline {r['n_baseline_days']} días ({r['period']})", "",
         "## Dependencia (Kendall tau, baseline)", "| Par | τ |", "|---|---|"]
    for k, v in r["kendall_tau"].items():
        L.append(f"| {k} | {v} |")

    c = r["catalog"]
    L += ["", "## Catálogo de días compound (P,SM,Qs simultáneamente > p90) [A]",
          f"- Umbrales p90: P>{c['thresholds']['P']} mm, SM>{c['thresholds']['SM']} kg/m², Qs>{c['thresholds']['Qs']} mm/d.",
          f"- Días compound en baseline: **{c['n_compound_days']}** de {c['n_total_days']} ({c['compound_day_freq_pct']}%).",
          f"- Máx. severidad histórica: **{c['hist_max_severity']}** ({c['hist_max_date']}).",
          f"- Severidad del evento: **{c['event_severity']}** → supera al **{c['event_rank_pct']}%** de días históricos."]

    j = r["joint_exceedance_p90"]
    L += ["", "## Probabilidad conjunta [B]",
          f"- P(P>p90, SM>p90, Qs>p90) = **{j['p_joint']}** (IC95% {j['ci']}).",
          f"- Si independientes: {j['p_independent_if_no_dependence']} → amplificación por dependencia **×{j['amplification_ratio']}**."]

    L += ["", "## Análisis condicional [D] (test clave H1/H2)",
          "| Umbral | P(Qs>q | P>p) | P(Qs>q | P>p & SM>s) | Δ |", "|---|---|---|---|"]
    for key, a in [("90%", r["conditional_p90"]), ("95%", r["conditional_p95"])]:
        L.append(f"| {key} | {a['P_Qs_extreme_given_P_extreme']} ({a['ci_P']}) | "
                 f"{a['P_Qs_extreme_given_P_and_SM_extreme']} ({a['ci_PSM']}) | {a['delta']} |")

    L += ["", "## Estacionalidad: GEV sobre febreros (retorno condicional a estación seca)",
          f"- P: RP condicional a feb = **{r['feb_GEV_P']['return_period_yr']} a** (empírico {r['feb_GEV_P']['rp_empirical_yr']} a, n={r['feb_GEV_P']['n_feb_years']}).",
          f"- Qs: RP condicional a feb = **{r['feb_GEV_Qs']['return_period_yr']} a** (empírico {r['feb_GEV_Qs']['rp_empirical_yr']} a, n={r['feb_GEV_Qs']['n_feb_years']})."]

    u = r["urra_definition"]
    L += ["", "## Definición precisa del '19×' (Urrá, XM)",
          f"- Pico aportes: **{u['peak_m3s']} m³/s** ({u['peak_date']}).",
          f"- vs media global 2005–2025: **×{u['ratio_vs_full_mean']}** ({u['full_mean_m3s']} m³/s).",
          f"- vs media de FEBRERO: **×{u['ratio_vs_feb_mean']}** ({u['feb_mean_m3s']} m³/s).",
          f"- vs mediahist del MISMO día (XM): **×{u['ratio_vs_mediahist_same_day']}** ({u['mediahist_same_day_m3s']} m³/s).",
          f"- Feb-2026 completo: media {u['feb2026_mean_m3s']} m³/s = **×{u['feb2026_vs_feb_mean']}** la media de febrero.",
          "", "## Figuras", "- `figures/compound_severity.png`, `conditional_probability.png`."]
    (RES / "compound_analysis_report.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()


