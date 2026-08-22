#!/usr/bin/env python
"""10_copula_analysis.py — paso C (D-17): cópulas con selección AIC + retorno conjunto.

Compara 5 familias (Gaussian, t, Clayton, Gumbel, Frank) por AIC sobre pseudo-observaciones,
para los pares (P,Qs), (P,SM), (SM,Qs). Luego estima la probabilidad de excedencia conjunta
P(U>u, V>v) en umbrales p90/p95/p99 (empírica + por cada cópula) y su periodo de retorno conjunto.

Uso: .venv\\Scripts\\python.exe scripts/10_copula_analysis.py
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
from scipy import stats, optimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
FIG = ROOT / "figures"
for d in (RES, FIG):
    d.mkdir(parents=True, exist_ok=True)

EXCL = ("2026-01-15", "2026-02-28")


def load():
    chirps = pd.read_csv(PROC / "basin_daily_CHIRPS.csv",
                         parse_dates=["time"], index_col="time")["precip_mm_day"]
    gldas = pd.read_csv(PROC / "basin_daily_GLDAS.csv", parse_dates=["time"], index_col="time")
    df = pd.DataFrame({"P": chirps, "SM": gldas["GLDAS_SoilMoi0_10cm_kg_m2"],
                       "Qs": gldas["GLDAS_Qs_runoff_mm_day"]}).dropna()
    base = df.loc[~((df.index >= pd.Timestamp(EXCL[0])) & (df.index <= pd.Timestamp(EXCL[1])))]
    return base


def pseudo(x):
    x = np.asarray(x, float)
    return (stats.rankdata(x) / (len(x) + 1.0))


def _clip(u):
    return np.clip(np.asarray(u, float), 1e-9, 1 - 1e-9)


# ---------------- densidades de cópula ----------------
def d_gaussian(u, v, rho):
    x = stats.norm.ppf(_clip(u)); y = stats.norm.ppf(_clip(v))
    num = 1 / (2 * np.pi * np.sqrt(1 - rho ** 2)) * np.exp(
        -(x ** 2 - 2 * rho * x * y + y ** 2) / (2 * (1 - rho ** 2)))
    return num / (stats.norm.pdf(x) * stats.norm.pdf(y))


def d_t(u, v, rho, nu):
    x = stats.t.ppf(_clip(u), nu); y = stats.t.ppf(_clip(v), nu)
    X = np.stack([x, y], axis=-1)
    num = stats.multivariate_t(loc=None, shape=np.array([[1, rho], [rho, 1]]), df=nu).pdf(X)
    return num / (stats.t.pdf(x, nu) * stats.t.pdf(y, nu))


def d_clayton(u, v, th):
    u, v = _clip(u), _clip(v)
    s = u ** (-th) + v ** (-th) - 1
    return (1 + th) * (u * v) ** (-th - 1) * s ** (-1 / th - 2)


def d_gumbel(u, v, th):
    u, v = _clip(u), _clip(v)
    x, y = -np.log(u), -np.log(v)
    S = x ** th + y ** th
    B = S ** ((1 - th) / th)
    C = np.exp(-S ** (1 / th))
    return C * (x * y) ** (th - 1) / (u * v) * (B ** 2 + (th - 1) * B / S)


def d_frank(u, v, th):
    u, v = _clip(u), _clip(v)
    if abs(th) < 1e-8:
        return np.ones_like(u)
    e = np.exp(-th)
    denom = 1 - e - (1 - np.exp(-th * u)) * (1 - np.exp(-th * v))
    return th * (1 - e) * np.exp(-th * (u + v)) / denom ** 2


# ---------------- CDFs de cópula ----------------
def c_gaussian(u, v, rho):
    z = stats.norm.ppf(_clip(np.asarray([u, v], float)))
    return float(stats.multivariate_normal(mean=None, cov=np.array([[1, rho], [rho, 1]])).cdf(z))


def c_t(u, v, rho, nu):
    t = stats.t.ppf(_clip(np.asarray([u, v], float)), nu)
    return float(stats.multivariate_t(loc=None, shape=np.array([[1, rho], [rho, 1]]), df=nu).cdf(t))


def c_clayton(u, v, th):
    return (u ** (-th) + v ** (-th) - 1) ** (-1 / th)


def c_gumbel(u, v, th):
    return np.exp(-((-np.log(u)) ** th + (-np.log(v)) ** th) ** (1 / th))


def c_frank(u, v, th):
    e = np.exp(-th)
    return -1 / th * np.log(1 + (np.exp(-th * u) - 1) * (np.exp(-th * v) - 1) / (e - 1))


DENS = {"gaussian": d_gaussian, "t": d_t, "clayton": d_clayton, "gumbel": d_gumbel, "frank": d_frank}
CDFS = {"gaussian": c_gaussian, "t": c_t, "clayton": c_clayton, "gumbel": c_gumbel, "frank": c_frank}
FAMILIES = ["gaussian", "t", "clayton", "gumbel", "frank"]


def fit_family(u, v, family):
    if family == "gaussian":
        f = lambda p: -np.sum(np.log(d_gaussian(u, v, p[0])))
        r = optimize.minimize_scalar(lambda p: f([p]), bounds=(-0.99, 0.99), method="bounded")
        params = [r.x]; ll = -f([r.x]); k = 1
    elif family == "t":
        def f(p):
            rho, nu = p
            if nu < 1 or abs(rho) > 0.98:
                return 1e12
            try:
                return -np.sum(np.log(d_t(u, v, rho, nu)))
            except Exception:
                return 1e12
        r = optimize.minimize(f, x0=[0.5, 5.0], bounds=[(-0.98, 0.98), (1, 60)], method="L-BFGS-B")
        params = list(r.x); ll = -f(r.x); k = 2
    elif family == "clayton":
        f = lambda p: -np.sum(np.log(d_clayton(u, v, p[0])))
        r = optimize.minimize_scalar(lambda p: f([p]), bounds=(0.05, 20), method="bounded")
        params = [r.x]; ll = -f([r.x]); k = 1
    elif family == "gumbel":
        f = lambda p: -np.sum(np.log(d_gumbel(u, v, p[0])))
        r = optimize.minimize_scalar(lambda p: f([p]), bounds=(1.0, 20), method="bounded")
        params = [r.x]; ll = -f([r.x]); k = 1
    else:  # frank
        f = lambda p: -np.sum(np.log(d_frank(u, v, p[0])))
        r = optimize.minimize_scalar(lambda p: f([p]), bounds=(-20, 20), method="bounded")
        params = [r.x]; ll = -f([r.x]); k = 1
    return {"params": [round(float(x), 4) for x in params], "loglik": round(float(ll), 2),
            "AIC": round(float(2 * k - 2 * ll), 2), "k": k}


def copula_cdf(family, params, u, v):
    if family == "gaussian":
        return c_gaussian(u, v, params[0])
    if family == "t":
        return c_t(u, v, params[0], params[1])
    if family == "clayton":
        return c_clayton(u, v, params[0])
    if family == "gumbel":
        return c_gumbel(u, v, params[0])
    return c_frank(u, v, params[0])


def joint_p_and(family, params, q):
    """P(U>q, V>q) = 1 - 2q + C(q,q) (escenario AND)."""
    return 1 - 2 * q + copula_cdf(family, params, q, q)


def empirical_joint(base, a, b, q):
    pa = np.quantile(base[a], q); pb = np.quantile(base[b], q)
    return float(np.mean((base[a] > pa) & (base[b] > pb)))


def analyze_pair(base, a, b):
    u = pseudo(base[a].values); v = pseudo(base[b].values)
    tau = float(stats.kendalltau(base[a], base[b]).statistic)
    fits = {f: fit_family(u, v, f) for f in FAMILIES}
    best = min(FAMILIES, key=lambda f: fits[f]["AIC"])
    out = {"tau": round(tau, 3), "best_family": best, "best_AIC": fits[best]["AIC"],
           "best_params": fits[best]["params"], "AIC_table": {f: fits[f]["AIC"] for f in FAMILIES},
           "joint_exceedance": {}}
    for q in (0.90, 0.95, 0.99):
        out["joint_exceedance"][str(q)] = {
            "empirical": round(empirical_joint(base, a, b, q), 5),
            "copula_best": round(joint_p_and(best, fits[best]["params"], q), 5),
            "copula_sensitivity": {f: round(joint_p_and(f, fits[f]["params"], q), 5) for f in FAMILIES},
            "rp_best_yr": round(1 / (joint_p_and(best, fits[best]["params"], q) * 365.25), 1)}
    return out


def fig_pairs(base):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (a, b) in zip(axs, [("P", "Qs"), ("P", "SM"), ("SM", "Qs")]):
        u = pseudo(base[a].values); v = pseudo(base[b].values)
        tau = stats.kendalltau(base[a], base[b]).statistic
        ax.scatter(u, v, s=4, alpha=0.4, color="#08519c")
        ax.set_title(f"{a}–{b} (τ={tau:.2f})", fontsize=9)
        ax.set_xlabel(f"u({a})"); ax.set_ylabel(f"u({b})")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(alpha=0.3)
    fig.suptitle("Pseudo-observaciones (escala cópula) — dependencia entre drivers")
    fig.tight_layout(); fig.savefig(FIG / "copula_pairs.png", dpi=300); plt.close(fig)


def main():
    base = load()
    report = {"generated": pd.Timestamp.now().isoformat(timespec="seconds"), "n_days": int(len(base))}
    pairs = {}
    for a, b in [("P", "Qs"), ("P", "SM"), ("SM", "Qs")]:
        pairs[f"{a}_{b}"] = analyze_pair(base, a, b)
    report["pairs"] = pairs

    fig_pairs(base)
    (RES / "copula_analysis.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_md(report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("[ok] 10_copula_analysis.py terminado")


def write_md(r):
    L = ["# Cópulas — selección AIC + retorno conjunto (Fase 6, paso C)", "",
         f"Generado: {r['generated']} · baseline {r['n_days']} días (2000–2025, sin evento)", "",
         "## Selección de familia (AIC; menor = mejor) — 5 modelos comparados",
         "| Par | τ | Mejor | AIC | Gaussian | t | Clayton | Gumbel | Frank |", "|---|---|---|---|---|---|---|---|---|"]
    for k, p in r["pairs"].items():
        t = p["AIC_table"]
        L.append(f"| {k} | {p['tau']} | **{p['best_family']}** | {p['best_AIC']} | "
                 f"{t['gaussian']} | {t['t']} | {t['clayton']} | {t['gumbel']} | {t['frank']} |")

    L += ["", "## Probabilidad de excedencia conjunta P(U>q, V>q) y RP conjunto (AND)",
          "| Par | Umbral | Empírica | Cópula (mejor) | RP (años) |", "|---|---|---|---|---|"]
    for k, p in r["pairs"].items():
        for q, v in p["joint_exceedance"].items():
            L.append(f"| {k} | {q} | {v['empirical']} | {v['copula_best']} | {v['rp_best_yr']} |")

    L += ["", "## Sensibilidad del RP conjunto a la familia de cópula (umbral 99%)",
          "| Par | Gaussian | t | Clayton | Gumbel | Frank |", "|---|---|---|---|---|---|"]
    for k, p in r["pairs"].items():
        s = p["joint_exceedance"]["0.99"]["copula_sensitivity"]
        L.append(f"| {k} | {s['gaussian']} | {s['t']} | {s['clayton']} | {s['gumbel']} | {s['frank']} |")
    L += ["", "- P(U>q,V>q) = probabilidad diaria de excedencia conjunta (escenario AND); RP = 1/(P×365.25).",
          "- Si P(U>q,V>q) ≫ (1−q)² (independencia), la dependencia infla el retorno conjunto.",
          "", "## Figuras", "- `figures/copula_pairs.png`."]
    (RES / "copula_analysis_report.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()


