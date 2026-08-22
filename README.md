# cordoba_flood

Reproducible analysis of the exceptional **1–6 February 2026 flood of the Sinú River (Córdoba, Colombia)** as a multivariate **compound event**. The pipeline combines multi-source Earth observations and reanalysis — CHIRPS/IMERG (precipitation), GLDAS-2.1 (soil moisture, runoff), MERRA-2 (water vapor, temperature), ERA5-Land (long-record soil moisture), XM/Urrá operational records, and Copernicus EMS SAR flood mapping — to quantify the event's rarity (percentiles, extreme-value analysis), its compound character (copulas, conditional probabilities) and its spatial coherence with the observed flood footprint.

> This repository contains **only the code**. No datasets, figures, tables, or manuscript files are included (see `.gitignore`).

## Repository layout

```
cordoba_flood/
├── README.md
├── .gitignore
├── requirements.txt
└── scripts/
    ├── 02_download_xm_urra.py      # XM / Urrá reservoir operational records
    ├── 04_spatial_processing.py    # Basin mask (HydroBASINS level 12)
    ├── 05_pilot_acquire_qc.py      # Pilot: IMERG + GLDAS + MERRA-2 (event window)
    ├── 06_historical_acquire.py    # GLDAS + MERRA-2 full historical record
    ├── 07_chirps_acquire.py        # CHIRPS daily precipitation (1981–present)
    ├── 08_event_analysis.py        # Percentiles + EVT (GEV/GPD) + IMERG-vs-CHIRPS
    ├── 09_compound_analysis.py     # Compound catalogue, joint/conditional probabilities
    ├── 10_copula_analysis.py       # Copula selection (AIC) + joint exceedance
    ├── 11_spatial_ems.py           # Spatial overlay of the Copernicus EMS flood extent
    └── 12_era5land_crosscheck.py   # ERA5-Land soil-moisture long-record cross-check
```

The scripts expect the following directories to exist (they are created automatically when needed):

```
data/raw/          # downloaded/raw inputs (HydroBASINS, Copernicus EMS, ...)
data/interim/      # per-source intermediate files
data/processed/    # basin-mean daily series and derived files
results/           # per-analysis JSON / Markdown reports
figures/           # output figures
```

## Requirements

- **Python 3.12** (Windows or Linux; the scripts are OS-agnostic).
- Install dependencies:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
pip install -r requirements.txt
```

## External data access (credentials required)

| Source | Variable | Script(s) | Access |
|---|---|---|---|
| XM (Colombia grid operator) | Urrá reservoir inflow/spill/storage | `02` | Public API via `pydataxm` (no key) |
| NASA GES DISC / Earthdata | IMERG, GLDAS-2.1, MERRA-2 | `05`, `06` | Free NASA Earthdata Login (`earthaccess`) |
| CHIRPS (UCSB / ClimateSERV) | precipitation | `07` | ClimateSERV API (free registration) |
| Copernicus EMS | EMSR865 flood extent/depth | `11` | Manual download (mapping.emergency.copernicus.eu) |
| Copernicus CDS | ERA5-Land monthly soil moisture | `12` | CDS API key in `~/.cdsapirc` (license accepted) |
| HydroBASINS / HydroSHEDS | basin polygons | `04` | Manual download of `hybas_sa_lev01-12_v1c.zip` |

Set the NASA Earthdata credentials with `earthaccess.login()` on first run, and place the CDS key in `~/.cdsapirc` (see the [CDS API docs](https://cds.climate.copernicus.eu/how-to-api)).

## Execution order

Run from the repository root:

```bash
# 1. Build the basin mask (needs the HydroBASINS zip in data/raw/)
python scripts/04_spatial_processing.py

# 2. Acquire the core datasets (downloads; require the credentials above)
python scripts/05_pilot_acquire_qc.py     # event-window pilot + QC
python scripts/06_historical_acquire.py   # GLDAS + MERRA-2 historical
python scripts/07_chirps_acquire.py       # CHIRPS historical
python scripts/02_download_xm_urra.py     # Urrá reservoir records
python scripts/12_era5land_crosscheck.py  # ERA5-Land (CDS)

# 3. Analysis (read data/processed/*.csv; write results/ and figures/)
python scripts/08_event_analysis.py
python scripts/09_compound_analysis.py
python scripts/10_copula_analysis.py
python scripts/11_spatial_ems.py          # needs Copernicus EMS files in data/raw/copernicus_ems/
```

Each acquisition script is **checkpointed**: already-downloaded files are skipped on re-run. Analysis scripts are deterministic (fixed random seed) and overwrite their `results/*.json` and `figures/*.png` outputs.

## Outputs

- `data/processed/basin_daily_CHIRPS.csv`, `basin_daily_GLDAS.csv`, `basin_daily_MERRA2.csv`, `basin_monthly_ERA5LAND_swvl1.csv`
- `data/processed/sinu_basin.gpkg` (basin mask, HydroBASINS level 12)
- `results/event_analysis.json`, `compound_analysis.json`, `copula_analysis.json`, `spatial_ems.json`, `era5land_crosscheck.json`
- `figures/*.png` (11 figures)

## Citation

Code is provided for reproducibility of the analysis. Please cite the associated manuscript (in preparation) if you reuse the pipeline.

