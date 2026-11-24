# Open-data industrial HSE screening index for Bangladesh: audit code

Code and derived outputs for the manuscript "Data-integrity artefacts beneath apparent weight robustness
in an audited open-data industrial safety screening index for Bangladesh".

## Data

The two raw inputs are third-party data and are not redistributed. Download them into `data/raw/`:

| File | Source | SHA-256 |
|---|---|---|
| `bd_air_quality_hourly.csv.gz` | Mendeley Data, https://doi.org/10.17632/9j447cynb9.2 (v2) | `90c01dad834b3603a57e7b918881fecb2c95ede96585b2264b143c2a32ccc192` |
| `osh_facilities_bd.csv` | Open Supply Hub, Bangladesh country export (downloaded 2026-09-04) | `53080becd7827da8bf45884e3a7ae061b1c32ab0c9a29adebc18d92acb19f4e3` |

Three assessed sites belong to one private company. They are identified only by facility type and
district, at sub-district or area-centroid precision.

## Reproduce

    pip install -r requirements.txt scipy matplotlib
    python scripts/build_dataset.py   # integrity screen, facility counts, exposure tables
    python scripts/analysis.py        # manuscript Sections 5.1 and 5.3-5.6 -> reports/audit/
    python scripts/figs.py            # Figs. 3-7 -> reports/figures/
    python -m pytest -q               # 87 unit tests

Random seed 20261124, 10,000 Monte Carlo draws. Model definitions are in the header of
`scripts/analysis.py`; provenance and checksums are in `reports/audit/DATA_PROVENANCE.md`.

## Licence

MIT.
