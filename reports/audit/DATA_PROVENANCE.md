# Data provenance for the audit (manuscript Sections 5.1-5.6)

| File | Source | SHA-256 | Local file date |
|---|---|---|---|
| data/raw/bd_air_quality_hourly.csv.gz | Mendeley Data, DOI 10.17632/9j447cynb9.2, version 2 | 90c01dad834b3603a57e7b918881fecb2c95ede96585b2264b143c2a32ccc192 | 2026-09-04 |
| data/raw/osh_facilities_bd.csv | Open Supply Hub, Bangladesh country export | 53080becd7827da8bf45884e3a7ae061b1c32ab0c9a29adebc18d92acb19f4e3 | 2026-09-04 |

The Open Supply Hub download date is taken from the local file date (2026-09-04); confirm it against
the download record before citing it as the retrieval date.

## Reproduction

    python scripts/build_dataset.py   # integrity screen, facility counts, exposure tables
    python scripts/analysis.py        # Sections 5.3-5.6, writes reports/audit/
    python scripts/figs.py            # Figs. 3-7, writes reports/figures/

Fixed random seed 20261124, 10,000 draws. A fresh run on 2026-11-24 from the two raw files above
reproduced the duplicate series (Azimpur-Dhaka 28,968 h; Dohar-Faridpur 28,992 h; Barisal-Gaurnadi 9,783 h),
the Dhaka segments (2000-2021 mean 93.8; 2022 118.5; 2023 50.2), all twelve CSHRI values, Table 4,
the monthly and shift means and the 435 AQI-trigger days.

## Splice

The Dhaka daily mean falls from 156.2 ug/m3 on 2022-08-03 to 12.0 ug/m3 on 2022-08-04, the first day
of the other 29 series. The 1 January to 3 August 2022 mean is 166.8 ug/m3.
