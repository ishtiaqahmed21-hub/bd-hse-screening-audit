"""Audit of the original site index and the revised two-pillar framework (manuscript Sections 5.1, 5.3-5.6).

Run after scripts/build_dataset.py:  python scripts/analysis.py
Inputs : data/processed/site_risk_index.csv, data/reference/{sites,bd_cities}.csv,
         data/raw/bd_air_quality_hourly.csv.gz (splice check only)
Outputs: reports/audit/*.csv and reports/audit/summary.json

Monte Carlo definitions (manuscript Table 3), one draw = one independent sample of every factor:
  w_E        ~ Uniform(0.2, 0.8); H weight = 1 - w_E
  PM2.5 err  multiplicative lognormal, factor = exp(Z), Z ~ N(0, 0.20^2), median 1.
             One factor per distinct air-quality series: sites drawing on the same series share it.
  H indicator facility count n, or worker total converted to facility equivalents
             n_w = workers / (sum workers / sum facilities over the 12 sites); p = 0.5 each.
  H transform log(1 + n) or linear n; p = 0.5 each.
  Normalise  fixed goalposts (E: 5-60 ug/m3; H: 0-1000 facilities, on the transformed scale) or
             min-max over the 12 sites; p = 0.5 each, applied to both pillars together.
  Aggregate  arithmetic w_E*E + (1-w_E)*H, or geometric (E+1)^w_E (H+1)^(1-w_E) - 1; p = 0.5 each.
All factors are sampled independently. Data uncertainty (PM2.5 error) and structural uncertainty
(the other five factors) are reported separately in first_order_shares.csv.
Ranks: 1 = highest score, ties share the best rank. Upper tier = rank 1-4, lower tier = rank 9-12.
First-order share of factor X for one site = Var(E[rank | X]) / Var(rank); continuous factors are
binned into 20 quantile bins. Shares are averaged over sites; the remainder is interaction.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "audit"
OUT.mkdir(parents=True, exist_ok=True)
SEED, N_DRAWS = 20261124, 10_000
W_ORIG = np.array([0.30, 0.25, 0.25, 0.20])        # PM2.5, facilities, workers, access


def minmax(x):
    x = np.asarray(x, float)
    span = x.max() - x.min()
    return np.zeros_like(x) if span == 0 else 100 * (x - x.min()) / span


def original_index(pm, fac, wrk, acc, w=W_ORIG):
    return np.c_[minmax(pm), minmax(fac), minmax(wrk), minmax(acc)] @ w


def revised_pillars(pm, n, transform="log", norm="fixed"):
    t = np.log1p if transform == "log" else (lambda v: np.asarray(v, float))
    if norm == "fixed":
        e = 100 * np.clip((pm - 5) / 55, 0, 1)
        h = 100 * np.clip(t(n) / t(1000.0), 0, 1)
    else:
        e, h = minmax(pm), minmax(t(n))
    return e, h


def aggregate(e, h, w_e=0.5, how="arith"):
    if how == "arith":
        return w_e * e + (1 - w_e) * h
    return (e + 1) ** w_e * (h + 1) ** (1 - w_e) - 1


def ranks(score):
    return rankdata(-np.asarray(score), method="min")


def circle_overlap_pct(d, r=10.0):
    if d >= 2 * r:
        return 0.0
    a = 2 * r * r * np.arccos(d / (2 * r)) - d / 2 * np.sqrt(4 * r * r - d * d)
    return 100 * a / (np.pi * r * r)


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(h))


def splice_check():
    aq = pd.read_csv(ROOT / "data/raw/bd_air_quality_hourly.csv.gz",
                     usecols=["city_name", "datetime", "pm2_5"], parse_dates=["datetime"])
    dhaka = aq[aq.city_name == "Dhaka"].set_index("datetime").pm2_5.resample("D").mean()
    others_start = aq.loc[aq.city_name != "Dhaka", "datetime"].min().normalize()
    splice_day = dhaka.diff().loc["2022"].idxmin()
    return {
        "splice_date": str(splice_day.date()),
        "other_series_start": str(others_start.date()),
        "dhaka_mean_2022_01_01_to_day_before_splice": round(dhaka.loc["2022-01-01":splice_day - pd.Timedelta(days=1)].mean(), 1),
        "dhaka_daily_mean_day_before": round(dhaka.loc[splice_day - pd.Timedelta(days=1)], 1),
        "dhaka_daily_mean_splice_day": round(dhaka.loc[splice_day], 1),
    }


def main():
    rng = np.random.default_rng(SEED)
    idx = pd.read_csv(ROOT / "data/processed/site_risk_index.csv")
    sites = pd.read_csv(ROOT / "data/reference/sites.csv")
    cities = pd.read_csv(ROOT / "data/reference/bd_cities.csv")
    d = sites[["site_id", "lat", "lon"]].merge(idx.drop(columns=["lat", "lon"]), on="site_id")
    # company sites are reported by facility type and district only (manuscript Ethics statement)
    d["site_name"] = d.site_id.map({"S01": "Food plant A (Narayanganj)", "S02": "Food plant B (Gazipur)",
                                    "S03": "Corporate office (Dhaka)"}).fillna(d.site_name)
    pm = d.mean_pm2_5.to_numpy(float)
    fac = d.facilities_within_r.to_numpy(float)
    wrk = d.workers_within_r.to_numpy(float) / 1000          # thousands, as in Table 5
    acc = d.urban_centre_km.to_numpy(float)
    # duplicate series (Section 5.1) are one series: map each city to the first of its duplicate pair
    dup = pd.read_csv(ROOT / "data/processed/qa_duplicate_series.csv")
    alias = dict(zip(dup.city_b, dup.city_a))
    series = d.nearest_monitor_city.map(lambda c: alias.get(c, c)).to_numpy()
    summary = {"seed": SEED, "n_draws": N_DRAWS, "distinct_air_series": int(len(set(series)))}

    # ---- 5.1 splice ------------------------------------------------------------------
    summary["splice"] = splice_check()

    # ---- 5.3 reproduction and structural audit ----------------------------------------
    cshri = original_index(pm, fac, wrk, acc)
    summary["cshri_max_abs_diff_vs_pipeline"] = float(np.abs(cshri - d.cshri).max())
    comps = np.c_[pm, fac, wrk, acc]
    norm = np.c_[minmax(pm), minmax(fac), minmax(wrk), minmax(acc)]
    corr_p = np.corrcoef(comps.T)
    corr_s = spearmanr(comps).statistic
    k = 4
    alpha = k / (k - 1) * (1 - norm.var(axis=0, ddof=1).sum() / norm.sum(1).var(ddof=1))
    summary.update({
        "pearson_fac_wrk": round(pearsonr(fac, wrk)[0], 3),
        "spearman_fac_wrk": round(spearmanr(fac, wrk)[0], 3),
        "eigenvalues": np.round(np.linalg.eigvalsh(corr_p), 4).tolist(),
        "realised_influence_r2": dict(zip(["pm25", "facilities", "workers", "access"],
                                          [round(pearsonr(norm[:, j], cshri)[0] ** 2, 3) for j in range(4)])),
        "cronbach_alpha": round(alpha, 3),
        "distinct_pm25_values": int(len(np.unique(np.round(pm, 2)))),
    })
    pd.DataFrame(np.triu(corr_p) + np.tril(corr_s, -1), index=["PM2.5", "Facilities", "Workers", "Access"],
                 columns=["PM2.5", "Facilities", "Workers", "Access"]).round(3).to_csv(OUT / "correlations.csv")

    dist = [haversine(r_.lat, r_.lon, cities.Latitude.values, cities.Longitude.values).min() for r_ in d.itertuples()]
    summary["sites_on_gazetteer_points"] = int(sum(x < 0.05 for x in dist))
    pairs = []
    for i in range(len(d)):
        for j in range(i + 1, len(d)):
            km = haversine(d.lat[i], d.lon[i], d.lat[j], d.lon[j])
            pairs.append((d.site_name[i], d.site_name[j], round(km, 1), round(circle_overlap_pct(km))))
    ov = pd.DataFrame(pairs, columns=["site_a", "site_b", "distance_km", "shared_area_pct"])
    ov.sort_values("shared_area_pct", ascending=False).head(6).to_csv(OUT / "neighbourhood_overlap.csv", index=False)

    # ---- 5.4 revised framework -------------------------------------------------------
    e, h = revised_pillars(pm, fac)
    r = aggregate(e, h)
    geo = aggregate(e, h, how="geo")
    pareto = [not any((e[j] >= e[i]) & (h[j] >= h[i]) & ((e[j] > e[i]) | (h[j] > h[i]))
                      for j in range(len(d)) if j != i) for i in range(len(d))]
    table = pd.DataFrame({
        "site_id": d.site_id, "site_name": d.site_name, "facilities": fac, "workers_000": wrk.round(1),
        "pm25": pm.round(2), "access_km": acc, "cshri_pipeline": d.cshri, "cshri_recomputed": cshri.round(1),
        "E": e.round(1), "H": h.round(1), "R": r.round(1),
        "rank_original": ranks(cshri), "rank_revised": ranks(r), "pareto_front": pareto,
    }).sort_values("R", ascending=False)
    table.to_csv(OUT / "table5_components_scores.csv", index=False)
    summary.update({
        "spearman_original_vs_revised": round(spearmanr(cshri, r)[0], 3),
        "spearman_arith_vs_geo": round(spearmanr(r, geo)[0], 3),
        "max_rank_shift": int(np.abs(ranks(cshri) - ranks(r)).max()),
        "pareto_front": d.site_name[pareto].tolist(),
    })

    # ---- 5.5 Monte Carlo -------------------------------------------------------------
    n_sites = len(d)
    w = rng.dirichlet(np.ones(4), N_DRAWS)
    rk_orig = np.array([ranks(norm @ wi) for wi in w])
    groups, g_idx = np.unique(series, return_inverse=True)
    wrk_eq = wrk / (wrk.sum() / fac.sum())
    f = {
        "w_E": rng.uniform(0.2, 0.8, N_DRAWS),
        "pm_error": rng.normal(0, 0.20, (N_DRAWS, len(groups))),
        "H_indicator": rng.integers(0, 2, N_DRAWS),      # 0 facilities, 1 workers
        "H_transform": rng.integers(0, 2, N_DRAWS),      # 0 log, 1 linear
        "normalisation": rng.integers(0, 2, N_DRAWS),    # 0 fixed, 1 min-max
        "aggregation": rng.integers(0, 2, N_DRAWS),      # 0 arithmetic, 1 geometric
    }

    def run(sigma_scale=1.0):
        rk, sc = np.empty((N_DRAWS, n_sites), int), np.empty((N_DRAWS, n_sites))
        for t in range(N_DRAWS):
            pm_t = pm * np.exp(sigma_scale * f["pm_error"][t, g_idx])
            n_t = wrk_eq if f["H_indicator"][t] else fac
            e_t, h_t = revised_pillars(pm_t, n_t, "lin" if f["H_transform"][t] else "log",
                                       "minmax" if f["normalisation"][t] else "fixed")
            sc[t] = aggregate(e_t, h_t, f["w_E"][t], "geo" if f["aggregation"][t] else "arith")
            rk[t] = ranks(sc[t])
        return rk, sc

    rk_rev, sc_rev = run()

    def summarise(rk):
        return pd.DataFrame({
            "rank_median": np.median(rk, 0), "rank_p05": np.percentile(rk, 5, 0), "rank_p95": np.percentile(rk, 95, 0),
            "p_upper": (rk <= 4).mean(0).round(3), "p_lower": (rk >= 9).mean(0).round(3)})

    mc = pd.concat([d[["site_id", "site_name"]], pd.Series(r.round(1), name="R"),
                    pd.Series(np.percentile(sc_rev, 5, 0).round(1), name="R_p05"),
                    pd.Series(np.percentile(sc_rev, 95, 0).round(1), name="R_p95"),
                    summarise(rk_rev).add_suffix("_revised"), summarise(rk_orig).add_suffix("_original")], axis=1)
    mc.sort_values("R", ascending=False).to_csv(OUT / "table7_monte_carlo.csv", index=False)

    def first_order(rk, x, bins=20):
        x = np.asarray(x)
        key = pd.qcut(x, bins, labels=False, duplicates="drop") if np.unique(x).size > 2 else x
        shares = []
        for s in range(rk.shape[1]):
            v = rk[:, s].var()
            if v == 0:
                continue
            cond = pd.Series(rk[:, s]).groupby(key).transform("mean")
            shares.append(cond.var(ddof=0) / v)
        return float(np.mean(shares))

    shares = {name: first_order(rk_rev, f[name]) for name in ["w_E", "H_indicator", "H_transform", "normalisation", "aggregation"]}
    # PM2.5 error: for each site, condition on the error of the series that site draws on
    pm_sh = []
    for s_ in range(n_sites):
        v = rk_rev[:, s_].var()
        if v:
            key = pd.qcut(f["pm_error"][:, g_idx[s_]], 20, labels=False)
            pm_sh.append(pd.Series(rk_rev[:, s_]).groupby(key).transform("mean").var(ddof=0) / v)
    shares["pm_error (data)"] = float(np.mean(pm_sh))
    shares["interactions (remainder)"] = max(0.0, 1 - sum(shares.values()))
    pd.Series(shares, name="first_order_share").round(3).to_csv(OUT / "first_order_shares.csv")
    summary["first_order_shares"] = {k_: round(v, 3) for k_, v in shares.items()}

    sweep = {}
    for sig in (0.0, 0.10, 0.20, 0.30):
        rk_s, _ = run(sigma_scale=sig / 0.20)
        sweep[sig] = (rk_s <= 4).mean(0).round(3)
    pd.DataFrame(sweep, index=d.site_name).to_csv(OUT / "pm_error_sweep_p_upper.csv")

    # ---- 5.6 candidate-set dependence ------------------------------------------------
    loo = []
    em, hm = revised_pillars(pm, fac, norm="minmax")
    for i in range(n_sites):
        keep = np.arange(n_sites) != i
        o = original_index(pm[keep], fac[keep], wrk[keep], acc[keep])
        ek, hk = revised_pillars(pm[keep], fac[keep], norm="minmax")
        ef, hf = revised_pillars(pm[keep], fac[keep])
        loo.append((d.site_name[i], np.abs(o - cshri[keep]).max().round(2),
                    np.abs(aggregate(ek, hk) - aggregate(em, hm)[keep]).max().round(2),
                    np.abs(aggregate(ef, hf) - r[keep]).max().round(2)))
    loo = pd.DataFrame(loo, columns=["dropped_site", "original_max_shift", "revised_minmax_max_shift", "revised_fixed_max_shift"])
    loo.to_csv(OUT / "leave_one_out.csv", index=False)
    summary["loo_max_shift"] = {c: float(loo[c].max()) for c in loo.columns[1:]}

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
