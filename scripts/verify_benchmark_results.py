import json

import numpy as np
import pandas as pd

import pyperfanalytics as pa


def compare():
    # Load data
    df = pd.read_csv("data/test_data.csv", index_col=0, parse_dates=True)

    # Load R benchmarks
    with open("data/r_benchmarks.json") as f:
        r_bench = json.load(f)

    assets = ["AGG", "GLD", "SPY"]
    rf = df["BIL"]
    rb = df["SPY"]

    comparison_results = []

    for asset in assets:
        ra = df[asset]
        bench = r_bench[asset]

        # Calculate Python metrics
        py_metrics = {
            "Return.annualized": pa.return_annualized(ra),
            "StdDev.annualized": pa.std_dev_annualized(ra),
            "SharpeRatio": pa.sharpe_ratio(ra, Rf=rf, annualize=True),
            "SortinoRatio": pa.sortino_ratio(ra, MAR=rf),
            "OmegaRatio": pa.omega_ratio(ra, L=rf),
            "RachevRatio": pa.rachev_ratio(ra),
            # Match R script workaround: use mean Rf as scalar MAR
            "ProspectRatio": pa.prospect_ratio(ra, MAR=rf.mean()),
            "AdjustedSharpeRatio": pa.adjusted_sharpe_ratio(ra),
            # Negate VaR/ES to match R's sign convention (loss = negative)
            "VaR.Hist": -pa.var_historical(ra, p=0.95),
            "VaR.Gaus": -pa.var_gaussian(ra, p=0.95),
            "VaR.Mod": -pa.var_modified(ra, p=0.95),
            "ES.Hist": -pa.es_historical(ra, p=0.95),
            "ES.Gaus": -pa.es_gaussian(ra, p=0.95),
            "ES.Mod": -pa.es_modified(ra, p=0.95),
            "MaxDrawdown": pa.max_drawdown(ra),
            "UlcerIndex": pa.ulcer_index(ra),
            "CAPM.beta": pa.capm_beta(ra, rb, Rf=rf),
            # R's CAPM.alpha is periodic, not annualized
            "CAPM.alpha": pa.capm_alpha(ra, rb, Rf=rf),
            "SpecificRisk": pa.specific_risk(ra, rb, Rf=rf),
            "SystematicRisk": pa.systematic_risk(ra, rb, Rf=rf),
        }

        for metric, r_val in bench.items():
            py_val = py_metrics.get(metric)
            diff = abs(py_val - r_val) if py_val is not None else np.nan
            status = "✅" if diff < 1e-4 else "❌"  # Use 1e-4 for diverse data (float diffs)

            comparison_results.append(
                {"Asset": asset, "Metric": metric, "Python": py_val, "R": r_val, "Diff": diff, "Status": status}
            )

    res_df = pd.DataFrame(comparison_results)
    print(res_df.to_string(index=False))

    failed = res_df[res_df["Status"] == "❌"]
    if not failed.empty:
        print(f"\nDiscrepancies found in {len(failed)} metrics!")
        # print(failed)
    else:
        print("\nAll cross-verification checks passed! ✅")


if __name__ == "__main__":
    compare()
