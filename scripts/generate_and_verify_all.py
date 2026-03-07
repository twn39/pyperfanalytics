import pandas as pd
import pyperfanalytics as pa
import os
import subprocess
import numpy as np

# 1. Setup
output_dir = "images"
os.makedirs(output_dir, exist_ok=True)
for f in os.listdir(output_dir):
    if f.endswith(".png"):
        os.remove(os.path.join(output_dir, f))

# 2. Data Selection: yfinance_etfs.csv (Monthly Returns)
data_file = "data/yfinance_etfs.csv"
df = pd.read_csv(data_file, index_col=0, parse_dates=True)

# Select primary assets: GLD (Gold) and SPY (S&P 500 Benchmark)
strategies = ["GLD", "SPY"]
data = df[strategies].dropna()

# Notable dates for Event Study (Monthly)
# 2020-03-31: COVID crash month
# 2022-01-31: Market peak before 2022 bear
event_dates = ["2020-03-31", "2022-01-31"]

def generate_pair(name, py_func, r_cmd):
    py_path = os.path.join(output_dir, f"py_{name}.png")
    r_path = os.path.join(output_dir, f"r_{name}.png")
    
    # Generate Python
    try:
        fig = py_func()
        fig.write_image(py_path, width=1200, height=800, scale=2)
        py_status = "OK"
    except Exception as e:
        print(f"Error in Py {name}: {e}")
        py_status = "FAIL"
    
    # Generate R
    full_r_code = f"""
    library(PerformanceAnalytics)
    df <- read.csv("{data_file}", row.names=1, check.names=FALSE)
    xts_data <- as.xts(df, order.by=as.Date(rownames(df)))
    sub_data <- xts_data[, c("GLD", "SPY")]
    png("{r_path}", width=1200, height=800, res=120)
    {r_cmd}
    dev.off()
    """
    subprocess.run(['Rscript', '-e', full_r_code], capture_output=True)
    
    # Verify
    py_exists = os.path.exists(py_path)
    r_exists = os.path.exists(r_path)
    status = "PASS" if py_exists and r_exists else "FAIL"
    print(f"[{status}] {name:30} | Py: {py_status:4} | R: {'OK' if r_exists else 'MISSing'}")

# --- Tasks updated for Monthly Returns (width=12 for annual scale) ---
tasks = [
    ("performance_summary", lambda: pa.charts_performance_summary(data), "charts.PerformanceSummary(sub_data)"),
    ("cum_returns", lambda: pa.chart_cum_returns(data), "chart.CumReturns(sub_data)"),
    ("bar_returns", lambda: pa.chart_bar_returns(data), "chart.Bar(sub_data)"),
    ("drawdown", lambda: pa.chart_drawdown(data), "chart.Drawdown(sub_data)"),
    ("relative_performance", lambda: pa.chart_relative_performance(data[["GLD"]], data[["SPY"]]), "chart.RelativePerformance(sub_data[,1], sub_data[,2])"),
    ("capture_ratios", lambda: pa.chart_capture_ratios(df[["GLD", "TLT"]], df[["SPY"]]), "chart.CaptureRatios(xts_data[,c('GLD','TLT')], xts_data[,'SPY'])"),
    ("rolling_performance", lambda: pa.chart_rolling_performance(data, width=12), "chart.RollingPerformance(sub_data, width=12)"),
    ("bar_var_gld", lambda: pa.chart_bar_var(data[["GLD"]], width=12), "chart.BarVaR(sub_data[,1,drop=FALSE], width=12)"),
    ("var_sensitivity_gld", lambda: pa.chart_var_sensitivity(data[["GLD"]]), "chart.VaRSensitivity(sub_data[,1,drop=FALSE])"),
    ("rolling_regression_beta", lambda: pa.chart_rolling_regression(data[["GLD"]], data[["SPY"]], width=12, attribute="Beta"), "chart.RollingRegression(sub_data[,1], sub_data[,2], width=12, attribute='Beta')"),
    ("charts_rolling_regression", lambda: pa.charts_rolling_regression(data[["GLD"]], data[["SPY"]], width=12), "charts.RollingRegression(sub_data[,1], sub_data[,2], width=12)"),
    ("rolling_correlation", lambda: pa.chart_rolling_correlation(data[["GLD"]], data[["SPY"]], width=12), "chart.RollingCorrelation(sub_data[,1], sub_data[,2], width=12)"),
    ("risk_return_scatter", lambda: pa.chart_risk_return_scatter(df, Rf=0.01/12), "chart.RiskReturnScatter(xts_data, Rf=0.01/12)"),
    ("histogram", lambda: pa.chart_histogram(data[["GLD"]]), "chart.Histogram(sub_data[,1], methods=c('add.density', 'add.normal', 'add.risk'))"),
    ("boxplot", lambda: pa.chart_boxplot(df, sort_by='mean'), "chart.Boxplot(xts_data, sort.by='mean')"),
    ("qqplot", lambda: pa.chart_qqplot(data[["GLD"]]), "chart.QQPlot(sub_data[,1])"),
    ("acf_plus", lambda: pa.chart_acf_plus(data[["GLD"]]), "chart.ACFplus(sub_data[,1])"),
    ("events", lambda: pa.chart_events(data[["GLD"]], dates=event_dates, prior=12, post=12), "chart.Events(sub_data[,1,drop=FALSE], dates=c('2020-03-31', '2022-01-31'), prior=12, post=12)"),
    ("snail_trail", lambda: pa.chart_snail_trail(data, width=12, stepsize=3), "chart.SnailTrail(sub_data, width=12, stepsize=3)"),
    ("stacked_bar", lambda: pa.chart_stacked_bar(df.iloc[:60, :]), "chart.StackedBar(xts_data[1:60,])"),
    ("ecdf", lambda: pa.chart_ecdf(data[["GLD"]]), "chart.ECDF(sub_data[,1])"),
    ("scatter", lambda: pa.chart_scatter(data[["GLD"]], data[["SPY"]]), "chart.Scatter(sub_data[,1], sub_data[,2])"),
]

print(f"Starting generation of {len(tasks)} pairs using {data_file}...")
for name, py_func, r_cmd in tasks:
    generate_pair(name, py_func, r_cmd)

print(f"\nFinal count in images/: {len(os.listdir(output_dir))}")
