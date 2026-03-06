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

# 2. Data Selection: test_data_v3.csv (Daily Returns)
data_file = "data/test_data_v3.csv"
df = pd.read_csv(data_file, index_col=0, parse_dates=True)

# Select primary assets: NVDA (High Vol) and QQQ (Benchmark)
strategies = ["NVDA", "QQQ"]
data = df[strategies].dropna()

# Notable dates for Event Study
# 2020-03-23: COVID bottom
# 2022-01-03: Market peak
event_dates = ["2020-03-23", "2022-01-03"]

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
    sub_data <- xts_data[, c("NVDA", "QQQ")]
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

# --- Tasks updated for Daily Returns (width increased for daily scale) ---
tasks = [
    ("performance_summary", lambda: pa.charts_performance_summary(data), "charts.PerformanceSummary(sub_data)"),
    ("cum_returns", lambda: pa.chart_cum_returns(data), "chart.CumReturns(sub_data)"),
    ("bar_returns", lambda: pa.chart_bar_returns(data), "chart.Bar(sub_data)"),
    ("drawdown", lambda: pa.chart_drawdown(data), "chart.Drawdown(sub_data)"),
    ("relative_performance", lambda: pa.chart_relative_performance(data[["NVDA"]], data[["QQQ"]]), "chart.RelativePerformance(sub_data[,1], sub_data[,2])"),
    ("capture_ratios", lambda: pa.chart_capture_ratios(df[["NVDA", "AMD", "TSLA"]], df[["QQQ"]]), "chart.CaptureRatios(xts_data[,c('NVDA','AMD','TSLA')], xts_data[,'QQQ'])"),
    ("rolling_performance", lambda: pa.chart_rolling_performance(data, width=252), "chart.RollingPerformance(sub_data, width=252)"),
    ("bar_var_nvda", lambda: pa.chart_bar_var(data[["NVDA"]], width=252), "chart.BarVaR(sub_data[,1,drop=FALSE], width=252)"),
    ("var_sensitivity_nvda", lambda: pa.chart_var_sensitivity(data[["NVDA"]]), "chart.VaRSensitivity(sub_data[,1,drop=FALSE])"),
    ("rolling_regression_beta", lambda: pa.chart_rolling_regression(data[["NVDA"]], data[["QQQ"]], width=252, attribute="Beta"), "chart.RollingRegression(sub_data[,1], sub_data[,2], width=252, attribute='Beta')"),
    ("charts_rolling_regression", lambda: pa.charts_rolling_regression(data[["NVDA"]], data[["QQQ"]], width=252), "charts.RollingRegression(sub_data[,1], sub_data[,2], width=252)"),
    ("rolling_correlation", lambda: pa.chart_rolling_correlation(data[["NVDA"]], data[["QQQ"]], width=252), "chart.RollingCorrelation(sub_data[,1], sub_data[,2], width=252)"),
    ("risk_return_scatter", lambda: pa.chart_risk_return_scatter(df.drop(columns=['BIL']), Rf=0.01/252), "chart.RiskReturnScatter(xts_data[,colnames(xts_data)!='BIL'], Rf=0.01/252)"),
    ("histogram", lambda: pa.chart_histogram(data[["NVDA"]]), "chart.Histogram(sub_data[,1], methods=c('add.density', 'add.normal', 'add.risk'))"),
    ("boxplot", lambda: pa.chart_boxplot(df.drop(columns=['BIL']), sort_by='mean'), "chart.Boxplot(xts_data[,colnames(xts_data)!='BIL'], sort.by='mean')"),
    ("qqplot", lambda: pa.chart_qqplot(data[["NVDA"]]), "chart.QQPlot(sub_data[,1])"),
    ("acf_plus", lambda: pa.chart_acf_plus(data[["NVDA"]]), "chart.ACFplus(sub_data[,1])"),
    ("events", lambda: pa.chart_events(data[["NVDA"]], dates=event_dates, prior=60, post=60), "chart.Events(sub_data[,1,drop=FALSE], dates=c('2020-03-23', '2022-01-03'), prior=60, post=60)"),
    ("snail_trail", lambda: pa.chart_snail_trail(data, width=252, stepsize=63), "chart.SnailTrail(sub_data, width=252, stepsize=63)"),
    ("stacked_bar", lambda: pa.chart_stacked_bar(df.iloc[:100, [0,3,5]]), "chart.StackedBar(xts_data[1:100, c(1,4,6)])"),
    ("ecdf", lambda: pa.chart_ecdf(data[["NVDA"]]), "chart.ECDF(sub_data[,1])"),
    ("scatter", lambda: pa.chart_scatter(data[["NVDA"]], data[["QQQ"]]), "chart.Scatter(sub_data[,1], sub_data[,2])"),
]

print(f"Starting generation of {len(tasks)} pairs using {data_file}...")
for name, py_func, r_cmd in tasks:
    generate_pair(name, py_func, r_cmd)

print(f"\nFinal count in images/: {len(os.listdir(output_dir))}")
