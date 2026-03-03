library(PerformanceAnalytics)
data(managers)

# Function to write results to CSV
write_bench <- function(tab, name) {
  write.csv(tab, paste0("tests/benchmarks/", name, ".csv"), row.names = TRUE)
}

# Ensure directory exists
dir.create("tests/benchmarks", showWarnings = FALSE)

# Generate benchmarks for new tables
tab_auto <- table.Autocorrelation(managers[,1:8])
write_bench(tab_auto, "table_autocorrelation")

tab_corr <- table.Correlation(managers[,1:6], managers[,8,drop=FALSE])
write_bench(tab_corr, "table_correlation")

tab_dist <- table.Distributions(managers[,1:8])
write_bench(tab_dist, "table_distributions")

tab_downside <- table.DownsideRiskRatio(managers[,1:8])
write_bench(tab_downside, "table_downside_risk_ratio")

tab_drawdowns <- table.DrawdownsRatio(managers[,1:8])
write_bench(tab_drawdowns, "table_drawdowns_ratio")

tab_stats <- table.Stats(managers[,1:8])
write_bench(tab_stats, "table_stats")

# Also for specific functions
mad_val <- apply(managers[,1:8], 2, MeanAbsoluteDeviation)
write.csv(data.frame(MAD=mad_val), "tests/benchmarks/mean_absolute_deviation.csv")

ster_val <- apply(managers[,1:8], 2, SterlingRatio)
write.csv(data.frame(SterlingRatio=ster_val), "tests/benchmarks/sterling_ratio.csv")
