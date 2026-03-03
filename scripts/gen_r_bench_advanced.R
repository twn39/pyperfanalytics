library(PerformanceAnalytics)
data(managers)
data(edhec)

# Hurst Index
hurst = HurstIndex(managers[, 1:6])
write.csv(hurst, "tests/benchmarks/hurst_index.csv")

# Smoothing Index
smoothing = SmoothingIndex(managers[, 1:6])
write.csv(smoothing, "tests/benchmarks/smoothing_index.csv")

# Prob Outperformance
prob_out = table.ProbOutPerformance(edhec[, 1], edhec[, 2])
write.csv(prob_out, "tests/benchmarks/table_prob_outperformance.csv", row.names=FALSE)

# Rolling Periods
rolling = table.RollingPeriods(edhec[, 1:6], periods=c(12, 24, 36))
write.csv(rolling, "tests/benchmarks/table_rolling_periods.csv")

# to.period.contributions
# We need a portfolio contribution series
weights = c(.05, .1, .3, .4, .15)
res_qtr_rebal = Return.portfolio(managers["2002::", 1:5], weights=weights, rebalance_on="quarters", verbose=TRUE)
contrib = res_qtr_rebal$contribution
period_contrib = to.period.contributions(contrib, period="years")
write.csv(period_contrib, "tests/benchmarks/to_period_contributions.csv")

# Also export the input contribution data with proper indices
write.csv(as.data.frame(contrib), "tests/benchmarks/input_contribution.csv")
