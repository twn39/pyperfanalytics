invisible(Sys.setlocale("LC_TIME", "C"))
#!/usr/bin/env Rscript
# Generate R benchmarks for tables.py regression tests.
# Covers: table.CAPM, table.DownsideRisk, table.CaptureRatios, table.UpDownRatios,
#         table.CalendarReturns, table.HigherMoments, table.ProbSummary (skewness/kurtosis),
#         table.Variability, table.Drawdowns, table.ProbSharpeRatio, table.RollingPeriods
# Run with: Rscript scripts/gen_tables_benchmarks.R

library(PerformanceAnalytics)
library(jsonlite)

# --- Load managers dataset (same as used in existing tests) ---
managers <- read.csv("third_party/PerformanceAnalytics/data/managers.csv",
                     row.names = 1)
# Keep rows matching 1996-2006 (same slice used in test_r_comp_phase3.py)
row_dates <- as.Date(rownames(managers))
managers <- managers[row_dates >= as.Date("1996-01-01") & row_dates <= as.Date("2006-12-31"), ]
managers_idx <- row_dates[row_dates >= as.Date("1996-01-01") & row_dates <= as.Date("2006-12-31")]
managers_xts <- xts::xts(managers, order.by = managers_idx)

Ra <- managers_xts[, 1:6]       # HAM1..HAM6
Rb <- managers_xts[, 8, drop=FALSE]   # SP500 TR
Rf <- managers_xts[, 10, drop=FALSE]  # Rf column

results <- list()

# ---- table.CAPM ----
cat("table.CAPM\n", file = stderr())
tc <- table.CAPM(Ra[, 1:3], Rb, scale = 12, Rf = 0)
results$table_capm <- as.list(as.data.frame(tc))
# Store row names
results$table_capm_rownames <- rownames(tc)
results$table_capm_colnames <- colnames(tc)

# ---- table.DownsideRisk ----
cat("table.DownsideRisk\n", file = stderr())
tdr <- table.DownsideRisk(Ra[, 1:3], scale = 12, Rf = 0)
results$table_downside_risk <- as.list(as.data.frame(tdr))
results$table_downside_risk_rownames <- rownames(tdr)
results$table_downside_risk_colnames <- colnames(tdr)

# ---- table.CaptureRatios ----
cat("table.CaptureRatios\n", file = stderr())
tcr <- table.CaptureRatios(Ra[, 1:3], Rb)
results$table_capture_ratios <- as.list(as.data.frame(tcr))
results$table_capture_ratios_rownames <- rownames(tcr)
results$table_capture_ratios_colnames <- colnames(tcr)

# ---- table.UpDownRatios ----
cat("table.UpDownRatios\n", file = stderr())
tudr <- table.UpDownRatios(Ra[, 1:3], Rb)
results$table_up_down_ratios <- as.list(as.data.frame(tudr))
results$table_up_down_ratios_rownames <- rownames(tudr)
results$table_up_down_ratios_colnames <- colnames(tudr)

# ---- table.CalendarReturns (HAM1 only) ----
cat("table.CalendarReturns\n", file = stderr())
# R's table.CalendarReturns expects a single series
tcal <- table.CalendarReturns(Ra[, 1, drop=FALSE])
# Convert to a flat dict: year -> {month -> val}
cal_df <- as.data.frame(tcal)
results$table_calendar_returns <- lapply(seq_len(nrow(cal_df)), function(i) {
  as.list(cal_df[i, ])
})
results$table_calendar_returns_rownames <- rownames(tcal)
results$table_calendar_returns_colnames <- colnames(tcal)

# ---- table.HigherMoments ----
cat("table.HigherMoments\n", file = stderr())
thm <- table.HigherMoments(Ra[, 1:3], Rb)
results$table_higher_moments <- as.list(as.data.frame(thm))
results$table_higher_moments_rownames <- rownames(thm)
results$table_higher_moments_colnames <- colnames(thm)

# ---- table.ProbSummary (skewness/kurtosis) ----
cat("table.ProbSummary (skewness/kurtosis methods)\n", file = stderr())
# R equivalent: manual calculation using skewness() and kurtosis() with different methods
# Our Python table_prob_skewness_kurtosis calls 8 method combinations
cols_for_sk <- colnames(Ra[, 1:3])
sk_result <- list()
for (col in cols_for_sk) {
  x <- na.omit(Ra[, col])
  sk_result[[col]] <- list(
    skewness_moment  = as.numeric(skewness(x, method = "moment")),
    skewness_fisher  = as.numeric(skewness(x, method = "fisher")),
    skewness_sample  = as.numeric(skewness(x, method = "sample")),
    kurtosis_moment  = as.numeric(kurtosis(x, method = "moment")),
    kurtosis_fisher  = as.numeric(kurtosis(x, method = "fisher")),
    kurtosis_sample  = as.numeric(kurtosis(x, method = "sample")),
    kurtosis_excess  = as.numeric(kurtosis(x, method = "excess")),
    kurtosis_sample_excess = as.numeric(kurtosis(x, method = "sample_excess"))
  )
}
results$table_prob_summary <- sk_result

# ---- table.Variability: skipped (R's table.Variability has a bug: 'freq' not found) ----
# Covered separately via individual function tests.

# ---- table.Drawdowns (top 5 for HAM1) ----
cat("table.Drawdowns\n", file = stderr())
td <- table.Drawdowns(Ra[, 1, drop=FALSE], top = 5)
td_list <- lapply(seq_len(nrow(td)), function(i) {
  row <- as.list(td[i, ])
  row$From <- as.character(row$From)
  row$Trough <- as.character(row$Trough)
  row$To <- if (is.na(row$To)) NULL else as.character(row$To)
  row$Depth <- as.numeric(row$Depth)
  row$Length <- as.integer(row$Length)
  row$`To Trough` <- as.integer(row$`To Trough`)
  row$Recovery <- if (is.na(row$Recovery)) NULL else as.integer(row$Recovery)
  row
})
results$table_drawdowns <- td_list

# ---- table.Variability: manual benchmark via component functions ----
cat("table.Variability (manual)\n", file = stderr())
cols_v <- colnames(Ra[, 1:3])
var_result <- list()
for (col in cols_v) {
  x <- na.omit(Ra[, col])
  ann_std  <- as.numeric(StdDev.annualized(x, scale = 12))
  per_std  <- ann_std / sqrt(12)
  mad_val  <- as.numeric(MeanAbsoluteDeviation(x))
  var_result[[col]] <- list(
    MeanAbsoluteDeviation = mad_val,
    PeriodStdDev          = per_std,
    AnnualizedStdDev      = ann_std
  )
}
results$table_variability_manual <- var_result

cat(toJSON(results, null = "null", na = "null", auto_unbox = TRUE, pretty = TRUE, digits = NA))
