#!/usr/bin/env Rscript
# Generate R benchmark values for edge case tests.
# Run with: Rscript scripts/gen_edge_case_benchmarks.R > data/edge_case_benchmarks.json
library(PerformanceAnalytics)
library(jsonlite)

make_xts <- function(vals, start = "2020-01-31", freq = "month") {
  dates <- seq.Date(as.Date(start), by = freq, length.out = length(vals))
  xts::xts(vals, order.by = dates)
}

safe <- function(expr) {
  tryCatch(expr, error = function(e) NULL, warning = function(w) suppressWarnings(expr))
}

# ---- Define edge case series ----
cases <- list(
  all_positive   = c(0.01, 0.02, 0.03, 0.02, 0.01, 0.02, 0.03, 0.01, 0.02, 0.02),
  all_negative   = c(-0.01, -0.02, -0.03, -0.02, -0.01, -0.02, -0.03, -0.01, -0.02, -0.02),
  starts_with_loss = c(-0.05, 0.03, 0.04, -0.02, 0.01, 0.06, -0.01, 0.02, 0.03, -0.01),
  constant_pos   = rep(0.01, 12),
  constant_neg   = rep(-0.01, 12),
  two_obs        = c(0.05, -0.03),
  single_obs     = c(0.05),
  mixed_zeros    = c(0.0, 0.0, 0.01, -0.01, 0.0, 0.02, -0.02, 0.0, 0.01, 0.0),
  high_vol       = c(0.15, -0.12, 0.20, -0.18, 0.10, -0.08, 0.25, -0.20, 0.12, -0.15),
  benchmark_proxy = c(0.01, 0.015, 0.008, 0.012, 0.009, 0.011, 0.013, 0.007, 0.010, 0.012)
)

results <- list()

run_case <- function(name, R_vec, Rb_vec = NULL, scale = 12) {
  R <- make_xts(R_vec)
  if (!is.null(Rb_vec)) {
    Rb <- make_xts(Rb_vec)
  }

  res <- list()

  # Core returns
  res$Return.annualized <- tryCatch(as.numeric(Return.annualized(R, scale = scale)), error = function(e) NA)
  res$Return.cumulative <- tryCatch(as.numeric(Return.cumulative(R)), error = function(e) NA)
  res$StdDev.annualized <- tryCatch(as.numeric(StdDev.annualized(R, scale = scale)), error = function(e) NA)

  # Risk
  res$SharpeRatio.StdDev <- tryCatch(as.numeric(SharpeRatio(R, Rf = 0, FUN = "StdDev", annualize = FALSE)), error = function(e) NA)
  res$DownsideDeviation.0 <- tryCatch(as.numeric(DownsideDeviation(R, MAR = 0)), error = function(e) NA)
  res$SortinoRatio.0 <- tryCatch(as.numeric(SortinoRatio(R, MAR = 0)), error = function(e) NA)
  res$SemiDeviation <- tryCatch(as.numeric(SemiDeviation(R)), error = function(e) NA)

  # VaR / ES
  res$VaR.hist <- tryCatch(as.numeric(VaR(R, p = 0.95, method = "historical")), error = function(e) NA)
  res$VaR.gaus <- tryCatch(as.numeric(VaR(R, p = 0.95, method = "gaussian")), error = function(e) NA)
  res$ES.hist <- tryCatch(as.numeric(ES(R, p = 0.95, method = "historical")), error = function(e) NA)
  res$ES.gaus <- tryCatch(as.numeric(ES(R, p = 0.95, method = "gaussian")), error = function(e) NA)

  # Drawdowns
  res$MaxDrawdown <- tryCatch(as.numeric(maxDrawdown(R)), error = function(e) NA)
  res$UlcerIndex <- tryCatch(as.numeric(UlcerIndex(R)), error = function(e) NA)
  res$PainIndex <- tryCatch(as.numeric(PainIndex(R)), error = function(e) NA)
  res$AverageDrawdown <- tryCatch(as.numeric(AverageDrawdown(R)), error = function(e) NA)

  # Higher moments
  res$skewness.moment <- tryCatch(as.numeric(skewness(R, method = "moment")), error = function(e) NA)
  res$kurtosis.excess <- tryCatch(as.numeric(kurtosis(R, method = "excess")), error = function(e) NA)

  # Attribution (only when we have a benchmark)
  if (!is.null(Rb_vec)) {
    res$CAPM.beta <- tryCatch(as.numeric(CAPM.beta(R, Rb, Rf = 0)), error = function(e) NA)
    res$CAPM.alpha <- tryCatch(as.numeric(CAPM.alpha(R, Rb, Rf = 0)), error = function(e) NA)
    res$TrackingError <- tryCatch(as.numeric(TrackingError(R, Rb, scale = scale)), error = function(e) NA)
    res$ActivePremium <- tryCatch(as.numeric(ActivePremium(R, Rb, scale = scale)), error = function(e) NA)
  }

  res
}

# benchmark proxy (for attribution tests)
bench <- cases$benchmark_proxy

for (nm in names(cases)) {
  cat(sprintf("Processing: %s\n", nm), file = stderr())
  # Only add Rb for cases with enough obs
  has_bench <- length(cases[[nm]]) == length(bench) && nm != "benchmark_proxy"
  results[[nm]] <- run_case(nm, cases[[nm]],
                             Rb_vec = if (has_bench) bench else NULL)
}

# Also save the raw series values for verification
results$`_series` <- lapply(cases, as.list)

cat(toJSON(results, null = "null", na = "null", auto_unbox = TRUE, pretty = TRUE))
