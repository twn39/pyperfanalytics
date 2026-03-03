suppressPackageStartupMessages({
  library(PerformanceAnalytics)
  library(xts)
})

df <- read.csv("data/yfinance_etfs.csv", row.names=1)
dates <- as.Date(rownames(df))
data_xts <- xts(df, order.by=dates)

weights <- c(0.4, 0.4, 0.2)

b1 <- Return.portfolio(data_xts, weights=weights, geometric=TRUE, rebalance_on=NA)
write.csv(as.data.frame(b1), "tests/benchmarks/yf_port_geom_none.csv")

b2 <- Return.portfolio(data_xts, weights=weights, geometric=FALSE, rebalance_on=NA)
write.csv(as.data.frame(b2), "tests/benchmarks/yf_port_arith_none.csv")

b3 <- Return.portfolio(data_xts, weights=weights, geometric=TRUE, rebalance_on="years")
write.csv(as.data.frame(b3), "tests/benchmarks/yf_port_geom_years.csv")

cat("Successfully generated yfinance R benchmarks.\n")
