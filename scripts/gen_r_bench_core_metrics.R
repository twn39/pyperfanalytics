library(PerformanceAnalytics)
library(jsonlite)

# Load data
df <- read.csv("data/test_data.csv", row.names=1)
# Convert to xts
data_xts <- as.xts(df, order.by=as.Date(rownames(df)))

# Assets: AGG, GLD, SPY
# Rf: BIL
Ra <- data_xts[, c("AGG", "GLD", "SPY")]
Rb <- data_xts[, "SPY"]
Rf <- data_xts[, "BIL"]

metrics <- list()

for (col in colnames(Ra)) {
    asset_r <- Ra[, col]
    rf_r <- Rf
    rf_mean <- mean(rf_r)
    
    # Basic
    metrics[[col]]$Return.annualized <- as.numeric(Return.annualized(asset_r))
    metrics[[col]]$StdDev.annualized <- as.numeric(StdDev.annualized(asset_r))
    
    # Ratios
    # Use mean Rf where functions are sensitive to vector Rf or for consistency in this test
    metrics[[col]]$SharpeRatio <- as.numeric(SharpeRatio(asset_r, Rf=rf_r, FUN="StdDev", annualize=TRUE))
    metrics[[col]]$SortinoRatio <- as.numeric(SortinoRatio(asset_r, MAR=rf_mean))
    metrics[[col]]$OmegaRatio <- as.numeric(Omega(asset_r, L=rf_r))
    metrics[[col]]$RachevRatio <- as.numeric(RachevRatio(asset_r))
    # Note: ProspectRatio in PA has a bug with vector MAR. Using scalar mean(Rf).
    metrics[[col]]$ProspectRatio <- as.numeric(ProspectRatio(asset_r, MAR=rf_mean))
    metrics[[col]]$AdjustedSharpeRatio <- as.numeric(AdjustedSharpeRatio(asset_r))
    
    # Risk
    metrics[[col]]$VaR.Hist <- as.numeric(VaR(asset_r, p=0.95, method="historical"))
    metrics[[col]]$VaR.Gaus <- as.numeric(VaR(asset_r, p=0.95, method="gaussian"))
    metrics[[col]]$VaR.Mod <- as.numeric(VaR(asset_r, p=0.95, method="modified"))
    
    metrics[[col]]$ES.Hist <- as.numeric(ES(asset_r, p=0.95, method="historical"))
    metrics[[col]]$ES.Gaus <- as.numeric(ES(asset_r, p=0.95, method="gaussian"))
    metrics[[col]]$ES.Mod <- as.numeric(ES(asset_r, p=0.95, method="modified"))
    
    metrics[[col]]$MaxDrawdown <- as.numeric(maxDrawdown(asset_r))
    metrics[[col]]$UlcerIndex <- as.numeric(UlcerIndex(asset_r))
    
    # Attribution
    metrics[[col]]$CAPM.beta <- as.numeric(CAPM.beta(asset_r, Rb, Rf=rf_r))
    metrics[[col]]$CAPM.alpha <- as.numeric(CAPM.alpha(asset_r, Rb, Rf=rf_r))
    metrics[[col]]$SpecificRisk <- as.numeric(SpecificRisk(asset_r, Rb, Rf=rf_r))
    metrics[[col]]$SystematicRisk <- as.numeric(SystematicRisk(asset_r, Rb, Rf=rf_r))

    # New Metrics
    metrics[[col]]$BurkeRatio <- as.numeric(BurkeRatio(asset_r, Rf=rf_mean))
    metrics[[col]]$ModifiedBurkeRatio <- as.numeric(BurkeRatio(asset_r, Rf=rf_mean, modified=TRUE))
    metrics[[col]]$Modigliani <- as.numeric(Modigliani(asset_r, Rb, Rf=rf_mean))
    metrics[[col]]$FamaBeta <- as.numeric(FamaBeta(asset_r, Rb))

    metrics[[col]]$MeanAbsoluteDeviation <- as.numeric(MeanAbsoluteDeviation(asset_r))
    metrics[[col]]$DownsideFrequency <- as.numeric(DownsideFrequency(asset_r, MAR=rf_mean))
    metrics[[col]]$M2Sortino <- as.numeric(M2Sortino(asset_r, Rb, MAR=rf_mean))
    metrics[[col]]$MSquared <- as.numeric(MSquared(asset_r, Rb, Rf=rf_mean))
    metrics[[col]]$MSquaredExcess_geom <- as.numeric(MSquaredExcess(asset_r, Rb, Rf=rf_mean, Method="geometric"))
    metrics[[col]]$MSquaredExcess_arith <- as.numeric(MSquaredExcess(asset_r, Rb, Rf=rf_mean, Method="arithmetic"))
    metrics[[col]]$NetSelectivity <- as.numeric(NetSelectivity(asset_r, Rb, Rf=rf_mean))
    metrics[[col]]$OmegaExcessReturn <- as.numeric(OmegaExcessReturn(asset_r, Rb, MAR=rf_mean))
    metrics[[col]]$OmegaSharpeRatio <- as.numeric(OmegaSharpeRatio(asset_r, MAR=rf_mean))
    metrics[[col]]$DownsideSharpeRatio <- as.numeric(DownsideSharpeRatio(asset_r, rf=rf_mean))
    metrics[[col]]$Return.cumulative <- as.numeric(Return.cumulative(asset_r))
    metrics[[col]]$Kappa <- as.numeric(Kappa(asset_r, MAR=rf_mean, l=2))
    metrics[[col]]$AnnualizedExcessReturn <- as.numeric(Return.annualized.excess(asset_r, Rb))
    
    metrics[[col]]$AverageDrawdown <- as.numeric(AverageDrawdown(asset_r))
    metrics[[col]]$AverageLength <- as.numeric(AverageLength(asset_r))
    metrics[[col]]$AverageRecovery <- as.numeric(AverageRecovery(asset_r))
    metrics[[col]]$DrawdownDeviation <- as.numeric(DrawdownDeviation(asset_r))
    metrics[[col]]$CDD <- as.numeric(CDD(asset_r, p=0.95))
    metrics[[col]]$CDaR.beta <- as.numeric(CDaR.beta(asset_r, Rb, p=0.95))
    
    # R's CDaR.alpha hardcodes 12 for annualization. We calculate manually with scale=252.
    cdar_b <- as.numeric(CDaR.beta(asset_r, Rb, p=0.95))
    ra_exp <- (1 + mean(asset_r))^252 - 1
    rb_exp <- (1 + mean(Rb))^252 - 1
    metrics[[col]]$CDaR.alpha <- as.numeric(ra_exp - cdar_b * rb_exp)
    
    # Phase 10 Metrics
    refSR_val <- 0.0
    if (col == "AGG") refSR_val <- -0.01
    
    metrics[[col]]$ProbSharpeRatio <- as.numeric(ProbSharpeRatio(asset_r, Rf=rf_r, refSR=refSR_val)$sr_prob)
    metrics[[col]]$MinTrackRecord <- as.numeric(MinTrackRecord(asset_r, refSR=refSR_val, Rf=rf_mean)$min_TRL)
    metrics[[col]]$HerfindahlIndex <- as.numeric(PerformanceAnalytics:::HerfindahlIndex(asset_r))
    
    scale <- 252
    metrics[[col]]$table.InformationRatio <- list(
        "Tracking Error" = as.numeric(TrackingError(asset_r, Rb)/sqrt(scale)),
        "Annualised Tracking Error" = as.numeric(TrackingError(asset_r, Rb)),
        "Information Ratio" = as.numeric(InformationRatio(asset_r, Rb))
    )
    
    sys_val <- as.numeric(SystematicRisk(asset_r, Rb, Rf=rf_r))
    spec_val <- as.numeric(SpecificRisk(asset_r, Rb, Rf=rf_r))
    metrics[[col]]$table.SpecificRisk <- list(
        "Specific Risk" = spec_val,
        "Systematic Risk" = sys_val,
        "Total Risk" = sqrt(sys_val^2 + spec_val^2)
    )
}

# Save to JSON
write_json(metrics, "data/r_benchmarks.json", auto_unbox=TRUE, pretty=TRUE, digits=NA)
cat("Benchmarks saved to data/r_benchmarks.json\n")
