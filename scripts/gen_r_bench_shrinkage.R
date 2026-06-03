library(PerformanceAnalytics)
library(jsonlite)

# Load EDHEC dataset
data_path <- "data/edhec.csv"
edhec <- read.csv(data_path, row.names = 1, check.names = FALSE)
# Convert row names to Date if possible (not strictly needed, but coredata helps)
R_full <- as.matrix(edhec)

# We use 3 assets for R and 1 factor
R <- R_full[, 1:3]
f_single <- R_full[, 4, drop = FALSE]
f_multi <- R_full[, 4:5]

# Output container
results <- list()

# M2.struct targets
results$m2_struct_Indep <- M2.struct(R, "Indep")
results$m2_struct_IndepId <- M2.struct(R, "IndepId")
results$m2_struct_observedfactor_single <- M2.struct(R, "observedfactor", f = f_single)
results$m2_struct_observedfactor_multi <- M2.struct(R, "observedfactor", f = f_multi)
results$m2_struct_CC <- M2.struct(R, "CC")

# M3.struct targets (as vector for precision verification, and as matrix)
results$m3_struct_latent1factor <- M3.struct(R, "latent1factor", as.mat = FALSE)
results$m3_struct_Indep <- M3.struct(R, "Indep", as.mat = FALSE)
results$m3_struct_IndepId <- M3.struct(R, "IndepId", as.mat = FALSE)
results$m3_struct_observedfactor_single <- M3.struct(R, "observedfactor", f = f_single, as.mat = FALSE)
results$m3_struct_observedfactor_multi <- M3.struct(R, "observedfactor", f = f_multi, as.mat = FALSE)
results$m3_struct_CC <- M3.struct(R, "CC", as.mat = FALSE)
results$m3_struct_CS <- M3.struct(R, "CS", as.mat = FALSE)

# M3.struct unbiased versions
results$m3_struct_Indep_unbiased <- M3.struct(R, "Indep", unbiasedMarg = TRUE, as.mat = FALSE)
results$m3_struct_IndepId_unbiased <- M3.struct(R, "IndepId", unbiasedMarg = TRUE, as.mat = FALSE)

# M4.struct targets
results$m4_struct_Indep <- M4.struct(R, "Indep", as.mat = FALSE)
results$m4_struct_IndepId <- M4.struct(R, "IndepId", as.mat = FALSE)
results$m4_struct_observedfactor_single <- M4.struct(R, "observedfactor", f = f_single, as.mat = FALSE)
results$m4_struct_observedfactor_multi <- M4.struct(R, "observedfactor", f = f_multi, as.mat = FALSE)
results$m4_struct_CC <- M4.struct(R, "CC", as.mat = FALSE)

# M2.shrink single and multi-targets
m2_sh_1 <- M2.shrink(R, targets = 1)
results$m2_shrink_t1 <- list(M2sh = m2_sh_1$M2sh, lambda = m2_sh_1$lambda)

m2_sh_all <- M2.shrink(R, targets = c(1, 2, 3, 4), f = f_single)
results$m2_shrink_tall <- list(M2sh = m2_sh_all$M2sh, lambda = m2_sh_all$lambda)

# M3.shrink single and multi-targets (with different setups)
m3_sh_t1 <- M3.shrink(R, targets = 1, as.mat = FALSE)
results$m3_shrink_t1 <- list(M3sh = m3_sh_t1$M3sh, lambda = m3_sh_t1$lambda)

m3_sh_tall <- M3.shrink(R, targets = c(1, 2, 3, 4, 5, 6), f = f_single, as.mat = FALSE)
results$m3_shrink_tall <- list(M3sh = m3_sh_tall$M3sh, lambda = m3_sh_tall$lambda)

m3_sh_unbiased <- M3.shrink(R, targets = c(1, 2, 6), unbiasedMSE = TRUE, as.mat = FALSE)
results$m3_shrink_unbiased <- list(M3sh = m3_sh_unbiased$M3sh, lambda = m3_sh_unbiased$lambda)

# M4.shrink single and multi-targets
m4_sh_t1 <- M4.shrink(R, targets = 1, as.mat = FALSE)
results$m4_shrink_t1 <- list(M4sh = m4_sh_t1$M4sh, lambda = m4_sh_t1$lambda)

m4_sh_tall <- M4.shrink(R, targets = c(1, 2, 3, 4), f = f_single, as.mat = FALSE)
results$m4_shrink_tall <- list(M4sh = m4_sh_tall$M4sh, lambda = m4_sh_tall$lambda)

# Write to file
write_json(results, "data/r_benchmarks_shrinkage.json", auto_unbox = TRUE, pretty = TRUE, digits = 16)
cat("Benchmark results written to data/r_benchmarks_shrinkage.json\n")
