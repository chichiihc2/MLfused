# Parallel simulation driver for MLfused
# Run with: Rscript run_simulation.r

## ------------------------------------------------------------
## Libraries
## ------------------------------------------------------------
library(MLfused)
library(parallel)
library(dplyr)
library(tidyverse)
library(readr)
library(pbapply)

source(system.file("scripts", "simulation.r", package = "MLfused"))

name       <- "simulation_fixed_coef"
date_stamp <- format(Sys.Date(), "%Y%m%d")

out_dir <- file.path("results", paste0(name, "_", date_stamp))

# --- REMOVE OLD RESULTS (CLEAN START) ---
if (dir.exists(out_dir)) {
  cat("Removing old 'results' folder...\n")
  unlink(out_dir, recursive = TRUE)
}
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

## ------------------------------------------------------------
## Global Parameters & Grids
## ------------------------------------------------------------
n2 <- 10000
p  <- 4
K  <- 3
L  <- 2
beta_true  <- c(0.2, -0.1)
alpha_true <- c(0.35, -0.25)

# Fixed coefficients (not randomly drawn)
Theta_true <- matrix(c(1, -1, -1, 1, 1, 1, -1, 1), nrow = 4, ncol = 2, byrow = TRUE)
mu1.shift  <- c(0.3, -0.2, 0.4, 0)

# 1. Outer Grid
phi_list  <- list(c(1, 2, 3), c(1, 2, 3, 4))
seed_list <- 1:200
rho_val   <- 0.8
vext_vals <- c(1, 2)

job_grid <- expand.grid(
  seed  = seed_list,
  idx   = seq_along(phi_list),
  rho   = rho_val,
  V.ext = vext_vals,
  KEEP.OUT.ATTRS = FALSE
)

# 2. Inner Grids
n1_grid_vals    <- c(500)
shift_vals      <- c(0, 0.2)
nq_vals         <- c(0,1,2)
V.int <- 1

maxit_val         <- 500
tol_val           <- 1e-6
learning_rate_val <- 0.1
B_boot_val        <- 0

cat("Total Parallel Jobs:", nrow(job_grid), "\n")

## ------------------------------------------------------------
## Worker Function
## ------------------------------------------------------------
simulate_one <- function(i) {
  job <- job_grid[i, ]

  phi_idx <- phi_list[[job$idx]]
  rho     <- job$rho

  outfile <- file.path(out_dir, sprintf("seed_%04d_phi_%d_vext_%.1f.rds", job$seed, job$idx, job$V.ext))

  tryCatch({
    res <- run_simulation(
      seed          = job$seed,
      n1_grid       = n1_grid_vals,
      n.q_grid      = nq_vals,
      shift.levels  = shift_vals,
      phi_idx       = phi_idx,
      V.ext         = job$V.ext,
      V.int         = V.int,
      rho           = rho,
      bootstrap     = FALSE,
      maxit         = maxit_val,
      tol           = tol_val,
      learning.rate = learning_rate_val,
      B_boot        = B_boot_val,
      Theta_true    = Theta_true,
      beta_true     = beta_true,
      alpha_true    = alpha_true,
      mu1.shift     = mu1.shift,
      p             = p,
      K             = K
    )

    df    <- res$out
    df_ci <- res$out_ci

    df$phi_idx    <- paste(phi_idx, collapse = "-")
    df$rho        <- rho
    df_ci$phi_idx <- paste(phi_idx, collapse = "-")
    df_ci$rho     <- rho

    saveRDS(list(out = df, out_ci = df_ci), outfile)
    return(NULL)

  }, error = function(e) {
    message("Error job ", i, ": ", conditionMessage(e))
    return(NULL)
  })
}


## ------------------------------------------------------------
## Cluster Initialization
## ------------------------------------------------------------
num_cores <- min(30, detectCores() - 2)
cat("Initializing Cluster with", num_cores, "cores...\n")

cl <- makeCluster(num_cores)

# 1. THREAD SAFETY
clusterEvalQ(cl, {
  Sys.setenv(OMP_NUM_THREADS = "1")
  Sys.setenv(MKL_NUM_THREADS = "1")
  Sys.setenv(OPENBLAS_NUM_THREADS = "1")
})

# 2. LOAD LIBRARIES & SOURCE FILES
clusterEvalQ(cl, {
  library(MLfused)
  library(dplyr)
  library(tidyverse)
  library(readr)
  library(nnet)

  source(system.file("scripts", "simulation.r", package = "MLfused"))
})

# 3. EXPORT DATA
clusterExport(cl, varlist = c(
  "job_grid", "phi_list", "out_dir",
  "n1_grid_vals", "shift_vals", "nq_vals",
  "rho_val", "n2", "p", "K", "L",
  "beta_true", "alpha_true",
  "Theta_true", "mu1.shift",
  "vext_vals", "V.int",
  "maxit_val", "tol_val", "learning_rate_val",
  "B_boot_val"
))

## ------------------------------------------------------------
## Execution
## ------------------------------------------------------------
cat("Starting simulation...\n")

pblapply(
  X   = seq_len(nrow(job_grid)),
  FUN = simulate_one,
  cl  = cl
)

stopCluster(cl)
cat("Simulation finished. Combining results...\n")


## ------------------------------------------------------------
## Combine Results
## ------------------------------------------------------------
files <- list.files(out_dir, pattern = "\\.rds$", full.names = TRUE)

if (length(files) > 0) {
  all_out    <- list()
  all_out_ci <- list()

  for (f in files) {
    r <- readRDS(f)
    all_out[[length(all_out) + 1]]       <- r$out
    all_out_ci[[length(all_out_ci) + 1]] <- r$out_ci
  }

  combined_out    <- bind_rows(all_out)
  combined_out_ci <- bind_rows(all_out_ci)

  saveRDS(combined_out,    file.path("results", paste0("simulation_", name, "_", date_stamp, "_out.rds")))
  saveRDS(combined_out_ci, file.path("results", paste0("simulation_", name, "_", date_stamp, "_out_ci.rds")))

  cat("Success! Saved to:\n")
} else {
  cat("Warning: No results found.\n")
}
