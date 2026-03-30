# Visualization script for simulation results
# Reads combined .rds output and generates PDF figures

# ---- 0. Libraries & setup ----------------------------------------
library(MLfused)
library(tidyverse)
library(dplyr)
library(ggplot2)
library(kableExtra)
library(patchwork)

name       <- "simulation_fixed_coef"
date_stamp <- {
  f <- list.files("results", pattern = paste0("^simulation_", name, "_\\d{8}_out\\.rds$"))
  if (length(f) == 0) stop("No dated result files found for: ", name)
  max(sub(paste0("^simulation_", name, "_(\\d{8})_out\\.rds$"), "\\1", f))
}
date_stamp="20260314"

fig_dir    <- file.path("results", paste0("fig_", name, "_", date_stamp))
dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)

# ---- 1. Load data ------------------------------------------------
results <- readRDS(paste0("results/simulation_", name, "_", date_stamp, "_out.rds")) %>%
  filter(ml_method == "xgboost")

out_ci <- readRDS(paste0("results/simulation_", name, "_", date_stamp, "_out_ci.rds")) %>%
  filter(ml_method == "xgboost")

# ---- 2. Labels ---------------------------------------------------
results <- results %>%
  mutate(
    n.q_label = factor(paste0("H = ", n.q)),
    phi_label = dplyr::recode(
      phi_idx,
      "1-2-3-4" = "Full features",
      "1-2-3"   = "Omit one feature"
    ),
    shift.level = factor(
      shift.level,
      levels = c(0, 0.2),
      labels = c("Shift = 0", "Shift = 0.2")
    ),
    V.ext_label = factor(
      V.ext,
      levels = c(1, 2),
      labels = c("V.ext = 0.5 (matched)", "V.ext = 1 (variance shift)")
    ),
    case_label = case_when(
      V.ext == 1 & shift.level == "Shift = 0"   ~ "Case 1: No Shift",
      V.ext == 2.0 & shift.level == "Shift = 0"   ~ "Case 2: Var Shift",
      V.ext == 1 & shift.level == "Shift = 0.2" ~ "Case 3: Mean Shift",
      V.ext == 2 & shift.level == "Shift = 0.2" ~ "Case 4: Mean+Var Shift",
      TRUE ~ paste0("V.ext=", V.ext, ", ", shift.level)
    ),
    mean_shift_label = ifelse(shift.level == "Shift = 0.2", "Mean Shift = TRUE",  "Mean Shift = FALSE"),
    var_shift_label  = ifelse(V.ext >= 2,                 "Variance Shift = TRUE", "Variance Shift = FALSE")
  )

out_ci <- out_ci %>%
  mutate(
    n.q_label = factor(paste0("H = ", n.q)),
    phi_label = dplyr::recode(
      phi_idx,
      "1-2-3-4" = "Full features",
      "1-2-3"   = "Omit one feature"
    ),
    shift.level = factor(
      shift.level,
      levels = c(0, 0.2),
      labels = c("Shift = 0", "Shift = 0.2")
    ),
    V.ext_label = factor(
      V.ext,
      levels = c(1, 2),
      labels = c("V.ext = 0.5 (matched)", "V.ext = 1 (variance shift)")
    ),
    case_label = case_when(
      V.ext == 1 & shift.level == "Shift = 0"   ~ "Case 1: No Shift",
      V.ext == 2.0 & shift.level == "Shift = 0"   ~ "Case 2: Var Shift",
      V.ext == 1 & shift.level == "Shift = 0.2" ~ "Case 3: Mean Shift",
      V.ext == 2.0 & shift.level == "Shift = 0.2" ~ "Case 4: Mean+Var Shift",
      TRUE ~ paste0("V.ext=", V.ext, ", ", shift.level)
    ),
    mean_shift_label = ifelse(shift.level == "Shift = 0.2", "Mean Shift = TRUE",  "Mean Shift = FALSE"),
    var_shift_label  = ifelse(V.ext >= 2.0,                 "Variance Shift = TRUE", "Variance Shift = FALSE")
  )

a=out_ci%>%group_by(method, adj,variable,i,j) %>% summarise(n = n(), cp=mean(covered,na.rm=T), )


# ---- 3. Mean Error Improvement data prep -------------------------
result.original <- results %>%
  filter(method == "Original") %>%
  group_by(phi_label, n1, n.q, shift.level, V.ext, V.ext_label) %>%
  summarise(
    mean_error_original = mean(error,      na.rm = TRUE),
    avg_length_original = mean(avg_length, na.rm = TRUE),
    .groups = "drop"
  )

results%>%count(conv)
result.fused <- results %>%
  filter( method == "Fused.hard",conv==0) %>%
  group_by(phi_label, n1, n.q, method, shift.level, adj, V.ext, V.ext_label) %>%
  summarise(
    mean_error    = mean(error,      na.rm = TRUE),
    coverage_mean = mean(coverage,   na.rm = TRUE),
    avg_length    = mean(avg_length, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    n.q_label        = factor(paste0("H = ", n.q)),
    mean_shift_label = ifelse(shift.level == "Shift = 0.2", "Mean Shift = TRUE",  "Mean Shift = FALSE"),
    var_shift_label  = ifelse(V.ext >= 2,                 "Variance Shift = TRUE", "Variance Shift = FALSE")
  )

final.result <- left_join(
  result.fused, result.original,
  by = c("phi_label", "n1", "n.q", "shift.level", "V.ext", "V.ext_label")
) %>%
  mutate(
    mean_error_imp   = (mean_error_original - mean_error) / mean_error_original,
    mean_shift_label = ifelse(shift.level == "Shift = 0.2", "Mean Shift = TRUE",  "Mean Shift = FALSE"),
    var_shift_label  = ifelse(V.ext >= 2.0,                 "Variance Shift = TRUE", "Variance Shift = FALSE")
  )


# ---- 4. Plot 1: Mean Error Improvement ---------------------------
p1 <- ggplot(
  final.result %>% filter(adj == "unadj"),
  aes(
    x     = n1,
    y     = mean_error_imp,
    group = interaction(phi_label, n.q),
    shape = n.q_label
  )
) +
  geom_hline(yintercept = 0, color = "grey50", linetype = "solid") +
  geom_point(size = 2.5, position = position_dodge(width = 15)) +
  scale_shape_manual(values = c(0, 1, 2, 8)) +
  facet_grid(mean_shift_label + var_shift_label ~ phi_label) +
  labs(
    title = "Mean Error Improvement over Original (Fixed Coef; Fused.hard)",
    x     = "Primary Sample Size (n)",
    y     = "Mean Error Improvement",
    shape = expression(H[m])
  ) +
  theme_bw(base_size = 12) +
  theme(
    legend.position  = "bottom",
    panel.grid.minor = element_blank()
  )

p1

ggsave(file.path(fig_dir, paste0("Mean_Error_Improvement_", name, ".pdf")),
       p1, width = 8, height = 6)
ggsave(file.path(fig_dir, paste0("Mean_Error_Improvement_", name, ".png")),
       p1, width = 8, height = 6)


# ---- 5. CI Coverage data prep ------------------------------------
orig_cov <- results %>%
  filter(method == "Original") %>%
  group_by(n1, n.q, phi_label, shift.level, V.ext, V.ext_label) %>%
  summarise(cov_orig = mean(coverage, na.rm = TRUE), .groups = "drop") %>%
  mutate(
    n.q_label        = factor(paste0("H = ", n.q)),
    mean_shift_label = ifelse(shift.level == "Shift = 0.2", "Mean Shift = TRUE",  "Mean Shift = FALSE"),
    var_shift_label  = ifelse(V.ext >= 1.0,                 "Variance Shift = TRUE", "Variance Shift = FALSE")
  )

adj_levels <- c("unadj", "adj", "adj.noq", "score", "score.adj",
                "thm3", "thm3.adj", "thm3.score", "thm3.score.adj", "boot")
adj_labels <- c("Unadj", "Adj", "Adj.noq", "Score", "Score.adj",
                "Thm3", "Thm3.adj", "Thm3.Score", "Thm3.Score.Adj", "Boot")
adj_colors <- c(
  "Unadj"          = "#E69F00",
  "Adj"            = "#0072B2",
  "Adj.noq"        = "#56B4E9",
  "Score"          = "#D55E00",
  "Score.adj"      = "#CC79A7",
  "Thm3"           = "#009E73",
  "Thm3.adj"       = "#F0E442",
  "Thm3.Score"     = "#44AA99",
  "Thm3.Score.Adj" = "#DDCC77",
  "Boot"           = "#882255"
)

cov_plot <- result.fused %>%
  filter(adj %in% adj_levels) %>%
  mutate(
    adj_label = factor(
      dplyr::recode(adj, !!!setNames(adj_labels, adj_levels)),
      levels = adj_labels
    )
  )


# ---- 6. Plot 2: CI Coverage — unadj vs adj, MVLR dot baseline ----
p2 <- ggplot(
  cov_plot,
  aes(
    x     = n1,
    y     = coverage_mean,
    group = interaction(n.q, method, adj_label),
    shape = n.q_label,
    color = adj_label
  )
) +
  geom_hline(yintercept = 0.95, color = "grey40", linetype = "dotted") +
  geom_point(
    data = orig_cov,
    aes(x = n1, y = cov_orig, group = interaction(n.q, phi_label, mean_shift_label, var_shift_label)),
    inherit.aes = FALSE,
    shape  = 4,
    size   = 2.5,
    color  = "grey30"
  ) +
  geom_point(size = 2.5, position = position_dodge(width = 15)) +
  scale_shape_manual(values = c(0, 1, 2, 8)) +
  scale_color_manual(
    values = adj_colors,
    name   = "SE type"
  ) +
  facet_grid(mean_shift_label + var_shift_label ~ phi_label) +
  labs(
    title = "CI Coverage by SE Type (Fixed Coef; \u2715 = MVLR baseline)",
    x     = "Primary Sample Size (n)",
    y     = "Coverage",
    shape = expression(H[m])
  ) +
  theme_bw(base_size = 12) +
  theme(
    legend.position  = "bottom",
    panel.grid.minor = element_blank()
  )

p2

ggsave(file.path(fig_dir, paste0("CI_coverage_", name, ".pdf")),
       p2, width = 8, height = 6)
ggsave(file.path(fig_dir, paste0("CI_coverage_", name, ".png")),
       p2, width = 8, height = 6)


results%>%group_by(method,n.q)%>%summarise(max(error),n=n(),robust_prop=mean(error<4,na.rm=T), na_rate=mean(is.na(error.beta)))

# ---- 7. Convergence Rate data prep -------------------------------
conv_tbl <- results %>%
  filter(method == "Fused.hard", adj == "unadj") %>%
  group_by(method, phi_label, n1, shift.level, n.q, V.ext, V.ext_label) %>%
  summarise(
    pct_conv  = mean(conv == 0, na.rm = TRUE),
    .groups   = "drop",
    robust_prop=mean(error<4,na.rm=T),
    na.rate=mean(is.na(error.beta))
  ) %>%
  mutate(
    n.q_label        = factor(paste0("H = ", n.q)),
    mean_shift_label = ifelse(shift.level == "Shift = 0.2", "Mean Shift = TRUE",  "Mean Shift = FALSE"),
    var_shift_label  = ifelse(V.ext >= 1.0,                 "Variance Shift = TRUE", "Variance Shift = FALSE")
  )


# ---- 8. Plot 3: Convergence Rate ---------------------------------
p3 <- ggplot(
  conv_tbl,
  aes(
    x     = n1,
    y     =  robust_prop,
    group = interaction(phi_label, n.q),
    shape = n.q_label
  )
) +
  geom_hline(yintercept = 1, color = "grey50", linetype = "solid") +
  geom_point(size = 2.5, position = position_dodge(width = 15)) +
  scale_shape_manual(values = c(0, 1, 2, 8)) +
  scale_y_continuous(labels = scales::percent, limits = c(0.94, 1.001)) +
  facet_grid(mean_shift_label + var_shift_label ~ phi_label) +
  labs(
    title = "Convergence Rate (Fixed Coef; Fused.hard)",
    x     = "Primary Sample Size (n)",
    y     = "% Converged",
    shape = expression(H[m])
  ) +
  theme_bw(base_size = 12) +
  theme(
    legend.position  = "bottom",
    panel.grid.minor = element_blank()
  )

p3
ggsave(file.path(fig_dir, paste0("Convergence_", name, ".pdf")),
       p3, width = 8, height = 6)
ggsave(file.path(fig_dir, paste0("Convergence_", name, ".png")),
       p3, width = 8, height = 6)


# ---- 9. Convergence Summary --------------------------------------
cat("=== Convergence Summary ===\n")
print(
  conv_tbl %>%
    group_by(method) %>%
    summarise(pct_conv = mean(pct_conv), .groups = "drop")
)



#-------CP histogram: Original-unadj and Fused.hard by SE method-----------
cp_adj_levels <- c("unadj", "score", "thm3", "thm3.score", "boot")
cp_adj_labels <- c("Unadj", "Score", "Thm3", "Thm3.Score", "Boot")

# Summarise CP per grouping key
cp_summarised <- out_ci %>%
  filter(abs(est - truth) < 5) %>%
  group_by(method, adj, i, j, n1, n.q, shift.level, phi_label, variable, V.ext, V.ext_label) %>%
  summarise(cp = mean(covered, na.rm = TRUE), .groups = "drop")


# Original data to be overlaid in every fused panel
cp_orig <- cp_summarised %>%
  filter(method == "Original", adj == "unadj")

# Fused.hard by SE method
cp_fused <- cp_summarised %>%
  filter(method == "Fused.hard", adj %in% cp_adj_levels) %>%
  mutate(
    adj_label = factor(
      dplyr::recode(adj, !!!setNames(cp_adj_labels, cp_adj_levels)),
      levels = cp_adj_labels
    )
  )

p_fused <- ggplot(cp_fused, aes(x = cp)) +
  geom_density(
    data = cp_orig,
    fill = "#0072B2", color = "#0072B2", alpha = 0.20, linewidth = 0.6
  ) +
  geom_density(
    aes(fill = adj_label, color = adj_label),
    alpha = 0.25, linewidth = 0.7
  ) +
  geom_vline(xintercept = 0.95, color = "black", linetype = "dashed", linewidth = 0.7) +
  scale_fill_manual(
    values = setNames(
      c("#E69F00", "#D55E00", "#009E73", "#44AA99", "#882255"),
      cp_adj_labels
    ),
    name = "SE type"
  ) +
  scale_color_manual(
    values = setNames(
      c("#E69F00", "#D55E00", "#009E73", "#44AA99", "#882255"),
      cp_adj_labels
    ),
    guide = "none"
  ) +
  scale_x_continuous(
    limits = c(0, 1),
    breaks = c(0, 0.5, 0.95, 1),
    labels = c("0", "0.5", "0.95", "1")
  ) +
  facet_grid(adj_label ~ phi_label, scales = "free_y") +
  labs(
    title = "Fused.hard with Original (unadj) overlay",
    x = "Coverage Probability",
    y = "Density"
  ) +
  theme_bw(base_size = 11) +
  theme(
    panel.grid.minor = element_blank(),
    strip.text = element_text(size = 9)
  )
p_fused

ggsave(file.path(fig_dir, paste0("CP_histogram_", name, ".pdf")),
       p_fused, width = 7, height = 11)
ggsave(file.path(fig_dir, paste0("CP_histogram_", name, ".png")),
       p_fused, width = 7, height = 11)


# ---- 11. Theta LaTeX Tables ------------------------------------------

# Extend theta_tbl with CP (coverage probability)
theta_tbl_cp <- out_ci %>%
  filter(variable == "theta", method %in% c("Fused.hard", "Original"),
         (method == "Fused.hard" & adj == "boot") | (method == "Original" & adj == "unadj"),
         abs(est - truth) < 4,se<10) %>%
  group_by(method, i, j, n1, n.q, shift.level, phi_label, V.ext, V.ext_label) %>%
  summarise(
    truth   = first(truth),
    bias    = mean(est - truth),
    sd_est  = sd(est),
    se_mean = mean(se,na.rm=T),
    cp      = mean(covered, na.rm = TRUE),
    .groups = "drop"
  )

view(theta_tbl_cp)

beta_tbl_cp<- out_ci %>%
  filter(variable == "beta", method %in% c("Fused.hard", "Original"),
         (method == "Fused.hard" & adj == "boot") | (method == "Original" & adj == "unadj"),
         abs(est - truth) < 4) %>%
  filter(!(method == "Original" & n.q != 0)) %>%
  group_by(method, i, n1, n.q, shift.level, phi_label, V.ext, V.ext_label) %>%
  summarise(
    truth   = first(truth),
    bias    = mean(est - truth),
    sd_est  = sd(est),
    se_mean = mean(se),
    cp      = mean(covered, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(j = NA_integer_)

view(beta_tbl_cp)


make_combined_latex_table <- function(n1_val, theta_tbl, beta_tbl) {
  method_order  <- c("MVLR", "Fused (H=0)", "Fused (H=1)", "Fused (H=2)")
  metric_order  <- c("Bias", "SD", "SE", "CP")
  scenario_list <- list(
    list(ms = FALSE, vs = FALSE, label = "No Shift"),
    list(ms = FALSE, vs = TRUE,  label = "Variance Shift"),
    list(ms = TRUE,  vs = FALSE, label = "Mean Shift"),
    list(ms = TRUE,  vs = TRUE,  label = "Mean + Variance Shift")
  )

  build_phi_block <- function(phi, sl, vext) {
    th <- theta_tbl %>%
      filter(n1 == n1_val, phi_label == phi, shift.level == sl, V.ext == vext,
             (method == "Fused.hard" & n.q %in% c(0, 1, 2)) |
             (method == "Original"   & n.q == 0)) %>%
      mutate(
        method_label = case_when(
          method == "Fused.hard" ~ paste0("Fused (H=", n.q, ")"),
          TRUE ~ "MVLR"),
        param = paste0("psi_i", i, "_j", j)
      ) %>%
      select(method_label, param, bias, sd_est, se_mean, cp)

    be <- beta_tbl %>%
      filter(n1 == n1_val, phi_label == phi, shift.level == sl, V.ext == vext,
             (method == "Fused.hard" & n.q %in% c(0, 1, 2)) |
             method == "Original") %>%
      mutate(
        method_label = case_when(
          method == "Fused.hard" ~ paste0("Fused (H=", n.q, ")"),
          TRUE ~ "MVLR"),
        param = paste0("phi_", i)
      ) %>%
      select(method_label, param, bias, sd_est, se_mean, cp)

    bind_rows(th, be) %>%
      pivot_longer(cols = c(bias, sd_est, se_mean, cp),
                   names_to = "metric", values_to = "value") %>%
      mutate(
        metric = dplyr::recode(metric,
          "bias" = "Bias", "sd_est" = "SD", "se_mean" = "SE", "cp" = "CP"),
        metric = factor(metric, levels = metric_order)
      ) %>%
      select(method_label, metric, param, value) %>%
      pivot_wider(names_from = param, values_from = value)
  }

  build_scenario_block <- function(ms, vs) {
    sl   <- if (ms) "Shift = 0.2" else "Shift = 0"
    vext <- if (vs) 1.0 else 0.5

    block_full <- build_phi_block("Full features",    sl, vext)
    block_omit <- build_phi_block("Omit one feature", sl, vext)

    phi_cols   <- c("phi_1", "phi_2")
    psi_cols   <- c(paste0("psi_i", 1:4, "_j1"), paste0("psi_i", 1:4, "_j2"))
    block_cols <- c(phi_cols, psi_cols)

    block_full %>%
      rename_with(~ paste0("full_", .), all_of(block_cols)) %>%
      left_join(
        block_omit %>% rename_with(~ paste0("omit_", .), all_of(block_cols)),
        by = c("method_label", "metric")
      ) %>%
      mutate(method_label = factor(method_label, levels = method_order)) %>%
      arrange(metric, method_label)
  }

  blocks     <- lapply(scenario_list, function(s) build_scenario_block(s$ms, s$vs))
  row_counts <- sapply(blocks, nrow)
  row_ends   <- cumsum(row_counts)
  row_starts <- c(1, head(row_ends, -1) + 1)

  all_param_cols <- c(
    paste0("full_", c("phi_1", "phi_2", paste0("psi_i", 1:4, "_j1"), paste0("psi_i", 1:4, "_j2"))),
    paste0("omit_", c("phi_1", "phi_2", paste0("psi_i", 1:4, "_j1"), paste0("psi_i", 1:4, "_j2")))
  )

  dat_fmt <- bind_rows(blocks) %>%
    select(metric, method_label, all_of(all_param_cols)) %>%
    mutate(across(all_of(all_param_cols), ~ sprintf("%.3f", .)))

  if (nrow(dat_fmt) == 0) return(NULL)

  param_names <- c(
    "$\\varphi_1$", "$\\varphi_2$",
    paste0("$\\psi_{", 1:4, "1}$"),
    paste0("$\\psi_{", 1:4, "2}$")
  )

  tbl <- kbl(dat_fmt, format = "latex", booktabs = TRUE, escape = FALSE,
             col.names = c("Metric", "Method", param_names, param_names),
             align = c("l", "l", rep("r", 20))) %>%
    add_header_above(
      c(" " = 2,
        "$\\varphi$" = 2, "$j=1$" = 4, "$j=2$" = 4,
        "$\\varphi$" = 2, "$j=1$" = 4, "$j=2$" = 4),
      escape = FALSE) %>%
    add_header_above(
      c(" " = 2, "Full features" = 10, "Omit one feature" = 10),
      escape = FALSE) %>%
    collapse_rows(columns = 1, latex_hline = "none", valign = "middle")

  for (k in seq_along(scenario_list)) {
    tbl <- pack_rows(tbl, scenario_list[[k]]$label,
                     row_starts[k], row_ends[k],
                     bold = TRUE, latex_gap_space = "0.3em")
  }

  tbl %>% kable_styling(latex_options = c("hold_position", "scale_down"))
}


# ---- 12. Generate 3 LaTeX tables (one per n1) ----------------------------
for (n1v in c(400, 500, 600)) {
  tbl <- make_combined_latex_table(n1_val    = n1v,
                                   theta_tbl = theta_tbl_cp,
                                   beta_tbl  = beta_tbl_cp)
  if (!is.null(tbl)) {
    fname <- file.path(fig_dir, paste0("theta_table_n", n1v, "_", name, ".tex"))
    cat(tbl, file = fname)
    cat("Saved:", fname, "\n")
  }
}
