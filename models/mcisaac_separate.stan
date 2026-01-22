
data {
  // RADT Data
  int<lower=1> N_R;
  array[N_R] int y_R_TP; // True Positives
  array[N_R] int n_R_D;  // Total Diseased
  array[N_R] int y_R_FP; // False Positives
  array[N_R] int n_R_ND; // Total Non-Diseased

  // McIsaac Data
  int<lower=1> N_M;
  array[N_M] int y_M_TP;
  array[N_M] int n_M_D;
  array[N_M] int y_M_FP;
  array[N_M] int n_M_ND;
}

parameters {
  // RADT parameters
  // 1 = Logit Sensitivity, 2 = Logit FPR
  vector[2] mu_R; 
  vector<lower=0>[2] tau_R; 
  cholesky_factor_corr[2] L_Omega_R;
  matrix[2, N_R] z_R;

  // McIsaac parameters
  // 1 = Logit Sensitivity, 2 = Logit FPR
  vector[2] mu_M; 
  vector<lower=0>[2] tau_M; 
  cholesky_factor_corr[2] L_Omega_M;
  matrix[2, N_M] z_M;
}

transformed parameters {
  // Reconstruct Random Effects
  matrix[2, N_R] eta_R;
  matrix[2, N_M] eta_M;
  
  matrix[2, 2] L_Sigma_R = diag_pre_multiply(tau_R, L_Omega_R);
  matrix[2, 2] L_Sigma_M = diag_pre_multiply(tau_M, L_Omega_M);
  
  // Non-centered parameterization
  eta_R = rep_matrix(mu_R, N_R) + L_Sigma_R * z_R;
  eta_M = rep_matrix(mu_M, N_M) + L_Sigma_M * z_M;
}

model {
  // Priors
  mu_R ~ normal(0, 5); 
  tau_R ~ normal(0, 1);
  L_Omega_R ~ lkj_corr_cholesky(1.0);
  to_vector(z_R) ~ std_normal();

  mu_M ~ normal(0, 5);
  tau_M ~ normal(0, 1);
  L_Omega_M ~ lkj_corr_cholesky(1.0);
  to_vector(z_M) ~ std_normal();
  
  // Likelihoods
  for (i in 1:N_R) {
    // RADT
    y_R_TP[i] ~ binomial_logit(n_R_D[i],  eta_R[1, i]); // Sensitivity
    y_R_FP[i] ~ binomial_logit(n_R_ND[i], eta_R[2, i]); // FPR
  }

  for (j in 1:N_M) {
    // McIsaac
    y_M_TP[j] ~ binomial_logit(n_M_D[j],  eta_M[1, j]); // Sensitivity
    y_M_FP[j] ~ binomial_logit(n_M_ND[j], eta_M[2, j]); // FPR
  }
}

generated quantities {
  // Summary estimates
  real TPR_R_summary = inv_logit(mu_R[1]);
  real FPR_R_summary = inv_logit(mu_R[2]);
  real Spec_R_summary = 1 - FPR_R_summary;
  
  real TPR_M_summary = inv_logit(mu_M[1]);
  real FPR_M_summary = inv_logit(mu_M[2]);
  real Spec_M_summary = 1 - FPR_M_summary;
  
  // Predictive
  vector[2] z_pred_R;
  vector[2] z_pred_M;
  for(k in 1:2) {
    z_pred_R[k] = normal_rng(0, 1);
    z_pred_M[k] = normal_rng(0, 1);
  }

  // Calculate latent effects for new study
  // Re-calculate L_Sigma locally for generated quantities block
  matrix[2, 2] Sigma_chol_R = diag_pre_multiply(tau_R, L_Omega_R);
  matrix[2, 2] Sigma_chol_M = diag_pre_multiply(tau_M, L_Omega_M);
  
  vector[2] eta_pred_R = mu_R + Sigma_chol_R * z_pred_R;
  vector[2] eta_pred_M = mu_M + Sigma_chol_M * z_pred_M;

  real pred_TPR_R = inv_logit(eta_pred_R[1]);
  real pred_FPR_R = inv_logit(eta_pred_R[2]);
  real pred_Spec_R = 1 - pred_FPR_R;

  real pred_TPR_M = inv_logit(eta_pred_M[1]);
  real pred_FPR_M = inv_logit(eta_pred_M[2]);
  real pred_Spec_M = 1 - pred_FPR_M;
  
  // Ranking
  
  // Difference (McIsaac - RADT)
  real diff_Sens = TPR_M_summary - TPR_R_summary;
  real diff_Spec = Spec_M_summary - Spec_R_summary;
  
  // Probabilities of superior performance (0 or 1 per draw)
  // Is McIsaac Sensitivity > RADT Sensitivity?
  real prob_better_Sens = (TPR_M_summary > TPR_R_summary) ? 1.0 : 0.0;
  
  // Is McIsaac Specificity > RADT Specificity?
  real prob_better_Spec = (Spec_M_summary > Spec_R_summary) ? 1.0 : 0.0;
  
  // Is McIsaac better on BOTH counts?
  real prob_better_Both = (prob_better_Sens * prob_better_Spec);
  
  // Diagnostic odds ratios (DOR)
  real log_DOR_R = mu_R[1] - mu_R[2];
  real DOR_R     = exp(log_DOR_R);
  
  real log_DOR_M = mu_M[1] - mu_M[2];
  real DOR_M     = exp(log_DOR_M);
  
  // Hypothetical composite rules
  // What performance WOULD be if we combined them assuming independence.
  
  real JTPR_indep = TPR_R_summary * TPR_M_summary;
  real JFPR_indep = FPR_R_summary * FPR_M_summary;
  
  real Sens_and = JTPR_indep;
  real Spec_and = 1 - JFPR_indep;
  
  real Sens_or = TPR_R_summary + TPR_M_summary - JTPR_indep;
  real Spec_or = 1 - FPR_R_summary - FPR_M_summary + JFPR_indep;
  
  real log_DOR_and = logit(Sens_and) + logit(Spec_and);
  real DOR_and     = exp(log_DOR_and);
  
  real log_DOR_or  = logit(Sens_or) + logit(Spec_or);
  real DOR_or      = exp(log_DOR_or);
}
