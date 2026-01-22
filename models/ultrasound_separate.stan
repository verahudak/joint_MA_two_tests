 data {
  int<lower=1> N_total;
  // Combined data from all the studies
  // Femur
  array[N_total] int y_F_TP; // True Positives
  array[N_total] int n_F_D;  // Total Diseased
  array[N_total] int y_F_FP; // False Positives
  array[N_total] int n_F_ND; // Total Non-Diseased

  // Humerus
  array[N_total] int y_H_TP; 
  array[N_total] int n_H_D;  
  array[N_total] int y_H_FP; 
  array[N_total] int n_H_ND; 
}

parameters {
  // Femur parameters
  // 1 = Logit Sensitivity, 2 = Logit FPR
  vector[2] mu_F; 
  vector<lower=0>[2] tau_F; 
  cholesky_factor_corr[2] L_Omega_F;
  matrix[2, N_total] z_F;

  // Humerus parameters
  // 1 = Logit Sensitivity, 2 = Logit FPR
  vector[2] mu_H; 
  vector<lower=0>[2] tau_H; 
  cholesky_factor_corr[2] L_Omega_H;
  matrix[2, N_total] z_H;
}

transformed parameters {

  // Reconstruct Random Effects
  matrix[2, N_total] eta_F;
  matrix[2, N_total] eta_H;
  matrix[2, 2] L_Sigma_F = diag_pre_multiply(tau_F, L_Omega_F);
  matrix[2, 2] L_Sigma_H = diag_pre_multiply(tau_H, L_Omega_H);

  // Non-centered parameterization
  eta_F = rep_matrix(mu_F, N_total) + L_Sigma_F * z_F;
  eta_H = rep_matrix(mu_H, N_total) + L_Sigma_H * z_H;

}

model {
  // Priors
  mu_F ~ normal(0, 5); 
  tau_F ~ normal(0, 1);
  L_Omega_F ~ lkj_corr_cholesky(1.0);
  to_vector(z_F) ~ std_normal();
  
  mu_H ~ normal(0, 5);
  tau_H ~ normal(0, 1);
  L_Omega_H ~ lkj_corr_cholesky(1.0);
  to_vector(z_H) ~ std_normal();

  // Likelihoods
  for (i in 1:N_total) {
    // Femur
    y_F_TP[i] ~ binomial_logit(n_F_D[i],  eta_F[1, i]); // Sensitivity
    y_F_FP[i] ~ binomial_logit(n_F_ND[i], eta_F[2, i]); // FPR
    // Humerus
    y_H_TP[i] ~ binomial_logit(n_H_D[i],  eta_H[1, i]); // Sensitivity
    y_H_FP[i] ~ binomial_logit(n_H_ND[i], eta_H[2, i]); // FPR
  }
}

generated quantities {

  // Summary estimayes
  real TPR_F_summary = inv_logit(mu_F[1]);
  real FPR_F_summary = inv_logit(mu_F[2]);
  real Spec_F_summary = 1 - FPR_F_summary;
  real TPR_H_summary = inv_logit(mu_H[1]);
  real FPR_H_summary = inv_logit(mu_H[2]);
  real Spec_H_summary = 1 - FPR_H_summary;

  // Predicitve
  vector[2] z_pred_F;
  vector[2] z_pred_H;
  for(k in 1:2) {
    z_pred_F[k] = normal_rng(0, 1);
    z_pred_H[k] = normal_rng(0, 1);
  }

  // Calculate latent effects for new study
  // Re-calculate L_Sigma locally for generated quantities block
  matrix[2, 2] Sigma_chol_F = diag_pre_multiply(tau_F, L_Omega_F);
  matrix[2, 2] Sigma_chol_H = diag_pre_multiply(tau_H, L_Omega_H);
  vector[2] eta_pred_F = mu_F + Sigma_chol_F * z_pred_F;
  vector[2] eta_pred_H = mu_H + Sigma_chol_H * z_pred_H;
  
  real pred_TPR_F = inv_logit(eta_pred_F[1]);
  real pred_FPR_F = inv_logit(eta_pred_F[2]);
  real pred_Spec_F = 1 - pred_FPR_F;
  real pred_TPR_H = inv_logit(eta_pred_H[1]);
  real pred_FPR_H = inv_logit(eta_pred_H[2]);
  real pred_Spec_H = 1 - pred_FPR_H;

  // Ranking
  // Difference (Femur - Humerus)
  real diff_Sens = TPR_F_summary - TPR_H_summary;
  real diff_Spec = Spec_F_summary - Spec_H_summary;

  // Probabilities of superior performance (0 or 1 per draw)
  // Is Femur Sensitivity > Humerus Sensitivity?
  real prob_better_Sens = (TPR_F_summary > TPR_H_summary) ? 1.0 : 0.0;
  // Is Femur Specificity > Humerus Specificity?
  real prob_better_Spec = (Spec_F_summary > Spec_H_summary) ? 1.0 : 0.0;
  // Is Femur better on BOTH counts?
  real prob_better_Both = (prob_better_Sens * prob_better_Spec);

  // Diagnostic odds ratios (DOR)
  real log_DOR_F = mu_F[1] - mu_F[2];
  real DOR_F     = exp(log_DOR_F);
  real log_DOR_H = mu_H[1] - mu_H[2];
  real DOR_H     = exp(log_DOR_H);

  // Hypothetical composite rules
  // What performance WOULD be if we combined them assuming independence.
  real JTPR_indep = TPR_F_summary * TPR_H_summary;
  real JFPR_indep = FPR_F_summary * FPR_H_summary;
  real Sens_and = JTPR_indep;
  real Spec_and = 1 - JFPR_indep;
  real Sens_or = TPR_F_summary + TPR_H_summary - JTPR_indep;
  real Spec_or = 1 - FPR_F_summary - FPR_H_summary + JFPR_indep;
  real log_DOR_and = logit(Sens_and) + logit(Spec_and);
  real DOR_and     = exp(log_DOR_and);
  real log_DOR_or  = logit(Sens_or) + logit(Spec_or);
  real DOR_or      = exp(log_DOR_or);
}

