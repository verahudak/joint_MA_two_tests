data {
  int<lower=1> N_full;
  int<lower=1> N_marginal;
  int<lower=1> N_partial;
  
  // Data for full studies
  array[N_full, 4] int y_full_d;
  array[N_full, 4] int y_full_nd;
  
  // Data for marginal studies
  array[N_marginal] int y_F_d;
  array[N_marginal] int N_d_F;
  array[N_marginal] int y_F_nd;
  array[N_marginal] int N_nd_F;
  
  array[N_marginal] int y_H_d;
  array[N_marginal] int N_d_H;
  array[N_marginal] int y_H_nd;
  array[N_marginal] int N_nd_H;
  
  // Data for partial studies
  array[N_partial, 4] int y_partial_d; 
  array[N_partial] int y_F_nd_partial; 
  array[N_partial] int N_nd_partial;   
  array[N_partial] int y_H_nd_partial; 
}

parameters {
  // 1=TPR_F, 2=TPR_H, 3=FPR_F, 4=FPR_H
  vector[4] mu; 
  
  // Covariance decomposition 
  cholesky_factor_corr[4] L_Omega;
  vector<lower=0>[4] tau; 
  
  // Random noise
  matrix[4, N_full] z_full;
  matrix[4, N_marginal] z_marginal;
  matrix[4, N_partial] z_partial;
}

transformed parameters {
  // Reconstruct random effects
  matrix[4, N_full] eta_full;
  matrix[4, N_marginal] eta_marginal;
  matrix[4, N_partial] eta_partial;
  
  matrix[4, 4] L_Sigma;
  
  L_Sigma = diag_pre_multiply(tau, L_Omega);
  
  // eta = mu + (diag(tau) * L * z)
  eta_full = rep_matrix(mu, N_full) + L_Sigma * z_full;
  eta_marginal = rep_matrix(mu, N_marginal) + L_Sigma * z_marginal;
  eta_partial = rep_matrix(mu, N_partial) + L_Sigma * z_partial;
}

model {
  // Priors
  mu ~ normal(0, 5);          
  tau ~ normal(0, 1);
  L_Omega ~ lkj_corr_cholesky(1.0); 
  
  // Standard normals
  to_vector(z_full) ~ std_normal();
  to_vector(z_marginal) ~ std_normal();
  to_vector(z_partial) ~ std_normal();
  
  // Likelihood: full studies
  for (k in 1:N_full) {
    real TPR_F = inv_logit(eta_full[1, k]);
    real TPR_H = inv_logit(eta_full[2, k]);
    // Assume independence
    real JTPR = TPR_F * TPR_H;
    
    real FPR_F = inv_logit(eta_full[3, k]);
    real FPR_H = inv_logit(eta_full[4, k]);
    // Assume independence
    real JFPR = FPR_F * FPR_H;
    
    vector[4] p_d;
    vector[4] p_nd;
    
    p_d[1] = JTPR;
    p_d[2] = TPR_F - JTPR;
    p_d[3] = TPR_H - JTPR;
    p_d[4] = 1 - (p_d[1] + p_d[2] + p_d[3]);
    
    p_nd[1] = JFPR;
    p_nd[2] = FPR_F - JFPR;
    p_nd[3] = FPR_H - JFPR;
    p_nd[4] = 1 - (p_nd[1] + p_nd[2] + p_nd[3]);
    
    y_full_d[k] ~ multinomial(p_d);
    y_full_nd[k] ~ multinomial(p_nd);
  }
  
  // Likelihood: marginal studies
  for (j in 1:N_marginal) {
    real TPR_F = inv_logit(eta_marginal[1, j]);
    real TPR_H = inv_logit(eta_marginal[2, j]);
    real FPR_F = inv_logit(eta_marginal[3, j]);
    real FPR_H = inv_logit(eta_marginal[4, j]);
    
    y_F_d[j] ~ binomial(N_d_F[j], TPR_F);
    y_F_nd[j] ~ binomial(N_nd_F[j], FPR_F);
    y_H_d[j] ~ binomial(N_d_H[j], TPR_H);
    y_H_nd[j] ~ binomial(N_nd_H[j], FPR_H);
  }
  
  // Likelihood: partial studies
  for (m in 1:N_partial) {
    real TPR_F = inv_logit(eta_partial[1, m]);
    real TPR_H = inv_logit(eta_partial[2, m]);
    // Diseased: Independence assumption
    real JTPR = TPR_F * TPR_H;
    
    vector[4] p_d;
    p_d[1] = JTPR;
    p_d[2] = TPR_F - JTPR;
    p_d[3] = TPR_H - JTPR;
    p_d[4] = 1 - sum(p_d[1:3]);
    y_partial_d[m] ~ multinomial(p_d);
    
    // Disease free (binomials)
    real FPR_F = inv_logit(eta_partial[3, m]);
    real FPR_H = inv_logit(eta_partial[4, m]);
    
    y_F_nd_partial[m] ~ binomial(N_nd_partial[m], FPR_F);
    y_H_nd_partial[m] ~ binomial(N_nd_partial[m], FPR_H);
  }
}

generated quantities {
  // Summary estimates 
  real TPR_F_summary = inv_logit(mu[1]);
  real TPR_H_summary = inv_logit(mu[2]);
  real FPR_F_summary = inv_logit(mu[3]);
  real FPR_H_summary = inv_logit(mu[4]);
  
  real Spec_F_summary = 1 - FPR_F_summary;
  real Spec_H_summary = 1 - FPR_H_summary;
  
  // Predictive distributions
  vector[4] eta_pred;
  vector[4] z_pred;
  for(i in 1:4) z_pred[i] = normal_rng(0, 1);
  eta_pred = mu + L_Sigma * z_pred;
  
  real pred_TPR_F = inv_logit(eta_pred[1]);
  real pred_TPR_H = inv_logit(eta_pred[2]);
  real pred_FPR_F = inv_logit(eta_pred[3]);
  real pred_FPR_H = inv_logit(eta_pred[4]);
  real pred_Spec_F = 1 - pred_FPR_F;
  real pred_Spec_H = 1 - pred_FPR_H;
  
  // Full correlation matrix
  matrix[4, 4] Omega_corr;
  Omega_corr = multiply_lower_tri_self_transpose(L_Omega);
  
  // Summary JTRP and JFPR (Assuming Independence)
  real JTPR_sum = TPR_F_summary * TPR_H_summary;
  real JFPR_sum = FPR_F_summary * FPR_H_summary;
  
  // Tests in sequence: AND rule
  real Sens_and = JTPR_sum;
  real Spec_and = 1 - JFPR_sum;
  
  // Tests in sequence: OR rule
  real Sens_or = TPR_F_summary + TPR_H_summary - JTPR_sum;
  real Spec_or = 1 - FPR_F_summary - FPR_H_summary + JFPR_sum;
  
  // Comparative test performance
  
  // Difference in sens/spec (Femur - Humerus)
  real diff_Sens = TPR_F_summary - TPR_H_summary;
  real diff_Spec = Spec_F_summary - Spec_H_summary;
  
  // Probabilities of superior performance
  // Is Femur better than Humerus? (1 if yes, 0 if no)
  real prob_better_Sens = (TPR_F_summary > TPR_H_summary) ? 1.0 : 0.0;
  real prob_better_Spec = (Spec_F_summary > Spec_H_summary) ? 1.0 : 0.0;
  // Sens_F > Sens_H AND Spec_F > Spec_H
  real prob_better_Both = (prob_better_Sens * prob_better_Spec);
  
  // Diagnostic odds ratios (DOR)
  real log_DOR_F = mu[1] - mu[3];
  real DOR_F = exp(log_DOR_F);
  real log_DOR_H = mu[2] - mu[4];
  real DOR_H = exp(log_DOR_H);
  
  // Composite Rules DOR 
  real log_DOR_and = logit(Sens_and) + logit(Spec_and);
  real DOR_and = exp(log_DOR_and);
  
  real log_DOR_or = logit(Sens_or) + logit(Spec_or);
  real DOR_or = exp(log_DOR_or);
}

