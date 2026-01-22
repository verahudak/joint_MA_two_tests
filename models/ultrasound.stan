functions {
  // Helper to solve the quadratic equation for joint probabilities
  // p1 and p2 are the TPRs or FPRs
  real get_joint_prob(real p1, real p2, real OR) {
    real J;
    real a; 
    real b; 
    real c;
    real discriminant;
    
    // If OR = 1
    if (abs(OR - 1.0) < 0.00001) {
      J = p1 * p2;
    } else {
      a = OR - 1.0;
      b = -OR * (p1 + p2) - 1.0 + (p1 + p2);
      c = OR * p1 * p2;
      
      discriminant = sqrt(b^2 - 4.0 * a * c);
      
      // Using the negative root (-b - discriminant) / 2a
      J = (-b - discriminant) / (2.0 * a);
    }
    return J;
  }
}

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
  // Global means
  vector[6] mu; 
  
  // Covariance decomposition
  cholesky_factor_corr[6] L_Omega;
  vector<lower=0>[6] tau; 
  
  // Random noise 
  matrix[6, N_full] z_full;
  matrix[6, N_marginal] z_marginal;
  matrix[6, N_partial] z_partial;
}

transformed parameters {
  // Reconstruct the actual random effects
  matrix[6, N_full] eta_full;
  matrix[6, N_marginal] eta_marginal;
  matrix[6, N_partial] eta_partial;
  
  matrix[6, 6] L_Sigma;
  
  L_Sigma = diag_pre_multiply(tau, L_Omega);
  
  // eta = mu + (diag(tau) * L * z)
  eta_full = rep_matrix(mu, N_full) + L_Sigma * z_full;
  eta_marginal = rep_matrix(mu, N_marginal) + L_Sigma * z_marginal;
  eta_partial = rep_matrix(mu, N_partial) + L_Sigma * z_partial;
}

model {
  // Uninformative priors 
  mu ~ normal(0, 5);          
  tau ~ normal(0, 1);
  L_Omega ~ lkj_corr_cholesky(1.0); 
  
  // Standard normals for random noise
  to_vector(z_full) ~ std_normal();
  to_vector(z_marginal) ~ std_normal();
  to_vector(z_partial) ~ std_normal();
  
  // Likelihood: full studies
  for (k in 1:N_full) {
    real TPR_F = inv_logit(eta_full[1, k]);
    real TPR_H = inv_logit(eta_full[2, k]);
    real OR1   = exp(eta_full[3, k]);
    real FPR_F = inv_logit(eta_full[4, k]);
    real FPR_H = inv_logit(eta_full[5, k]);
    real OR0   = exp(eta_full[6, k]);
    
    real JTPR = get_joint_prob(TPR_F, TPR_H, OR1);
    real JFPR = get_joint_prob(FPR_F, FPR_H, OR0);
    
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
    real FPR_F = inv_logit(eta_marginal[4, j]);
    real FPR_H = inv_logit(eta_marginal[5, j]);
    
    y_F_d[j] ~ binomial(N_d_F[j], TPR_F);
    y_F_nd[j] ~ binomial(N_nd_F[j], FPR_F);
    y_H_d[j] ~ binomial(N_d_H[j], TPR_H);
    y_H_nd[j] ~ binomial(N_nd_H[j], FPR_H);
  }
  
  // Likelihood: partial studies
  for (m in 1:N_partial) {
    real TPR_F = inv_logit(eta_partial[1, m]);
    real TPR_H = inv_logit(eta_partial[2, m]);
    real OR1   = exp(eta_partial[3, m]);
    real FPR_F = inv_logit(eta_partial[4, m]);
    real FPR_H = inv_logit(eta_partial[5, m]);
    
    // Diseased (full multinomial)
    real JTPR = get_joint_prob(TPR_F, TPR_H, OR1);
    vector[4] p_d;
    p_d[1] = JTPR;
    p_d[2] = TPR_F - JTPR;
    p_d[3] = TPR_H - JTPR;
    p_d[4] = 1 - sum(p_d[1:3]);
    y_partial_d[m] ~ multinomial(p_d);
    
    // Disease free (binomials)
    y_F_nd_partial[m] ~ binomial(N_nd_partial[m], FPR_F);
    y_H_nd_partial[m] ~ binomial(N_nd_partial[m], FPR_H);
  }
}

generated quantities {
  // Summary estimates
  real TPR_F_summary = inv_logit(mu[1]);
  real TPR_H_summary = inv_logit(mu[2]);
  real OR1_summary   = exp(mu[3]);
  real FPR_F_summary = inv_logit(mu[4]);
  real FPR_H_summary = inv_logit(mu[5]);
  real OR0_summary   = exp(mu[6]);
  
  real Spec_F_summary = 1 - FPR_F_summary;
  real Spec_H_summary = 1 - FPR_H_summary;
  
  // Predictive distributions
  vector[6] eta_pred;
  vector[6] z_pred;
  for(i in 1:6) z_pred[i] = normal_rng(0, 1);
  eta_pred = mu + L_Sigma * z_pred;
  
  real pred_TPR_F = inv_logit(eta_pred[1]);
  real pred_TPR_H = inv_logit(eta_pred[2]);
  real pred_FPR_F = inv_logit(eta_pred[4]);
  real pred_FPR_H = inv_logit(eta_pred[5]);
  real pred_Spec_F = 1 - pred_FPR_F;
  real pred_Spec_H = 1 - pred_FPR_H;
  
  // Covariance matrix
  matrix[6, 6] Omega_corr;
  Omega_corr = multiply_lower_tri_self_transpose(L_Omega);
  
  // Composite rules (AND/OR)
  real JTPR_sum = get_joint_prob(TPR_F_summary, TPR_H_summary, OR1_summary);
  real JFPR_sum = get_joint_prob(FPR_F_summary, FPR_H_summary, OR0_summary);
  
  real Sens_and = JTPR_sum;
  real Spec_and = 1 - JFPR_sum;
  
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
  real log_DOR_F = mu[1] - mu[4];
  real DOR_F = exp(log_DOR_F);
  real log_DOR_H = mu[2] - mu[5];
  real DOR_H = exp(log_DOR_H);
  
  // Composite Rules DOR 
  real log_DOR_and = logit(Sens_and) + logit(Spec_and);
  real DOR_and = exp(log_DOR_and);
  
  real log_DOR_or = logit(Sens_or) + logit(Spec_or);
  real DOR_or = exp(log_DOR_or);
}
