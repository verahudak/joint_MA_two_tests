functions {
  // Helper to solve the quadratic equation for joint probabilities
  // p1 and p2 are the individual TPRs or FPRs, and OR is either OR0 or OR1
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
  // Dimensions
  int<lower=0> N_full;
  int<lower=0> N_m_only;
  int<lower=0> N_marginal;
  int<lower=0> N_partial_paired;
  int<lower=0> N_nishiyama;
  int<lower=0> N_edmonson;
  int<lower=0> N_lindgren;

  // Full 2x2x2 studies (Cohen, Ezike)
  array[N_full, 4] int y_full_d;
  array[N_full, 4] int y_full_nd;

  // M-only studies (Palla)
  array[N_m_only] int tp_m_only;
  array[N_m_only] int fp_m_only;
  array[N_m_only] int N1_m_only;
  array[N_m_only] int N0_m_only;

  // Marginal studies (Ad el Ghany)
  array[N_marginal, 4] int y_M_vs_GS_marg;
  array[N_marginal, 4] int y_R_vs_GS_marg;
  array[N_marginal, 4] int y_M_vs_R_marg;

  // Partially paired studies (Felsenstein)
  array[N_partial_paired, 4] int y_R_vs_GS_pp;
  array[N_partial_paired, 4] int y_M_vs_GS_pp_S;
  array[N_partial_paired, 4] int y_M_vs_R_pp_S;

  // Nishiyama (no GS study)
  array[N_nishiyama, 4] int y_nishiyama;
  
  // Edmonson (check the negatives with extra info)
  array[N_edmonson, 6] int y_edmonson;
  
  // Lindgren (check the negatives)
  array[N_lindgren, 4] int y_lindgren;

}

parameters {
  // Global means
  vector[6] mu; 
  
  // Covariance decomposition
  cholesky_factor_corr[6] L_Omega;
  vector<lower=0>[6] tau; 
  
  // Random noise to add to random effects mean
  matrix[6, N_full] z_full;
  matrix[6, N_m_only] z_m_only;
  matrix[6, N_marginal] z_marginal;
  matrix[6, N_partial_paired] z_partial_paired;
  matrix[6, N_nishiyama] z_nishiyama;
  matrix[6, N_edmonson] z_edmonson;
  matrix[6, N_lindgren] z_lindgren;
  
  // Study-specific prevalences 
  vector<lower=0, upper=1>[N_marginal] pi_marginal;
  vector<lower=0, upper=1>[N_partial_paired] pi_partial_paired;
  vector<lower=0, upper=1>[N_nishiyama] pi_nishiyama;
  vector<lower=0, upper=1>[N_edmonson] pi_edmonson;
  vector<lower=0, upper=1>[N_lindgren] pi_lindgren;
}

transformed parameters {
  // Reconstruct the actual random effects
  matrix[6, N_full] eta_full;
  matrix[6, N_m_only] eta_m_only;
  matrix[6, N_marginal] eta_marginal;
  matrix[6, N_partial_paired] eta_partial_paired;
  matrix[6, N_nishiyama] eta_nishiyama;
  matrix[6, N_edmonson] eta_edmonson;
  matrix[6, N_lindgren] eta_lindgren;
  
  matrix[6, 6] L_Sigma;
  
  L_Sigma = diag_pre_multiply(tau, L_Omega);
  
  // eta = mu + (diag(tau) * L * z)
  eta_full = rep_matrix(mu, N_full) + L_Sigma * z_full;
  eta_m_only = rep_matrix(mu, N_m_only) + L_Sigma * z_m_only;
  eta_marginal = rep_matrix(mu, N_marginal) + L_Sigma * z_marginal;
  eta_partial_paired = rep_matrix(mu, N_partial_paired) + L_Sigma * z_partial_paired;
  eta_nishiyama = rep_matrix(mu, N_nishiyama) + L_Sigma * z_nishiyama;
  eta_edmonson = rep_matrix(mu, N_edmonson) + L_Sigma * z_edmonson;
  eta_lindgren = rep_matrix(mu, N_lindgren) + L_Sigma * z_lindgren;
}

model {
  // Uninformative priors for random effects mean and variance covariance
  mu ~ normal(0, 5);         
  tau ~ normal(0, 1);
  L_Omega ~ lkj_corr_cholesky(1.0); 
  
  // Standard normals for random noise
  to_vector(z_full) ~ std_normal();
  to_vector(z_m_only) ~ std_normal();
  to_vector(z_marginal) ~ std_normal();
  to_vector(z_partial_paired) ~ std_normal();
  to_vector(z_nishiyama) ~ std_normal();
  to_vector(z_edmonson) ~ std_normal();
  to_vector(z_lindgren) ~ std_normal();
  
  // Likelihood: full studies
  for (k in 1:N_full) {
    
    real TPR_M = inv_logit(eta_full[1, k]);
    real TPR_R = inv_logit(eta_full[2, k]);
    real OR1   = exp(eta_full[3, k]);
    real FPR_M = inv_logit(eta_full[4, k]);
    real FPR_R = inv_logit(eta_full[5, k]);
    real OR0   = exp(eta_full[6, k]);
    
    real JTPR = get_joint_prob(TPR_M, TPR_R, OR1);
    real JFPR = get_joint_prob(FPR_M, FPR_R, OR0);

    // Probabilities for diseased and disease free populations
    vector[4] p_d;
    vector[4] p_nd;
    
    // Diseased
    p_d[1] = JTPR;
    p_d[2] = TPR_M - JTPR;
    p_d[3] = TPR_R - JTPR;
    p_d[4] = 1.0 - sum(p_d[1:3]);

    // Non-diseased
    p_nd[1] = JFPR;
    p_nd[2] = FPR_M - JFPR;
    p_nd[3] = FPR_R - JFPR;
    p_nd[4] = 1.0 - sum(p_nd[1:3]);
    
    y_full_d[k] ~ multinomial(p_d);
    y_full_nd[k] ~ multinomial(p_nd);
    
  }
  
  // Likelihood: M only studies
  for (i in 1:N_m_only) {
    real TPR_M = inv_logit(eta_m_only[1, i]);
    real FPR_M = inv_logit(eta_m_only[4, i]);
    
    tp_m_only[i] ~ binomial(N1_m_only[i], TPR_M);
    fp_m_only[i] ~ binomial(N0_m_only[i], FPR_M);
  }
    
    // Likelihood: marginal studies
  for (j in 1:N_marginal) {
    
    real TPR_M = inv_logit(eta_marginal[1, j]);
    real TPR_R = inv_logit(eta_marginal[2, j]);
    real OR1   = exp(eta_marginal[3, j]);
    real FPR_M = inv_logit(eta_marginal[4, j]);
    real FPR_R = inv_logit(eta_marginal[5, j]);
    real OR0   = exp(eta_marginal[6, j]);

    real JTPR = get_joint_prob(TPR_M, TPR_R, OR1);
    real JFPR = get_joint_prob(FPR_M, FPR_R, OR0);
    
    real pi = pi_marginal[j];
    
    // M vs GS
    int TP_M = y_M_vs_GS_marg[j, 1]; 
    int FP_M = y_M_vs_GS_marg[j, 2]; 
    int FN_M = y_M_vs_GS_marg[j, 3]; 
    int TN_M = y_M_vs_GS_marg[j, 4]; 
    
    int N_dis_M = TP_M + FN_M;
    int N_health_M = FP_M + TN_M;
    TP_M ~ binomial(N_dis_M, TPR_M);  
    FP_M ~ binomial(N_health_M, FPR_M);   

    // R vs GS
    int TP_R = y_R_vs_GS_marg[j, 1]; 
    int FP_R = y_R_vs_GS_marg[j, 2]; 
    int FN_R = y_R_vs_GS_marg[j, 3]; 
    int TN_R = y_R_vs_GS_marg[j, 4];
    int N_dis_R = TP_R + FN_R;
    int N_health_R = FP_R + TN_R;
    TP_R ~ binomial(N_dis_R, TPR_R);
    FP_R ~ binomial(N_health_R, FPR_R); 

    // M vs R
    // Get the observed count of people positive on both tests (x11)
    int x11 = y_M_vs_R_marg[j, 1]; 
    // Get the total N for this table
    int N_total = sum(y_M_vs_R_marg[j]);
    // Probability of being positive on BOTH
    real p11_prob = pi * JTPR + (1-pi) * JFPR; 
    // The Likelihood
    x11 ~ binomial(N_total, p11_prob);
  }
  
  // Likelihood: partially paired studies (Felsenstein)
for (k in 1:N_partial_paired) {

  // Get Parameters
  real TPR_M = inv_logit(eta_partial_paired[1, k]);
  real TPR_R = inv_logit(eta_partial_paired[2, k]);
  real OR1   = exp(eta_partial_paired[3, k]);
  real FPR_M = inv_logit(eta_partial_paired[4, k]);
  real FPR_R = inv_logit(eta_partial_paired[5, k]);
  real OR0   = exp(eta_partial_paired[6, k]);
   
  real JTPR = get_joint_prob(TPR_M, TPR_R, OR1);
  real JFPR = get_joint_prob(FPR_M, FPR_R, OR0);
  
  real pi = pi_partial_paired[k];

  // R vs GS
  int TP_R_full = y_R_vs_GS_pp[k, 1];
  int FP_R_full = y_R_vs_GS_pp[k, 2];
  TP_R_full ~ binomial(TP_R_full + y_R_vs_GS_pp[k, 3], TPR_R);
  FP_R_full ~ binomial(FP_R_full + y_R_vs_GS_pp[k, 4], FPR_R);

  // M vs GS (Subset S)
  int TP_M_sub = y_M_vs_GS_pp_S[k, 1];
  int FP_M_sub = y_M_vs_GS_pp_S[k, 2];

  // Diseased in S (All D included)
  TP_M_sub ~ binomial(TP_M_sub + y_M_vs_GS_pp_S[k, 3], TPR_M);

  // Healthy in S (Conditional on R+)
  real prob_M_pos_given_S_ND = JFPR / FPR_R;
  FP_M_sub ~ binomial(FP_M_sub + y_M_vs_GS_pp_S[k, 4], prob_M_pos_given_S_ND);

  // M vs R (Subset S)
  int x11_sub = y_M_vs_R_pp_S[k, 1];
  int N_total_sub = sum(y_M_vs_R_pp_S[k]);
  
  // Probability of inclusion in S
  real P_S = pi + (1-pi) * FPR_R;
  // Probability of joint posiitve in subset S
  real p11_prob = (pi * JTPR + (1-pi) * JFPR) / P_S;

  x11_sub ~ binomial(N_total_sub, p11_prob);
}
  
  // Likelihood: Nishiyama
  for (m in 1:N_nishiyama) {
    real TPR_M = inv_logit(eta_nishiyama[1, m]);
    real TPR_R = inv_logit(eta_nishiyama[2, m]);
    real OR1   = exp(eta_nishiyama[3, m]);
    real FPR_M = inv_logit(eta_nishiyama[4, m]);
    real FPR_R = inv_logit(eta_nishiyama[5, m]);
    real OR0   = exp(eta_nishiyama[6, m]);

    real JTPR = get_joint_prob(TPR_M, TPR_R, OR1);
    real JFPR = get_joint_prob(FPR_M, FPR_R, OR0);
    real pi = pi_nishiyama[m];

    vector[4] p;
    p[1] = pi * JTPR + (1-pi) * JFPR;
    p[2] = pi * (TPR_M - JTPR) + (1-pi) * (FPR_M - JFPR);
    p[3] = pi * (TPR_R - JTPR) + (1-pi) * (FPR_R - JFPR);
    p[4] = 1.0 - sum(p[1:3]);
    y_nishiyama[m] ~ multinomial(p);
  }
  // Likelihood: Edmonson
  for (e in 1:N_edmonson) {
    real TPR_M = inv_logit(eta_edmonson[1, e]);
    real TPR_R = inv_logit(eta_edmonson[2, e]);
    real OR1   = exp(eta_edmonson[3, e]);
    real FPR_M = inv_logit(eta_edmonson[4, e]);
    real FPR_R = inv_logit(eta_edmonson[5, e]);
    real OR0   = exp(eta_edmonson[6, e]);

    real JTPR = get_joint_prob(TPR_M, TPR_R, OR1);
    real JFPR = get_joint_prob(FPR_M, FPR_R, OR0);
    real pi = pi_edmonson[e];

    vector[6] p;
    p[1] = pi * JTPR + (1-pi) * JFPR;
    p[2] = pi * (TPR_R - JTPR) + (1-pi) * (FPR_R - JFPR);
    p[3] = pi * (TPR_M - JTPR);
    p[4] = pi * (1 - TPR_M - TPR_R + JTPR);
    p[5] = (1-pi) * (FPR_M - JFPR);
    p[6] = (1-pi) * (1 - FPR_M - FPR_R + JFPR);
    y_edmonson[e] ~ multinomial(p);
  }

  // 7. Lindgren
  for (l in 1:N_lindgren) {
    real TPR_M = inv_logit(eta_lindgren[1, l]);
    real FPR_M = inv_logit(eta_lindgren[4, l]);
    real FPR_R = inv_logit(eta_lindgren[5, l]);
    real OR0   = exp(eta_lindgren[6, l]);

    real JFPR = get_joint_prob(FPR_M, FPR_R, OR0);
    real pi = pi_lindgren[l];

    vector[4] p;
    p[1] = pi * TPR_M + (1-pi) * JFPR;
    p[2] = pi * (1 - TPR_M) + (1-pi) * (FPR_R - JFPR);
    p[3] = (1-pi) * (FPR_M - JFPR);
    p[4] = 1.0 - sum(p[1:3]);
    y_lindgren[l] ~ multinomial(p);
  }

}

generated quantities {
  // Summary estimates
  real TPR_M_summary = inv_logit(mu[1]);
  real TPR_R_summary = inv_logit(mu[2]);
  real OR1_summary   = exp(mu[3]);
  real FPR_M_summary = inv_logit(mu[4]);
  real FPR_R_summary = inv_logit(mu[5]);
  real OR0_summary   = exp(mu[6]);
  
  real Spec_M_summary = 1 - FPR_M_summary;
  real Spec_R_summary = 1 - FPR_R_summary;

  // Joint summary estimates
  real JTPR_summary = get_joint_prob(TPR_M_summary, TPR_R_summary, OR1_summary);
  real JFPR_summary = get_joint_prob(FPR_M_summary, FPR_R_summary, OR0_summary);

  // Summary JTRP and JFPT
  
  real Sens_and = JTPR_summary;
  real Spec_and = 1 - JFPR_summary;
  
  real Sens_or = TPR_M_summary + TPR_R_summary - JTPR_summary;
  real Spec_or = 1 - FPR_M_summary - FPR_R_summary + JFPR_summary;
  
  // Reconstruct Covariance Matrix (Sigma)
  matrix[6, 6] Sigma;
  Sigma = multiply_lower_tri_self_transpose(L_Sigma);

  // Predictive distribution 
  vector[6] eta_pred;
  vector[6] z_pred;
  
  // Generate a new random effect vector from standard normal
  for(i in 1:6) z_pred[i] = normal_rng(0, 1);
  
  // Project using the covariance structure
  eta_pred = mu + L_Sigma * z_pred;
  
  real pred_TPR_M = inv_logit(eta_pred[1]);
  real pred_TPR_R = inv_logit(eta_pred[2]);
  real pred_FPR_M = inv_logit(eta_pred[4]);
  real pred_FPR_R = inv_logit(eta_pred[5]);
  
  real pred_Spec_M = 1 - pred_FPR_M;
  real pred_Spec_R = 1 - pred_FPR_R;
  
  // Comparative test performance
  
  // Difference in sens/spec (McIsaac - RADT)
  real diff_Sens = TPR_M_summary - TPR_R_summary;
  real diff_Spec = Spec_M_summary - Spec_R_summary;
  
  // Probabilities of superior performance
  // Is McIsaac better than RADT? (1 if yes, 0 if no)
  real prob_better_Sens = (TPR_M_summary > TPR_R_summary) ? 1.0 : 0.0;
  real prob_better_Spec = (Spec_M_summary > Spec_R_summary) ? 1.0 : 0.0;
  // Sens_M > Sens_R AND Spec_M > Spec_R
  real prob_better_Both = (prob_better_Sens * prob_better_Spec);
  
  // Diagnostic odds ratios (DOR)
  real log_DOR_M = mu[1] - mu[4];
  real DOR_M = exp(log_DOR_M);
  real log_DOR_R = mu[2] - mu[5];
  real DOR_R = exp(log_DOR_R);
  
  // Composite Rules DOR 
  real log_DOR_and = logit(Sens_and) + logit(Spec_and);
  real DOR_and = exp(log_DOR_and);
  
  real log_DOR_or = logit(Sens_or) + logit(Spec_or);
  real DOR_or = exp(log_DOR_or);
}
