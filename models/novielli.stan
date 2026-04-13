data {
  int<lower=1> N_studies;
  // Stratified counts
  // [N_studies, 1]: Test 1 is negative, [N_studies, 2]: Test 1 is positive
  int d[N_studies, 2];    // Number of diseased patients per Test 1 stratum
  int h[N_studies, 2];    // Number of healthy patients per Test 1 stratum
  
  int tp[N_studies, 2];   // Test 2 True Positives per Test 1 stratum
  int tn[N_studies, 2];   // Test 2 True Negatives per Test 1 stratum 
}

parameters {
  // 1. Test 1 accuracy
  vector[2] mu_t1;              // [logit_sens, logit_spec]
  vector<lower=0>[2] sigma_t1;
  
  // 2. Test 2 accuracy (nested within Test 1 results)
  // [Test1_Stratum, parameter] where parameter: 1=sens, 2=spec
  vector[2] mu_t2_cond[2];      
  vector<lower=0>[2] sigma_t2_cond[2];
  corr_matrix[2] Rho_t2[2];     // Correlation between sens/spec
  
  // Study-level random effects
  vector[2] theta_t1[N_studies];
  vector[2] theta_t2[N_studies, 2];
}

model {
  // Vague priors
  mu_t1 ~ normal(0, 5);
  sigma_t1 ~ normal(0, 1);
  
  for (f in 1:2) {
    mu_t2_cond[f] ~ normal(0, 5);
    sigma_t2_cond[f] ~ normal(0, 1);
    Rho_t2[f] ~ lkj_corr(1);
  }

  for (i in 1:N_studies) {
    // Likelihood for Test 1
    d[i, 2] ~ binomial_logit(d[i, 1] + d[i, 2], theta_t1[i, 1]);
    h[i, 1] ~ binomial_logit(h[i, 1] + h[i, 2], theta_t1[i, 2]);
    
    theta_t1[i] ~ normal(mu_t1, sigma_t1);

    for (f in 1:2) {
      // Likelihood for Test 2 conditioned on Test 1 result
      tp[i, f] ~ binomial_logit(d[i, f], theta_t2[i, f, 1]);
      tn[i, f] ~ binomial_logit(h[i, f], theta_t2[i, f, 2]);
      
      theta_t2[i, f] ~ multi_normal_cholesky(mu_t2_cond[f], 
                       diag_pre_multiply(sigma_t2_cond[f], 
                       cholesky_decompose(Rho_t2[f])));
    }
  }
}

generated quantities {
  /*
     BLOCK 1: FEMUR FIRST (Condition Humerus on Femur)
  */
  /*
  real sens_femur = inv_logit(mu_t1[1]);
  real spec_femur = inv_logit(mu_t1[2]);
  
  real sens_hum_if_fem_neg = inv_logit(mu_t2_cond[1, 1]);
  real spec_hum_if_fem_neg = inv_logit(mu_t2_cond[1, 2]);
  real sens_hum_if_fem_pos = inv_logit(mu_t2_cond[2, 1]);
  real spec_hum_if_fem_pos = inv_logit(mu_t2_cond[2, 2]);

  real sens_humerus_overall = (sens_femur * sens_hum_if_fem_pos) + 
                              ((1 - sens_femur) * sens_hum_if_fem_neg);
  real spec_humerus_overall = (spec_femur * spec_hum_if_fem_neg) + 
                              ((1 - spec_femur) * spec_hum_if_fem_pos);

  real joint_sens_or_f1st = 1 - ((1 - sens_femur) * (1 - sens_hum_if_fem_neg));
  real joint_spec_or_f1st = spec_femur * spec_hum_if_fem_pos;
  
  real joint_sens_and_f1st = sens_femur * sens_hum_if_fem_pos;
  real joint_spec_and_f1st = 1 - ((1 - spec_femur) * (1 - spec_hum_if_fem_neg));
  */

  // BLOCK 2: HUMERUS FIRST (Condition Femur on Humerus)

  real sens_humerus = inv_logit(mu_t1[1]);
  real spec_humerus = inv_logit(mu_t1[2]);
  
  real sens_fem_if_hum_neg = inv_logit(mu_t2_cond[1, 1]);
  real spec_fem_if_hum_neg = inv_logit(mu_t2_cond[1, 2]);
  real sens_fem_if_hum_pos = inv_logit(mu_t2_cond[2, 1]);
  real spec_fem_if_hum_pos = inv_logit(mu_t2_cond[2, 2]);

  real sens_femur_overall = (sens_humerus * sens_fem_if_hum_pos) + 
                            ((1 - sens_humerus) * sens_fem_if_hum_neg);
  real spec_femur_overall = (spec_humerus * spec_fem_if_hum_neg) + 
                            ((1 - spec_humerus) * spec_fem_if_hum_pos);

  real joint_sens_or_h1st = 1 - ((1 - sens_humerus) * (1 - sens_fem_if_hum_neg));
  real joint_spec_or_h1st = spec_humerus * spec_fem_if_hum_pos;
  
  real joint_sens_and_h1st = sens_humerus * sens_fem_if_hum_pos;
  real joint_spec_and_h1st = 1 - ((1 - spec_humerus) * (1 - spec_fem_if_hum_neg));

}