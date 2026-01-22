full_ma_ultrasound <- "model {

# Priors for the 6D mean vector (Eta.Xi)
# 1=logit TPR_F, 2=logit TPR_H, 3=logOR1, 4=logit FPR_F, 5=logit FPR_H, 6=logOR0
for(i in 1:6) {
  Eta.Xi[i] ~ dnorm(0, 0.01)
}

# Priors for the 6x6 precision matrix (Omega)
Omega[1:6, 1:6] ~ dwish(R_mat[1:6, 1:6], 6) # Vague Wishart prior
Tau[1:6, 1:6] <- inverse(Omega) # Covariance matrix

# Likelihoods for each study design

# For the N_full studies with Full 2x2x2 Cross-Verified Data
for(k in 1:N_full) {
  # Study-specific parameters from multivariate normal
  eta.xi_full[k, 1:6] ~ dmnorm(Eta.Xi[], Omega[,])
  
  # Transform from logit/log scale to probability scale
  TPR_F_full[k] <- ilogit(eta.xi_full[k,1]) # Femur Sensitivity
  TPR_H_full[k] <- ilogit(eta.xi_full[k,2]) # Humerus Sensitivity
  FPR_F_full[k] <- ilogit(eta.xi_full[k,4]) # Femur False Positive Rate
  FPR_H_full[k] <- ilogit(eta.xi_full[k,5]) # Humerus False Positive Rate
  OR1_full[k] <- exp(eta.xi_full[k,3])     # Odds Ratio (diseased group)
  OR0_full[k] <- exp(eta.xi_full[k,6])     # Odds Ratio (non-diseased group)
  
  # Calculate Joint TPR (JTPR) from marginals and OR1 (diseased group)
  is_indep1[k] <- step(OR1_full[k] - 0.99999) * step(1.00001 - OR1_full[k])
  p_indep1[k] <- TPR_F_full[k] * TPR_H_full[k]
  a1[k] <- OR1_full[k] - 1
  b1[k] <- -OR1_full[k]*(TPR_F_full[k] + TPR_H_full[k]) - 1 + (TPR_F_full[k] + TPR_H_full[k])
  d1[k] <- OR1_full[k] * TPR_F_full[k] * TPR_H_full[k]
  discriminant1[k] <- sqrt(b1[k]*b1[k] - 4*a1[k]*d1[k])
  safe_denom1[k] <- 2 * (a1[k] + is_indep1[k])
  quadratic1[k] <- (-b1[k] - discriminant1[k]) / safe_denom1[k]
  JTPR_full[k] <- is_indep1[k] * p_indep1[k] + (1 - is_indep1[k]) * quadratic1[k]
  
  # Calculate Joint FPR (JFPR) from marginals and OR0 (non-diseased group)
  is_indep0[k] <- step(OR0_full[k] - 0.99999) * step(1.00001 - OR0_full[k])
  p_indep0[k] <- FPR_F_full[k] * FPR_H_full[k]
  a0[k] <- OR0_full[k] - 1
  b0[k] <- -OR0_full[k]*(FPR_F_full[k] + FPR_H_full[k]) - 1 + (FPR_F_full[k] + FPR_H_full[k])
  d0[k] <- OR0_full[k] * FPR_F_full[k] * FPR_H_full[k]
  discriminant0[k] <- sqrt(b0[k]*b0[k] - 4*a0[k]*d0[k])
  safe_denom0[k] <- 2 * (a0[k] + is_indep0[k])
  quadratic0[k] <- (-b0[k] - discriminant0[k]) / safe_denom0[k]
  JFPR_full[k] <- is_indep0[k] * p_indep0[k] + (1 - is_indep0[k]) * quadratic0[k]

  # Likelihood for diseased group
  y_full_d[k, 1:4] ~ dmulti(p_d[k, 1:4], N_d_full[k])
  p_d[k,1] <- JTPR_full[k]                      # H+F+
  p_d[k,2] <- TPR_F_full[k] - JTPR_full[k]      # H-F+
  p_d[k,3] <- TPR_H_full[k] - JTPR_full[k]      # H+F-
  p_d[k,4] <- 1 - (p_d[k,1] + p_d[k,2] + p_d[k,3]) # H-F-

  # Likelihood for non-diseased group 
  y_full_nd[k, 1:4] ~ dmulti(p_nd[k, 1:4], N_nd_full[k])
  p_nd[k,1] <- JFPR_full[k]                       # H+F+
  p_nd[k,2] <- FPR_F_full[k] - JFPR_full[k]       # H-F+
  p_nd[k,3] <- FPR_H_full[k] - JFPR_full[k]       # H+F-
  p_nd[k,4] <- 1 - (p_nd[k,1] + p_nd[k,2] + p_nd[k,3]) # H-F-
}

# For the N_marginal studies with only separate marginals
for(j in 1:N_marginal) {

  eta.xi_marginal[j, 1:6] ~ dmnorm(Eta.Xi[], Omega[,])
  TPR_F_marginal[j] <- ilogit(eta.xi_marginal[j,1]) # Femur Sensitivity
  TPR_H_marginal[j] <- ilogit(eta.xi_marginal[j,2]) # Humerus Sensitivity
  FPR_F_marginal[j] <- ilogit(eta.xi_marginal[j,4]) # Femur FPR
  FPR_H_marginal[j] <- ilogit(eta.xi_marginal[j,5]) # Humerus FPR
  
  # Likelihoods defined with separate denominators for each test
  y_F_d[j] ~ dbin(TPR_F_marginal[j], N_d_F[j])   # Femur data in diseased
  y_F_nd[j] ~ dbin(FPR_F_marginal[j], N_nd_F[j]) # Femur data in disease free
   
  y_H_d[j] ~ dbin(TPR_H_marginal[j], N_d_H[j])   # Humerus data in diseased
  y_H_nd[j] ~ dbin(FPR_H_marginal[j], N_nd_H[j]) # Humerus data in disease free

}

# For the N_partial study (Vintzileos) (2x2x2 data for diseased, separate marginals for non-diseased)
for(m in 1:N_partial) {
  
  eta.xi_partial[m, 1:6] ~ dmnorm(Eta.Xi[], Omega[,])
  TPR_F_partial[m] <- ilogit(eta.xi_partial[m,1])
  TPR_H_partial[m] <- ilogit(eta.xi_partial[m,2])
  OR1_partial[m] <- exp(eta.xi_partial[m,3])

  FPR_F_partial[m] <- ilogit(eta.xi_partial[m,4])
  FPR_H_partial[m] <- ilogit(eta.xi_partial[m,5])
  
  # Note: eta.xi_partial[m,6] (OR0) is not used in this likelihood.
   
  # Calculate Joint TPR (JTPR) from marginals and OR1 (diseased group)
  is_indep1_partial[m] <- step(OR1_partial[m] - 0.99999) * step(1.00001 - OR1_partial[m])
  p_indep1_partial[m] <- TPR_F_partial[m] * TPR_H_partial[m]
  a1_partial[m] <- OR1_partial[m] - 1
  b1_partial[m] <- -OR1_partial[m]*(TPR_F_partial[m] + TPR_H_partial[m]) - 1 + (TPR_F_partial[m] + TPR_H_partial[m])
  d1_partial[m] <- OR1_partial[m] * TPR_F_partial[m] * TPR_H_partial[m]
  discriminant1_partial[m] <- sqrt(b1_partial[m]*b1_partial[m] - 4*a1_partial[m]*d1_partial[m]) 
  safe_denom1_partial[m] <- 2 * (a1_partial[m] + is_indep1_partial[m])
  quadratic1_partial[m] <- (-b1_partial[m] - discriminant1_partial[m]) / safe_denom1_partial[m]
  JTPR_partial[m] <- is_indep1_partial[m] * p_indep1_partial[m] + (1 - is_indep1_partial[m]) * quadratic1_partial[m]
  
  # Diseased group: full 2x2 table
  y_partial_d[m, 1:4] ~ dmulti(p_d_partial[m, 1:4], N_d_partial[m])
  p_d_partial[m,1] <- JTPR_partial[m]                      # H+F+
  p_d_partial[m,2] <- TPR_F_partial[m] - JTPR_partial[m]  # H-F+
  p_d_partial[m,3] <- TPR_H_partial[m] - JTPR_partial[m]  # H+F-
  p_d_partial[m,4] <- 1 - (p_d_partial[m,1] + p_d_partial[m,2] + p_d_partial[m,3]) # H-F-

  # Disease free group: separate marginals
  y_F_nd_partial[m] ~ dbin(FPR_F_partial[m], N_nd_partial[m])   # Femur data in disease free
  y_H_nd_partial[m] ~ dbin(FPR_H_partial[m], N_nd_partial[m])   # Humerus data in disease free
}

# Summary estimates and predictive distributions

# Summary estimates
TPR_F <- ilogit(Eta.Xi[1])
TPR_H <- ilogit(Eta.Xi[2])
OR1 <- exp(Eta.Xi[3])
FPR_F <- ilogit(Eta.Xi[4])
FPR_H <- ilogit(Eta.Xi[5])
OR0 <- exp(Eta.Xi[6])
Spec_F <- 1 - FPR_F
Spec_H <- 1 - FPR_H

# Predictive distributions
eta.xi_pred[1:6] ~ dmnorm(Eta.Xi[], Omega[,])

# Transform to get the predicted outcomes for the new study
pred_TPR_F <- ilogit(eta.xi_pred[1])
pred_TPR_H <- ilogit(eta.xi_pred[2])
pred_FPR_F <- ilogit(eta.xi_pred[4])
pred_FPR_H <- ilogit(eta.xi_pred[5])

# Calculate predicted specificities
pred_Spec_F <- 1 - pred_FPR_F
pred_Spec_H <- 1 - pred_FPR_H

# Accuracy of tests in sequence
# JTPR from summary marginals and OR1
is_indep1_summary <- step(OR1 - 0.99999) * step(1.00001 - OR1)
p_indep1_summary <- TPR_F * TPR_H
a1_summary <- OR1 - 1
b1_summary <- -OR1*(TPR_F + TPR_H) - 1 + (TPR_F + TPR_H)
d1_summary <- OR1 * TPR_F * TPR_H
discriminant1_summary <- sqrt(b1_summary*b1_summary - 4*a1_summary*d1_summary)
safe_denom1_summary <- 2 * (a1_summary + is_indep1_summary)
quadratic1_summary <- (-b1_summary - discriminant1_summary) / safe_denom1_summary
JTPR_summary <- is_indep1_summary * p_indep1_summary + (1 - is_indep1_summary) * quadratic1_summary

# JFPR from summary marginals and OR0 
is_indep0_summary <- step(OR0 - 0.99999) * step(1.00001 - OR0)
p_indep0_summary <- FPR_F * FPR_H
a0_summary <- OR0 - 1
b0_summary <- -OR0*(FPR_F + FPR_H) - 1 + (FPR_F + FPR_H)
d0_summary <- OR0 * FPR_F * FPR_H
discriminant0_summary <- sqrt(b0_summary*b0_summary - 4*a0_summary*d0_summary)
safe_denom0_summary <- 2 * (a0_summary + is_indep0_summary)
quadratic0_summary <- (-b0_summary - discriminant0_summary) / safe_denom0_summary
JFPR_summary <- is_indep0_summary * p_indep0_summary + (1 - is_indep0_summary) * quadratic0_summary


# Sens/Spec for AND rule
# Positive if F+ AND H+
Sens_and <- JTPR_summary
Spec_and <- 1 - JFPR_summary

# Sens/Spec for OR rule
# Positive if F+ OR H+
Sens_or <- TPR_F + TPR_H - JTPR_summary
Spec_or <- 1 - FPR_F - FPR_H + JFPR_summary

}"



full_ma_mcisaac <- "model {

# Priors for the 6D mean vector (Eta.Xi)
# 1=TPR_M, 2=TPR_R, 3=logOR1, 4=FPR_M, 5=FPR_R, 6=logOR0
for(i in 1:6) {
  Eta.Xi[i] ~ dnorm(0, 0.01)
}

# Priors for the 6x6 covariance matrix 
Omega[1:6, 1:6] ~ dwish(R_mat[1:6, 1:6], 7) # Vague Wishart prior
Sigma[1:6, 1:6] <- inverse(Omega)


# Likelihoods for each study design

# For the 2 studies with Full 2x2x2 Cross-Verified Data
for(k in 1:N_full) {
  eta.xi_full[k, 1:6] ~ dmnorm(Eta.Xi[], Omega[,])
  
  TPR_M_full[k] <- ilogit(eta.xi_full[k,1])
  TPR_R_full[k] <- ilogit(eta.xi_full[k,2])
  FPR_M_full[k] <- ilogit(eta.xi_full[k,4])
  FPR_R_full[k] <- ilogit(eta.xi_full[k,5])
  OR1_full[k] <- exp(eta.xi_full[k,3]) # Odds Ratio (diseased)
  OR0_full[k] <- exp(eta.xi_full[k,6]) # Odds Ratio (non-diseased)
  
  # OR1
  is_indep1[k] <- step(OR1_full[k] - 0.99999) * step(1.00001 - OR1_full[k])
  p_indep1[k] <- TPR_M_full[k] * TPR_R_full[k]
  a1[k] <- OR1_full[k] - 1
  b1[k] <- -OR1_full[k]*(TPR_M_full[k] + TPR_R_full[k]) - 1 + (TPR_M_full[k] + TPR_R_full[k])
  d1[k] <- OR1_full[k] * TPR_M_full[k] * TPR_R_full[k]
  discriminant1[k] <- sqrt(b1[k]*b1[k] - 4*a1[k]*d1[k])
  safe_denom1[k] <- 2 * (a1[k] + is_indep1[k])
  quadratic1[k] <- (-b1[k] - discriminant1[k]) / safe_denom1[k]
  JTPR_full[k] <- is_indep1[k] * p_indep1[k] + (1 - is_indep1[k]) * quadratic1[k]
  
  # OR0
  is_indep0[k] <- step(OR0_full[k] - 0.99999) * step(1.00001 - OR0_full[k])
  p_indep0[k] <- FPR_M_full[k] * FPR_R_full[k]
  a0[k] <- OR0_full[k] - 1
  b0[k] <- -OR0_full[k]*(FPR_M_full[k] + FPR_R_full[k]) - 1 + (FPR_M_full[k] + FPR_R_full[k])
  d0[k] <- OR0_full[k] * FPR_M_full[k] * FPR_R_full[k]
  discriminant0[k] <- sqrt(b0[k]*b0[k] - 4*a0[k]*d0[k])
  safe_denom0[k] <- 2 * (a0[k] + is_indep0[k])
  quadratic0[k] <- (-b0[k] - discriminant0[k]) / safe_denom0[k]
  JFPR_full[k] <- is_indep0[k] * p_indep0[k] + (1 - is_indep0[k]) * quadratic0[k]

  y_full_d[k, 1:4] ~ dmulti(p_d[k, 1:4], N_d_full[k])
  p_d[k,1] <- JTPR_full[k]
  p_d[k,2] <- TPR_M_full[k] - JTPR_full[k]
  p_d[k,3] <- TPR_R_full[k] - JTPR_full[k]
  p_d[k,4] <- 1 - (p_d[k,1] + p_d[k,2] + p_d[k,3])

  y_full_nd[k, 1:4] ~ dmulti(p_nd[k, 1:4], N_nd_full[k])
  p_nd[k,1] <- JFPR_full[k]
  p_nd[k,2] <- FPR_M_full[k] - JFPR_full[k]
  p_nd[k,3] <- FPR_R_full[k] - JFPR_full[k]
  p_nd[k,4] <- 1 - (p_nd[k,1] + p_nd[k,2] + p_nd[k,3])
}

# For the 1 study with M only data: Palla
for(i in 1:N_m_only) {
  eta.xi_m_only[i, 1:6] ~ dmnorm(Eta.Xi[], Omega[,])
  tp_m_only[i] ~ dbin(ilogit(eta.xi_m_only[i,1]), N1_m_only[i])
  fp_m_only[i] ~ dbin(ilogit(eta.xi_m_only[i,4]), N0_m_only[i])
}

# For the not cross verified study with data on all 3 tables
for(j in 1:N_not_cross_verified) {

  eta.xi_ncv[j, 1:6] ~ dmnorm(Eta.Xi[], Omega[,])
  TPR_M_ncv[j] <- ilogit(eta.xi_ncv[j,1])
  TPR_R_ncv[j] <- ilogit(eta.xi_ncv[j,2])
  FPR_M_ncv[j] <- ilogit(eta.xi_ncv[j,4])
  FPR_R_ncv[j] <- ilogit(eta.xi_ncv[j,5])
  OR1_ncv[j]   <- exp(eta.xi_ncv[j,3])
  OR0_ncv[j]   <- exp(eta.xi_ncv[j,6])

  is_indep1_ncv[j] <- step(OR1_ncv[j] - 0.99999) * step(1.00001 - OR1_ncv[j])
  p_indep1_ncv[j]  <- TPR_M_ncv[j] * TPR_R_ncv[j]
  a1_ncv[j] <- OR1_ncv[j] - 1
  b1_ncv[j] <- -OR1_ncv[j]*(TPR_M_ncv[j] + TPR_R_ncv[j]) - 1 + (TPR_M_ncv[j] + TPR_R_ncv[j])
  d1_ncv[j] <- OR1_ncv[j] * TPR_M_ncv[j] * TPR_R_ncv[j]
  discriminant1_ncv[j] <- sqrt(max(0, b1_ncv[j]*b1_ncv[j] - 4*a1_ncv[j]*d1_ncv[j]))
  safe_denom1_ncv[j] <- 2 * (a1_ncv[j] + is_indep1_ncv[j])
  quadratic1_ncv[j] <- (-b1_ncv[j] - discriminant1_ncv[j]) / safe_denom1_ncv[j]
  JTPR_ncv[j] <- is_indep1_ncv[j] * p_indep1_ncv[j] + (1 - is_indep1_ncv[j]) * quadratic1_ncv[j]

  is_indep0_ncv[j] <- step(OR0_ncv[j] - 0.99999) * step(1.00001 - OR0_ncv[j])
  p_indep0_ncv[j]  <- FPR_M_ncv[j] * FPR_R_ncv[j]
  a0_ncv[j] <- OR0_ncv[j] - 1
  b0_ncv[j] <- -OR0_ncv[j]*(FPR_M_ncv[j] + FPR_R_ncv[j]) - 1 + (FPR_M_ncv[j] + FPR_R_ncv[j])
  d0_ncv[j] <- OR0_ncv[j] * FPR_M_ncv[j] * FPR_R_ncv[j]
  discriminant0_ncv[j] <- sqrt(max(0, b0_ncv[j]*b0_ncv[j] - 4*a0_ncv[j]*d0_ncv[j]))
  safe_denom0_ncv[j] <- 2 * (a0_ncv[j] + is_indep0_ncv[j])
  quadratic0_ncv[j] <- (-b0_ncv[j] - discriminant0_ncv[j]) / safe_denom0_ncv[j]
  JFPR_ncv[j] <- is_indep0_ncv[j] * p_indep0_ncv[j] + (1 - is_indep0_ncv[j]) * quadratic0_ncv[j]

  # prevalence
  pi_ncv[j] ~ dunif(0,1)
  
  # Underlying probabilities
  p_ncv_d[j,1] <- pi_ncv[j] * JTPR_ncv[j]
  p_ncv_d[j,2] <- pi_ncv[j] * (TPR_M_ncv[j] - JTPR_ncv[j])
  p_ncv_d[j,3] <- pi_ncv[j] * (TPR_R_ncv[j] - JTPR_ncv[j]) 
  p_ncv_d[j,4] <- pi_ncv[j] * (1 - TPR_M_ncv[j] - TPR_R_ncv[j] + JTPR_ncv[j])

  p_ncv_nd[j,1] <- (1 - pi_ncv[j]) * JFPR_ncv[j]
  p_ncv_nd[j,2] <- (1 - pi_ncv[j]) * (FPR_M_ncv[j] - JFPR_ncv[j])  
  p_ncv_nd[j,3] <- (1 - pi_ncv[j]) * (FPR_R_ncv[j] - JFPR_ncv[j]) 
  p_ncv_nd[j,4] <- (1 - pi_ncv[j]) * (1 - FPR_M_ncv[j] - FPR_R_ncv[j] + JFPR_ncv[j])


  y_M_vs_GS_ncv[j, 1:4] ~ dmulti(p_M_vs_GS[j,], N_M_vs_GS_ncv[j])
  y_R_vs_GS_ncv[j, 1:4] ~ dmulti(p_R_vs_GS[j,], N_R_vs_GS_ncv[j])
  y_M_vs_R_ncv[j, 1:4] ~ dmulti(p_M_vs_R[j,], N_M_vs_R_ncv[j])
  
  # M vs GS table probabilities
  p_M_vs_GS[j,1] <- p_ncv_d[j,1] + p_ncv_d[j,2] 
  p_M_vs_GS[j,2] <- p_ncv_nd[j,1] + p_ncv_nd[j,2] 
  p_M_vs_GS[j,3] <- p_ncv_d[j,3] + p_ncv_d[j,4]  
  p_M_vs_GS[j,4] <- p_ncv_nd[j,3] + p_ncv_nd[j,4] 

  # R vs GS table probabilities
  p_R_vs_GS[j,1] <- p_ncv_d[j,1] + p_ncv_d[j,3]  
  p_R_vs_GS[j,2] <- p_ncv_nd[j,1] + p_ncv_nd[j,3] 
  p_R_vs_GS[j,3] <- p_ncv_d[j,2] + p_ncv_d[j,4]  
  p_R_vs_GS[j,4] <- p_ncv_nd[j,2] + p_ncv_nd[j,4] 

 
  # Calculate the marginal probability of M+ and R+ from the GS tables
  total_M_pos <- p_M_vs_GS[j,1] + p_M_vs_GS[j,2]  # P(M+)
  total_R_pos <- p_R_vs_GS[j,1] + p_R_vs_GS[j,2]  # P(R+)

  # Anchor the (M+/R+) cell using the full underlying probabilities
  p_M_vs_R[j,1] <- p_ncv_d[j,1] + p_ncv_nd[j,1]

  # Fill remaining cells by subtraction to maintain consistency
  p_M_vs_R[j,2] <- total_M_pos - p_M_vs_R[j,1]      # P(M+, R-) = P(M+) - P(M+, R+)
  p_M_vs_R[j,3] <- total_R_pos - p_M_vs_R[j,1]      # P(M-, R+) = P(R+) - P(M+, R+)
  p_M_vs_R[j,4] <- 1 - (p_M_vs_R[j,1] + p_M_vs_R[j,2] + p_M_vs_R[j,3])

}

# For the partially paired design with a non-random subset (Felsenstein)
for (k in 1:N_partial_paired) {

  eta.xi_pp[k, 1:6] ~ dmnorm(Eta.Xi[], Omega[,])
  TPR_M_pp[k] <- ilogit(eta.xi_pp[k,1])
  TPR_R_pp[k] <- ilogit(eta.xi_pp[k,2])
  FPR_M_pp[k] <- ilogit(eta.xi_pp[k,4])
  FPR_R_pp[k] <- ilogit(eta.xi_pp[k,5])
  OR1_pp[k]   <- exp(eta.xi_pp[k,3])
  OR0_pp[k]   <- exp(eta.xi_pp[k,6])
  
  is_indep1_pp[k] <- step(OR1_pp[k] - 0.99999) * step(1.00001 - OR1_pp[k])
  p_indep1_pp[k]  <- TPR_M_pp[k] * TPR_R_pp[k]
  a1_pp[k] <- OR1_pp[k] - 1
  b1_pp[k] <- -OR1_pp[k]*(TPR_M_pp[k] + TPR_R_pp[k]) - 1 + (TPR_M_pp[k] + TPR_R_pp[k])
  d1_pp[k] <- OR1_pp[k] * TPR_M_pp[k] * TPR_R_pp[k]
  discriminant1_pp[k] <- sqrt(max(0, b1_pp[k]*b1_pp[k] - 4*a1_pp[k]*d1_pp[k]))
  safe_denom1_pp[k] <- 2 * (a1_pp[k] + is_indep1_pp[k])
  quadratic1_pp[k] <- (-b1_pp[k] - discriminant1_pp[k]) / safe_denom1_pp[k]
  JTPR_pp[k] <- is_indep1_pp[k] * p_indep1_pp[k] + (1 - is_indep1_pp[k]) * quadratic1_pp[k]

  is_indep0_pp[k] <- step(OR0_pp[k] - 0.99999) * step(1.00001 - OR0_pp[k])
  p_indep0_pp[k]  <- FPR_M_pp[k] * FPR_R_pp[k]
  a0_pp[k] <- OR0_pp[k] - 1
  b0_pp[k] <- -OR0_pp[k]*(FPR_M_pp[k] + FPR_R_pp[k]) - 1 + (FPR_M_pp[k] + FPR_R_pp[k])
  d0_pp[k] <- OR0_pp[k] * FPR_M_pp[k] * FPR_R_pp[k]
  discriminant0_pp[k] <- sqrt(max(0, b0_pp[k]*b0_pp[k] - 4*a0_pp[k]*d0_pp[k]))
  safe_denom0_pp[k] <- 2 * (a0_pp[k] + is_indep0_pp[k])
  quadratic0_pp[k] <- (-b0_pp[k] - discriminant0_pp[k]) / safe_denom0_pp[k]
  JFPR_pp[k] <- is_indep0_pp[k] * p_indep0_pp[k] + (1 - is_indep0_pp[k]) * quadratic0_pp[k]
  
  # Study-specific prevalence
  pi_pp[k] ~ dunif(0,1)

  # Underlying absolute probabilities
  p_d_pp[k,1] <- pi_pp[k] * JTPR_pp[k]
  p_d_pp[k,2] <- pi_pp[k] * (TPR_M_pp[k] - JTPR_pp[k])
  p_d_pp[k,3] <- pi_pp[k] * (TPR_R_pp[k] - JTPR_pp[k])
  p_d_pp[k,4] <- pi_pp[k] * (1 - TPR_M_pp[k] - TPR_R_pp[k] + JTPR_pp[k])

  p_nd_pp[k,1] <- (1-pi_pp[k]) * JFPR_pp[k]
  p_nd_pp[k,2] <- (1-pi_pp[k]) * (FPR_M_pp[k] - JFPR_pp[k])
  p_nd_pp[k,3] <- (1-pi_pp[k]) * (FPR_R_pp[k] - JFPR_pp[k])
  p_nd_pp[k,4] <- (1-pi_pp[k]) * (1 - FPR_M_pp[k] - FPR_R_pp[k] + JFPR_pp[k])

  # R vs GS table (for the FULL population) 
  p_R_vs_GS_full[k,1] <- p_d_pp[k,1] + p_d_pp[k,3]   # P(R+, D)
  p_R_vs_GS_full[k,2] <- p_nd_pp[k,1] + p_nd_pp[k,3] # P(R+, ND)
  p_R_vs_GS_full[k,3] <- p_d_pp[k,2] + p_d_pp[k,4]   # P(R-, D)
  p_R_vs_GS_full[k,4] <- p_nd_pp[k,2] + p_nd_pp[k,4] # P(R-, ND)
  
  y_R_vs_GS_pp[k, 1:4] ~ dmulti(p_R_vs_GS_full[k,], N_total_pp[k])

  # Tables for the RESTRICTED SUBSET
  # Probability of being in the subset S = {R=1 or GS=1}
  # P(S) = P(GS=1) + P(R=1, GS=0)
  P_S_pp[k] <- pi_pp[k] + p_nd_pp[k,1] + p_nd_pp[k,3]

  # M vs GS (conditional on subset S)
  p_M_vs_GS_S[k,1] <- (p_d_pp[k,1] + p_d_pp[k,2]) / P_S_pp[k]     
  p_M_vs_GS_S[k,2] <- p_nd_pp[k,1] / P_S_pp[k]                     
  p_M_vs_GS_S[k,3] <- (p_d_pp[k,3] + p_d_pp[k,4]) / P_S_pp[k]      
  p_M_vs_GS_S[k,4] <- p_nd_pp[k,3] / P_S_pp[k]                     
  
  y_M_vs_GS_pp_S[k, 1:4] ~ dmulti(p_M_vs_GS_S[k,], N_subset_pp[k])

  # M vs R (conditional on subset S)
  total_M_pos_S <- p_M_vs_GS_S[k,1] + p_M_vs_GS_S[k,2]    
  total_R_pos_S <- (p_R_vs_GS_full[k,1] + p_R_vs_GS_full[k,2]) / P_S_pp[k]

  p_M_vs_R_S[k,1] <- (p_d_pp[k,1] + p_nd_pp[k,1]) / P_S_pp[k]  # P(M+, R+ | S)
  p_M_vs_R_S[k,2] <- total_M_pos_S - p_M_vs_R_S[k,1]
  p_M_vs_R_S[k,3] <- total_R_pos_S - p_M_vs_R_S[k,1]
  p_M_vs_R_S[k,4] <- 1 - (p_M_vs_R_S[k,1] + p_M_vs_R_S[k,2] + p_M_vs_R_S[k,3])
  
  y_M_vs_R_pp_S[k, 1:4] ~ dmulti(p_M_vs_R_S[k,], N_subset_pp[k])
}

# For Nishiyama (imperfect reference standard)
for(m in 1:N_nishiyama) {
  eta.xi_nishiyama[m, 1:6] ~ dmnorm(Eta.Xi[], Omega[,])
  
  TPR_M_n <- ilogit(eta.xi_nishiyama[m,1])
  TPR_R_n <- ilogit(eta.xi_nishiyama[m,2])
  FPR_M_n <- ilogit(eta.xi_nishiyama[m,4])
  FPR_R_n <- ilogit(eta.xi_nishiyama[m,5])
  OR1_n <- exp(eta.xi_nishiyama[m,3])
  OR0_n <- exp(eta.xi_nishiyama[m,6])
  
  # OR1
  is_indep1_n <- step(OR1_n - 0.99999) * step(1.00001 - OR1_n)
  p_indep1_n <- TPR_M_n * TPR_R_n
  a1_n <- OR1_n - 1
  b1_n <- -OR1_n*(TPR_M_n + TPR_R_n) - 1 + (TPR_M_n + TPR_R_n)
  d1_n <- OR1_n * TPR_M_n * TPR_R_n
  discriminant1_n <- sqrt(b1_n*b1_n - 4*a1_n*d1_n)
  safe_denom1_n <- 2 * (a1_n + is_indep1_n)
  quadratic1_n <- (-b1_n - discriminant1_n) / safe_denom1_n
  JTPR_n <- is_indep1_n * p_indep1_n + (1 - is_indep1_n) * quadratic1_n
  
  # OR0
  is_indep0_n <- step(OR0_n - 0.99999) * step(1.00001 - OR0_n)
  p_indep0_n <- FPR_M_n * FPR_R_n
  a0_n <- OR0_n - 1
  b0_n <- -OR0_n*(FPR_M_n + FPR_R_n) - 1 + (FPR_M_n + FPR_R_n)
  d0_n <- OR0_n * FPR_M_n * FPR_R_n
  discriminant0_n <- sqrt(b0_n*b0_n - 4*a0_n*d0_n)
  safe_denom0_n <- 2 * (a0_n + is_indep0_n)
  quadratic0_n <- (-b0_n - discriminant0_n) / safe_denom0_n
  JFPR_n <- is_indep0_n * p_indep0_n + (1 - is_indep0_n) * quadratic0_n
  
  y_nishiyama[m, 1:4] ~ dmulti(p_nishiyama[m, 1:4], N_total_nishiyama[m])
  p_nishiyama[m,1] <- pi_nishiyama[m] * JTPR_n + (1-pi_nishiyama[m]) * JFPR_n
  p_nishiyama[m,2] <- pi_nishiyama[m] * (TPR_M_n - JTPR_n) + (1-pi_nishiyama[m]) * (FPR_M_n - JFPR_n)
  p_nishiyama[m,3] <- pi_nishiyama[m] * (TPR_R_n - JTPR_n) + (1-pi_nishiyama[m]) * (FPR_R_n - JFPR_n)
  p_nishiyama[m,4] <- 1 - (p_nishiyama[m,1] + p_nishiyama[m,2] + p_nishiyama[m,3])
  
  pi_nishiyama[m] ~ dunif(0,1)
}

# For Edmonson (check the negatives with extra info)
for(e in 1:N_edmonson) {
  eta.xi_edmonson[e, 1:6] ~ dmnorm(Eta.Xi[], Omega[,])
  
  TPR_M_e <- ilogit(eta.xi_edmonson[e,1])
  TPR_R_e <- ilogit(eta.xi_edmonson[e,2])
  FPR_M_e <- ilogit(eta.xi_edmonson[e,4])
  FPR_R_e <- ilogit(eta.xi_edmonson[e,5])
  OR1_e <- exp(eta.xi_edmonson[e,3])
  OR0_e <- exp(eta.xi_edmonson[e,6])

  # OR1
  is_indep1_e <- step(OR1_e - 0.99999) * step(1.00001 - OR1_e)
  p_indep1_e <- TPR_M_e * TPR_R_e
  a1_e <- OR1_e - 1
  b1_e <- -OR1_e*(TPR_M_e + TPR_R_e) - 1 + (TPR_M_e + TPR_R_e)
  d1_e <- OR1_e * TPR_M_e * TPR_R_e
  discriminant1_e <- sqrt(b1_e*b1_e - 4*a1_e*d1_e)
  safe_denom1_e <- 2 * (a1_e + is_indep1_e)
  quadratic1_e <- (-b1_e - discriminant1_e) / safe_denom1_e
  JTPR_e <- is_indep1_e * p_indep1_e + (1 - is_indep1_e) * quadratic1_e
  
  # OR0
  is_indep0_e <- step(OR0_e - 0.99999) * step(1.00001 - OR0_e)
  p_indep0_e <- FPR_M_e * FPR_R_e
  a0_e <- OR0_e - 1
  b0_e <- -OR0_e*(FPR_M_e + FPR_R_e) - 1 + (FPR_M_e + FPR_R_e)
  d0_e <- OR0_e * FPR_M_e * FPR_R_e
  discriminant0_e <- sqrt(b0_e*b0_e - 4*a0_e*d0_e)
  safe_denom0_e <- 2 * (a0_e + is_indep0_e)
  quadratic0_e <- (-b0_e - discriminant0_e) / safe_denom0_e
  JFPR_e <- is_indep0_e * p_indep0_e + (1 - is_indep0_e) * quadratic0_e
  
  y_edmonson[e, 1:6] ~ dmulti(p_edmonson[e, 1:6], N_total_edmonson[e])

  p_edmonson[e,1] <- pi_edmonson[e]*JTPR_e + (1-pi_edmonson[e])*JFPR_e
  p_edmonson[e,2] <- pi_edmonson[e]*(TPR_R_e - JTPR_e) + (1-pi_edmonson[e])*(FPR_R_e - JFPR_e)
  p_edmonson[e,3] <- pi_edmonson[e]*(TPR_M_e - JTPR_e)
  p_edmonson[e,4] <- pi_edmonson[e]*(1 - TPR_M_e - TPR_R_e + JTPR_e)
  p_edmonson[e,5] <- (1-pi_edmonson[e])*(FPR_M_e - JFPR_e)
  p_edmonson[e,6] <- (1-pi_edmonson[e])*(1 - FPR_M_e - FPR_R_e + JFPR_e)
  
  pi_edmonson[e] ~ dunif(0,1)
}

# For Lindgren (check the negatives)
for(l in 1:N_lindgren) {
  eta.xi_lindgren[l, 1:6] ~ dmnorm(Eta.Xi[], Omega[,])

  TPR_M_l <- ilogit(eta.xi_lindgren[l,1])
  FPR_M_l <- ilogit(eta.xi_lindgren[l,4])
  FPR_R_l <- ilogit(eta.xi_lindgren[l,5])
  OR0_l <- exp(eta.xi_lindgren[l,6])

  # OR0
  is_indep0_l <- step(OR0_l - 0.99999) * step(1.00001 - OR0_l)
  p_indep0_l <- FPR_M_l * FPR_R_l
  a0_l <- OR0_l - 1
  b0_l <- -OR0_l*(FPR_M_l + FPR_R_l) - 1 + (FPR_M_l + FPR_R_l)
  d0_l <- OR0_l * FPR_M_l * FPR_R_l
  discriminant0_l <- sqrt(b0_l*b0_l - 4*a0_l*d0_l)
  safe_denom0_l <- 2 * (a0_l + is_indep0_l)
  quadratic0_l <- (-b0_l - discriminant0_l) / safe_denom0_l
  JFPR_l <- is_indep0_l * p_indep0_l + (1 - is_indep0_l) * quadratic0_l
  
  y_lindgren[l, 1:4] ~ dmulti(p_lindgren[l, 1:4], N_total_lindgren[l])

  p_lindgren[l,1] <- pi_lindgren[l]*TPR_M_l + (1-pi_lindgren[l])*JFPR_l
  p_lindgren[l,2] <- pi_lindgren[l]*(1-TPR_M_l) + (1-pi_lindgren[l])*(FPR_R_l - JFPR_l)
  p_lindgren[l,3] <- (1-pi_lindgren[l])*(FPR_M_l - JFPR_l)
  p_lindgren[l,4] <- 1 - p_lindgren[l,1] - p_lindgren[l,2] - p_lindgren[l,3]
  
  pi_lindgren[l] ~ dunif(0,1)
}


# Summary estimates

TPR_M <- ilogit(Eta.Xi[1])
TPR_R <- ilogit(Eta.Xi[2])
OR1 <- exp(Eta.Xi[3])
FPR_M <- ilogit(Eta.Xi[4])
FPR_R <- ilogit(Eta.Xi[5])
OR0 <- exp(Eta.Xi[6])
Spec_M <- 1 - FPR_M
Spec_R <- 1 - FPR_R

# Calculate Joint TPR (JTPR) from summary marginals and OR1 (diseased group)
is_indep1_summary <- step(OR1 - 0.99999) * step(1.00001 - OR1)
p_indep1_summary <- TPR_M * TPR_R
a1_summary <- OR1 - 1
b1_summary <- -OR1*(TPR_M + TPR_R) - 1 + (TPR_M + TPR_R)
d1_summary <- OR1 * TPR_M * TPR_R
discriminant1_summary <- sqrt(b1_summary*b1_summary - 4*a1_summary*d1_summary)
safe_denom1_summary <- 2 * (a1_summary + is_indep1_summary)
quadratic1_summary <- (-b1_summary - discriminant1_summary) / safe_denom1_summary
JTPR_summary <- is_indep1_summary * p_indep1_summary + (1 - is_indep1_summary) * quadratic1_summary

# Calculate Joint FPR (JFPR) from summary marginals and OR0 (non-diseased group)
is_indep0_summary <- step(OR0 - 0.99999) * step(1.00001 - OR0)
p_indep0_summary <- FPR_M * FPR_R
a0_summary <- OR0 - 1
b0_summary <- -OR0*(FPR_M + FPR_R) - 1 + (FPR_M + FPR_R)
d0_summary <- OR0 * FPR_M * FPR_R
discriminant0_summary <- sqrt(b0_summary*b0_summary - 4*a0_summary*d0_summary)
safe_denom0_summary <- 2 * (a0_summary + is_indep0_summary)
quadratic0_summary <- (-b0_summary - discriminant0_summary) / safe_denom0_summary
JFPR_summary <- is_indep0_summary * p_indep0_summary + (1 - is_indep0_summary) * quadratic0_summary

# 'And' rule (positive if M+ AND R+)
Sens_and <- JTPR_summary
Spec_and <- 1 - JFPR_summary

# 'Or' rule (positive if M+ OR R+)
Sens_or <- TPR_M + TPR_R - JTPR_summary
Spec_or <- 1 - FPR_M - FPR_R + JFPR_summary

# Predictive distributions
eta.xi_pred[1:6] ~ dmnorm(Eta.Xi[], Omega[,])

# Transform to get the predicted outcomes for the new study
pred_TPR_M <- ilogit(eta.xi_pred[1])
pred_TPR_R <- ilogit(eta.xi_pred[2])
pred_FPR_M <- ilogit(eta.xi_pred[4])
pred_FPR_R <- ilogit(eta.xi_pred[5])

# Calculate predicted specificities
pred_Spec_M <- 1 - pred_FPR_M
pred_Spec_R <- 1 - pred_FPR_R

}"








