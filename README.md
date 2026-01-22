# joint_MA_two_tests

This repository contains the R and Stan code accompanying the paper: "A joint meta-analysis framework for the accuracy of two diagnostic tests accounting for varying study designs."

The framework implements a Bayesian hierarchical model to jointly estimate the sensitivity and specificity of two diagnostic tests while accounting for varying study and reporting designs and possibel conditional dependence of tests via log odds ratios.

## Repository Structure

**models**: contains the Stan models for the primary joint meta analysis model with varying study designs and dependence via log odds ratios for both case studies. Additionally, for comparison, Stan models for the joint meta-analysis of two tests assuming conditional independence of tests and a separate bivariate meta-analysis for tests, for both case studies. There is also an alternative implementation of the primary joint dependence model using JAGS. 

**data**: data formatted for the joint and separate models for both case studies. 

**scripts**: R scripts to load data, configure the Stan models, and extract summary parameters.
