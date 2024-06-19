# Functions to fit and use statistical distributions for PESE


## Description
Acronyms:
1) PPI transform: Probit Probability Integral transform. Maps data into space where marginal distribution is standard normal.

Three kinds of functions:
1) Fitting functions (e.g., `fit_bounded_rank_histogram`)
2) PPI transform functions (e.g., `ppi_bounded_rank_histogram`)
3) Inverse PPI transform functions (e.g., `inv_ppi_bounded_rank_histogram`)



## Distributions available
1) `bounded_boxcar_rank_histogram`
2) `boxcar_rank_histogram` (INEFFICIENT AND BUGGY)