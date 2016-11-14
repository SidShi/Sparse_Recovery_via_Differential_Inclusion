# Sparse_Recovery_via_Differential_Inclusion
MATLAB implementations of Bregman ISS and LB ISS methods

Functions right now do not work quite well.  

Main problems come from the dimensions and appearances of matrices and transformation of object structures between R and MATLAB. I will keep on debugging till all problems are fixed and all functions work perfectly well.  

Function list:  

- cross validation for iss (cv_iss)  
- cross validation for lb (cv_lb)  
- iss method (iss)  
- lb method (lb) together with three helper functions (lb_lasso, lb_logistic, lb_multilogistic)  
- prediction of lb object (predict_lb)
