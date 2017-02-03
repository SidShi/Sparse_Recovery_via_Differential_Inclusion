# Sparse_Recovery_via_Differential_Inclusion
MATLAB and C implementations of Bregman Inverse Scale Space (refered later here and in code as ISS) and Linearized Bregman ISS methods (refered later here and in code as LB)


# Function list:  

- iss.m: ISS method, output is an LB object  
- lb.m: LB method, output is an LB object  
- cv_iss.m: cross validation for ISS method 
- cv_lb.m: cross validation for LB method  
- lb_lasso.m: LB method for LASSO case; used when data follows Gaussian distribution; not used directly but as a helper function  
- lb_logistic.m: LB method for logistic case; used when data follows binomial distribution; not used directly but as a helper function  
- lb_multilogistic.m: LB method for multilogistic case; used when data follows multinomial distribution; not used directly but as a helper function  
- predict_lb.m: prediction of an LB object
- plot_lb.m: plot an LB object  


# Function inner connections:  

- cv_iss.m uses iss.m and predict_lb.m  
- cv_lb.m uses lb.m and predict_lb.m  
- lb.m uses three helper functions (lb_lasso.m, lb_logistic.m, lb_multilogistic.m) and three C mex functions (lblasso_void.c, lblogistic_void.c, lbmultilogistic_void.c)  
- lb_lasso.m uses lblasso_void.c  
- lb_logistic.m uses lblogistic_void.c  
- lb_multilogistic.m uses lbmultilogistic_void.c  


# Other files in the folder:  

- Mex gateway functions: lblasso_void.mexw64, lblogistic_void.mexw64, lbmultilogistic_void.mexw64  
- Windows 64 GSL library file: libgsl-0.dll, libgslcblas-0.dll  


# Some guidelines of using the functions in this folder:  

For the mex functions to work, one needs to have GSL libraries successfully installed and set up in the system. Since GSL is originally designed for Linux, if one wants to run it on Windows, he/she can go online and download the Win32 version. Win64 version can be made up with partial Win64 files online. There are tutorials out there.  

Since MATLAB does not provide with default function value, all parameters need to be determined when running the functions. If one parameter is not known or needs to be set to a default value, use a single letter in the place of the parameter. I use 'm' to test my functions.  


Below is the driver I use to test my functions:  
```
n = 200; p = 100; k = 30; sigma = 1;
% Build a matrix A that follows multivariate normal distribution
Sigma = 1/(3*p)*ones(p,p);
Sigma(logical(eye(size(Sigma)))) = 1;
A = mvnrnd(zeros(n,p),Sigma);

% Build a response variable b
u_ref = zeros(p,1);
supp_ref = (1:k)';
u_ref(supp_ref) = normrnd(0,1,[1 k])';
u_ref(supp_ref) = u_ref(supp_ref)+sign(u_ref(supp_ref));
b = A*u_ref+sigma*normrnd(0,1,[1 n])';

% test the iss.m function
object = iss(A,b,false,false,'m');

% test the cv_iss.m function
cv_iss(A,b,'m','m',false,false,'m','m',@iss,@predict_lb)

% test the plot_lb.m function for iss object
plot_lb(object,'m','m','m')

% test the lb.m function for Gaussian distribution and corresponding plot_lb.m function
kappa = 16;
alpha = 1/160;
object = lb(A,b,kappa,alpha,'m','m','m',20,'gaussian',false,'m',false,false,'m',@lb_lasso,@lb_logistic,@lb_multilogistic);
plot_lb(object,'m','m','m')

% test the cv_lb.m function for gaussian distribution
cv_lb(A,b,10,1/20,'m','m','m','m','m','m',false,false,'m','m','m',@lb,@predict_lb,@lb_lasso,@lb_logistic,@lb_multilogistic)


% Build a predictor variable X
X = normrnd(0,1,500,100);

% Build a response variable y
alpha = [normrnd(0,1,3,30),zeros(3,70)];
P = exp(alpha*X');
psum = sum(P,1);
for i = 1:size(P,1)
    P(i,:) = P(i,:)./psum;
end
y = zeros(500,1);
rd = rand(1,500);
y(rd<P(1,:)) = 1;
y(rd>(1-P(3,:))) = -1;

% test the lb.m function for multinomial distribution and corresponding plot_lb.m function
result = lb(X,y,5,0.1,'m','m','m','m','multinomial',true,'m',false,false,'m',@lb_lasso,@lb_logistic,@lb_multilogistic);
plot_lb(result,'m','m','m')

% test the cv_lb.m function for binomial distribution
alpha = [ones(1,30), zeros(1,70)];
y = 2*(rand(500,1)<(1/(1+exp(-X*alpha')))')-1;
figure
cv_lb(X,y,5,1,'m','m','m','m','binomial','m',false,false,'m','m','m',@lb,@predict_lb,@lb_lasso,@lb_logistic,@lb_multi_logistic)
```


# Attribution  
- Author: Tianyi Shi  
- Purpose: Introduce some new signal processing/variable selection methods to MATLAB  
- Reference Paper: *Sparse Recovery via Differential Inclusions*, Stanley Osher, Feng Ruan, Jiechao Xiong, Yuan Yao, and Wotao Yin  
- Reference Code: *LIBRA* package in R, Feng Ruan, Jiechao Xiong and Yuan Yao, https://cran.r-project.org/web/packages/Libra/index.html  
- Project Mentor: Wotao Yin (University of California Los Angeles), Jiechao Xiong (Peking University)  
- Special Thanks: Ke Ma (Chinese Academy of Sciences)
- Contact: shitianyisid@gmail.com  
