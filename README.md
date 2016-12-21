# Sparse_Recovery_via_Differential_Inclusion
MATLAB implementations of Bregman ISS and LB ISS methods



Function list:  

- cross validation for iss (cv_iss.m)  
- cross validation for lb (cv_lb.m)  
- iss method (iss.m)  
- lb method (lb.m) together with three helper functions (lb_lasso.m, lb_logistic.m, lb_multilogistic.m) and three C mex functions (lblasso_void.c, lblogistic_void.c, lbmultilogistic_void.c)  
- prediction of lb object (predict_lb.m)
- plot lb object (plot_lb.m)  

Other files in the folder:  

- Mex gateway functions: lblasso_void.mexw64, lblogistic_void.mexw64, lbmultilogistic_void.mexw64  
- Windows 64 GSL library file: libgsl-0.dll, libgslcblas-0.dll  


For the mex functions to work, one needs to have GSL libraries successfully installed and set up in the computer. Since GSL is originally designed for Linux, if one wants to run it on Windows, he/she can go online and download the Win32 version. Win64 version can be made up with partial Win64 files online. There are tutorials out there.  


Since MATLAB does not provide with default function value, all parameters need to be determined when running the functions. If one parameter is not known or needs to be set to a default value, use a single letter in the place of the parameter. I use 'm' to test my functions.  


Below is the driver I use to test my functions:  
```
n = 200; p = 100; k = 30; sigma = 1;
Sigma = 1/(3*p)*ones(p,p);
Sigma(logical(eye(size(Sigma)))) = 1;
A = mvnrnd(zeros(n,p),Sigma);
u_ref = zeros(p,1);
supp_ref = (1:k)';
u_ref(supp_ref) = normrnd(0,1,[1 k])';
u_ref(supp_ref) = u_ref(supp_ref)+sign(u_ref(supp_ref));
b = A*u_ref+sigma*normrnd(0,1,[1 n])';

%object = iss(A,b,false,false,'m');
%cv_iss(A,b,'m','m',false,false,'m','m',@iss,@predict_lb)
%plot_lb(object,'m','m','m')

%kappa = 16;
%alpha = 1/160;
%object = lb(A,b,kappa,alpha,'m','m','m',20,'gaussian',false,'m',false,false,'m',@lb_lasso,@lb_logistic,@lb_multilogistic);
%plot_lb(object,'m','m','m')

%cv_lb(A,b,10,1/20,'m','m','m','m','m','m',false,false,'m','m','m',@lb,@predict_lb,@lb_lasso,@lb_logistic,@lb_multilogistic)


X = normrnd(0,1,500,100);
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

result = lb(X,y,5,0.1,'m','m','m','m','multinomial',true,'m',false,false,'m',@lb_lasso,@lb_logistic,@lb_multilogistic);
plot_lb(result,'m','m','m')


alpha = [ones(1,30), zeros(1,70)];
% y <- 2*as.numeric(runif(500)<1/(1+exp(-X %*% alpha)))-1
y = 2*(rand(500,1)<(1/(1+exp(-X*alpha')))')-1;
%cv.lb(X,y,kappa=5,alpha=1,family="binomial",intercept=FALSE,normalize = FALSE)
%function object = cv_lb(X,y,kappa,alpha,K,tlist,nt,trate,family,group,intercept,normalize,plot_it,se,index,lb,predict,lb_lasso,lb_logistic,lb_multilogistic)
figure
cv_lb(X,y,5,1,'m','m','m','m','binomial','m',false,false,'m','m','m',@lb,@predict_lb,@lb_lasso,@lb_logistic,@lb_multi_logistic)
```
