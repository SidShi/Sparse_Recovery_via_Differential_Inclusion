function object = iss(X,y,intercept,normalize,nvar)
% ISS solver for linear model with lasso penalty

% Solver for the entire solution path of coefficients for ISS.  
% The ISS solver computes the whole regularization path for 
% lasso-penalty for linear model. It gives the piecewise constant
% solution path for Bregman Inverse Scale Space Differential 
% Inclusion. It is the asymptotic limit of LB method with kaapa 
% goes to infinity and alpha goes to zero.

% Input:
% X: An n-by-p matrix of predictors
% y: Response Variable
% intercept: If TRUE, an intercept is included in the model (and not penalized), otherwise no intercept is included. Default is TRUE.
% normalize: if TRUE, each variable is scaled to have L2 norm square-root n. Default is TRUE.
% nvar: Maximal number of variables allowed in the model
% Output:
% A structure array with fields: function call, family, path, intercept,
% alpha, kappa, t, meanx, normx, group

%% check if there are undetermined values
if (ischar(intercept))
    intercept = true;
end
if (ischar(normalize))
    normalize = true;
end
if (ischar(nvar))
    nvar = min(size(X));
end

%% check if there are illegible variables
if (~ismatrix(X) || ~isvector(y))
    error('X must be a matrix and y must be a vector')
end
if (size(X,1) ~= length(y))
    error('Number of rows of X must equal to the length of y')
end

%% initialization
n = size(X,1);
p = size(X,2);
one = ones(1,n);

if (intercept)
    meanx = (one*X)/n;
    for i = 1:size(X,1)
        X(i,:) = X(i,:) - meanx;
    end
    mu = mean(y);
    y = squeeze(y-mu);
else
    meanx = zeros(1,p);
    mu = 0;
    y = squeeze(y);
end
if (normalize)
    normx = sqrt(one*(X.^2)/n);
    for i = 1:size(X,1)
        X(i,:) = X(i,:)./normx;
    end
else
    normx = ones(1,p);
end

maxitr = 20*min(n,p);
res = y;
rho = zeros(1,p);
active = false(1,p);
t = 0;
hist_t = zeros(1,maxitr+1);
hist_path = zeros(maxitr+1,p);

%% Main step for the iss algorithm
for i = 1:maxitr
    use = i;
    ng = X'*res;
    n1 = (1-rho)./ng';
    n2 = (-1-rho)./ng';
    delta = zeros(1,p);
    for j = 1:p
        if (n1(j) >= n2(j))
            delta(j) = n1(j);
        else
            delta(j) = n2(j);
        end
    end
    delta(active) = Inf;
    delta_t = min(delta);
    add = find(delta == delta_t);
    
    t = t+delta_t;
    rho(~active) = rho(~active)+delta_t*ng(~active)';
    rho(add) = round(rho(add));
    active(add) = true;
    hist_t(i+1) = t;
    
    
    [obj_nnls_y,dum,res] = lsqnonneg(X(:,active)*diag(rho(active)),y);
    obj_nnls = obj_nnls_y./rho(active)';
    hist_path(i+1,:) = hist_path(i);
    hist_path(i+1,active) = obj_nnls;
    active(active) = (obj_nnls ~= 0);
    if (sum(active) >= min(min(nvar,n),p))
        break
    end
end

hist_path = hist_path(linspace(1,use+1,use+1),:);
for i = 1:size(hist_path,1)
    hist_path(i,:) = hist_path(i,:)./normx;
end
hist_path = hist_path';

a0 = mu-meanx*hist_path;
hist_t = hist_t(linspace(1,use+1,use+1))*n;

%% build 'lb' class object
call = 'iss';
field = 'lb';
family = 'gaussian';
group = false;
kappa = Inf;
alpha = 0;
value = {call,family,hist_path,a0,alpha,kappa,hist_t,meanx,normx,group};
object = struct(field,value);
end