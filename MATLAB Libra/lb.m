function object = lb(X,y,kappa,alpha,c,tlist,nt,trate,family,group,index,intercept,normalize,print,lb_lasso,lb_logistic,lb_multilogistic)
% Linearized Bregman solver for linear, binomial, multinomial models
% with lasso, group lasso or column lasso penalty.
 
% Solver for the entire solution path of coefficients for Linear Bregman iteration.
 
% The Linearized Bregman solver computes the whole regularization path
% for different types of lasso-penalty for gaussian, binomial and 
% multinomial models through iterations. It is the Euler forward 
% discretized form of the continuous Bregman Inverse Scale Space 
% Differential Inclusion. For binomial models, the response variable y
% is assumed to be a vector of two classes which is transformed in to \{1,-1\}.
% For the multinomial models, the response variable y can be a vector of k classes
% or a n-by-k matrix that each entry is in \{0,1\} with 1 indicates 
% the class. Under all circumstances, two parameters, kappa 
% and alpha need to be specified beforehand. The definitions of kappa 
% and alpha are the same as that defined in the reference paper. 
% Parameter alpha is defined as stepsize and kappa is the damping factor
% of the Linearized Bregman Algorithm that is defined in the reference paper.

% Input:
% X: An n-by-p matrix of predictors
% y: Response Variable
% kappa: The damping factor of the Linearized Bregman Algorithm that is defined in the reference paper. See details. 
% alpha: Parameter in Linearized Bregman algorithm which controls the step-length of the discretized solver for the Bregman Inverse Scale Space. 
% c: Normalized step-length. If alpha is missing, alpha is automatically generated by 
% alpha=n*c/(kappa*||X^T*X||_2). It should be in (0,2) for family = "gaussian"(Default is 1), (0,8) for family = "binomial"(Default is 4), (0,4) for family = "multinomial"(Default is 2).
% If beyond these range the path may be oscillated at large t values.
% family: Response type
% group: Whether to use a group penalty, Default is FALSE.
% index: For group models, the index is a vector that determines the group of the parameters. Parameters of the same group should have equal value in index. Be careful that multinomial group model default assumes the variables in same column are in the same group, and a empty value of index means each variable is a group.
% intercept: if TRUE, an intercept is included in the model (and not penalized), otherwise no intercept is included. Default is TRUE.
% normalize: if TRUE, each variable is scaled to have L2 norm square-root n. Default is TRUE.
% tlist: Parameters t along the path.
% nt: Number of t. Used only if tlist is missing. Default is 100.
% trate: tmax/tmin. Used only if tlist is missing. Default is 100.
% print: If TRUE, the percentage of finished computation is printed.
% Output:
% A structure array with fields: function call, family, path, intercept,
% alpha, kappa, t, meanx, normx, group

%% check if there are undetermined variables
if (ischar(nt))
    nt = 100;
end
if (ischar(trate))
    trate = 100;
end
if (length(family) == 1)
    family = 'gaussian';
end
if (ischar(group))
    group = false;
end
if (ischar(intercept))
    intercept = true;
end
if (ischar(normalize))
    normalize = true;
end
if (ischar(print))
    print = false;
end

%% check if there are inelligible values
if (~ismatrix(X))
    error('X must be a matrix!')
end
if (~strcmp(family,'multinomial'))
    if (~isvector(y))
        error('y must be a vector unless in multinomial model!')
    end
    if (size(X,1) ~= length(y))
        error('Number of rows of X must equal to the length of y')
    end
    if (strcmp(family,'binomial'))
        for i = 1:length(y)
            if (abs(y(i)) ~= 1)
                error('y must be in {1,-1}')
            end
        end
    end
else
    if (isvector(y))
        if (size(X,1) ~= length(y))
            error('Number of rows of X must be equal to the length of y!')
        end
        y_unique = unique(y,'stable');
        y_unimat = zeros(length(y),length(y_unique));
        for i = 1:length(y_unique)
            for j = 1:length(y)
                if (y(j) == y_unique(i))
                    y_unimat(j,i) = 1;
                end
            end
        end
        y = y_unimat;
    elseif (ismatrix(y))
        if (size(X,1) ~= size(y,1))
            error('Number of rows of X and y must equal!')
        end
        for i = 1:size(y,1)
            for j = 1:size(y,2)
                if (y(i,j) ~= 0 && y(i,j) ~= 1)
                    error('y should be an indicator matrix!')
                end
            end
        end
        rsum = sum(y,2);
        for i = 1:length(rsum)
            if (rsum(i) ~= 1)
                error('y should be an indicator matrix!')
            end
        end
    else
        error('y must be a vector or a matrix!')
    end
end

if (group)
    if (ischar(index))
        if (strcmp(family,'multinomial'))
            index = NaN(1,size(X,2));
        else
            group = false;
            disp('Index is missing, using group = false instead!')
        end
    end
    if (~isvector(index))
        error('Index must be a vector!')
    end
    if (length(index) ~= size(X,2))
        if (~strcmp(family,'multinomial') || sum(isNaN(index)) == 0)
            error('Length of index must equal to the number of columns of X')
        end
    end
end

%% initialization
n = size(X,1);
p = size(X,2);
one = ones(1,n);

if (intercept)
    meanx = one*X/n;
    for i = 1:size(X,1)
        X(i,:) = X(i,:) - meanx;
    end
else
    meanx = zeros(1,p);
end
if (normalize)
    normx = sqrt(one*(X.^2)/n);
    for i = 1:size(X,1)
        X(i,:) = X(i,:)./normx;
    end
else
    normx = ones(1,p);
end

if (tlist == 'm')
    tlist = ones(1,nt);
    tlist = -tlist;
else
    nt = length(tlist);
end

alpha0_rate = 1;
if (ischar(alpha))
    sigma = norm(X);
    if (ischar(c))
        if (strcmp(family,'gaussian'))
            c = 1;
        elseif (strcmp(family,'binomial'))
            c = 4;
        else
            c = 2;
        end
    end
    alpha = n*c/kappa/sigma^2;
    if (intercept)
        alpha0_rate = sigma^2/n;
    end
end
if (intercept && ~normalize)
    alpha0_rate = max(squeeze(one*(X.^2)))/n;
end

%% use helper functions to implement the main 'lb' algorithm
if (strcmp(family, 'gaussian'))
    object = lb_lasso(X,y',kappa,alpha,alpha0_rate,tlist,nt,trate,intercept,group,index,print,meanx,normx);
elseif (strcmp(family,'binomial'))
    object = lb_logistic(X,y,kappa,alpha,alpha0_rate,tlist,nt,trate,intercept,group,index,print,meanx,normx);
elseif (strcmp(family,'multinomial'))
    object = lb_multilogistic(X,y,kappa,alpha,alpha0_rate,tlist,nt,trate,intercept,group,index,print,meanx,normx);
else
    error('No such family type!')
end

%% post-handle and process data, complete the 'lb' class object
if (intercept)
    if (~strcmp(family,'multinomail'))
        object(4).lb = object(3).lb(p+1,:);
        object(3).lb(p+1,:) = [];
    else
        object(4).lb = object(3).lb(:,p+1,:);
        object(3).lb(:,p+1,:) = [];
    end
else
    if (strcmp(family,'multinomial'))
        object(4).lb = zeros(size(y,2),nt);
    else
        object(4).lb = zeros(1,nt);
    end
end

if (strcmp(family,'multinomial'))
    for i = 1:nt
        for j = 1:size(object(3).lb(:,:,i),1)
            object(3).lb(j,:,i) = object(3).lb(j,:,i)./normx;
        end
    end
    if (intercept)
        temp = object(3).lb;
        for i = 1:nt
            temp(:,:,i) = temp(:,:,i)*meanx;
        end
        object(4).lb = object(4).lb-temp;
    end
end

object(8).lb = meanx;
object(9).lb = normx;
object(2).lb = family;
object(10).lb = group;

end