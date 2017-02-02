function object = cv_lb(X,y,kappa,alpha,K,tlist,nt,trate,family,group,intercept,normalize,plot_it,se,index,lb,predict,lb_lasso,lb_logistic,lb_multilogistic)
% CV for lb

% Cross-validation method to tuning the parameter t for lb.
% K-fold cross-validation method is used to tuning the parameter t for ISS.
% Mean square error is used for linear model. Miss-classification error
% is used for binomial and multinomial model.

% Input:
% X: An n-by-p matrix of predictors
% y: Response Variable
% kappa: The damping factor of the Linearized Bregman Algorithm that is defined in the reference paper. See details. 
% alpha: Parameter in Linearized Bregman algorithm which controls the step-length of the discretized solver for the Bregman Inverse Scale Space. 
% K: Folds number for CV. Default is 5.
% tlist: Parameters t along the path.
% nt: Number of t. Used only if tlist is missing. Default is 100.
% trate: tmax/tmin. Used only if tlist is missing. Default is 100.
% family: Response type
% group: Whether to use a group penalty, Default is FALSE.
% intercept: If TRUE, an intercept is included in the model (and not penalized), otherwise no intercept is included. Default is TRUE.
% normalize: if TRUE, each variable is scaled to have L2 norm square-root n. Default is TRUE.
% plot_it: Plot it? Default is TRUE
% se: Include standard error bands? Default is TRUE
% Output: 
% A structure array is returned. The array contains a vector of parameter t, crossvalidation error cv_error, and the estimated standard deviation for it cv_sd

%% check if there are undetermined values
if (ischar(K))
    K = 5;
end
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
if (ischar(plot_it))
    plot_it = true;
end
if (ischar(se))
    se = true;
end

%% check if inputs are elligible
if (size(X,1) == 1 || size(X,2) == 1)
    error('X must be a matrix!')
end
if (~strcmp(family,'multinomial'))
    if (~isvector(y))
        error('y must be a vector unless in a multinomial model!')
    end
    if (size(X,1) ~= length(y))
        error('Number of rows of X must be equal to the length of y!')
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
        if (sum(sum(y,2) == 1) ~= size(y,1))
            error('y should be an indicator matrix!')
        end
    else
        error('y must be a vector or a matrix!')
    end
end

if (group)
    if (ischar(index))
        if (strcmp(family,'multinomial'))
            index = NaN(1,size(X,2)-intercept);
        else
            group = false;
            disp('Index is missing, using group = false instead!')
        end
    end
    if (length(index) == 1)
        error('Index must be a vector!')
    end
    if (length(index) + intercept ~= size(X,2))
        error('Length of index must be the same as the number of columns of X minus the intercept!')
    end
end

%% initialize and generate folds
n = size(X,1);
si = floor(n/K);
mo = n-K*si;
randn = randperm(n);
folds(K) = struct();
for i = 1:mo
    folds(i).group = zeros(1,si+1);
end
for i = (mo+1):K
    folds(i).group = zeros(1,si);
end
for i = 1:K
    for j = 1:si
        folds(i).group(j) = randn(i+(j-1)*K);
    end
end
for i = 1:mo
    folds(i).group(si+1) = randn(i+si*K);
end

if (tlist == 'm')
    obj = lb(X,y,kappa,alpha,'m','m',nt,trate,family,group,index,intercept,normalize,'m',lb_lasso,lb_logistic,lb_multilogistic);
    tlist = obj(7).lb;
end

%% run the lb algorithm and do cross validation
residmat = zeros(length(tlist),K);
for i = 1:K
    omit = folds(i).group;
    Xuse = X;
    Xuse(omit,:) = [];
    if (strcmp(family,'multinomial'))
        yuse = y;
        yuse(omit,:) = [];
        fit = lb(Xuse,yuse,kappa,alpha,'m',tlist,nt,trate,family,group,index,intercept,normalize,'m',lb_lasso,lb_logistic,lb_multilogistic);
    else
        yuse = y;
        yuse(omit) = [];
        fit = lb(Xuse,yuse,kappa,alpha,'m',tlist,nt,trate,family,group,index,intercept,normalize,'m',lb_lasso,lb_logistic,lb_multilogistic);
    end
    fi = predict(fit,X(omit,:),tlist,'m');
    fit = fi(5).predict;
    
    if (strcmp(family,'binomial'))
        dum1 = fit;
        dum3 = (fit > 0.5) + ones(size(dum1,1),size(dum1,2));
        yomit = y(omit);
        for z = 1:numel(dum3)
            dum2 = mod(z,length(yomit));
            if dum2 == 0
                dum2 = length(yomit);
            end
            dum3(z) = yomit(dum2) - dum3(z);
        end
        residmat(:,i) = mean(dum1.^2/4,1)';
    elseif (strcmp(family,'gaussian'))
        dum1 = fit;
        yomit = y(omit);
        for z = 1:numel(dum1)
            dum2 = mod(z,length(yomit));
            if dum2 == 0
                dum2 = length(yomit);
            end
            dum1(z) = yomit(dum2) - dum1(z);
        end
        residmat(:,i) = mean(dum1.^2,1)';
    end
    if (strcmp(family,'multinomial'))
        t = size(fit,3);
        for j = 1:length(tlist)
            for k = 1:t
                dum1 = fit(:,:,k);
                dum2 = dum1;
                for p = 1:size(dum1,2)
                    for q = 1:size(dum1,1)
                        if (dum1(q,p) == max(dum1(q,:)))
                            dum2(q,p) = 1;
                        else
                            dum2(q,p) = 0;
                        end
                    end
                end
            end
            residmat(j,i) = sum(sum((y(omit,:)-dum2).^2))/2/length(omit);
        end
    end
end

cv_error = sum(residmat,2)/size(residmat,2);
cv_sd = sqrt(var(residmat,0,2)/K);

%% plot and generate 'cv_lb' class object
if (plot_it)
    plot(tlist,cv_error)
    xlabel('t')
    ylabel('Cross-Validated MSE')
end
hold on
if (se)
    plot([tlist;tlist],[cv_error'+cv_sd';cv_error'-cv_sd'],'r')
end

field = 'cv_lb';
value = {tlist,cv_error,cv_sd};
object = struct(field,value);
end