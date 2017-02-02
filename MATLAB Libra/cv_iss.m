function object = cv_iss(X,y,K,t,intercept,normalize,plot_it,se,iss,predict)
% CV for ISS

% Cross-validation method to tuning the parameter t for ISS.
% K-fold cross-validation method is used to tuning the parameter $t$ for ISS.
% Mean square error is used as prediction error.

% Input:
% X: An n-by-p matrix of predictors
% y: Response Variable
% K: Folds number for cv, default will be set to 5 if not specified
% t: A vector of predecided tuning parameter
% intercept: If TRUE, an intercept is included in the model (and not penalized), otherwise no intercept is included. Default is TRUE.
% normalize: if TRUE, each variable is scaled to have L2 norm square-root n. Default is TRUE.
% plot.it Plot it? Default is TRUE
% se Include standard error bands? Default is TRUE
% Output:
% A structure array is returned. The struct contains a vector of parameter t, crossvalidation error cv_error, and the estimated standard deviation for it cv_sd


%% check if there are undetermined values
if (ischar(K))
    K = 5;
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
if (~ismatrix(X) || ~isvector(y))
    error('X must be a matrix and y must be a vector')
end
if (size(X,1) ~= length(y))
    error('Number of rows of X must equal to the length of y')
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

if (t == 'm')
    one = ones(1,n);
    if (intercept)
        meanx = squeeze(one*X)/n;
        for i = 1:size(X,1)
            X(i,:) = X(i,:) - meanx;
        end
    end
    if (normalize)
        normx = sqrt(squeeze(one*(X.^2))/n);
        for i = 1:size(X,1)
            X(i,:) = X(i,:)./normx;
        end
    end
    t = linspace(1,100,100)/max(abs(y'*X))*n;
end

%% run the iss algorithm and do cross validation
residmat = zeros(length(t),K);
for i = 1:K
    omit = folds(i).group;
    Xuse = X;
    Xuse(omit,:) = [];
    yuse = y;
    yuse(omit) = [];
    fit = iss(Xuse,yuse,intercept,normalize,'m');
    fi = predict(fit,X(omit,:),t,'m');
    fit = fi(5).predict;
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

cv_error = sum(residmat,2)/size(residmat,2);
cv_sd = sqrt(var(residmat,0,2)/K);

%% plot and generate 'cv_iss' class object
if (plot_it)
    plot(t,cv_error)
    %ylim([cv_error+cv_sd, cv_error-cv_sd])
    xlabel('t')
    ylabel('Cross-Validated MSE')
end
hold on
if (se)
    plot([t;t],[cv_error'+cv_sd';cv_error'-cv_sd'],'r')
end

field = 'cv_iss';
value = {t,cv_error,cv_sd};
object = struct(field,value);
end