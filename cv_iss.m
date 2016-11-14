function object = cv_iss(X,y,K,t,intercept,normalize,plot,iss,predict)
% if t is not specified, use 'm' in the input
if (~ismatrix(X) || ~isvector(y))
    error('X must be a matrix and y must be a vector')
end
if (size(X,1) ~= length(y))
    error('Number of rows of X must equal to the length of y')
end

n = size(X,1);
si = floor(n/K);
mo = n-K*si;
randn = randperm(n);
folds(n) = struct();
for i = 1:mo
    folds(i).group = zeros(1,si+1);
end
for i = (mo+1):n
    folds(i).group = zeros(1,si);
end
for i = 1:n
    for j = 1:si
        folds(i).group(j) = randn(i+(j-1)*K);
    end
end
for i = 1:mo
    folds(i).group(si+1) = randn(i+(si-1)*K);
end

if (t == 'm')
    one = ones(1,n);
    if (intercept)
        meanx = squeeze(one*X)/n;
        for i = 1:size(X,2)
            X(:,i) = X(:,i) - meanx;
        end
    end
    if (normalize)
        normx = sqrt(squeeze(one*(X.^2))/n);
        for i = 1:size(X,2)
            X(:,i) = X(:,i)./normx;
        end
    end
    t = linspace(1,100)/max(abs(y*X))*n;
end

residmat = zeros(length(t),K);
for i = 1:K
    omit = folds(i).group;
    Xuse = X;
    Xuse(omit,:) = [];
    yuse = y;
    yuse(omit) = [];
    fit = iss(Xuse,yuse,intercept,normalize);
    fi = predict(fit,X(omit,:),t);
    fit = fi(5).predict;
    residmat(:,i) = mean((y(omit)-fit)^2);
end

cv_error = zeros(1,size(residmat,1));
cv_sd = zeros(1,size(residmat,1));
for i = 1:size(residmat,1)
    cv_error(i) = mean(residmat(i,:));
    cv_sd(i) = sqrt(var(residmat(i,:))/K);
end

if (plot)
    plot(t,cv_error)
    ylim([cv_error+cv_sd, cv_error-cv_sd])
    xlabel('t')
    ylabel('Cross-Validated MSE')
end

field = 'iss';
value = {t,cv_error,cv_sd};
object = struct(field,value);
end