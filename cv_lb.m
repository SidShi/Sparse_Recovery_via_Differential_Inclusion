function object = cv_lb(X,y,kappa,alpha,K,tlist,nt,trate,family,group,intercept,normalize,plot,index,lb,predict)
% any input for which the user does not know what to put should be some
% meaningless stuff like 'm'

if (~ismatrix(X))
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
        if (sum(any(sum(y,2) ~= 1)))
            error('y should be an indicator matrix!')
        end
    else
        error('y must be a vector or a matrix!')
    end
end

if (group)
    if (index == 'm')
        if (strcmp(family,'multinomial'))
            index = NaN;
        else
            group = false;
            disp('Index is missing, using group = false instead!')
        end
    end
    if (~isvector(index))
        error('Index must be a vector!')
    end
    if (length(index) + intercept ~= size(X,2))
        error('Length of index must be the same as the number of columns of X minus the intercept!')
    end
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

if (tlist == 'm')
    obj = lb(X,y,kappa,alpha,nt,trate,family,group,intercept,normalize);
    tlist = obj(7).lb;
end

residmat = zeros(length(tlist),K);
for i = 1:K
    omit = folds(i).group;
    Xuse = X;
    Xuse(omit,:) = [];
    if (strcmp(family,'multinomial'))
        yuse = y;
        yuse(omit,:) = [];
        fit = lb(Xuse,yuse,kappa,alpha,tlist,nt,trate,family,group,intercept,normalize);
    else
        yuse = y;
        yuse(omit) = [];
        fit = lb(Xuse,yuse,kappa,alpha,tlist,nt,trate,family,group,intercept,normalize);
    end
    fi = predict(fit,X(omit,:),tlist);
    fit = fi(5).predict;
    
    if (strcmp(family,'binomial'))
        residmat(:,i) = mean(y(omit)-((fit>0.5)+1)^2/2);
    elseif (strcmp(family,'gaussian'))
        residmat(:,i) = mean((y(omit)-fit)^2);
    end
    if (strcmp(family,'multinomial'))
        aidmat1 = zeros(1,length(t));
        for j = 1:length(t)
            aidmat3 = fit(:,:,j);
            aidmat4 = aidmat3;
            for m = 1:size(aidmat3,1)
                for n = 1:size(aidmat3,2)
                    if (aidmat3(m,n) == max(aidmat3(m,:)))
                        aidmat4(m,n) = 1;
                    else
                        aidmat4(m,n) = 0;
                    end
                end
            end
            aidmat1(j) = sum((y(omit,:)-aidmat4).^2)/2/length(omit);
        end
        residmat(:,i) = aidmat1;
    end
end

cv_error = zeros(1,size(residmat,1));
cv_sd = zeros(1,size(residmat,1));
for i = 1:size(residmat,1)
    cv_error(i) = mean(residmat(i,:));
    cv_sd(i) = sqrt(var(residmat(i,:))/K);
end

if (plot)
    plot(tlist,cv_error)
    ylim([cv_error+cv_sd, cv_error-cv_sd])
    xlabel('t')
    ylabel('Cross-Validated MSE')
end

field = 'lb';
value = {tlist,cv_error,cv_sd};
object = struct(field,value);
end