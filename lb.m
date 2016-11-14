function object = lb(X,y,kappa,alpha,c,tlist,nt,trate,family,group,index,intercept,normalize,print,lb_lasso,lb_logistic,lb_multilogistic)

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
        if (~strcmp(family,'multinomial') || sum(isNaN(index)) ~= 0)
            error('Length of index must equal to the number of columns of X')
        end
    end
end

n = size(X,1);
p = size(X,2);
one = ones(1,n);

if (intercept)
    meanx = one*X/n;
    for i = 1:size(X,2)
        X(:,i) = X(:,i) - meanx;
    end
else
    meanx = zeros(1,p);
end
if (normalize)
    normx = sqrt(squeeze(one*(X.^2))/n);
    for i = 1:size(X,2)
        X(:,i) = X(:,i)./normx;
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
if (alpha == 'm')
    sigma = norm(X);
    if (strcmp(family,'gaussian'))
        c = 1;
    elseif (strcmp(family,'binomial'))
        c = 4;
    else
        c = 2;
    end
    alpha = n*c/kappa/sigma^2;
    if (intercept)
        alpha0_rate = sigma^2/n;
    end
end
if (intercept && ~normalize)
    alpha0_rate = max(squeeze(one*(X.^2)))/n;
end

if (strcmp(family, 'gaussian'))
    object = lb_lasso(X,y,kappa,alpha,alpha0_rate,tlist,nt,trate,intercept,group,index,print);
elseif (strcmp(family,'binomial'))
    object = lb_logistic(X,y,kappa,alpha,alpha0_rate,tlist,nt,trate,intercept,group,index,print);
elseif (strcmp(family,'multinomial'))
    object = lb_multilogistic(X,y,kappa,alpha,alpha0_rate,tlist,nt,trate,intercept,group,index,print);
else
    error('No such family type!')
end

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
        for j = 1:size(object(4).lb(:,:,i),2)
            object(4).lb(:,j,i) = object(4).lb(:,j,i)./normx;
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