function object = iss(X,y,intercept,normalize,nvar,nnnpls)

if (~ismatrix(X) || ~isvector(y))
    error('X must be a matrix and y must be a vector')
end
if (size(X,1) ~= length(y))
    error('Number of rows of X must equal to the length of y')
end
n = size(X,1);
p = size(X,2);
one = ones(1,n);

if (intercept)
    meanx = squeeze(one*X)/n;
    for i = 1:size(X,2)
        X(:,i) = X(:,i) - meanx;
    end
    mu = mean(y);
    y = squeeze(y-mu);
else
    meanx = zeros(1,p);
    mu = 0;
    y = squeeze(y);
end
if (normalize)
    normx = sqrt(squeeze(one*(X.^2))/n);
    for i = 1:size(X,2)
        X(:,i) = X(:,i)./normx;
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
hist_rho = zeros(maxitr+1,p);
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
    rho(~active) = rho(~active)+delta_t*ng(~active);
    rho(add) = round(rho(add));
    active(add) = true;
    hist_rho(i+1,:) = rho;
    hist_t(i+1) = t;
    
    % need to work on calling "nnnpls" from R
    obj_nnls = nnnpls(X(:,active),y,rho(active));
    hist_path(i+1,:) = hist_path(i);
    hist_path(i+1,active) = obj_nnls(1).nnnpls;
    res = obj_nnls(3).nnnpls;
    active(active) = (obj_nnls(1).nnnpls ~= 0);
    if (sum(active) >= min(nvar,n,p))
        break
    end
end

hist_path = hist_path(linspace(1,use+1,use+1),:);
for i = 1:size(hist_path,2)
    hist_path(:,i) = hist_path(:,i)./normx;
end
hist_path = hist_path';

a0 = mu-meanx*hist_path;
hist_rho = hist_rho(linspace(1,use+1,use+1),:);
hist_rho = hist_rho';
hist_t = hist_t(linspace(1,use+1,use+1))*n;

call = 'iss';
field = 'lb';
value = {call,family,hist_path,a0,alpha,kappa,hist_t,meanx,normx,group};
object = struct(field,value);
end