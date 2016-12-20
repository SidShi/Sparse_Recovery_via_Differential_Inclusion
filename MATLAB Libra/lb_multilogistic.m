function object = lb_multilogistic(X,y,kappa,alpha,alpha0_rate,tlist,nt,trate,intercept,group,index,print,meanx,normx)

if (ischar(intercept))
    intercept = true;
end
if (ischar(print))
    print = false;
end

row = size(X,1);
col = size(X,2);
if (~isnumeric(intercept) || intercept == 0)
    intercept = 0;
else
    intercept = 1;
end
if (~group)
    group_split = 0;
    group_split_length = 0;
elseif (sum(isnan(index)) > 0)
    group_split = 0;
    group_split_length = 1;
else
    [dum1,ord] = sort(index);
    [dum2,ord_rev] = sort(ord);
    X = X(:,ord);
    group_size = histc(index,unique(index));
    group_split = [0,cumsum(group_size)];
    group_split_length = length(group_split);
end

category = size(y,2);
result_r = zeros(1,nt*(col+intercept)*category);

lbmultilogistic_void(X,row,col,y,category,kappa,alpha,alpha0_rate,result_r,group_split,group_split_length...
    ,intercept,tlist,nt,trate,print);

path_multi = zeros(category,col+intercept,nt);
tem = result_r;
for i = 0:(nt-1)
    path_multi(:,:,i+1) = reshape(tem(linspace(1+i*category*(col+intercept),(i+1)*category*(col+intercept),(col+intercept)*category)),[category,col+intercept]);
end

te = path_multi(:,linspace(1,col,col),:);
if (group && (sum(isnan(index)) < length(index)))
   path_multi(:,linspace(1,col,col),:)  = te(:,ord_rev,:);
end

field = 'lb';
call = 'lb_multilogistic';
family = 'Multinomial';
t = tlist;
value = {call, family, path_multi, intercept, alpha, kappa, t, meanx, normx, group};
object = struct(field,value);
end