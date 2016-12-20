function object = lb_lasso(X,y,kappa,alpha,alpha0_rate,tlist,nt,trate,intercept,group,index,print,meanx,normx)

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
else
    [dum1,ord] = sort(index);
    [dum2,ord_rev] = sort(ord);
    X = X(:,ord);
    group_size = histc(index,unique(index));
    group_split = [0,cumsum(group_size)];
    group_split_length = length(group_split);
end
if (~isnumeric(print) || print == 0)
    print = 0;
else
    print = 1;
end
result_r = zeros(1,nt*(col+intercept));


lblasso_void(X,row,col,y,kappa,alpha,alpha0_rate,result_r,group_split,group_split_length...
    ,intercept,tlist,nt,trate,print);
path = reshape(result_r,[],nt);

if (group)
    a = path(linspace(1,col,col),:);
    path(linspace(1,col,col),:) = a(ord_rev,:);
end



field = 'lb';
call = 'lb_lasso';
family = 'Lasso';
%t = tli;
t = tlist;
value = {call, family, path, intercept, alpha, kappa, t, meanx, normx, group};
object = struct(field,value);
end