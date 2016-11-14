function object = lb_logistic(X,y,kappa,alpha,alpha0_rate,tlist,nt,trate,intercept,group,index,print)

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
    group_split = [0,cumsum(groupsize)];
    group_split_length = length(group_split);
end
if (~isnumeric(print) || print == 0)
    print = 0;
else
    print = 1;
end
result_r = false(1,nt*(col+intercept));

solution = coder.ceval('LB_logistic',X,row,col,y,kappa,alpha,alpha0_rate,result_r,group_split,group_split_length...
    ,intercept,tlist,nt,trate,print);

path = reshape(solution(8),[],nt);
if (group)
    a = path(linspace(1,col,col),:);
    path(linspace(1,col,col),:) = a(ord_rev,:);
end

field = 'lb';
call = 'lb_logistic';
t = solution(12);
value = {call, family, path, intercept, alpha, kappa, t, meanx, normx, group};
object = struct(field,value);
end