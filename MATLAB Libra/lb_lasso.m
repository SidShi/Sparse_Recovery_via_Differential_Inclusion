function object = lb_lasso(X,y,kappa,alpha,alpha0_rate,tlist,nt,trate,intercept,group,index,print,meanx,normx)
% helper function for lb algorithm, used when the distribution is gaussian

% Input:
% X: An n-by-p matrix of predictors
% y: Response Variable
% kappa: The damping factor of the Linearized Bregman Algorithm that is defined in the reference paper. See details. 
% alpha: Parameter in Linearized Bregman algorithm which controls the step-length of the discretized solver for the Bregman Inverse Scale Space. 
% alpha0_rate: starting value
% tlist: Parameters t along the path.
% nt: Number of t. Used only if tlist is missing. Default is 100.
% trate: tmax/tmin. Used only if tlist is missing. Default is 100.
% intercept: if TRUE, an intercept is included in the model (and not penalized), otherwise no intercept is included. Default is TRUE.
% group: Whether to use a group penalty, Default is FALSE.
% index: For group models, the index is a vector that determines the group of the parameters. Parameters of the same group should have equal value in index. Be careful that multinomial group model default assumes the variables in same column are in the same group, and a empty value of index means each variable is a group.
% print: If TRUE, the percentage of finished computation is printed.
% meanx: column mean from main 'lb' function
% normx: column norm from main 'lb' function

% Output:
% A structure array with fields: function call, family, path, intercept,
% alpha, kappa, t, meanx, normx, group

%% check if there are undetermined variables
if (ischar(intercept))
    intercept = true;
end
if (ischar(print))
    print = false;
end

%% initialization
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

%% use mex C function to do the algorithm
lblasso_void(X,row,col,y,kappa,alpha,alpha0_rate,result_r,group_split,group_split_length...
    ,intercept,tlist,nt,trate,print);
path = reshape(result_r,[],nt);

if (group)
    a = path(linspace(1,col,col),:);
    path(linspace(1,col,col),:) = a(ord_rev,:);
end

%% generate the 'lb' class object
field = 'lb';
call = 'lb_lasso';
family = 'Lasso';
%t = tli;
t = tlist;
value = {call, family, path, intercept, alpha, kappa, t, meanx, normx, group};
object = struct(field,value);
end