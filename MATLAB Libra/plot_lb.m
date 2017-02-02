function plot_lb(x, xtype, omit_zeros, eps)
% Plot method for lb objects
% Produce a plot of an LB fit. The default is a complete coefficient path.
% The default plot uses the fraction of L1 norm as the x. 
% For multinomial case, the sum of absolute values of different class's 
% coefficients are caculated to represent each variable.
% The intercept term is not ploted

% Input:
% x: lb object
% xtype: The x-axis type. "t" or "norm". Default is "t".
% omit_zeros: When the number of variables  is much greater than the number of observations, many coefficients will never be nonzero; this logical avoids plotting these zero coefficents
% eps: Definition of zero above, default is 1e-10
% Output:
% none, just do the plot

%% check if there are undetermined values
if (xtype == 'm')
    xtype = 't';
end
if (ischar(omit_zeros))
    omit_zeros = true;
end
if (ischar(eps))
    eps = 10^(-10);
end

%% Retrieve values
object = x;
if (~strcmp(object(2).lb, 'multinomial'))
    coef = object(3).lb;
else
    dum1 = object(3).lb;
    coef = zeros(size(dum1,2),size(dum1,3));
    for i = 1:size(dum1,3)
        coef(:,i) = sum(abs(dum1(:,:,i)),1)';
    end
end

if (omit_zeros)
    c1 = (abs(coef)*ones(size(coef,2),1))';
    nonzeros = c1 > eps;
    dum2 = 1:length(nonzeros);
    cnums = dum2(nonzeros);
    coef = coef(nonzeros,:);
else
    cnums = 1:size(coef,1);
    stepid = 1:size(coef,2);
end

if (xtype == 't')
    s = object(7).lb;
else
    s = sum(abs(coef),1);
    s = s/max(s);
end

%% do the plot
if (object(6).lb == Inf)
    for i = 1:size(coef,1)
        stairs(s,coef(i,:))
        hold on
    end
else
    for i = 1:size(coef,1)
        plot(s,coef(i,:))
        hold on
    end
end
plot(0:ceil(max(s)),zeros(1,ceil(max(s))+1))

end