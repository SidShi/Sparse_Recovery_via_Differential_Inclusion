function pre = predict_lb(object, newx, t, type)
% Predict response variable for new data given a lb object

% The default plot uses the fraction of L1 norm as the x. 
% For multinomial case, the sum of absolute values of different class's 
% coefficients are caculated to represent each variable.
% The intercept term is not ploted

% Input:
% object: lb object
% newx: New data matrix that each row is a data or a vector. If missing, type switched to coefficients
% t: The parmeter for object to determin which coeffiecients used for prediction. Linear interpolation is used if t is not in object. If missing, all the coeffiecients along the path is used to predict.
% type: To predict response of newx or just fit coeffients on the path.
% Output:
% A structure array containing t and other variables. For type="fit", the rediction response
% "fit" is returned. For "binomial", a vector of the probabilities for newx 
% falling into class +1 is redurned. For "multinomial", a matrix with each column means
% the probabilities for newx falling into the corresponding class. If type="coefficients"
% coefficients "beta" and intercepts "a0" are returned.

if (ischar(type))
    type = 'fit';
end
if (ischar(newx) && strcmp(type,'fit'))
    warning('Type=fit with no newx argument; type switched to coefficients')
    type = 'coefficients';
end

path = object(3).lb;
a0 = object(4).lb;
if (ischar(t))
    t = object(7).lb;
    newbetas = path;
    newa0 = a0;
else
    t0 = object(7).lb;
    for i = 1:length(t)
        if (t(i) < min(t0))
            t(i) = min(t0);
        end
        if (t(i) > max(t0))
            t(i) = max(t0);
        end
    end
    coord = interp1(t0,linspace(1,length(t0),length(t0)),t);
    left = floor(coord);
    right = ceil(coord);
    cright = (t - t0(left))./(t0(right) - t0(left));
    cleft = (t0(right) - t)./(t0(right) - t0(left));
    if (~strcmp(object(2).lb, 'multinomial'))
        if (object(6).lb == Inf) 
            newbetas = path(:,left);
            newa0 = a0(left);
        else
            newbetas = path(:,left);
            for i = 1:length(left)
                newbetas(:,i) = cleft.*path(:,left(i))'+cright.*path(:,right(i))';
            end
            newa0 = cleft.*a0(left)+cright.*a0(right);
            for i = 1:length(left)
                if (left(i) == right(i))
                    newbetas(:,i) = path(:,left(i));
                    newa0(i) = a0(left(i));
                end
            end
            newbetas = squeeze(newbetas);
            newa0 = squeeze(newa0);
        end
    else
        newbetas = path;
        newa0 = a0(:,left(i));
        for i = 1:length(t)
            newbetas(:,:,i) = cleft(i)*path(:,:,left(i))+cright(i)*path(:,:,right(i));
        end
        for i = 1:length(left)
            newa0(:,i) = cleft'.*a0(:,left(i))+cright'.*a0(:,right(i));
        end
        for i = 1:length(left)
            if (left(i) == right(i))
                newbetas(:,:,i) = path(:,:,left(i));
                newa0(:,i) = a0(:,left(i));
            end
        end
    end
    
    if (strcmp(type,'fit'))
        n = size(newx,1);
        if (strcmp(object(2).lb,'gaussian'))
            predict = newx*newbetas+ones(n,1)*newa0;
        elseif (strcmp(object(2).lb,'binomial'))
            predict = 1./(1+exp(-newx*newbetas-ones(n,1)*newa0));
        elseif (strcmp(object(2).lb,'multinomial'))
            dum = newx*newbetas(:,:,1)';
            predict = zeros(size(dum,1),size(dum,2),length(t));
            for i = 1:length(t)
                predict(:,:,i) = newx*newbetas(:,:,i)'+ones(1,n)*newa0(:,i)';
            end
            for i = 1:length(t)
                for j = 1:size(predict(:,:,i),2)
                    rsc = 1/sum(predict(:,:,i),2);
                    predict(:,j,i) = predict(:,j,i)./rsc;
                end
            end
        end
    end
end

field = 'predict';
if (strcmp(type,'coefficients'))
    predict = 'm';
end
value = {type, t, newbetas, newa0, predict};
pre = struct(field, value);
end