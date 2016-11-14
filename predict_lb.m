function pre = predict_lb(object, newx, t, type)
% if user does not have a newx, then input would be a simple character such
% as 'm' to distinct from useful input
% if user does not have a t, then input would also be a simple character
% such as 'm' to distinct from useful input
% same for type

if (ischar(newx) && strcmp(type,'fit'))
    warning('Type=fit with no newx argument; type switched to coefficients')
    type = 'coefficients';
end

path = object(3).lb;
a0 = obect(4).lb;
if (ischar(t))
    t = object(7).lb;
    newbetas = path;
    newa0 = a0;
else
    t0 = object(2).lb;
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
    right = ceiling(coord);
    cright = (t - t0(left))./(t0(right) - t0(left));
    cleft = (t0(right) - t)./(t0(right) - t0(left));
    if (~strcmp(object(2).lb, 'multinomial'))
        if (object(6).lb == Inf) 
            newbetas = path(:,left);
            newa0 = a0(left);
        else
            newbetas = path;
            for i = 1:length(left)
                newbetas(:,i) = cleft(i)*path(:,left(i))+cright(i)*path(:,right(i));
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
        newa0 = a0;
        for i = 1:length(t)
            newbetas(:,:,i) = cleft(i)*path(:,:,left(i))+cright(i)*path(:,:,right(i));
        end
        for i = 1:length(left)
            newa0(:,i) = cleft(i)*a0(:,left(i))+cright*a0(:,right(i));
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
        if (size(newa0,1) == 1)
            hem = repmat(newa0,n,1);
        else
            hem = zeros(1,size(newa0,1)*size(newa0,2));
            for i = size(newa0,1)*size(newa0,2)
                hem(i) = newa0(i);
            end
        end
        if (strcmp(object(2).lb,'gaussian'))
            predict = newx*newbetas+hem;
        elseif (strcmp(object(2).lb,'binomial'))
            predict = 1./(1+exp(-newx*newbetas-hem));
        elseif (strcmp(object(2).lb,'multinomial'))
            predict = zeros(numel(newa0),n);
            for i = 1:length(t)
                h = exp(newx*(newbetas(:,:,i))+ones(n)*newa0(:,i));
                for j = 1:numel(h)
                    predict(j,i) = h(j);
                end
            end
            %predict <- sapply(1:length(t),function(x)
            %(t(scale(t(predict[,,x]), center=FALSE, scale=1/rowSums(predict[,,x])))))
        end
    end
end

field = 'predict';
value = {type, t, newbetas, newa0, predict};
pre = struct(field, value);
end