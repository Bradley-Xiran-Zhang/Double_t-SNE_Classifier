function [notpass, k, d, x_lim] = Dtsne_regression(lower_map, lab)
    
%     Input:
%     lower_map: the 2-dimensional t-SNE map of the neighborhood
%     lab: label information of the points in lower_map
%     
%     Output:
%     notpass: whether the regression lines pass the goodness-of-fit test
%     k: parameters of the lines
%     d: distances between the testing point and local classes
%     x_lim: the range of the regression segment

    class = unique(lab);
    n_class = length(class);
    N = length(lab)/n_class;
    
    notpass = false;
    k = zeros(n_class, 2);
    d = zeros(n_class, 1);
    x_lim = zeros(n_class, 2);
    for cln = 1:n_class
        lowdim_cl = lower_map(lab == class(cln),:);
        x = lowdim_cl(:,1); xmean = mean(x);
        y = lowdim_cl(:,2); ymean = mean(y);
        lxx = sum(x.^2) - N * xmean^2;
        lxy = sum(x .* y) - N * xmean * ymean;
        lyy = sum(y.^2) - N * ymean^2;
                
        % y = a x + b
        a = (lyy-lxx+sqrt((lyy-lxx)^2+4*lxy^2))/(2*lxy);
        b = ymean - a*xmean;        
        k(cln,1) = a; k(cln,2) = b;
        
        % Goodness-of-Fit
        D = (y-a.*x-b)./sqrt(a^2+1); Dmean = mean(D);
        R2 = 1 - sum(D.^2)/sum((D-Dmean).^2);
        if R2 <= 0.8
            notpass = true;
            break;
        end
        
        % Use regression
        x_pro = (a.*(y-b)+x)./(a^2+1); lowdimp_xpro = (a*(lowdimp(2)-b)+lowdimp(1))/(a^2+1);
        x_limits = [min(x_pro), max(x_pro)];
        if lowdimp_xpro < x_limits(1)
            y_limits = a .* x_limits(1) + b;
            d(cln) = sqrt(sum(([x_limits(1), y_limits] - lowdimp).^2, 2));
        elseif lowdimp_xpro > x_limits(2)
            y_limits = a .* x_limits(2) + b;
            d(cln) = sqrt(sum(([x_limits(2), y_limits] - lowdimp).^2, 2));
        else
            d(cln) = abs(lowdimp(2)-a*lowdimp(1)-b)/sqrt(1+a^2);
        end
        x_lim(cln,:) = x_limits;
        clear x y lxx lyy lxy a b x_limits;
    end
end