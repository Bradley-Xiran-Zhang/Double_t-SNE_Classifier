function ydata = tsne_threshold_p(P, degrees_of_freedom)

%     Input:
%     P: pairwise affinities matrix for the original data
%     degrees_of_freedom: the degree of freedom of the t distribution
%     
%     Output:
%     ydata: the t-SNE map

    % Initialize some variables
    n = size(P, 1);                                     % number of instances
    momentum = 0.2;                                     % initial momentum
    final_momentum = 0.8;                               % value to which momentum is changed
    mom_switch_iter = 100;                              % iteration at which momentum is changed
    max_iter = 200;                                     % maximum number of iterations
    epsilon = 1;                                        % initial learning rate
    min_gain = .0001;                                   % minimum gain for delta-bar-delta
    
    % Make sure P-vals are set properly
    P(1:n + 1:end) = 0;                                 % set diagonal to zero
    P = 0.5 * (P + P');                                 % symmetrize P-values
    P = max(P ./ sum(P(:)), realmin);                   % make sure P-values sum to one
        
    % Initialize the solution
    ydata = .0001 * randn(n, 2);
    y_incs  = zeros(size(ydata));
    gains = ones(size(ydata));
    
    % Run the iterations
    for iter=1:max_iter
        
        % Compute joint probability that point i and j are neighbors
        sum_ydata = sum(ydata .^ 2, 2);
        numNeg1 = 1 ./ (1+bsxfun(@plus, sum_ydata, bsxfun(@plus, sum_ydata', -2 * (ydata * ydata'))) ./ degrees_of_freedom); % t_df
        clear sun_ydata;
        numNeg1(1:n+1:end) = 0;                                             % set diagonal to zero
        num = numNeg1 .^ ((degrees_of_freedom+1)/2);                        % get the real numerator
        Q = max(num ./ sum(num(:)), realmin);                               % normalize to get probabilities
        clear num;
        
        % Compute the gradients (faster implementation)
        L2 = (P - Q) .* numNeg1;
        clear Q numNeg1;
        y_grads = (2+2/degrees_of_freedom) * (diag(sum(L2, 1)) - L2) * ydata;
        clear L1 L2;
            
        % Update the solution
        gains = (gains + .2) .* (sign(y_grads) ~= sign(y_incs)) ...         % note that the y_grads are actually -y_grads
              + (gains * .8) .* (sign(y_grads) == sign(y_incs));
        gains(gains < min_gain) = min_gain;
        y_incs = momentum * y_incs - epsilon * (gains .* y_grads);
        ydata = ydata + y_incs;
        ydata = bsxfun(@minus, ydata, mean(ydata, 1));
        
        % Update the momentum if necessary
        if iter == mom_switch_iter
            momentum = final_momentum;
        end

        if ~rem(iter, 50)
            disp(num2str(iter));
        end
        
    end