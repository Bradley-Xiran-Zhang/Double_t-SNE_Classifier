function ydata = tsne_threshold(X, perplexity, degrees_of_freedom, T)

%     Input:
%     X: the original data
%     perplexity: the perplexity used in t-SNE calculation
%     degreess_of_freedom: the df of the t distribution
%     T: percentage of the thresholding to use, a number in [0,1]
%     
%     Output:
%     ydata: the t-SNE map

    % Compute thresholding
    no_dim = size(X,2);
    thres = 2 * sqrt(log(2 * no_dim)) * T;
    clear no_dim;

    % Normalize input data
    X = X - min(X(:));
    X = X / max(X(:));
    X = bsxfun(@minus, X, mean(X, 1));
    sigmaX = std(X,0,1);
    thres = thres .* sigmaX;
    clear sigmaX;
    
    % Compute pairwise distance matrix
    n = size(X,1);
    D = zeros(n);
    for i = 1:n
        t = abs(X((i+1):end,:) - X(i,:));
        J = t - thres;
        t(J < 0) = 0;
        clear J;
        D(:,i) = [zeros(i,1); sum(t .^ 2, 2)];
        clear t;
    end
    clear thres n;
    D = D + D';
    
    % Compute joint probabilities
    P = d2p_clean(D, perplexity, 1e-5); % compute affinities using fixed perplexity
    clear D
    
    % Run t-SNE
    ydata = tsne_bradley_p(P, degrees_of_freedom, no);
    
end