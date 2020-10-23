function [accuracy, cm, prediction] = Dtsne(Xtrain, Xtest, Ytrain, Ytest, Xtrainlab, Xtestlab, df, K, print_plot)
%     Input:
%     Xtrain: original training data
%     Xtest: original testing data
%     Ytrain: training data on the 2-dimensional t-SNE map
%     Ytrain: testing data on the 2-dimensional t-SNE map
%     Xtrainlab: label of the training data
%     Xtestlab: label of the testing data (only used to ccalculate the accuracy of the classification, does not influence the classification process)
%     df: degree of freedom chosen for the student t distribution
%     K: number of the nearest neighbors
%     print_plot: Boolean, whether to plot false classifications in the folder 'process'    
    
%     Output:
%     accuracy: Accuracy of the classification
%     cm: the confusion matrix
%     prediction: predicted label for the testing data

    %% Change Xtrainlab and Xtestlab to column vectors
    if size(Xtrainlab,2) > 1
        Xtrainlab = Xtrainlab';
    end
    if size(Xtestlab,2) > 1
        Xtestlab = Xtestlab';
    end
    %% Get basic information of classes
    classes = sort(unique(Xtrainlab)); % How many classes there are in the original labels
    length_classes = length(classes);                                    
    Xtrainlab_new = zeros(size(Xtrainlab));
    Xtestlab_new = zeros(size(Xtestlab));
    for i = 1:length_classes % Change the original labels to {1,...,n}
        Xtrainlab_new(Xtrainlab == classes(i)) = i;
        Xtestlab_new(Xtestlab == classes(i)) = i;
    end
    Xtrainlab = Xtrainlab_new; Xtestlab = Xtestlab_new;
    clear Xtrainlab_new Xtestlab_new;
    clear i;
    %% Prepare the palette for plots
    if print_plot == true
        colm = hot(length_classes);
        clear length_classes;
    end
    %% Classification
    testleng = size(Xtest, 1);
    prediction = zeros(testleng, 1)-1;
    for i = 1:testleng
        %% If plots are not required, output the process
        if print_plot==false && ~rem(i, 100)
            disp(num2str(i));
        end        
        %% Find the K-nearest neighbors
        distance = sum((Ytrain-Ytest(i,:)) .^ 2, 2); % Pairwise distance between X_i and all training points
        postr = tiedrank(distance) <= K; % Get the position of the K-nearest neighbors
        neighborslab = Xtrainlab(postr); % Label for the K-nearest neighbors
        cl_local = unique(neighborslab); % The local classes
        lengcl_local = length(cl_local); % How many classes there are in the neighborhood
        clear distance postr neighborslab;
        
        if lengcl_local == 1 % If all the points in the neighborhood come from the same class
            prediction(i) = cl_local; % Assign the testing point to the class
            clear neighbors cl_local lengcl_local;
        else            
            %% Find K-nearest neighbors from every local class
            neighbors = zeros(lengcl_local * K+1, size(Xtest, 2));
            for cln = 1:lengcl_local
                templocalcln = Ytrain(Xtrainlab == cl_local(cln),:);
                distance = sum((templocalcln-Ytest(i,:)) .^ 2, 2);
                [~, pos] = sort(distance, 'ascend');
                pos = pos(1:K); clear distance;
                Xtemplocalcln = Xtrain(Xtrainlab == cl_local(cln),:);
                neighbors((2-K:1) + cln*K,:) = Xtemplocalcln(pos,:);
            end
            clear cln templocalcln Xtemplocalcln pos;
            neighborslab = reshape(repmat(cl_local', K, 1),[],1);
            neighbors(1,:) = Xtest(i,:); % Set the first row of neighbors as testing X_i            
            %% Local projection
            perp = (size(neighbors, 1)-1)/3; % Perp = (n-1)/3
            Y = tsne_threshold(neighbors, perp, df, 0.2); % Run threshold t-SNE
            clear perp neighbors;
            lowdim = (Y-min(Y)) ./ (max(Y)-min(Y)); % Normalize the lower map into [0,1]^2
            clear Y;
            lowdimp = lowdim(1,:); % The projection Y_i of X_i
            lowdim(1,:) = [];            
            %% Regression
            [nearest_neighbor, k, d, x_lim] = Dtsne_regression(lowdim, neighborslab);
            % Predict based on the distance calculated
            if nearest_neighbor
                distance = sum((lowdim - lowdimp).^2, 2);
                [~, pos] = min(distance);
                prediction(i) = neighborslab(pos);
            else
                prediction(i) = cl_local(d == min(d));
            end
            clear cln d nearest_neighbor distance pos;            
            %% Plot if incorrect
            if prediction(i) ~= Xtestlab(i) && print_plot == true
                set(0, 'DefaultFigureVisible', 'off');
                scatter(lowdim(:,1), lowdim(:,2), 20, colm(neighborslab,:));
                title({['True: ', num2str(Xtestlab(i)), ' -> Predict: ', num2str(prediction(i))]});
                xlabel("x"); ylabel("y");
                hold on;
                scatter(lowdimp(1), lowdimp(2), 40, colm(Xtestlab(i),:), 'filled', '^');
                for cln = 1:lengcl_local
                    if ~isnan(k(cln,1))
                        plot(x_lim(cln,:), k(cln,1)*x_lim(cln,:)+k(cln,2),'Color', colm(cl_local(cln),:));
                    end
                end
                saveas(gcf, ['./process/', num2str(i), '.png']);
                hold off;
                clear cl neighbors postr lowdim lowdimp neighborslab cl_local lengcl_local;
            end            
        end
    end
    %% Change the labels back to the original names and calculate some results
    Xtestlab_res = zeros(size(Xtestlab));
    prediction_res = zeros(size(prediction));
    for i = 1:length(classes)
        Xtestlab_res(Xtestlab == i) = classes(i);
        prediction_res(prediction == i) = classes(i);
    end
    cm = confusionmat(Xtestlab_res, prediction_res);
    accuracy = sum(diag(cm))/testleng;
end