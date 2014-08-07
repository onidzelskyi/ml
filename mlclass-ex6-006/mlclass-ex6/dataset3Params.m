function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

%% 

C_vec = [0.01 0.03 0.1 0.3 1 3 10 30]';
sigma_vec = [0.01 0.03 0.1 0.3 1 3 10 30]';
error_train = zeros(length(C_vec)*length(sigma_vec), 1);
%error_val = zeros(length(C_vec)*length(sigma_vec), 1);
C_candidate = zeros(length(C_vec)*length(sigma_vec), 1);
sigma_candidate = zeros(length(C_vec)*length(sigma_vec), 1);
k = 1;
x1 = X(1,:);
x2 = X(2,:);

for i=1:length(C_vec)
    C = C_vec(i);
    for j=1:length(sigma_vec)
        sigma = sigma_vec(j);
        C_candidate(k) = C;
        sigma_candidate(k) = sigma;
        k = k + 1;
%        x1 = X(1,:);
%        x2 = X(2,:);
        model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
        predictions = svmPredict(model, Xval);
        prediction_error = mean(double(predictions ~= yval));
        error_train(k) = prediction_error;
    end
end


[min_val, index] = min(error_train(:))
C = C_candidate(index)
sigma = sigma_candidate(index)
% =========================================================================

end
