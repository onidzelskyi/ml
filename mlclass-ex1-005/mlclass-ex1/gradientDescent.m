function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

tmp = theta;
n = size(X,2); % number of features

  for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    
    %X2 = X(:,2)
    %tmp1 = theta(1) - alpha * (1/(m)) *sum((X*theta-y));
    %tmp2 = theta(2) - alpha * (1/(m)) *sum((X*theta-y).*X2);    
    %theta(1) = tmp1;
    %theta(2) = tmp2;

  for i=1:n
    X_i = X(:,i);
    tmp(i) = theta(i) - alpha * (1/(m)) *sum((X*theta-y).*X_i);      
  end

  theta = tmp;


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
