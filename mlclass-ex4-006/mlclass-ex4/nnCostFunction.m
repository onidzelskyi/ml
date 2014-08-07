function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%% create Y

Y = zeros(size(y,1),max(y));
for i=1:size(y,1)
    Y(i,y(i)) = 1;
end

%% Add ones to the X data matrix
%X = [ones(m, 1) X];

%% calc hypotezis

%H = zeros(m,num_labels);

%for i=1:m
    %a1 = X(i,:);
    %a1 = [1 a1];
    
    %z2 = a1*Theta1';
    %a2 = sigmoid(z2);
    %a2 = [1 a2];

    %z3 = a2*Theta2';
    %a3 = sigmoid(z3);
    %H(i,:) = sigmoid([1 sigmoid([1 X(i,:)]*Theta1')]*Theta2');
    %H(i,:) = a3(:);
%end
%H1 = [ones(size(X,1),1) X];
%H2 = H1*Theta1';
%H3 = sigmoid(H2);
%H4 = [ones(size(H3,1),1) H3];
%H5 = H4*Theta2';
%H6 = sigmoid(H5);
H = sigmoid([ones(m,1) sigmoid([ones(size(X,1),1) X]*Theta1')]*Theta2');
 

%% unregularized cost 

%J = 1/m * sum(sum((-1)*Y.*log(H) - (1-Y).*log(1-H)));

%% regularized cost 
%T1 = Theta1(:,2:input_layer_size+1);
%T2 = Theta2(:,2:num_labels+1);

%T1_sum = sum(sum(T1.^2,1),2);
%T1_sum = 0;
%for i=1:size(T1,1)
%    for j=1:size(T1,2)
%        T1_sum = T1_sum + T1(i,j)*T1(i,j);
%    end
%end

%T2_sum = sum(sum(T2.^2,2),1);
%T2_sum = 0;
%for i=1:size(T2,1)
%    for j=1:size(T2,2)
%        T2_sum = T2_sum + T2(i,j)*T2(i,j);
%    end
%end

%J = 1/m * sum(sum((-1)*Y.*log(H) - (1-Y).*log(1-H))) + (lambda/(2*m))*(T1_sum+T2_sum);
%a = Theta1(:,2:size(Theta1,2)+1);
%b = Theta2(:,2:size(Theta2,2)+1);

%% vectorized cost function

J = 1/m * sum(sum((-1)*Y.*log(H) - (1-Y).*log(1-H))) + (lambda/(2*m))*(sum(sum(Theta1(:,2:input_layer_size+1).^2))+sum(sum(Theta2(:,2:num_labels+1).^2)));

%% gradient

A1 = [ones(size(X,1),1) X];
Z2 = A1*Theta1';
A2 = sigmoid(Z2);
A2 = [ones(size(A2,1),1) A2];
Z3 = A2*Theta2';
A3 = sigmoid(Z3);
d3 = A3-Y;
d2 = (d3*Theta2(:,2:end)).*sigmoidGradient(Z2);
D2 = d3'*A2;
D1 = d2'*A1;
Theta1_grad = (1.0/m)*D1 + [zeros(size(Theta1,1),1) (lambda/m)*Theta1(:,2:end)];
Theta2_grad = (1.0/m)*D2 + [zeros(size(Theta2,1),1) (lambda/m)*Theta2(:,2:end)];

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
