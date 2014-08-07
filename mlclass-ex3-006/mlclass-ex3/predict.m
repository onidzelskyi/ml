function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

%X = [ones(m,1) X];
%tmp = zeros(m,size(Theta2,1));
%for i=1:m
%    A_1 = [1 X(i,:)];
%    A_2 = zeros(1,size(Theta1,1));
%    A_3 = zeros(1,size(Theta2,1));
%    for j=1:size(Theta1,1)
%        theta = Theta1(j,:);
%        A_2(j) = sigmoid(theta*A_1');
%    end
%    A_2 = [1 A_2];
%    AA_2 = [1;sigmoid(Theta1*A_1')];
%    
%    for j=1:size(Theta2,1)
%        theta = Theta2(j,:);
%        A_3(j) = sigmoid(theta*A_2');
%    end
%    AA_3 = sigmoid(Theta2*A_2');
    
%    tmp(i,:) = AA_3(:);
%end

%for j=1:m
%    tmp1 = tmp(j,:);
%    [b,c] = max(tmp1);
%    p(j) = c;
%end

%[d,p] = max(tmp,[],2);

%A_2 = [ones(m,1) sigmoid([ones(m,1) X]*Theta1')];
%A_3 = sigmoid(A_2*Theta2');
%[d,p] = max(A_3,[],2);

[d,p] = max(sigmoid([ones(m,1) sigmoid([ones(m,1) X]*Theta1')]*Theta2'),[],2);

% =========================================================================


end
