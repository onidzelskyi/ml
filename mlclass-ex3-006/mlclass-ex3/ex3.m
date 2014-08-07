%% Machine Learning Online Class - Exercise 3 | Part 1: One-vs-all

%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  linear exercise. You will need to complete the following functions 
%  in this exericse:
%
%     lrCostFunction.m (logistic regression cost function)
%     oneVsAll.m
%     predictOneVsAll.m
%     predict.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%

%% Initialization
clear ; close all; clc

%% Setup the parameters you will use for this part of the exercise
%input_layer_size  = 400;  % 20x20 Input Images of Digits
%num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)

%% =========== Part 1: Loading and Visualizing Data =============
%  We start the exercise by first loading and visualizing the dataset. 
%  You will be working with a dataset that contains handwritten digits.
%

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

%load('ex3data1.mat'); % training data stored in arrays X, y

%% Updated for VSN
load('data.mat'); % training data stored in arrays X, y
XX = double(X);
yy = y;
X_row = size(XX,1);
X_col = size(XX,2);
X_cell = mat2cell(XX,[X_row/2, X_row-X_row/2], [X_col]);
X_A = X_cell{1};
X_B = X_cell{2};

y_row = size(y,1);
y_col = size(y,2);
y_cell = mat2cell(y,[y_row/2, y_row-y_row/2],[y_col]);
y_a = y_cell{1};
y_b = y_cell{2};

X_test = double(X_test);


input_layer_size  = size(X,2);
num_labels = size(unique(y),1);
m = size(X, 1);

% Randomly select 100 data points to display
rand_indices = randperm(m);
sel = X(rand_indices(1:9), :);

displayData(sel);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ============ Part 2: Vectorize Logistic Regression ============
%  In this part of the exercise, you will reuse your logistic regression
%  code from the last exercise. You task here is to make sure that your
%  regularized logistic regression implementation is vectorized. After
%  that, you will implement one-vs-all classification for the handwritten
%  digit dataset.
%

fprintf('\nTraining One-vs-All Logistic Regression...\n')

lambda = 0.1;
X = X_A;
y = y_a;
[all_theta] = oneVsAll(X, y, num_labels, lambda);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% Train additional data

fprintf('Train additional data.\n');

X = X_B;
y = y_b;
[all_theta] = oneVsAll(X, y, num_labels, lambda, all_theta);
X = XX; y = yy;


%% ================ Part 3: Predict for One-Vs-All ================
%  After ...
pred = predictOneVsAll(all_theta, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

%%
rp = find(pred==y);     % finds the indexes of the cases that were mis-classified

for i=1:size(rp,1)
    % Display 
    fprintf('\nDisplaying Example Image\n');
    displayData(X(rp(i), :));
    fprintf('\nDigit: %d\n', pred(rp(i)));
    pause;
end

%% process test data
pred = predictOneVsAll(all_theta, X_test);

for i=1:size(pred,1)
    % Display 
    fprintf('\nDisplaying test Image\n');
    displayData(X_test(i, :));
    fprintf('\nDigit: %d\n', pred(i));
    pause;
end
