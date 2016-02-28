function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
theta_rows = size(theta, 1);
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


sum_temp = 0;
for i = 1:m
  sum_temp += -y(i) * log(sigmoid(X(i, :) * theta)) - (1 - y(i)) * log(1 - sigmoid(X(i, :) * theta));
endfor

J = (sum_temp / m) + (lambda * sum(theta(2:theta_rows) .* theta(2:theta_rows))) / (2 * m);

theta_temp = zeros(size(theta));
theta_temp(1) = sum((sigmoid(X * theta) - y)' * X(:, 1)) / m;

for j = 2:size(theta, 1)
  sum_temp = 0;
  for i = 1:m
    sum_temp += (sigmoid(X(i, :) * theta) - y(i)) * X(i, j);
  endfor
  theta_temp(j) = sum_temp / m + (theta(j) * lambda) / m;
endfor

grad = theta_temp;



% =============================================================

end
