function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%guardem h per a no tenir que calcular-ho cada cop
h = X * theta;
%fórmula donada al pdf. Theta comença al 2 perque ens diuen que no s'ha de regularitzar el terme theta_subzero que en matlab és theta(1) 
J = ((1/(2*m)) * sum((h-y).^2)) + ((lambda/(2*m)) * sum(theta(2:end).^2));

%gradient per a j=0
grad(1) = (1/m) * sum((h-y).*X(:,1));
%gradient per a j>0
grad(2:end) = (1/m)*(X(:,2:end)'*(h-y)) + (lambda/m)*theta(2:end);
%juntem gradients
grad = grad(:);

end
