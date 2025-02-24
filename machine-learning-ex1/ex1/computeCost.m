function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
%fprintf('%f %f \n', rows(X(1,:)),columns(X(1,:)));
%fprintf('%f %f \n', rows(theta),columns(theta));
%fprintf(theta);

%for i = 1:m
	%J = J + (X(i,:) * theta - y(i)).^2;
    %%J = J + (theta(1) + theta(2) .* X(i,2)-y(i)).*(theta(1) + theta(2) .* X(i,2)-y(i));
%end

%J = J / (2*m);

J = 1/(2.*m) .* (X*theta - y)' * (X*theta - y);


% =========================================================================

end
