function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %





    theta_tmp = zeros(length(theta), 1); 

    %theta2 = theta2 + (theta*X(i)-y(i)) .* X(i,2);


    for i = 1:m
        for j = 1:length(theta)
            theta_tmp(j) = theta_tmp(j) + (X(i,:)*theta-y(i)) .* X(i,j);
            %fprintf('%f %f \n', rows(X(i)),columns(X(i)));
            %fprintf('%f %f \n', rows(theta_tmp(j)),columns(theta_tmp(j)));
            %fprintf('%f %f \n', rows(X(i,j)),columns(X(i,j)));
            
        end
    end

    for j = 1:length(theta)
        theta_tmp(j) = theta(j) - alpha .* theta_tmp(j) / m;
    end

    %theta1 = theta(1) - alpha .* theta1 / m;
    %theta2 = theta(2) - alpha .* theta2 / m;
   %for i = 1:m
   %     theta1 = theta1 + (theta(1) + theta(2) .* X(i,2)-y(i));
   %     theta2 = theta2 + (theta(1) + theta(2) .* X(i,2)-y(i)) .* X(i,2);
   %end

    %theta1 = theta(1) - alpha .* theta1 / m;
    %theta2 = theta(2) - alpha .* theta2 / m;
 
    






    % ============================================================

    % Save the cost J in every iteration    
    J = computeCost(X, y, theta);
    J_history(iter) = J;
    %fprintf('%f \n',J)
    theta = theta_tmp;

end

end
