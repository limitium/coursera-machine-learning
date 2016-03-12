function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

fprintf("%f \n",size(theta));
fprintf("\n");        
fprintf("%f \n",size(theta));
fprintf("\n"); 
fprintf("%f \n",size(y));               
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    theta_t = zeros(size(theta));
    for i = 1:m
        theta_t(1,1) = theta_t(1,1) + (theta(1) .* X(i,1) + theta(2) .* X(i,2)-y(i)) .* X(i,1);
        theta_t(2,1) = theta_t(2,1) + (theta(1) .* X(i,1) + theta(2) .* X(i,2)-y(i)) .* X(i,2);
    end

    theta_t = theta - alpha .* theta_t ./ m;
    % ============================================================
                  
    % Save the cost J in every iteration    
    J = computeCost(X, y, theta);
    J_history(iter) = J;
    
    theta = theta_t;
end

end
