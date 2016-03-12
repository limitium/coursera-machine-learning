function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% fprintf("t %f \n",theta);
% fprintf("ts %f \n",size(theta));
% fprintf("x %f \n",X(1,:));
% fprintf("xs %f \n",size(X(1,:)));

% fprintf("tx %f \n",X(1,:)*theta);

%for i=1:m
%	J=J+(-y(i)*log(sigmoid(X(i,:)*theta))-(1-y(i))*log(sigmoid(X(i,:)*theta)))/m
%end




% fprintf("sG %f \n",size(grad));
% fprintf("sX %f \n",size(X));
% fprintf("sY %f \n",size(y));
% fprintf("sTh %f \n",size(theta));


% for j=1:length(grad)
	% for i=1:m
		% thetaX = X(i,:)*theta;
% 		fprintf("thT %f \n",theta');
% 		fprintf("Xi %f \n",X(i));
% 		fprintf("sXi %f \n",size(X(i)));
% 		fprintf("thX %f \n",thetaX);
% 		fprintf("sth %f \n",size(thetaX));
% 		Ht = sigmoid(thetaX);
		
% 		Xij = X(i,j);
% 		yi = y(i);
% 		err = Ht-yi;
% 		inSum = err * Xij;
% % fprintf("sH %f \n",size(Ht));
% % fprintf("i %f \n",i);
% % fprintf("j %f \n",j);
% % fprintf("ht %f \n",Ht);
% % fprintf("xij %f \n",Xij);
% % fprintf("yi %f \n",yi);
% % fprintf("err %f \n",err);
% % fprintf("inSum %f \n",inSum);
% % fprintf("sSum %f \n",size(inSum));
% % fprintf("sgrad %f \n",size(grad(j)));
		
% 		grad(j) = grad(j) + inSum;
% 	end
% end


predictions =  sigmoid(X*theta);
% 
leftPart = -y' * log(predictions);

rightPart = (1 - y') * log(1 - predictions);

J = (1 / m) * (leftPart - rightPart);

grad = (1 / m) * ((predictions - y)' * X);
fprintf("%f \n",J);

% =============================================================

end
