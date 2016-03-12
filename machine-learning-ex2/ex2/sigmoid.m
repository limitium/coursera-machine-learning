function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).
%fprintf("%f \n",size(z)(1));
%fprintf("%f \n",size(z)(2));
f=@(h) 1 ./ (1 + exp(-h)); % 1 / (1 + exp(-h));


g = f(z);
% for i=1:size(z)(1)
% 	%fprintf("I:%f sj:%f\n",i,size(z(2)));
% 	for j=1:size(z)(2)
% 		%fprintf("J:%f \n",j);
% 		%fprintf("%f \n",z(i,j));
% 		g(i,j)=1/(1+exp(-z(i,j)));
% 	end
% % =============================================================

% end
