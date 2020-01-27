function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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
h=sigmoid(X*theta);
firstTerm=y.*log(h);
secondTerm=(1-y).*log(1-h);
lambdaTerm=(theta).^2;
constant=lambda/(2*m);
constant2=lambda/m;

J=(1/m)*sum(-firstTerm-secondTerm)+constant*sum(lambdaTerm(2:size(lambdaTerm,1)));



for i=1:size(grad),
  if(i==1)
      grad(i)=(1/m)*sum((h-y)'*X(:,i));
  else
      grad(i)=(1/m)*sum((h-y)'*X(:,i))+constant2*theta(i);
  end
end;






% =============================================================

end
