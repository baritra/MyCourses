function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
                 
              

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



% Part 1 implementation of the cost function
XwithOnes = [ones(size(X, 1), 1), X]';
z2 = Theta1*XwithOnes;
alpha2 = sigmoid(z2);
alpha2WithOnes = [ones(1, size(alpha2, 2)); alpha2];
output = sigmoid(Theta2*alpha2WithOnes);
%alpha2 is the predicted value with dimension 10x5000. ith column basically is the output
%vector or the predicted result for the ith input. We need to calculate the cost by first 
%determining what this predicted vector actually predicts, i.e., which digit we are predicting
%through this vector. After that, we can calculate the cost function 
J = 0
for i = 1:columns(output)
  %The y-value represents the actual letter. Convert that to the 10-element vector with the correponding
  %element as 1 and the rest as 0. This will facilitate comparison with the actual predicted vector
  correctValue = zeros(num_labels,1);
  correctValue(y(i,1), 1) = 1;
  costi = 0;
  for j = 1:rows(output)
    costi += -correctValue(j, 1)*log(output(j,i)) - (1-correctValue(j, 1))*log(1-output(j,i));
  endfor
  J += costi;
endfor
J = J/m;
%Add the regularization
Theta1WithoutBias = Theta1(:,2:size(Theta1,2));
Theta2WithoutBias = Theta2(:, 2:size(Theta2, 2));
SumofThetaSquares = sum(sumsq(Theta1WithoutBias))+sum(sumsq(Theta2WithoutBias));
Regularization = SumofThetaSquares*lambda/(2*m);
J = J + Regularization



%back propagation algorithm
for i = 1:m
  %construct the y vector (correct value) corresponding to the ith input
  correctValue = zeros(num_labels,1);
  correctValue(y(i,1), 1) = 1;

  %the output of the neural net for example i
  alpha3_i = output(:,i);
  %error in the output units
  delta3_i = alpha3_i - correctValue;
  %error of the hidden layer
  delta2_i = (Theta2WithoutBias'*delta3_i).*sigmoidGradient(z2(:,i));
  %accumulate partial derivatives of cost function with respect to Theta2 for this training example
  alpha2WithOnes_i = alpha2WithOnes(:,i);
  Theta2_grad = Theta2_grad .+ delta3_i*alpha2WithOnes_i';
  XWithBias_i = XwithOnes(:,i);
  Theta1_grad = Theta1_grad .+ delta2_i*XWithBias_i';
endfor
% -------------------------------------------------------------

% =========================================================================
Theta1_grad = Theta1_grad/m;
Theta1_grad_regularizationTerms = [zeros(rows(Theta1), 1), Theta1(:, 2:end)*(lambda/m)];
Theta1_grad = Theta1_grad .+ Theta1_grad_regularizationTerms;
Theta2_grad_regularizationTerms = [zeros(rows(Theta2), 1), Theta2(:, 2:end)*(lambda/m)];
Theta2_grad = Theta2_grad/m .+ Theta2_grad_regularizationTerms;
% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
