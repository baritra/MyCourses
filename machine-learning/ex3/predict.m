function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%




%1st layer
XwithOnes = transpose([ones(size(X,1), 1), X]);
temp = sigmoid(Theta1*XwithOnes);

%2nd layer
%The alpha 1 values are in columns need to add a row of 1s
alpha1 = [ones(1, size(temp, 2)); temp];
predictions = transpose(sigmoid(Theta2*alpha1));

%Now the ith row of predictions has all the 10 output values for the ith input
[values, indices] = max(predictions, [], 2);
p = indices;







% =========================================================================


end
