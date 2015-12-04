function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
initialC = 0.01;
initialSigma = 0.05;

C = initialC;
sigma = initialSigma;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
optC = C;
optSigma = sigma;
minError = 1000;
i = 1;
do
  j = 1;
  C = initialC;
  do
    model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
    printf("checking model accuracy with C = %f sigma = %f", C, sigma);
    predictions = svmPredict(model, Xval);
    error = mean(double(predictions ~= yval))
    if (error < minError)
      minError = error;
      optC = C;
      optSigma = sigma;
    endif
    C = C*10;
    j = j+1;
   until(j == 5)
  sigma = sigma*10;
  i = i+1;
 until(i == 5)
 
C = optC
sigma = optSigma
% =========================================================================
end
