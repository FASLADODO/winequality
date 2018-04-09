function [svmModel, svmAcc_trg, svmAcc, svmRec, svmPre, svmF1] = trainSVMmodel(trainX, trainY, testX, testY)
%% SVM classifier model:

%Bayesian Hyperparameter Optimization - commented out to save time
%Linear: BoxConstraint 1.8416, KernelScale 0.11525, Cost function 0.2135
%Gaussian: BoxConstraint 560.62, KernelScale 0.13539, Cost function 0.14098
%Polynomial:  BoxConstaint 2.1875, KernelScale 3.6498, Cost function 0.18853     

%svmModel = fitcsvm(trainX,trainY,'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus'))
%svmModel = fitcsvm(trainX,trainY,'KernelFunction','gaussian','OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus', 'KFold', 10))
%svmModel = fitcsvm(trainX,trainY,'KernelFunction','polynomial','OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus'))

rng(123);
%Uses the optimal set of parameters from optimization:
start_time = clock;
svmModel = fitcsvm(trainX,trainY,'Standardize',true,'KernelFunction','gaussian','BoxConstraint',560.62, 'KernelScale', 0.13539);
end_time = clock; 
diff = etime(end_time, start_time);
cvSVMModel = crossval(svmModel); % Creates a 10-fold cross-validated model
cvSVMLoss = kfoldLoss(cvSVMModel);
svmAcc_trg = 1 - cvSVMLoss;

% Test set predictions:
[svmLabel,svmScore] = predict(svmModel,testX');

%Print results:
fprintf('\nSVM Confusion Matrix:\n')
svmC = confusionmat(testY,svmLabel);
disp(svmC);
svmAcc = (svmC(1,1)+svmC(2,2))/sum(sum(svmC)); %(TP+TN)/ALL 
svmRec = svmC(2,2)/(svmC(2,1)+svmC(2,2));       %TP/(TP+FN) VERIFY FORMULAS
svmPre = svmC(2,2)/(svmC(1,2)+svmC(2,2));       %TP/(TP+FP) VERIFY FORMULAS
svmF1  = 2/(1/svmPre + 1/svmRec);      % Harmonic mean between precision and recall
fprintf('Cross-validation error: %.1f%%\n',cvSVMLoss*100);
fprintf('SVM training accuracy: %.1f%%\n',svmAcc_trg*100);
fprintf('SVM test accuracy: %.1f%%\n',svmAcc*100);
fprintf('SVM classifier implemented. Training took %.2f seconds.\n', diff);



end
