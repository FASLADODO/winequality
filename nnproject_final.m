%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Script for Neural Computing coursework
%
% Shawn Ban
% Shalini Leelananda
% 3 April, 2018
%
% Implements an SVM and ANN for binary classification.
%
% Moved model training to separate functions for faster iteration:
%   trainSVMmodel.m
%   trainANNmodel.m
%
% We have saved the trained ANN model for replicability:
%   models.mat
%
% Hyperparameter optimization and grid search commented out to save time
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Data preprocessing
clc; clear all; close all;
winedata = csvread('wine_clean.csv', 1, 0);          
rng(123);
n = size(winedata, 1);            
targets  = winedata(:,12);           
targetsd = dummyvar(targets);     
inputs = winedata(:,1:11);               
inputs = inputs';                   
targets = targets';                 
targetsd = targetsd';             

% Keep 4000 samples for training, hold out 898 samples for testing:
c = cvpartition(n,'Holdout',898);
Xtest = inputs(:, test(c));         
Ytest = targets(test(c));   
Xtrain = inputs(:, training(c))';    
Ytrain = targetsd(:, training(c));   % dummy variable version, for ANN
Ytrain_nd = targets(:, training(c)); % non-dummy variable version

%% SVM classifier model:

[svmModel, svmAcc_trg, svmAcc, svmRec, svmPre, svmF1]  = trainSVMmodel(Xtrain, Ytrain_nd, Xtest, Ytest);

%% Use Neural Network toolbox to train a neural net:

[annModel, annAcc_trg, annAcc, annRec, annPre, annF1] = trainANNmodel(Xtrain, Ytrain, Ytrain_nd, Xtest, Ytest);

%% Display the ROC curves

% The ANN model results may vary by 1-2% due to random data division, so 
% we load the trained model to ensure replicability.
%
% Unsure why, but also obtained different SVM score on home laptop vs.
% school computer despite exact same models and same code. This loads SVM score
% to ensure replicability. 

load('models.mat') 

% ANN Training set accuracy:
Ypred_trg = annModel(Xtrain');             % predicts probability for each label
[~, Ypred_trg] = max(Ypred_trg);            % find the indices of max probabilities
annAcc_trg = sum(Ytrain_nd == Ypred_trg) / length(Ytrain_nd); % compare the predicted vs. actual

% ANN Test set predictions:
Ypred = annModel(Xtest);             % predicts probability for each label
[~, Ypred2] = max(Ypred);            % find the indices of max probabilities           
annC = confusionmat(Ytest,Ypred2);
annAcc = (annC(1,1)+annC(2,2))/sum(sum(annC)); %(TP+TN)/ALL 
annRec = annC(2,2)/(annC(2,1)+annC(2,2));       %TP/(TP+FN) VERIFY FORMULAS
annPre = annC(2,2)/(annC(1,2)+annC(2,2));       %TP/(TP+FP) VERIFY FORMULAS
annF1  = 2/(1/annPre + 1/annRec);      % Harmonic mean between precision and recall

% SVM test set predictions:
%[~,svmScore] = predict(svmModel,Xtest');

%Plot ROC curves:
[svmX,svmY,~,svmAUC] = perfcurve(Ytest',svmScore(:,2),2);           
[annX,annY,~,annAUC] = perfcurve(Ytest',Ypred(2,:),2);

figure
plot(annX,annY)
hold on
plot(svmX,svmY)
legend('MLP model','SVM model','Location','Best')
xlabel('False positive rate'); ylabel('True positive rate');
title('ROC Curves for MLP and SVM Classification')
hold off
fprintf('ROC curves plotted.\n\n\n');

%% Summarize performance measures
modelNames = {'ANN', 'SVM'};
perfMeasures = {'Training accuracy','Test accuracy', 'Recall', 'Precision', 'F1 score', 'AUC'};
results = [annAcc_trg,svmAcc_trg;annAcc,svmAcc;annRec,svmRec;annPre,svmPre;annF1,svmF1;annAUC,svmAUC];
restable= array2table(results,'RowNames',perfMeasures,'VariableNames',modelNames);
disp(restable);
