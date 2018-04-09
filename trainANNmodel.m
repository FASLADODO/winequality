function [annModel, annAcc_trg, annAcc, annRec, annPre, annF1] = trainANNmodel(trainX, trainY, trainYnd, testX, testY)
%% Use Neural Network toolbox to train a neural net
% Implements 10-fold cross-validation
rng(123);
indices = crossvalind('Kfold',trainYnd,10);
performance = zeros(1,10);
cvAccuracy = zeros(1,10);
trainY = trainY';
trainYnd = trainYnd';

for i = 1:10 
    testset = (indices == i); 
    trainset = ~testset;
    x = trainX(trainset,:)';
    t = trainY(trainset,:)';
    Xval = trainX(testset,:)';
    Yval = trainYnd(testset,:)';
 
    % Choose a Training Function
    % For a list of all training functions type: help nntrain
    % 'trainlm' is usually fastest.
    % 'trainbr' takes longer but may be better for challenging problems.
    % 'trainscg' uses less memory. Suitable in low memory situations.
    trainFcn = 'trainbr'; % Also tried: trainlm, trainscg, trainrp, traingdx
    
    % Create a Pattern Recognition Network
    hiddenLayerSize = [11 8]; % Also tried: [11 8], [11 8 4], [11 8 4 2]
    net = patternnet(hiddenLayerSize, trainFcn);
    
    %Set activation function:
    net.layers{1}.transferFcn = 'tansig'; % Also tried: logsig
    net.layers{2}.transferFcn = 'tansig';
    
    % Setup Division of Data for Training, Validation, Testing
    net.divideParam.trainRatio = 80/100;
    net.divideParam.valRatio = 10/100;
    net.divideParam.testRatio = 10/100;

    % Set parameters:
    net.trainParam.epochs = 1000; %Maximum number of epochs 
    net.trainParam.max_fail = 10; %Maximum validation failures
    net.trainParam.lr = 0.1; %Learning rate 
    net.trainParam.mc = 0.9; %Momentum
    net.trainParam.goal = 0.00001;  
    
    %net.performParam.regularization = 0.5; %Regularized performance function
    net.trainParam.mu = 0.005;
    %net.trainParam.min_grad = 1e-6
    %net.performFcn = 'msereg';
    %net.performParam.ratio=0.5; 
    
    % Train the Network
    [annModel,~] = train(net,x,t);

    % Test the Network
    y = annModel(x);
    %e = gsubtract(t,y);
    performance(1,i) = perform(annModel,t,y);
    %tind = vec2ind(t);
    %yind = vec2ind(y);
    %percentErrors = sum(tind ~= yind)/numel(tind);
    
    Ypred = annModel(Xval);             % predicts probability for each label
    [~, Ypred] = max(Ypred);            % find the indices of max probabilities
    cvAccuracy(1,i) = sum(Yval == Ypred) / length(Yval); % compare the predicted vs. actual
end

trainY = trainY';
trainYnd = trainYnd';

% Train the model on the full training set:
x = trainX';
t = trainY; 
start_time = clock;
[annModel,~] = train(net,x,t);
end_time = clock;
diff = etime(end_time, start_time);

% Training set accuracy:
Ypred_trg = annModel(trainX');             % predicts probability for each label
[~, Ypred_trg] = max(Ypred_trg);            % find the indices of max probabilities
annAcc_trg = sum(trainYnd == Ypred_trg) / length(trainYnd); % compare the predicted vs. actual

% Test set predictions:
Ypred = annModel(testX);             % predicts probability for each label
[~, Ypred2] = max(Ypred);            % find the indices of max probabilities
Ypred = Ypred(2, :);                 % take probability of positive class             

%Print results:
fprintf('\nANN Confusion Matrix:\n')
annC = confusionmat(testY,Ypred2);
disp(annC);
fprintf('10-fold cross-validation error: %.1f%%\n',(1 - mean(cvAccuracy))*100);
annAcc = (annC(1,1)+annC(2,2))/sum(sum(annC)); %(TP+TN)/ALL 
annRec = annC(2,2)/(annC(2,1)+annC(2,2));       %TP/(TP+FN) VERIFY FORMULAS
annPre = annC(2,2)/(annC(1,2)+annC(2,2));       %TP/(TP+FP) VERIFY FORMULAS
annF1  = 2/(1/annPre + 1/annRec);      % Harmonic mean between precision and recall
fprintf('ANN training accuracy: %.1f%%\n',annAcc_trg*100);
fprintf('ANN test accuracy: %.1f%%\n',annAcc*100);
fprintf(1, 'Network trained. Training took %.2f seconds\n', diff);
end