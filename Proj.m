function testResult = Proj()
     %% load data    
    load('Train.mat');
    load('Test.mat')
    load('X0.mat');
    load('Y0.mat');
    load('X1.mat');
    load('Y1.mat');
    load('X3.mat');
    load('Y3.mat');
    
    %% pca   
    % only use Xtrain
%     mean_p = Xtrain' * ones(size(Xtrain,1),1) / size(Xtrain, 1);  % p-1
%     cenXtrain = Xtrain - ones(size(Xtrain,1),1) * mean_p';
%     [coeff,score,latent] = pca(cenXtrain);  

    % add Xtest
%     Xall = [Xtrain;Xtest];
%     mean_p = Xall' * ones(size(Xall,1),1) / size(Xall, 1);  % p-1
%     cenXall = Xall - ones(size(Xall,1),1) * mean_p';
%     
%     pcNum = 500;
% 
%     [coeff,score,latent] = pca(cenXall);  
%     baseF = coeff(:, 1:pcNum);
%         
%     cenX0 = X0 - ones(size(X0,1),1) * mean_p';
%     cenX1 = X1 - ones(size(X1,1),1) * mean_p';
%     cenX3 = X3 - ones(size(X3,1),1) * mean_p';
%     
%     pcX0 = cenX0*baseF;
%     pcX1 = cenX1*baseF;
%     pcX3 = cenX3*baseF;
   
%     load('pca_500.mat');
    
    %% lda
%     Xall = [Xtrain;Xtest];
%     mean_p = Xall' * ones(size(Xall,1),1) / size(Xall, 1);  % p-1
%     cenXtrain = Xtrain - ones(size(Xtrain,1),1) * mean_p';  
%     options = [];
%     options.Fisherface = 1;
%     [eigvector, eigvalue, elapse] = LDA(Ytrain,options,cenXtrain);
%     
%     cenX0 = X0 - ones(size(X0,1),1) * mean_p';
%     cenX1 = X1 - ones(size(X1,1),1) * mean_p';
%     cenX3 = X3 - ones(size(X3,1),1) * mean_p';
%     
%   
%     pcX0 = cenX0*eigvector;
%     pcX1 = cenX1*eigvector;
%     pcX3 = cenX3*eigvector;

    %% pca + lda
    load('pca_500.mat');
    
    pcNum = 91;
    
    cenXtrain = Xtrain - ones(size(Xtrain,1),1) * mean_p';  
    pcXtrain = cenXtrain*coeff(:, 1:pcNum);
    options = [];
    options.Fisherface = 1;
    [eigvector, eigvalue, elapse] = LDA(Ytrain,options,pcXtrain);
    
    cenX0 = X0 - ones(size(X0,1),1) * mean_p';
    cenX1 = X1 - ones(size(X1,1),1) * mean_p';
    cenX3 = X3 - ones(size(X3,1),1) * mean_p';
    
  
    pcX0 = cenX0*coeff(:, 1:pcNum)*eigvector;
    pcX1 = cenX1*coeff(:, 1:pcNum)*eigvector;
    pcX3 = cenX3*coeff(:, 1:pcNum)*eigvector;
    
    % k-fold cv for pca+lda
%     cenXtrain = Xtrain - ones(size(Xtrain,1),1) * mean_p';  
%     
%     score = [];
%     for i_pc = 100:5:120    
%         pcXtrain = cenXtrain*coeff(:, 1:i_pc);
%         options = [];
%         options.Fisherface = 1;
%         [eigvector, eigvalue, elapse] = LDA(Ytrain,options,pcXtrain);
% 
%         cenX0 = X0 - ones(size(X0,1),1) * mean_p';
%         cenX1 = X1 - ones(size(X1,1),1) * mean_p';
%         cenX3 = X3 - ones(size(X3,1),1) * mean_p';
%         
%         pcX0 = cenX0*coeff(:, 1:i_pc)*eigvector;
%         pcX1 = cenX1*coeff(:, 1:i_pc)*eigvector;
%         pcX3 = cenX3*coeff(:, 1:i_pc)*eigvector;
%         
%         score = [score;i_pc, CrossValidation(pcX0, pcX1, pcX3, Y0, Y1, Y3, 50, 100)];
%      %   fprintf('pc num: %d, score %d \n', i_pc, score);
%     end
%     score

    %% k-fold cross validation
    
    % One-vs-All
%     C = [0.1, 1, 50, 100];
%     sigma = [1, 100, 1000];

%     C = [40, 50, 60, 80, 100];
%     sigma = [1, 100, 500, 1000];
% 
%     for i_C = 1:size(C, 2)
%         for i_sigma = 1: size(sigma, 2)
%             score = CrossValidation(pcX0, pcX1, pcX3, Y0, Y1, Y3, C(i_C), sigma(i_sigma));
%             fprintf('C: %d, sigma %d, score %d \n', C(i_C), sigma(i_sigma), score);
%         end
%     end
    
%     % poly kernel
%         C = [0.005, 0.01, 0.1, 0.5, 1];
%         order = [2, 3, 4, 5, 6];
%         for i_C = 1:size(C, 2)
%             for i_order = 1:size(order ,2)
%                 score = CrossValidation(pcX0, pcX1, pcX3, Y0, Y1, Y3, C(i_C), order(i_order));
%                 fprintf('C: %d, order: %d, score: %d \n', C(i_C), order(i_order), score);
%             end
%         end

%     C = [0.01, 1, 50, 1000];
%     sigma = [0.1, 1, 100, 1000];
%     
%     for i_C = 1:size(C, 2)
%         for i_sigma = 1: size(sigma, 2)
%             score = CrossValidation(pcX0, pcX1, pcX3, Y0, Y1, Y3, C(i_C), sigma(i_sigma));
%             fprintf('C: %d, sigma %d, score %d \n', C(i_C), sigma(i_sigma), score);
%         end
%     end

%     for i_pc = 95:105
%         score = CrossValidation(pcX0(:, 1:i_pc), pcX1(:, 1:i_pc), pcX3(:, 1:i_pc), Y0, Y1, Y3, 50, 100);
%         fprintf('pc num: %d, score %d \n', i_pc, score);
%     end

%     for i_pc = 2:2
%         score = CrossValidation(pcX0(:, 1:i_pc), pcX1(:, 1:i_pc), pcX3(:, 1:i_pc), Y0, Y1, Y3, 50, 100);
%         fprintf('pc num: %d, score %d \n', i_pc, score);
%     end

        

    %% Predict
%     pc_Num = 75;
%  
%     pcX0 = pcX0(:, 1:pc_Num);
%     pcX1 = pcX1(:, 1:pc_Num);
%     pcX3 = pcX3(:, 1:pc_Num);
%     
%     pcXtest = (Xtest- ones(size(Xtest,1),1) * mean_p')*baseF(:, 1:pc_Num);
%     
%     % svm
% %     % One-vs-All
% %     C = 50;
% %     sigma = 100;
% %     testResult = One_All_SVM(pcX0, pcX1, pcX3, Y0, Y1, Y3, pcXtest, C, sigma);
% %     dlmwrite('prediction.csv', testResult,'precision','%.8f');
% 
%     
%     % One-vs-One
%     C = 40;
%     sigma = 159;
%     pcXtest = (Xtest- ones(size(Xtest,1),1) * mean_p')*coeff(:, 1:pcNum)*eigvector;
%     testResult = One_One_SVM(pcX0, pcX1, pcX3, Y0, Y1, Y3, pcXtest, C, sigma);
%     dlmwrite('prediction.csv', testResult,'precision','%.8f');
%
      % One-vs-One
      C01 = 0.0100;
      gamma01 = 0.0828;
      C13 = 0.0562;
      gamma13 = 0.0177;
      C03 = 0.3162;
      gamma03 = 0.0134;
      pcXtest = (Xtest- ones(size(Xtest,1),1) * mean_p')*coeff(:, 1:pcNum)*eigvector;
      testResult = One_One_SSVM(pcX0, pcX1, pcX3, Y0, Y1, Y3, pcXtest, C01, gamma01, C13, gamma13, C03, gamma03);
      dlmwrite('prediction.csv', testResult,'precision','%.8f');
%       
%     % mn logistic
%     testResult = MNRfit(pcX0, pcX1, pcX3, Y0, Y1, Y3, pcXtest);
%     dlmwrite('prediction.csv', testResult,'precision','%.8f');

end