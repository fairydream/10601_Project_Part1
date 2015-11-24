function predict = One_One_SVM(pcX0, pcX1, pcX3, Y0, Y1, Y3, pcXtest, C, sigma)
    svmStruct01 = svmtrain([pcX0;pcX1], [Y0;Y1], 'kernel_function', 'rbf', 'boxconstraint',C, 'rbf_sigma',sigma);
    svmStruct13 = svmtrain([pcX1;pcX3], [Y1;Y3], 'kernel_function', 'rbf', 'boxconstraint',C, 'rbf_sigma',sigma);
    svmStruct03 = svmtrain([pcX0;pcX3], [Y0;Y3], 'kernel_function', 'rbf', 'boxconstraint',C, 'rbf_sigma',sigma);          

%     svmStruct01 = svmtrain([i_pcX0;i_pcX1], [i_Y0;i_Y1], 'kernel_function', 'polynomial', 'boxconstraint',C, 'polyorder', order);
%     svmStruct13 = svmtrain([i_pcX1;i_pcX3], [i_Y1;i_Y3], 'kernel_function', 'polynomial', 'boxconstraint',C,'polyorder', order);
%     svmStruct03 = svmtrain([i_pcX0;i_pcX3], [i_Y0;i_Y3], 'kernel_function', 'polynomial', 'boxconstraint',C,'polyorder', order);       

 
    testResult01 = svmclassify(svmStruct01, pcXtest);
    testResult13 = svmclassify(svmStruct13, pcXtest);
    testResult03 = svmclassify(svmStruct03, pcXtest);

    pos0 = (testResult01==0) + (testResult03==0);
    pos1 = (testResult01==1) + (testResult13==1);
    pos3 = (testResult13==3) + (testResult03==3);

    all_same_index = pos0==pos1;
    isP0 = pos0==2;
    isP1 = pos1==2;
    isP3 = pos3==2;
    predict = zeros(size(pcXtest, 1), 3);
    predict([isP0, isP1, isP3]) = 1;
    predict(all_same_index, :) = 1/3;
end