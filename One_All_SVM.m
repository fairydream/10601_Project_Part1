function predict = One_All_SVM(pcX0, pcX1, pcX3, Y0, Y1, Y3, pcXtest, C, sigma)
    N_all = size(pcX0, 1) + size(pcX1, 1) + size(pcX3, 1);
    
    svmStruct0 = svmtrain([pcX0;pcX1;pcX3], [ones(size(Y0)); zeros(N_all-size(Y0,1), 1)], 'kernel_function', 'rbf', 'boxconstraint',C, 'rbf_sigma',sigma);
    svmStruct1 = svmtrain([pcX1;pcX0;pcX3], [ones(size(Y1)); zeros(N_all-size(Y1,1), 1)], 'kernel_function', 'rbf', 'boxconstraint',C, 'rbf_sigma',sigma);
    svmStruct3 = svmtrain([pcX3;pcX0;pcX1], [ones(size(Y3)); zeros(N_all-size(Y3,1), 1)], 'kernel_function', 'rbf', 'boxconstraint',C, 'rbf_sigma',sigma);          

    testResult0 = svmclassify(svmStruct0, pcXtest);
    testResult1 = svmclassify(svmStruct1, pcXtest);
    testResult3 = svmclassify(svmStruct3, pcXtest);

    all_zeros_index = (testResult0==0) & (testResult1==0) & (testResult3==0) ;        

    predict = [testResult0, testResult1, testResult3];
    i_sum = sum(predict')';
    i_sum(all_zeros_index, :) = 1; 
    predict = predict./[i_sum, i_sum, i_sum];
    predict(all_zeros_index, :) = 1/3;
end