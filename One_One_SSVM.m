function predict = One_One_SSVM(pcX0, pcX1, pcX3, Y0, Y1, Y3, pcXtest, C01, gamma01, C13, gamma13, C03, gamma03)
    param01 = num2str(cell2mat(strcat(strcat('-c', {' '}, num2str(C01)), strcat(' -g', {' '}, num2str(gamma01)))));
    param13 = num2str(cell2mat(strcat(strcat('-c', {' '}, num2str(C13)), strcat(' -g', {' '}, num2str(gamma13)))));
    param03 = num2str(cell2mat(strcat(strcat('-c', {' '}, num2str(C03)), strcat(' -g', {' '}, num2str(gamma03)))));
    svmStruct01 = ssvm_train([Y0;Y1], [pcX0;pcX1], param01);
    svmStruct13 = ssvm_train([Y1;Y3], [pcX1;pcX3], param13);
    svmStruct03 = ssvm_train([Y0;Y3], [pcX0;pcX3], param03);
    
    origin = zeros(length(pcXtest),1);
    
    [testResult01, ~] = ssvm_predict(origin, pcXtest, svmStruct01);
    [testResult13, ~] = ssvm_predict(origin, pcXtest, svmStruct13);
    [testResult03, ~] = ssvm_predict(origin, pcXtest, svmStruct03);

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