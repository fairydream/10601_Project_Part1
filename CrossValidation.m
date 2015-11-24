function score = CrossValidation(pcX0, pcX1, pcX3, Y0, Y1, Y3, C, sigma)
    tot_score = 0;
    N_trial = 3;
    for j = 1:N_trial
        k = 10;
        indices0 = crossvalind('Kfold', size(pcX0, 1), k);
        indices1 = crossvalind('Kfold', size(pcX1, 1), k);
        indices3 = crossvalind('Kfold', size(pcX3, 1), k);
        for i = 1:k
            %% generate train & test data
            i_pcX0 = pcX0(indices0~=i, :);
            i_pcX1 = pcX1(indices1~=i, :);
            i_pcX3 = pcX3(indices3~=i, :);
            i_Y0 = Y0(indices0~=i, :);
            i_Y1 = Y1(indices1~=i, :);
            i_Y3 = Y3(indices3~=i, :);
            
            i_pcXtest = [pcX0(indices0==i, :); pcX1(indices1==i, :); pcX3(indices3==i, :)];
            i_Ytest = [Y0(indices0==i, :); Y1(indices1==i, :); Y3(indices3==i, :)];
            
            %% One-vs-One
            i_predict = One_One_SVM(i_pcX0, i_pcX1, i_pcX3, i_Y0, i_Y1, i_Y3, i_pcXtest, C, sigma);
            
            %% One-vs-All
%             i_predict = One_All_SVM(i_pcX0, i_pcX1, i_pcX3, i_Y0, i_Y1, i_Y3, i_pcXtest, C, sigma);
            
            %% MNRfit
%             i_predict = MNRfit(i_pcX0, i_pcX1, i_pcX3, i_Y0, i_Y1, i_Y3, i_pcXtest);


            %% score
            score_index = [i_Ytest==0, i_Ytest==1, i_Ytest==3];
            tot_score = tot_score + sum(i_predict(score_index));
        end
    end
    score = tot_score/N_trial;
end