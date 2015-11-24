function predict = MNRfit(pcX0, pcX1, pcX3, Y0, Y1, Y3, pcXtest)
%     Y = categorical([Y0;Y1;Y3]);
    Y = nominal([Y0;Y1;Y3]);
    Y = double(Y);

    B = mnrfit([pcX0;pcX1;pcX3], Y);
    
    pos = [exp([ones(size(pcXtest,1),1), pcXtest]*B), ones(size(pcXtest,1),1)];
    pos_sum = sum(pos')';
    predict = pos./[pos_sum, pos_sum, pos_sum];
end