function [X, Y] = GetClass(Xtrain, Ytrain, label)
    index = Ytrain(:,:) == label;
    X = Xtrain(index, :);
    Y = Ytrain(index, :);
end