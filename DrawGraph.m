function Graph = DrawGraph(Pixel, X, Y, Z)
    % Pixel, X, Y, Z are all column vectors
    width = range(X)+1;
    length = range(Y)+1;
    hight = range(Z)+1;
    
    X = X - min(X);
    Y = Y - min(Y);
    Z = Z - min(Z);

    Graph = zeros(width, length, hight);    
    
    Graph(Z*(width*length) + Y*width + X + 1) = Pixel;
        
%     for h = 1:hight
%         figure;
%         imshow(mat2gray(Graph(:,:,h)));
%     end
    
end