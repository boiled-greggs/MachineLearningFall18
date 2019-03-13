X = [1 0; 0 1; 0 -1; -1 0; 0 2; 0 -2; -2 0]
Y = [-1, -1, -1, -1, 1, 1, 1]

mdl = fitcknn(X,Y,'NumNeighbors',1)
mdl

hold on;
xlim([-2.5, 1.5]);
ylim([-1.5, 2.5]);
for i = 1:7
    if Y(i) == 1
        plot(X(i,1),X(i,2),'.r');
    else
        plot(X(i,1),X(i,2),'.b');
    end
end
[v,c] = voronoi(X(:,1), X(:,2), 'LineSpec')
voronoi(X(:,1), X(:,2))
%color = {'r' 'b' 'g' 'm' 'c' 'y' 'k' 'w'};
for i = 1:length(v)
    fill(v(1,i),v(1,2),char(randsample(color,1)));
end
