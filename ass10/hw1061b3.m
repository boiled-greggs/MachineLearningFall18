d = [1 0 -1 ; 0 1 -1 ; 0 -1 -1 ; -1 0 -1; 0 2 1; 0 -2 1; -2 0 1];


z = [sqrt(d(:,1).^2+d(:,2).^2), atan(d(:,2)./d(:,1)), d(:,3)];



figure; hold on;
for i = -3:0.05:3
    for j = -3:0.05:3
        I = sqrt(i.^2+j.^2);
        J = atan(j./i);
        [B,I] = mink(sqrt( (I-z(:,1)).^2 + (J - z(:,2)).^2),3);
        if sum(d(I,3)) > 0
            plot(i,j,'.b','MarkerSize',10)
        else
            plot(i,j,'.r','MarkerSize',10)
        end
    end
end


%{
redx = [-1.5,-1.5,-3,-3,-1.5,-1.5,3,3,-1.5];
redy = [-3,-1.5,-1.5,1.5,1.5,3,3,-3,-3];

patch([-3,-3,3,3],[-3,3,3,-3],[.5 .5 1 ]); 

xlim([-3,3])
ylim([-3,3])

hold on;
patch(redx,redy,[1 0.5 0.5]);


plot(redx,redy,'k')

%}
plot(d(1:4,1),d(1:4,2),'xr')
plot(d(5:7,1),d(5:7,2),'ob')
       