clear
format short g
 
load zip.train
[n1,d]=size(zip);
w=floor(sqrt(d-1));
 
load zipd.test
n2 = size(zipd,1);
N = n1 + n2;
 
data = [zip;zipd];
 
%Avg intensity of the end cols
F1 = @(x) (((sum(sum(abs(x(:,1:5)))) + sum(sum(abs(x(:,12:16))))) / 160)-0.5)*2;
%Avg intensity
F2 = @(x) sum(sum(x)) / 256 ;
 
r = randperm(N,N);
 
D = zeros(300,3); %Data points
Y = ones(300,1);
Dtest = zeros(N-300,3);
for i = 1:N
    x = reshape(data(r(i),2:end),w,w);
    if i < 301
        D(i,:) = [data(r(i),1), F1(x),F2(x) ];
        if data(r(i),1) ~= 1
            Y(i,1) = -1;
        end
    else 
        Dtest(i-300,:) = [data(r(i),1), F1(x),F2(x) ];
    end
end
 
 
 
%%%Legendre Transform
L = cell(1,9);
L{1} = @(x) 1;
L{2} = @(x) x;
for i = 2:8
    L{i+1} = @(x) ((2*i-1)/i).*x.*(L{i}(x)) - ((i-1)/i).*(L{i-1}(x))
end
 
Z = ones(300,45);
col = 2;
a = ones(45,1);
b = ones(45,1);
for tot = 1:8
    for i = 0:tot
        Z(:,col) = (L{i+1}(D(:,2))) .* (L{tot-i+1}(D(:,3)));
        a(col) = i+1;
        b(col) = tot-i+1;
        col = col + 1;
    end
end
 
%%%PSUEDO INVERSE THINGY
 
w = (inv(Z'*Z))*Z'*Y;
 
 
syms A B
R = [w(1)]
col = 2;
for tot = 1:8
    for i = 0:tot
        R = [R, w(col)*L{i+1}(A).*L{tot-i+1}(B)];
        col = col + 1;
    end
end
%X = @(x,y) arrayfun(@(av,bv) L{av}(x).*L{bv}(y),a,b);
R
X = matlabFunction(R)
F = @(x,y) sign(sum(X(x,y)));
 
fimplicit(F)
xlim([-1,1])
ylim([-1,1])
hold on;
 
for i = 1:300
    if D(i,1) == 1
        plot(D(i,2),D(i,3),'.g');
    else
        plot(D(i,2),D(i,3),'.y');
    end
end