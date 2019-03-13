
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
%F1 = @(x) (((sum(sum(abs(x(:,1:5)))) + sum(sum(abs(x(:,12:16))))) / 160)-0.5)*2;
F1 = @(x) 1*sum(sum(x(1:16,1:8) - x(1:16,9:16)))/256
%Avg intensity
F2 = @(x) sum(sum(x)) / 256 ;

r = randperm(N,N);

D = zeros(300,3); %Data points
Y = ones(300,1);
Dtest = zeros(N-300,3);
Ytest = ones(N-300,1);
for i = 1:N
    x = reshape(data(r(i),2:end),w,w);
    if i < 301
        D(i,:) = [data(r(i),1), F1(x),F2(x) ];
        if data(r(i),1) ~= 1
            Y(i,1) = -1;
        end
    else 
        Dtest(i-300,:) = [data(r(i),1), F1(x),F2(x) ];
        if data(r(i),1) ~= 1
            Ytest(i-300,1) = -1;
        end
    end
end

mn = min([min(D);min(Dtest)])
mx = max([max(D);max(Dtest)])
 
stretch1 = 2 / (mx(2) - mn(2))
shift1 = (mx(2) + mn(2))/2
Dtest(:,2) = ((Dtest(:,2)-shift1).*stretch1);
D(:,2) = ((D(:,2)-shift1).*stretch1);
 
stretch2 = 2 / (mx(3) - mn(3))
shift2 = (mx(3) + mn(3))/2
Dtest(:,3) = ((Dtest(:,3)-shift2).*stretch2);
D(:,3) = ((D(:,3)-shift2).*stretch2);


%%%Legendre Transform
L = cell(1,9);
L{1} = @(x) 1;
L{2} = @(x) x;
for i = 2:8
    L{i+1} = @(x) ((2*i-1)/i).*x.*(L{i}(x)) - ((i-1)/i).*(L{i-1}(x))
end

Z = ones(300,45);
notZ = ones(8998,45);
col = 2;
a = ones(45,1);
b = ones(45,1);
for tot = 1:8
    for i = 0:tot
        Z(:,col) = (L{i+1}(D(:,2))) .* (L{tot-i+1}(D(:,3)));
        notZ(:,col) = (L{i+1}(Dtest(:,2))) .* (L{tot-i+1}(Dtest(:,3)));
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
%F = @(x,y) sign(sum(X(x,y)));
F = @(x,y) sum(X(x,y));

fimplicit(F)
xlim([-1,1])
ylim([-1,1])

figure;
% Regularization
lambda = 2
w = (inv(Z'*Z + lambda*eye(45)))*Z'*Y
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
%F = @(x,y) sign(sum(X(x,y)));
F = @(x,y) sum(X(x,y));

fimplicit(F)
xlim([-1,1])
ylim([-1,1])

% CV error
Lambda = 0:0.01:2;
ECVall = ones(201,1);
Etestall = ones(201,1);
i = 1
for lambda=0:0.01:2
    wreg = (inv(Z'*Z + lambda*eye(45)))*Z'*Y;
    H = Z*(inv(Z'*Z + lambda*eye(45)))*Z';
    yhat = H*Y;
    E_CV = sum(((yhat-Y)./(ones(300,1)-diag(H))).^2)/300;
    Etest = sum((notZ*wreg - Ytest).^2)/(N-300);
    ECVall(i) = E_CV;
    Etestall(i) = Etest;
    i = i+1;
end

figure;
hold on;
plot(Lambda, ECVall);
plot(Lambda, Etestall);
ylim([0,.5]);
xlim([0,2]);
legend("E_{CV} vs \lambda","E_{test} vs \lambda");

[emin, index] = min(ECVall);
lambstar = Lambda(index);
wregs = (inv(Z'*Z + lambstar*eye(45)))*Z'*Y;
figure;
syms A B
R = [wregs(1)]
col = 2;
for tot = 1:8
    for i = 0:tot
        R = [R, wregs(col)*L{i+1}(A).*L{tot-i+1}(B)];
        col = col + 1;
    end
end
%X = @(x,y) arrayfun(@(av,bv) L{av}(x).*L{bv}(y),a,b);
R
X = matlabFunction(R)
%F = @(x,y) sign(sum(X(x,y)));
F = @(x,y) sum(X(x,y));

fimplicit(F)
xlim([-1,1])
ylim([-1,1])
hold on;
%{
for i = 1:300
    if D(i,1) == 1
        plot(D(i,2),D(i,3),'.r');
    else
        plot(D(i,2),D(i,3),'.b');
    end
end
%}
%{
for i = 1:8998
    if Dtest(i,1) == 1
        plot(Dtest(i,2),Dtest(i,3),'.g');
    else
        plot(Dtest(i,2),Dtest(i,3),'.y');
    end
end
%}

%{
X = matlabFunction(sign(sum(R)))

[XL, YL] = meshgrid(-1:0.01:1,-1:0.01:1);
figure; hold on;
for i = 1:201
    for j = 1:201
        if(X(XL(i,j),YL(i,j)) == 1)
            plot(XL(i,j),YL(i,j),'.b');
        else
            plot(XL(i,j),YL(i,j),'.r');
        end
    end
end

for i = 1:300
    if D(i,1) == 1
        plot(D(i,2),D(i,3),'.g');
    else
        plot(D(i,2),D(i,3),'.y');
    end
end

figure; hold on;
plus = X(XL,YL) == 1;
minus = X(XL,YL) == -1;
plot(XL(plus),YL(plus),'.b');
hold on;
plot(XL(minus),YL(minus),'.r');

for i = 1:300
    if D(i,1) == 1
        plot(D(i,2),D(i,3),'.g');
    else
        plot(D(i,2),D(i,3),'.y');
    end
end

%}
