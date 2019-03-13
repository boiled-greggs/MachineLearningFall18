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
F1 = @(x) 1*sum(sum(x(1:16,1:8) - x(1:16,9:16)))/256;
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

mn = min([min(D);min(Dtest)]);
mx = max([max(D);max(Dtest)]);
 
stretch1 = 2 / (mx(2) - mn(2));
shift1 = (mx(2) + mn(2))/2;
Dtest(:,2) = ((Dtest(:,2)-shift1).*stretch1);
D(:,2) = ((D(:,2)-shift1).*stretch1);
 
stretch2 = 2 / (mx(3) - mn(3));
shift2 = (mx(3) + mn(3))/2;
Dtest(:,3) = ((Dtest(:,3)-shift2).*stretch2);
D(:,3) = ((D(:,3)-shift2).*stretch2);

%%% RBF. k = num centers; r = 2/sqrt(k)
kmax = 50
E_CVs = zeros(1,kmax);
Ws = cell(1,kmax);
for k = 1:kmax
    r = 2/sqrt(k);
    E_CV = 0;
    for row = 1:300
        % Get first centers
        point = D(row,:);
        Dcv = repmat(D,1);
        Dcv(row,:) = [];
        yp = Y(row);
        Ycv = repmat(Y,1);
        Ycv(row,:) = [];
        center = Dcv(1,2:3)';
        
        for i = 2:k
            [a,loc] = max(min(sqrt((center(1,:) - Dcv(:,2)).^2 + (center(2,:) - Dcv(:,3)).^2),[],2));
            center = [center, Dcv(loc,2:3)'];
        end
        % get better centers
        x1 = Dcv(:, 2);
        x2 = Dcv(:, 3);
        
        for j = 1:50
            oldc = center;
            [a,loc] = min(sqrt((center(1,:) - Dcv(:,2)).^2 + (center(2,:) - Dcv(:,3)).^2),[],2);
            center = zeros(2,k);
            for i = 1:k
                center(1,i) = sum(x1(loc == i))/sum(loc == i);
                center(2,i) = sum(x2(loc == i))/sum(loc == i);
            end
        end
        
        % create RBF feature matrix Z
        Z = zeros(299, k+1);
        for i = 1:299
            Phi = ones(1,k+1);
            for j = 1:k
                Phi(1,j+1) = phi( sqrt((Dcv(i,2) - center(1,j))^2 + (Dcv(i,3) - center(2,j))^2) / r);
            end
            Z(i,:) = Phi;
        end
        w = ((Z'*Z) + 0*eye(k+1))\Z'*Ycv;
        
        % get cv error for this point
        phit = ones(k+1,1);
        for j = 1:k
            phit(j+1,1) = phi( sqrt((point(1,2) - center(1,j))^2 + (point(1,3) - center(2,j))^2) / r);
        end
        
        E_CV = E_CV + (sign(w'*phit) ~= yp);
        
    end
    E_CVs(k) = E_CV/300;
end

figure; hold on;
K = 1:kmax;
plot(K, E_CVs)
[val, loc] = min(E_CVs)

%%% FINAL W %%%
r = 2/sqrt(k);
% Get first centers
center = D(1,2:3)';
k = K(loc)
for i = 2:k
    [a,loc] = max(min(sqrt((center(1,:) - D(:,2)).^2 + (center(2,:) - D(:,3)).^2),[],2));
    center = [center, D(loc,2:3)'];
end
% get better centers
x1 = D(:, 2);
x2 = D(:, 3);

for j = 1:50
    oldc = center;
    [a,loc] = min(sqrt((center(1,:) - D(:,2)).^2 + (center(2,:) - D(:,3)).^2),[],2);
    center = zeros(2,k);
    for i = 1:k
        center(1,i) = sum(x1(loc == i))/sum(loc == i);
        center(2,i) = sum(x2(loc == i))/sum(loc == i);
    end
end

% create RBF feature matrix Z
Z = zeros(300, k+1);
for i = 1:300
    Phi = ones(1,k+1);
    for j = 1:k
        Phi(1,j+1) = phi( sqrt((D(i,2) - center(1,j))^2 + (D(i,3) - center(2,j))^2) / r);
    end
    Z(i,:) = Phi;
end
w = ((Z'*Z) + 0*eye(k+1))\Z'*Y;
    
%{
    figure;
    hold on;
    xlim([-1,1])
    ylim([-1,1])
    plot(center(1,:),center(2,:),'.b',"MarkerSize",10)
    voronoi(center(1,:), center(2,:));
    for i = 1:300
        if D(i,1) == 1
            plot(D(i,2),D(i,3),'.y', 'MarkerSize', 10);
        else
            plot(D(i,2),D(i,3),'.g', 'MarkerSize', 10);
        end
    end
%}

figure; hold on;
F = @(x,y) wPhi(x,y,center,r,k,w);
wPhi(D(1,2),D(1,3),center,r,k,w);
fimplicit(F)
xlim([-1,1])
ylim([-1,1])
for i = 1:300
    if D(i,1) == 1
        plot(D(i,2),D(i,3),'.r', 'MarkerSize', 10);
    else
        plot(D(i,2),D(i,3),'.b', 'MarkerSize', 10);
    end
end

Etest = 0;
for i = 1:8998
    Phitest = ones(k+1,1);
    for j = 1:k
        Phitest(j+1,1) = phi( sqrt((Dtest(i,2) - center(1,j))^2 + (Dtest(i,3) - center(2,j))^2) / r);
    end
    testclass = w'*Phitest;
    Etest = Etest + (sign(testclass) ~= Ytest(i));
end
Etest = Etest / 8998

function y = phi(x)
    y = exp(-.5*x^2);
end

function y = wPhi(x,y,c,r,k,w)
    Phi = ones(k+1,1);
    for j = 1:k
        Phi(j+1,1) = phi( sqrt((x - c(1,j))^2 + (y - c(2,j))^2) / r);
    end
    y = w'*Phi;
end



















