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

Y = Y'
X = D(:, 2:end);

N = 300;

Q = zeros(N,N);

for i=1:N
    for j=1:N
        %Q(i,j) = Y(i)*Y(j)*(1 + D(i,2:end)*D(j,2:end)')^8; % 
        Q(i,j) = Y(i)*Y(j)*(1 + D(i,2)*D(j,2) + D(i,3)*D(j,3))^8;
    end
end

Eins = ones(5000,1);
Eouts = ones(5000,1);

for C=493:1:493
    f = -1*ones(N,1);
    aeq = Y;
    beq = 0;
    A = [];
    b = [];
    lb = zeros(N,1);
    ub = ones(N,1)*C;

    alpha = quadprog(Q,f,A,b,aeq,beq,lb,ub);

    xs = ones(1,2);
    ys = 0;
    for i=1:N
        if alpha(i) > 0.00002 && alpha(i) < C
            xs = X(i,:);
            ys = Y(i);
            break
        end
    end

    wx = (Y'.*alpha)'*(1+X*xs').^8;

    b = 1/ys-wx;


    Ein = 0;
    for i=1:N
        Ein = Ein + (Y(i) ~= sign(hx(X(i,1), X(i,2), Y, alpha, X, b, N)));
    end
    Eins(C) = Ein;
    Eout = 0;
    for i=1:8998
        Eout = Eout + (Ytest(i) ~= sign(hx(Dtest(i,2), Dtest(i,3), Y, alpha, X, b, N)));
    end
    Eouts(C) = Eout/8998;
end

figure; hold on;
fimplicit( @(x,y) hx(x,y,Y,alpha,X,b,N))
xlim([-1,1]);
ylim([-1,1]);
for i = 1:300
    if D(i,1) == 1
        plot(D(i,2),D(i,3),'.y', 'MarkerSize', 10);
    else
        plot(D(i,2),D(i,3),'.g', 'MarkerSize', 10);
    end
end

function s = hx(x1, x2, Y, alpha, X, b, N)
    s = b;
    for i=1:N
        if alpha(i) > 0
            s = s + Y(i)*alpha(i)*(1 + X(i,1)*x1 + X(i,2)*x2)^8;
        end
    end
end











    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    