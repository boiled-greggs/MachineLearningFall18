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

%%% begin neural network lmao
%{
W1 = ones(3,2).*(.25);
W2 = ones(3,1).*(.25);

X0 = [1,x];

S1 = W1'*X0';
X1 = [1, tanh(S1)'];
%X1 = [1, S1'];
S2 = W2'*X1';
%X2 = tanh(S2);
X2 = S2;

%delta2 = .5*(X2 - y(1))*(1 - X2^2);
delta2 = .5*(X2 - y(1))*(1);
%delta1 = .5*(ones(3,1) - X1.^2)*(W2*delta2');
delta1 = (ones(1,3) - X1.^2)'.*(W2*delta2');

Ein = .25*(X2 - y(1))^2
G1 = zeros(3,2);
G2 = zeros(3,1);
G1 = X0'*delta1(2:end)';
G2 = X1'*delta2';
%}
% W1 = 3xm      W2 = mx1       X0 = 3x1     X1 = (m+1)x1       X2 = scalar
% S1 = mx1      S2 = scalar     delta2 = s   delta1 = (m+1)x1
m = 10;
iter = 2000000;
eta = .1;
N = 300;
X0 = D;
W1 = ones(3,m)*.25;
W2 = ones(m+1,1)*.25;
S1 = zeros(300,m);
S2 = zeros(300,1);
X1 = zeros(300,11);
X2 = zeros(300,1);
D2 = zeros(300,1);
D1 = zeros(300,m+1);
Eins = zeros(2000000,1);
its = 1:1:2000000;
lamb = .01/N;
tic
for j=1:iter
    Ein = 0;
    G1 = W1.*0;
    G2 = W2.*0;
    X1 = [ones(300,1), tanh(X0*W1)];
    X2 = X1*W2;
    D2 = .5*(X2 - Y);
    D1 = (1 - X1.^2).*(D2*W2');
    Ein = sum(1/(4*N)*(X2-Y).^2);
    Eins(j,1) = Ein;
    for i=1:300
        %{
        S1(i,:) = (W1'*X0(i,:)')';
        X1(i,:) = [1,tanh(S1(i,:))];
        S2(i,:) = W2'*X1(i,:)';
        X2(i,:) = S2(i,:);
        D2(i) = .5*(X2(i) - Y(i))*1;
        D1(i,:) = (ones(1,m+1) - X1(i,:).^2)'.*(W2*D2(i)');

        Ein = Ein + 1/(4*N)*(X2(i)-Y(i))^2;
        Eins(i,1) = Ein;
        %}
        G1xn = X0(i,:)'*D1(i,2:end) + 2*lamb/N*W1;
        G1 = G1 + 1/N*G1xn;
        G2xn = X1(i,:)'*D2(i) + 2*lamb/N*W2;
        G2 = G2 + 1/N*G2xn;
    end
    W1;
    W1 = W1 - eta*G1;
    W2;
    W2 = W2 - eta*G2;
end
toc

figure;
ylim([0,1]);
plot(its, Eins);
figure; hold on;
fimplicit(@(x,y) forw([1;x;y],W1,W2))
xlim([-1,1])
ylim([-1,1])
for i = 1:300
    if D(i,1) == 1
        plot(D(i,2),D(i,3),'.y', 'MarkerSize', 10);
    else
        plot(D(i,2),D(i,3),'.g', 'MarkerSize', 10);
    end
end

%%%%Forward
%X0 is 3xN      W1 is 3xm       S1 is mxN       X1 is (m+1)xN
%               W2 is (m+1)x1   S2 is 1xN       X2 is also 1xN
function X2 = forw(X0, W1, W2)
    N = size(X0,2);
	S1 = W1'*X0;
	X1 = vertcat(ones(1,N), tanh(S1)); %Sig
	S2 = W2'*X1;
	%X2 is output
	%X2 = tanh(S2);     %Sig
	%X2 = sign(S2);     %Sign
	X2 = tanh(S2);            %Linear
end

%{
for i=1:iter
    for j=1:300
        S1 = W1'*X0(j)';
        
        
    end
    
    
end
%}





















