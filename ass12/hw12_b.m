format short g

x = ones(1,2);
y = ones(1,1);
L = 2
% can just use tanh(x) to get tanh of x

W1 = ones(3,2).*(.25);
W2 = ones(3,1).*(.25);

X0 = [1,x];

S1 = W1'*X0';
X1 = [1, tanh(S1)'];
%X1 = [1, S1'];
S2 = W2'*X1';
X2 = tanh(S2);
%X2 = S2;

delta2 = .5*(X2 - y(1))*(1 - X2^2);
%delta2 = .5*(X2 - y(1))*(1);
%delta1 = .5*(ones(3,1) - X1.^2)*(W2*delta2');
delta1 = (ones(1,3) - X1.^2)'.*(W2*delta2');

Ein = .25*(X2 - y(1))^2

G1p = zeros(3,2);
G2p = zeros(3,1);
for i=1:3
    for j=1:2
        W1(i,j) = .2501;
        S1 = W1'*X0';
        X1 = [1, tanh(S1)'];
        %X1 = [1, S1'];
        S2 = W2'*X1';
        X2 = tanh(S2);
        %X2 = S2;

        Einp = .25*(X2 - y(1))^2;

        G1p(i,j) = (Einp - Ein)/.0001;
        W1(i,j)=.25;
    end
    W2(i,1) = .2501;
    S1 = W1'*X0';
    X1 = [1, tanh(S1)'];
    %X1 = [1, S1'];
    S2 = W2'*X1';
    X2 = tanh(S2);
    %X2 = S2;

    Einp = .25*(X2 - y(1))^2;

    G2p(i,1) = (Einp - Ein)/.0001;
    W2(i,1)=.25;
end



G1 = zeros(3,2);
G2 = zeros(3,1);
G1 = X0'*delta1(2:end)';
G2 = X1'*delta2';

