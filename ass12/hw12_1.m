clear
format short g

x = ones(1,2);
y = ones(1,1);
L = 2;

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
G1 = zeros(3,2);
G2 = zeros(3,1);
G1 = X0'*delta1(2:end)';
G2 = X1'*delta2';

%{
for l = 1:L
    Ss{l} = Ws{l}'*Xs{l}';
    Xs{l+1} = [1, tanh(Ss{l})'];
    Xs{l+1}
end

deltas = cell(1,2);

deltas{2} = .5*(Xs{3}(2) - y(1))*(1 - (Xs{3}(2))^2)
deltas{1} = ((ones(3,1) - Xs{2}'.^2)).*((W2*deltas{2}))
% 
deltas{1}(1,:) = [];

Ein = .25*(Xs{3}(2) - y(1))^2
g = cell(1,2);
g{1} = zeros(3,2);
g{2} = zeros(3,1);
g{1} = Xs{1}'*deltas{1}';
g{2} = Xs{2}'*deltas{2}';
%}







