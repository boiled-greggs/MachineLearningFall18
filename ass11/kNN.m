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

kmax = 3;
E_CVs = ones(ceil(kmax/2),1);
for k=1:2:kmax
    for row=1:300
        point = D(row,:);
        [B, I] = mink(sqrt( (point(1,2)-D(:,2)).^2 + (point(1,3)-D(:,3)).^2 ), k+1);
        I(1,:) = [];
        yhatrow = sign(sum(Y(I)));
        E_CVs(ceil(k/2)) = E_CVs(ceil(k/2)) + (yhatrow ~= Y(row));
    end
end

E_CVs = E_CVs./300

[a,b] = min(E_CVs)

Ks = 1:2:kmax;

plot(Ks, E_CVs)

kopt = b*2 - 1
figure; hold on;
for i = -1:0.025:1
    for j = -1:0.025:1
        %I = sqrt(i.^2+j.^2);
        %J = atan(j./i);
        [B,I] = mink(sqrt( (i-D(:,2)).^2 + (j - D(:,3)).^2),kopt);
        if sum(Y(I)) > 0
            plot(i,j,'.b','MarkerSize',20)
        else
            plot(i,j,'.r','MarkerSize',20)
        end
    end
end

Ein = 0
for i = 1:300
    [B, I] = mink(sqrt( (D(i,2)-D(:,2)).^2 + (D(i,3)-D(:,3)).^2 ), 4);
    I(1,:) = [];
    yhatrow = sign(sum(Y(I)));
    Ein = Ein + (yhatrow ~= Y(i));
    if D(i,1) == 1
        plot(D(i,2),D(i,3),'.y', 'MarkerSize', 10);
    else
        plot(D(i,2),D(i,3),'.g', 'MarkerSize', 10);
    end
end
Ein = Ein / 300

Etest = 0;
for i = 1:8998
    [B,I] = mink(sqrt( (Dtest(i,2)-D(:,2)).^2 + (Dtest(i,3) - D(:,3)).^2),kopt);
    if sign(sum(Y(I))) ~= Ytest(i)
        Etest = Etest + 1;
    end
end

Etest/8998




