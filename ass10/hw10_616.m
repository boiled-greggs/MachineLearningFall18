%d = rand(10000,2);

cen = rand(10,2);
d = [];
for i = 1:10
    d = [d; normrnd(cen(i,1),0.1,1000,1), normrnd(cen(i,2),0.1,1000,1)];
end

center = d(1,:)';
 
for i = 2:10
    [a,loc] = max(min(sqrt((center(1,:) - d(:,1)).^2 + (center(2,:) - d(:,2)).^2),[],2));
    center = [center, d(loc,:)'];
end
%{
plot(d(:,1),d(:,2),'.r')
xlim([0,1])
ylim([0,1])
hold on;

%plot(center(1,:),center(2,:),'.b',"MarkerSize",10)

%voronoi(center(1,:), center(2,:));
%}
%[a, loc] = min(sqrt((center(1,:) - d(:,1)).^2 + (center(2,:) - d(:,2)).^2),[],2)

x1 = d(:, 1);
x2 = d(:, 2);

for j = 1:50
    oldc = center;
    [a,loc] = min(sqrt((center(1,:) - d(:,1)).^2 + (center(2,:) - d(:,2)).^2),[],2);
    center = zeros(2,10);
    for i = 1:10
        center(1,i) = sum(x1(loc == i))/sum(loc == i);
        center(2,i) = sum(x2(loc == i))/sum(loc == i);
    end
end

figure;
hold on;
xlim([0,1])
ylim([0,1])
plot(center(1,:),center(2,:),'.b',"MarkerSize",10)
voronoi(center(1,:), center(2,:));


test = rand(10000,2);
tic
for i = 1:10000
    [a, locq] = min(sqrt((test(i,1) - d(:,1)).^2 + (test(i,2) - d(:,2)).^2),[],2);
    min(a);
end
toc

clusters = cell(1,10);
for i = 1:10
    clusters{1,i} = [x1(loc==i), x2(loc==i)];
end
radii = zeros(1,10);
for i = 1:10
    radii(i) = max(sqrt((center(1,i) - clusters{1,i}(:,1)).^2 + (center(2,i) - clusters{1,i}(:,2)).^2));
end

tic
[a, locq] = min(sqrt((test(:,1) - center(1,:)).^2 + (test(:,2) - center(2,:)).^2),[],2);
for i = 1:10000
    [lhs, nn] = min(sqrt((test(i,1) - clusters{1,locq(i)}(:,1)).^2 + (test(i,2) - clusters{1,locq(i)}(:,2)).^2),[],2);
    if sum(lhs <= sqrt((test(i,1) - center(1,:)).^2 + (test(i,2) - center(2,:)).^2) - radii) < 10
        [a,nn] = min(min(sqrt((test(i,1) - d(:,1)).^2 + (test(i,2) - d(:,2)).^2),[],2));
    end
end
toc

%{
[lhs,nn] = min(sqrt((q(i,1) - clusters{locq(i)}(:,1)).^2 + (q(i,2) - clusters{locq(i)}(:,2)).^2));
if lhs <= sqrt((q(i,1) - center(1,:)).^2 + (q(i,2) - center(2,:)).^2) - radii
    [a,nn] = min(min(sqrt((q(i,1) - d(:,1)).^2 + (q(i,2) - d(:,2)).^2),[],2));
end
%}



