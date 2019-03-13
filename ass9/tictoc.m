x1 = ([1;0])
x2 = ([-1;0])

figure; hold on;
xlim([-2,2]);
ylim([-2,2]);
Y1 = @(x) y1(x);
Y2 = @(x) y2(x);
fplot(@(x) x);
fplot(@(x) x^3);
plot(x1(1), x1(2), '.b');
plot(x2(1), x2(2), '*r');

function y = y1(x)
    y = x;
end
function y = y2(x)
    y = x^3;
end