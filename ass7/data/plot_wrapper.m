% This M-file constructs the individual images for 60 digits
% and plots them to a file.

clear
format short g
load zip.train
digits=zip(:,1);
grayscale=zip(:,2:end);

[n,d]=size(grayscale);
w=floor(sqrt(d));

for i=1:100
	[i, digits(i)]
	curimage=reshape(grayscale(i,:),w,w);
	curimage=curimage';
	if digits(i)==5
        l=displayimage(curimage);
    end
	sstr=['IndividualImages/image',int2str(i)];
%	eval(['print -deps ',sstr]);
end

function out=displayimage(curimage)
%This is a functions that creates a graphical image of a 
%of a digit which is a w x w grayscale matrix.

[m,n]=size(curimage);
%implus=(curimage<-0.1);
im=zeros(m,n,3);
for i=1:3	
	im(:,:,i)=0.5*(1-curimage);
%	im(:,:,i)=implus;
end

out=image(im);
h=gca;
set(h,'XTick',[]);
set(h,'YTick',[]);
end