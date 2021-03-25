%Classify an image Using AlexNet
%Upload an image.
file=uigetfile;
I=imread(file);
figure
imshow(I);
sz = netTransfer.Layers(1).InputSize;
I = imresize(I,sz(1:2));
%figure
%imshow(I);
label = classify(netTransfer,I);
figure("Name","Image Classification","NumberTitle","off");%%%%
imshow(I)
title(label);

