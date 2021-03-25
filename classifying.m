%Classify an image Using AlexNet
%Upload an image.
file=uigetfile;
I=imread(file);
figure
imshow(I);
%The image will resize at the first input layer
sz = netTransfer.Layers(1).InputSize;
I = imresize(I,sz(1:2));
label = classify(netTransfer,I);
%It classifies according to the trained dataset
figure("Name","Image Classification","NumberTitle","off");%%%%
imshow(I)
title(label);

