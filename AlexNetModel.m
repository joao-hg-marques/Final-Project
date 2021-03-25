%AlexNet%
%Unzip the dataset
unzip('Soil_Dataset.zip');
imds= imageDatastore('Soil_Dataset', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');
%Rezing the dataset at the first input layer
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized');
 numTrainImages = numel(imdsTrain.Labels);
 idx = randperm(numTrainImages,138);
%Ploting random images from the dataset to be trained
 figure;
for i = 1:8
  subplot(4,4,i)
 I = readimage(imdsTrain,idx(i));
 imshow(I)
end
%Loading AlexNet Deep Neuron Network
net = alexnet;
analyzeNetwork(net)
inputSize = net.Layers(1).InputSize;
layersTransfer = net.Layers(1:end-3);
numClasses = numel(categories(imdsTrain.Labels));
%layers
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];
pixelRange = [-30 30];%%%%%%%%%%%%%
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'ColorPreprocessing','gray2rgb',...%It was added "ColorPrepocessing 
    'DataAugmentation',imageAugmenter);
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);
options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'Plots','training-progress');
netTransfer = trainNetwork(augimdsTrain,layers,options);
%Classifying images that are in the dataset
[YPred,scores] = classify(netTransfer,augimdsValidation);
idx = randperm(numel(imdsValidation.Files),8);
figure;
for i = 1:8
    subplot(4,4,i)
    I = readimage(imdsValidation,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label));
end

