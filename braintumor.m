%%Load Data
imds = imageDatastore('tumor', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames'); 
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.8,'randomized');

%%
%%Load Pretrained Net
net = googlenet;
analyzeNetwork(net)

%%
%%input size
net.Layers(1)
inputSize = net.Layers(1).InputSize;

%%
%%Final Layer Replace
if isa(net,'SeriesNetwork') 
  lgraph = layerGraph(net.Layers); 
else
  lgraph = layerGraph(net);
end 

[learnableLayer,classLayer] = findLayersToReplace(lgraph);
[learnableLayer,classLayer] 

numClasses = numel(categories(imdsTrain.Labels));

if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
    
elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1,numClasses, ...
        'Name','new_conv', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
end

lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);

newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);

figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
plot(lgraph)
ylim([0,10])

%%
%Freeze Initial Layers
layers = lgraph.Layers;
connections = lgraph.Connections;

layers(1:10) = freezeWeights(layers(1:10));
lgraph = createLgraphUsingConnections(layers,connections);

%%
%Train
pixelRange = [-30 30];
scaleRange = [0.9 1.1];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange, ...
    'RandXScale',scaleRange, ...
    'RandYScale',scaleRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);

augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

miniBatchSize = 30;
valFrequency = 81;
options = trainingOptions('adam', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',10, ...
    'InitialLearnRate',3e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',valFrequency, ...
    'Verbose',false, ...
    'Plots','training-progress');

net = trainNetwork(augimdsTrain,lgraph,options);

%%
%Class image
[YPred,probs] = classify(net,augimdsValidation);
accuracy1 = mean(YPred == imdsValidation.Labels)

idx = randperm(numel(imdsValidation.Files),20);
figure
for i = 1:20
    subplot(4,5,i)
    I = readimage(imdsValidation,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label) + ", " + num2str(100*max(probs(idx(i),:)),3) + "%");
end

%%
%Features Train KNN,SVM...
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

layer = 'new_fc';
featuresTrain = activations(net,augimdsTrain,layer,'OutputAs','rows');
featuresValidation = activations(net,augimdsValidation,layer,'OutputAs','rows');
whos featuresTrain

YTrain = imdsTrain.Labels;
YValidation = imdsValidation.Labels;
classifier = fitcecoc(featuresTrain,YTrain);
YPred = predict(classifier,featuresValidation);
idx = [1 5 10 555];
figure
for i = 1:numel(idx)
    subplot(2,2,i)
    I = readimage(imdsValidation,idx(i));
    label = YPred(idx(i));
    imshow(I)
    title(char(label))
end

accuracy2 = mean(YPred == YValidation)