classdef Image_Classifier_exported < matlab.apps.AppBase

    % Properties that correspond to app components
    properties (Access = public)
        DemoAlexNetAppUIFigure         matlab.ui.Figure
        StartingImageClassifierButton  matlab.ui.control.Button
        ImageClassifierLabel           matlab.ui.control.Label
        ImageAxes                      matlab.ui.control.UIAxes
        RedAxes                        matlab.ui.control.UIAxes
        GreenAxes                      matlab.ui.control.UIAxes
        BlueAxes                       matlab.ui.control.UIAxes
    end

    
    properties (Access = private)
           net                            
    end
 
   methods (Access = private)
        
        function updateimage(app,imagefile)
           % For corn.tif, read the second image in the file 
           
           if strcmp(imagefile,'1.jfif')
                im = imread('1.jfif', 2);
            else
                try
                    im = imread(imagefile);
                catch ME
                    % If problem reading image, display error message
                    %uialert(app.DemoAlexNetAppUIFigure, ME.message, 'Image Error');
                    return;
                end            
            end 
        
            
            % Create histograms based on number of color channels
            switch size(im,3)
                case 1
                    % Display the grayscale image
                    imagesc(app.ImageAxes,im);
                                        
                    % Plot all histograms with the same data for grayscale
                    histr = histogram(app.RedAxes, im, 'FaceColor',[1 0 0],'EdgeColor', 'none');
                    histg = histogram(app.GreenAxes, im, 'FaceColor',[0 1 0],'EdgeColor', 'none');
                    histb = histogram(app.BlueAxes, im, 'FaceColor',[0 0 1],'EdgeColor', 'none');
                    
                case 3
                    % Display the truecolor image
                    imagesc(app.ImageAxes,im);
                                        
                    % Plot the histograms
                    histr = histogram(app.RedAxes, im(:,:,1), 'FaceColor', [1 0 0], 'EdgeColor', 'none');
                    histg = histogram(app.GreenAxes, im(:,:,2), 'FaceColor', [0 1 0], 'EdgeColor', 'none');
                    histb = histogram(app.BlueAxes, im(:,:,3), 'FaceColor', [0 0 1], 'EdgeColor', 'none');
                    
                otherwise
                    % Error when image is not grayscale or truecolor
                    uialert(app.DemoAlexNetAppUIFigure, 'Image must be grayscale or truecolor.', 'Image Error');
                    return;
            end
                % Get largest bin count
                maxr = max(histr.BinCounts);
                maxg = max(histg.BinCounts);
                maxb = max(histb.BinCounts);
                maxcount = max([maxr maxg maxb]);
                
                % Set y axes limits based on largest bin count
                app.RedAxes.YLim = [0 maxcount];
                app.RedAxes.YTick = round([0 maxcount/2 maxcount], 2, 'significant');
                app.GreenAxes.YLim = [0 maxcount];
                app.GreenAxes.YTick = round([0 maxcount/2 maxcount], 2, 'significant');
                app.BlueAxes.YLim = [0 maxcount];
                app.BlueAxes.YTick = round([0 maxcount/2 maxcount], 2, 'significant');
         
        end
        
        
        function alexNetModel(app)
            %Message 
            
             %Starting training data.
            %Gathering the Dataset
            unzip('Soil_Dataset_1.zip');
            imds= imageDatastore('Soil_Dataset_1', ...
                'IncludeSubfolders',true, ...
                'LabelSource','foldernames');
             
            [imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized');
             numTrainImages = numel(imdsTrain.Labels);
             randperm(numTrainImages,46);%
                     
            app.net = alexnet;
            %analyzeNetwork(net)
            %layer = imageInputLayer(inputSize);
            inputSize = app.net.Layers(1).InputSize;
            layersTransfer = app.net.Layers(1:end-3);
            numClasses = numel(categories(imdsTrain.Labels));
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
                'DataAugmentation',imageAugmenter);%DataAugmentation
            augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);
            
            options = trainingOptions('sgdm', ...
                'MiniBatchSize',10, ...
                'MaxEpochs',6, ...%
                'InitialLearnRate',1e-4, ... %1e-4
                'Shuffle','every-epoch', ...
                'ValidationData',augimdsValidation, ...
                'ValidationFrequency',3, ...%30
                'Verbose',false, ...
                'Plots','training-progress');
            
            %Training data completed
           
           netTransfer = trainNetwork(augimdsTrain,layers,options);
            
            [YPred,scores] = classify(netTransfer,augimdsValidation);
           
              
            %Classify an image Using AlexNet
            %Upload an image.
            % Display uigetfile dialog
            imagefile = {'*.jpg;*.tif;*.png;*.gif','All Image Files'};%%%%%%%
            [f, p] = uigetfile(imagefile);%%%%%%%%%%%%%%%%%%
            
                     
            % Make sure user didn't cancel uigetfile dialog
            if (ischar(p))%%%%%%%%%%%%%%%%%%%%%%%%%%
              fname = [p f];%%%%%%%%%%%%%%%%%%%%%%%
              updateimage(app,fname)%%%%%%%%%%%%%%
            end
            
            I=imread(fname);%%%%%%%%%%%%%%%
           
            sz = netTransfer.Layers(1).InputSize;%%%%%%%%%%%%%%%%
            I = imresize( I,sz(1:2));%%%%%%%%%%%%%%
            label = classify(netTransfer,I);%%%%%%%%%%%%%%%%
           
            %figure("Name","Image Classification","NumberTitle","off");%%%%
            %imshow(I)%%%%%%%%%%%%%%%%%
            %title(label);%%%%%%%%%%%%%%
            msgbox(char(label),"Image Classification","help"); 
            
        end 
        
     end
    
   

    % Callbacks that handle component events
    methods (Access = private)

        % Code that executes after component creation
        function startupFcn(app)
             % Configure image axes
            app.ImageAxes.Visible = 'off';
            app.ImageAxes.Colormap = gray(256);
            axis(app.ImageAxes, 'image');
            % Update the image and histograms
            updateimage(app, '');
                      
        
         
        end

        % Button pushed function: StartingImageClassifierButton
        function StartingImageClassifierButtonPushed(app, event)
            % Loading dataset
            msgbox("Loading Dataset and training data","Initiating ML","warn");
            %Starting training data.
            alexNetModel(app);
            
            
      
           
        end
    end

    % Component initialization
    methods (Access = private)

        % Create UIFigure and components
        function createComponents(app)

            % Create DemoAlexNetAppUIFigure and hide until all components are created
            app.DemoAlexNetAppUIFigure = uifigure('Visible', 'off');
            app.DemoAlexNetAppUIFigure.Color = [0.902 0.902 0.902];
            app.DemoAlexNetAppUIFigure.Position = [100 100 640 480];
            app.DemoAlexNetAppUIFigure.Name = 'Demo AlexNet App';
            app.DemoAlexNetAppUIFigure.Icon = 'alexNet-page.png';
            app.DemoAlexNetAppUIFigure.WindowStyle = 'modal';

            % Create StartingImageClassifierButton
            app.StartingImageClassifierButton = uibutton(app.DemoAlexNetAppUIFigure, 'push');
            app.StartingImageClassifierButton.ButtonPushedFcn = createCallbackFcn(app, @StartingImageClassifierButtonPushed, true);
            app.StartingImageClassifierButton.BackgroundColor = [0.9412 0.9412 0.9412];
            app.StartingImageClassifierButton.Position = [92 413 476 22];
            app.StartingImageClassifierButton.Text = 'Starting Image Classifier';

            % Create ImageClassifierLabel
            app.ImageClassifierLabel = uilabel(app.DemoAlexNetAppUIFigure);
            app.ImageClassifierLabel.FontSize = 18;
            app.ImageClassifierLabel.Position = [21 450 134 22];
            app.ImageClassifierLabel.Text = 'Image Classifier';

            % Create ImageAxes
            app.ImageAxes = uiaxes(app.DemoAlexNetAppUIFigure);
            app.ImageAxes.XTick = [];
            app.ImageAxes.XTickLabel = {'[ ]'};
            app.ImageAxes.YTick = [];
            app.ImageAxes.Position = [83 160 494 239];

            % Create RedAxes
            app.RedAxes = uiaxes(app.DemoAlexNetAppUIFigure);
            title(app.RedAxes, 'Red')
            xlabel(app.RedAxes, 'Intensity')
            ylabel(app.RedAxes, 'Pixels')
            app.RedAxes.PlotBoxAspectRatio = [1.82727272727273 1 1];
            app.RedAxes.XLim = [0 255];
            app.RedAxes.XTick = [0 128 255];
            app.RedAxes.Position = [1 0 216 151];

            % Create GreenAxes
            app.GreenAxes = uiaxes(app.DemoAlexNetAppUIFigure);
            title(app.GreenAxes, 'Green')
            xlabel(app.GreenAxes, 'Intensity')
            ylabel(app.GreenAxes, 'Pixels')
            app.GreenAxes.PlotBoxAspectRatio = [1.752 1 1];
            app.GreenAxes.XLim = [0 255];
            app.GreenAxes.XTick = [0 128 255];
            app.GreenAxes.Position = [225 5 196 138];

            % Create BlueAxes
            app.BlueAxes = uiaxes(app.DemoAlexNetAppUIFigure);
            title(app.BlueAxes, 'Blue')
            xlabel(app.BlueAxes, 'Intensity')
            ylabel(app.BlueAxes, 'Pixels')
            app.BlueAxes.PlotBoxAspectRatio = [1.7109375 1 1];
            app.BlueAxes.XLim = [0 255];
            app.BlueAxes.XTick = [0 128 255];
            app.BlueAxes.Position = [430 3 196 140];

            % Show the figure after all components are created
            app.DemoAlexNetAppUIFigure.Visible = 'on';
        end
    end

    % App creation and deletion
    methods (Access = public)

        % Construct app
        function app = Image_Classifier_exported

            % Create UIFigure and components
            createComponents(app)

            % Register the app with App Designer
            registerApp(app, app.DemoAlexNetAppUIFigure)

            % Execute the startup function
            runStartupFcn(app, @startupFcn)

            if nargout == 0
                clear app
            end
        end

        % Code that executes before app deletion
        function delete(app)

            % Delete UIFigure when app is deleted
            delete(app.DemoAlexNetAppUIFigure)
        end
    end
end