% ECS 171 HW2 Problem 5
% Daniel Ruiz


 load('hw2workspace.mat');
        yeastData = table2array(yeast(:,2:9));
        output = zeros(length(yeastData),10);
        outputClasses = {'CYT','NUC','MIT','ME3','ME2','ME1','EXC','VAC','POX','ERL'}';
        class = table2array(yeast(:,10));
        output = [];
        for j = 1:1484
            for i = 1:10
                if strcmp(class(j,1),outputClasses{i})
                    output(j,i) = 1;

                else
                    output(j,i) = 0;
                end
            end
        end
        yeastData = [output,yeastData];

        rng(100);
        yeastData(randperm(1484),:);

        xTrain = yeastData(:,11:end);
        yTrain = yeastData(:,1:10);

        epochs = 500;
        alpha = 0.01;
        numLayers = 1;
        numNodes = 3;
        
        
        
        unknownSample = [ 0.49 , 0.51 , 0.52 , 0.23 , 0.55 , 0.03 , 0.52 , 0.39];
        yTest = zeros(1,10);

        [firstA,firstW,firstPrediction,errorTrain,errorTest,accuracyTrain,accuracyTest,weightsNode1,hiddenNodes,predictionTrain,predictionTest] = neuralNetwork(xTrain,yTrain,epochs,alpha,unknownSample,yTest,numLayers,numNodes);
        
        predictionTrain = predictionTrain{epochs};
        predictionTest = predictionTest{epochs};
        figure(1)
        hold on;
        plot(1:10,predictionTest(:)','b*')
        xlabel('Class')
        ylabel('Probability')
        title('Class Prediction Probabilities')
        legend('Probability of class')

        clear Workspace;