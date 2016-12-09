% ECS 171 HW2 Problem 4
for l = 1:3
    for n = 1:4
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
        
        xTrain = yeastData(1:1039,11:end);
        yTrain = yeastData(1:1039,1:10);
        xTest = yeastData(1040:1484,11:end);
        yTest = yeastData(1040:1484,1:10);
        iterations = 500;
        alpha = 0.01;
        numLayers = l;
        numNodes = 3*n;

        [firstA,firstW,firstPrediction,errorTrain,errorTest,w,prediction] = neuralNetwork(xTrain,yTrain,iterations,alpha,xTest,yTest,numLayers,numNodes);

        testErrorMatrix(numLayers,n) = errorTest(iterations);
        trainErrorMatrix(numLayers,n) = errorTrain(iterations);

        clear Workspace;

    end
end
