% Prediction function for output

function [predictTrain,predictTest] = predictOutput(xTrain,yTrain,xTest,yTest,w,numLayers)
    for l = 1:(numLayers+1)
        if l == 1
            xBiasTrain = [ones(length(xTrain(:,1)),1),xTrain];
            predictTrain{l} = sigmoid(xBiasTrain*w{l});
            predictTrain{l} = [ones(length(predictTrain{l}(:,1)),1), predictTrain{l}];
            xBiasTest = [ones(length(xTest(:,1)),1),xTest];
            predictTest{l} = sigmoid(xBiasTest*w{l});
            predictTest{l} = [ones(length(predictTest{l}(:,1)),1), predictTest{l}];
        end
        if l == (numLayers+1)
            predictTrain{l} = sigmoid(predictTrain{l-1}*w{l});
            predictTest{l} = sigmoid(predictTest{l-1}*w{l});
            
        end
        if l ~= (numLayers+1) && l > 1
            predictTrain{l} = sigmoid(predictTrain{l-1}*w{l});
            predictTrain{l} = [ones(length(predictTrain{l}(:,1)),1), predictTrain{l}];
            predictTest{l} = sigmoid(predictTest{l-1}*w{l});
            predictTest{l} = [ones(length(predictTest{l}(:,1)),1), predictTest{l}];
        end
  
    end
    predictTrain = predictTrain{numLayers+1};
    predictTest = predictTest{numLayers+1};

end
