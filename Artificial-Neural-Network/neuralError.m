% training and test error function depending on the parameters


function [firstPrediction,eTrain,eTest,accuracyTrain,accuracyTest,predictionTest,predictionTrain] = neuralError(xTrain,yTrain,xTest,yTest,w,numLayers)
 
    [predictTrain,predictTest] = predictOutput(xTrain,yTrain,xTest,yTest,w,numLayers);
    firstPrediction = predictTest(1,:);
    eTrain1 = (predictTrain-yTrain).^2;
    eTrain2 = sum(eTrain1)/10;
    eTest1 = (predictTest-yTest).^2;
    eTest2 = sum(eTest1)/10;
    outputError = eTest2;
    eTest = sum(eTest2)/length(yTest);
    eTrain = sum(eTrain2)/length(yTrain);
    
    [outputTrain,I1Train] = max(predictTrain,[],2);
    [yTrain, I2Train] = max(yTrain,[],2);   
    [outputTest,I1Test] = max(predictTest,[],2);
    [yTest, I2Test] = max(yTest,[],2);
    accuracyTrain = sum(I1Train == I2Train)/length(I1Train);
    accuracyTest = sum(I1Test == I2Test)/length(I1Test);
    
    predictionTest = predictTest;
    predictionTrain = predictTrain;
end


