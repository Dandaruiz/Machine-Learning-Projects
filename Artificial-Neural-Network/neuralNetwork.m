%function to set up artificial neural network
function [firstA,firstW,firstPrediction,errorTrain,errorTest,accuracyTrain,accuracyTest,weightsNode1,outputNodes,predictionTrain,predictionTest] = neuralNetwork(xTrain,yTrain,iterations,alpha, xTest, yTest,numLayers,numNodes)


for l = 1:(numLayers+1)
     if l == 1
         rng(l);
         w{l} = rand(9, numNodes);
     end
     if l == (numLayers+1) 
         rng(l);
         w{l} = rand(numNodes+1, 10);
     end
     if l ~= (numLayers+1) && l > 1
         rng(l);
         w{l} = rand(numNodes+1,numNodes);
     end
         
end

 for j = 1:iterations
    %Use stochastic gradient descent and back propagation
    errTrain = 0;
    for i = 1:length(xTrain)
        
        %update all nodes 
        for l = 1:(numLayers+1)
            if l ==1
                a{l} = sigmoid([1, xTrain(i, :)]*w{l});
            else
                a{l} = sigmoid([1, a{l-1}]*w{l});
            end
        end
        
        if i ==1 && j == iterations
            firstW =  w;
            firstA = a;
        end
        
        %back propagation function
        [w,g,d] = backPropagation(a,xTrain,yTrain,i,w,alpha,numLayers);
        % output node
     
    end  
   %Calculate training and testing error
    [firstPrediction,eTrain,eTest,aTrain,aTest,predictTest,predictTrain] = neuralError(xTrain,yTrain,xTest,yTest,w,numLayers);
    accuracyTrain(j) = aTrain;
    accuracyTest(j) = aTest;
    errorTrain(j) = eTrain;
    errorTest(j) = eTest;
    weightsNode1(j,:)= w{1}(:,1)';
    outputNodes(j,:) = a{numLayers+1}(:);
    predictionTrain{j} = predictTrain;  
    predictionTest{j} = predictTest;
 end
    
end