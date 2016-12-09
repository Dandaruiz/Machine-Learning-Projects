% ECS 171 HW2 part 1
% Daniel Ruiz
% Construct three node one hidden layer artificial neural network


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

epochs = 500;
alpha = 0.01;
numLayers = 3;
numNodes = 3;

[firstA,firstW,firstPrediction,errorTrain,errorTest,accuracyTrain,accuracyTest,weightsNode1,outputNodes,predictionTrain,predictionTest] = neuralNetwork(xTrain,yTrain,epochs,alpha,xTest,yTest,numLayers,numNodes);

figure(1)
hold on;
plot(1:epochs,errorTrain,'g-','linewidth',2)
plot(1:epochs,errorTest,'b-','linewidth',2)
xlabel('Epochs')
ylabel('Error')
title('Training and Testing MSE')
legend('Training Error','Testing Error')


figure(2)
hold on;
for j = 1:(size(xTrain,2)+1)
      plot(1:epochs,weightsNode1(:,j))
      xlabel('Epochs')
      ylabel('Weight')
      title('Weights from inputs to hidden node 1')
end
legend('feature 1','feature 2','feature 3','feature 4','feature 5','feature 6','feature 7','feature 8','feature 9')

figure(3)
hold on;
plot(1:epochs,accuracyTrain,'r-','linewidth',2)
plot(1:epochs,accuracyTest,'b-','linewidth',2)
xlabel('Epochs')
ylabel('Accuracy')
title('Training and Testing Accuracy')
legend('Training Accuracy','Testing Accuracy')

figure(4)
hold on;
for j = 1:size(yTrain,2)
    plot(1:epochs,outputNodes(:,j))
    xlabel('Epochs')
    ylabel('Output')
    title('Node Output at each iteration')
end
legend('output 1','output 2','output 3','output 4','output 5','output 6','output 7','output 8','output 9')

clear Workspace




