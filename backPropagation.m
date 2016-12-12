% Back Propagation Function

function [w,g,d] = backPropagation(a,xTrain,yTrain,i,w,alpha,numLayers)

    for l = 1:(numLayers+1)
        if l == 1
            d{l} = a{numLayers+1}-yTrain(i, :);
            g{l} = [1, a{numLayers}]'*d{l}; 
            w{numLayers+1} = w{numLayers+1} - alpha * g{l};
        end
        
        if l == (numLayers+1)
            sum{l-1} = d{l-1}*w{numLayers-l+3}';
            d{l} = sum{l-1}(2:end).*(1 - a{numLayers-l+2}).*a{numLayers-l+2};
            g{l} = [1, xTrain(i, :)]'*d{l};
            w{numLayers-l+2} = w{numLayers-l+2}-alpha*g{l};
        end
        
        if l ~= (numLayers+1) && l > 1
          
            sum{l-1} = d{l-1}*w{numLayers-l+3}';
            d{l} = sum{l-1}(2:end).*(1 - a{numLayers-l+2}).*a{numLayers-l+2};
            g{l} = [1, a{numLayers-l+1}]'*d{l};
            w{numLayers-l+2} = w{numLayers-l+2}-alpha*g{l};
          
        end
    end

end