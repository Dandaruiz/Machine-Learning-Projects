Daniel Ruiz Machine Learning Projects

Artificial Neural Network:

Classified yeast samples using artificial neural network functions made from scratch in MATLAB.  

The yeast had 10 different output classes and was tested on several different ANN models:
1 hidden layer (3,6,9,12 nodes)
2 hidden layers (3,6,9,12 nodes)
3 hidden layers (3,6,9,12 nodes)

A UC Irvine study done on this data yielded a 55% accuracy using various models.  
This custom built ANN yielded a 54% accuracy, 1% off from the study.



Bayes Santander Competition:

Entered a Kaggle competition hosted by Santander Bank with a team of students.  The team used ANN, SVD, Collaborative Filtering and Bayes models to predict the product a Santander customer would want and then recommend that to the customer.  

The Bayes model, which was what I worked on, had the highest Kaggle score out of the group.  

The model was written in Python and utilized the sci-kit Python library for predictions.

The custom functions searched the user by ID to determine the products the customer already owned and discarded those products from the predictor, which returned the 7 highest probabilities for each product.

The seven highest probabilities were displayed to the customer as recommendations.