import time
import datetime
import numpy as np
import csv

from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
gnb = GaussianNB()

print('Loading data...')
data = np.loadtxt('../../samples/sample1.csv', delimiter = ',')
#test = np.loadtxt('../../test_ver2.csv', delimiter = ',') #currently cant be used i think we need to apply the parser to it first

X = preprocessing.normalize(np.matrix(data[:,  0:24]), norm='l2') #should we use l1 or l2 normalization?
Y = np.matrix(data[:, 24:48])


#the test data cant be used until the parser is applied to the test


#Xtest = preprocessing.normalize(np.matrix(test[:,  0:24]), norm='l2')
#Ytest = np.matrix(test[:, 24:48])

#for i in range(0,24): 
#perform fitting of model and classification for all 24 products
pred = gnb.fit(X, Y[:,1]).predict(X[1,:]) #.predict(Xtest)# the loaded test data would go in the prediction here

print(pred) # proof of concept, returns 0 indicating it does not think the very first person in sample1.csv will want the first product

print('No errors')