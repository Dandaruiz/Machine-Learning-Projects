import time
import datetime
import numpy as np
import pandas as pd
import csv

from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing

PATH_TO_USERS = '../users'
PATH_TO_FILE = '../samples/sample'



print('Loading users for testing...')
df = pd.read_csv(PATH_TO_USERS + '.csv', index_col=[1], skiprows = 1, header = None)
users = df.as_matrix()


print('Loading data sample1 for testing...')
df2 = pd.read_csv(PATH_TO_FILE + '1.csv', header = None)
sample = df2.as_matrix()
length = len(sample[:,1])
width = 48-24+1
Ytest = np.zeros(shape =(length, width))

Ytest[:,0] = (sample[:,1])
for i in range(24,48):
	Ytest[:,i-23] = sample[:,i]


#print(df.loc[1190172,3])

##################################################

for i in range(1,length):
	for j in range(1,24):
		
		if Ytest[i,0] in df[0]:
			if(Ytest[i,j] != df.loc[Ytest[i,0],j+1]):
				print('i: ',i, '\tj: ',j)
				print('Key is: ', Ytest[i,0])
				print('Ytest is: ', Ytest[i,j], '\tOriginal is: ', df.loc[Ytest[i,0],j+1])

