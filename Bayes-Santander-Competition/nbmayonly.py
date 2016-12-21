import numpy as np
import csv

import sys

from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing

sys.path.append('..')

import common


def parse_users(file_name):
    f = open(file_name)
    reader = csv.DictReader(f)

    user_dict = {}

    for row in reader:
        id = float(row[common.ID_COLS[1]])
        user_dict[id] = []

        for i in range(len(common.PRODUCTS)):
            user_dict[id].append(float(row[common.PRODUCTS[i]]))

    f.close()
    return user_dict


# Creating a list of predictors so each one can be partially fitted for a different product
gnb = []
for i in range(0, 24):
    gnb.append(GaussianNB())

PATH_TO_FILE = '../../data/samplemay'

# Reads in samples 2-14 and does partial fits
print('Loading data sample...')
data = np.loadtxt(PATH_TO_FILE + '.csv', delimiter=',')

X = preprocessing.normalize(np.matrix(data[:, 2:24]), norm='l1')
Y = np.matrix(data[:, 24:48])
for i in range(0, 24):
    print('Performing partial fit for product predictor ' + str(i + 1) + ' on sample...')
    gnb[i].partial_fit(X, Y[:, i], [0, 1])


user_product = parse_users('../../data/users.csv')
# Creating a test set with 1 million of the samples
print('Loading data sample1 for testing...')
data = np.loadtxt('../../data/test_features.csv', delimiter=',')
SRows = data.shape[0]

Xtest = preprocessing.normalize(np.matrix(data[:, 2:24]), norm='l1')
#Ytest = np.matrix(data[:, 24:48])

out = open('../../data/naive_bayes_out.csv', 'w+')
writer = csv.writer(out)

length = Xtest.shape[0]
width = 24
P = np.zeros(shape = (length,width))
for i in range(24):
    #P.append(gnb[i].predict_proba(Xtest)[:, 1], )
    #np.hstack((P, gnb[i].predict_proba(Xtest)[:, 1]))    
    #np.append(P, gnb[i].predict_proba(Xtest)[:, 1], axis = 1)
    P[:,i] = gnb[i].predict_proba(Xtest)[:,1]
#print()
#sys.exit()  
default = [0]*24
writer.writerow(['ncodpers','added_products'])
for t in range(0,Xtest.shape[0]):
        uid = data[t, 1]
        products = user_product.get(uid,default)
        p = [0] * 24
        for j in range(0, 24):
            if products[j] == 1:
                continue
            #print(P)
            #print(P[int(t)])  
            #print(P[int(t)][j])
            #sys.exit(0)

            p[j] = P[t][j]
        a = (-np.array(p)).argsort()[:7].tolist()
        print('Writing row', t)
        writer.writerow([int(uid), ' '.join([common.PRODUCTS[k] for k in a])])


out.close()


