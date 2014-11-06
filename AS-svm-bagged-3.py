
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from sklearn import svm, cross_validation
from sklearn.ensemble import BaggingRegressor


# In[2]:

train = pd.read_csv('training.csv')
test = pd.read_csv('sorted_test.csv')
labels = train[['Ca','P','pH','SOC','Sand']].values

train.drop(['Ca', 'P', 'pH', 'SOC', 'Sand', 'PIDN'], axis=1, inplace=True)
test.drop('PIDN', axis=1, inplace=True)


# In[3]:

#xtrain, xtest = np.array(train)[:,:3578], np.array(test)[:,:3578]
xtrain, xtest = np.array(train)[:,:3594], np.array(test)[:,:3594]

xtrain[:,3593]=(xtrain[:,3593]=='Topsoil')*1.0

print (xtrain[1,3593])

xtest[:,3593]=(xtest[:,3593]=='Topsoil')*1.0


# In[25]:

bages=BaggingRegressor(base_estimator=svm.SVR(C=10000.0),n_estimators=10,max_samples=0.95,max_features=0.9,bootstrap_features=True,oob_score=True,random_state=1,verbose=0,n_jobs=-1)


# In[26]:

preds = np.zeros((xtest.shape[0], 5))
for i in range(5):
    print "====="+str(i+1)+"====="
    bages.fit(xtrain, labels[:,i])
    preds[:,i] = bages.predict(xtest).astype(float)
    print bages.oob_score_


# In[27]:

print preds

for i in range(5):
    print "====="+str(i)+"====="
    print min(xx for xx in preds[:,i])
    print max(xx for xx in preds[:,i])


# In[28]:

sample = pd.read_csv('sample_submission.csv')
sample['Ca'] = preds[:,0]
sample['P'] = preds[:,1]
sample['pH'] = preds[:,2]
sample['SOC'] = preds[:,3]
sample['Sand'] = preds[:,4]

sample.to_csv('Bagged_SVMs-5.csv', index = False)


# In[8]:



