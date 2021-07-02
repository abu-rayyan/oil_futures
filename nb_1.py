# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 12:11:36 2021

@author: KHC
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestCentroid

df=pd.read_csv('features.csv')
x_df=df.drop(columns=['action']).values
y_df=df['action'].values
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(x_df, y_df)

#Predict the response for test dataset
y_pred = clf.predict(x_df)
#

clf_knn = NearestCentroid()
clf_knn.fit(x_df,y_df)
y_pred_knn = clf_knn.predict(x_df)
#print(clf.predict([[-0.8, -1]]))
#print(clf.predict_proba([[-0.8, -1]])) # predict the probability

from sklearn import metrics

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_df, y_pred))
print(" KNN Accuracy:",metrics.accuracy_score(y_df, y_pred_knn))