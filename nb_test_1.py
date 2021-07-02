# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 12:11:36 2021

@author: KHC
"""

import numpy as np
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X, Y)

print(clf.predict([[-0.8, -1]]))
print(clf.predict_proba([[-0.8, -1]])) # predict the probability