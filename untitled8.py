# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 18:13:29 2021

@author: KHC
"""
import pandas as pd
import numpy as np
df = pd.DataFrame({"name": ['Alfred', 'Batman', 'Catwoman'],

                   "toy": [np.nan, 'Batmobile', 'Bullwhip'],

                   "born": [pd.NaT, pd.Timestamp("1940-04-25"),

                            pd.NaT]})

#df.dropna(inplace=True)
df=df.dropna(inplace=False)