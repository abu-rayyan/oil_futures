#!/usr/bin/env python
# coding: utf-8

# In[1]:


#pip install seaborn


# In[2]:


import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sb
from sklearn.preprocessing import MinMaxScaler
import os
from sklearn.preprocessing import LabelEncoder
# from sklearn.pipeline import Pipeline


# In[3]:


# for dirname, _, filenames in os.walk(r'..\data'):
#     dir_name=dirname
dir_name="data"


# In[4]:


# dir_name=os.path.join('../','data_1')


# # Reading Data files
# 

# In[5]:


# df_1=pd.read_excel(os.path.join(dir_name,'cl_sell_buy.xlsx'),engine='openpyxl',)
df_1=pd.read_excel(os.path.join(dir_name,'cl_sell_buy.xlsx'))
df_2=pd.read_excel(os.path.join(dir_name,'diffs.xlsx'))  
df_3=pd.read_excel(os.path.join(dir_name,'margins.xlsx'))


# In[6]:


def delete_cols(col_names_list,date_list):
    for name in df_lag.columns:
            if 't-' in name or name in date_list or name=='action':
                pass
            else:
#                 print ('column name in else is:',name)
                del df_lag[name]


# In[7]:


def input_output(col_names_list):
    for name in df_lag.columns:
            if 'action' in name:
                del df_input[name]
            else:
#                 print ('column name in else is:',name)
                pass


# In[8]:


def normalize_df(df):

    # copy the data
    df_min_max_scaled = df.copy()

    # apply normalization techniques
    for column in df_min_max_scaled.columns:
        df_min_max_scaled[column] = (df_min_max_scaled[column] - df_min_max_scaled[column].min()) / (df_min_max_scaled[column].max() - df_min_max_scaled[column].min())    

    # view normalized data
    return(df_min_max_scaled)


# In[9]:


class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)


# In[10]:


start_date = '2015-01-10'


# In[11]:


mask = (df_1['date'] > start_date)
df_1 = df_1.loc[mask]
df_1.dropna(axis=0, inplace=True)


# In[12]:


df_1.head()


# In[13]:


df_2.head()


# In[14]:


mask = (df_2['ObservationDate'] > start_date)
df_2 = df_2.loc[mask]
df_2.dropna(axis=0, inplace=True)


# In[15]:


df_2.head()


# In[16]:


df_3.head()


# In[17]:


mask = (df_3['DateCalculated'] > start_date)
df_3 = df_3.loc[mask]
df_3.dropna(axis=0, inplace=True)
df_3.describe()


# In[18]:


df1_df2=pd.merge(
    df_1,
    df_2,
    left_on=['date'],
    right_on=['ObservationDate']
)


# In[19]:


combined_df=pd.merge(
    df1_df2,
    df_3,
    left_on=['date'],
    right_on=['DateCalculated']
)


# In[20]:


combined_df['year'] = pd.DatetimeIndex(combined_df['date']).year
combined_df['month'] = pd.DatetimeIndex(combined_df['date']).month
combined_df['day'] = pd.DatetimeIndex(combined_df['date']).day
combined_df['weekday'] = pd.DatetimeIndex(combined_df['date']).weekday   # Monday is 0 and Sunday is 6


# In[21]:


combined_df=combined_df.drop(columns=['date','ObservationDate','DateCalculated', 'Contract'])


# In[22]:


lags = range(1, 5)  # 5 lags

df_lag=combined_df.assign(**{'{} (t-{})'.format(col, t): combined_df[col].shift(t)
    for t in lags
    for col in combined_df
})


# In[23]:


df_lag.head()


# In[24]:


#input_df=combined_df.drop(columns=['action','ObservationDate','DateCalculated', 'Contract'])


# In[25]:


print (df_lag.columns)


# In[26]:


dates_list=['year','month','weekday','day']


# In[27]:


delete_cols(df_lag.columns,dates_list)


# In[28]:



#             df_lag.drop(labels=name, axis=1)
            #pass
            


# In[29]:


df_lag.head()


# In[30]:


print(df_lag.isnull().values.sum())


# In[31]:


print(df_lag.isnull().sum())


# In[32]:


df_lag.info()


# In[33]:


df_lag = df_lag.dropna()


# In[34]:


df_lag.info()


# In[35]:


df_input=df_lag.drop(columns='action',axis=1) # since these columns wer
df_output=df_lag['action']


# In[36]:


df_cat = df_input.select_dtypes(include=['object']).copy()


# In[37]:


df_cat.head()


# In[38]:


#categorical data
categorical_cols = df_cat.columns 


# In[39]:


df_input=MultiColumnLabelEncoder(columns = categorical_cols).fit_transform(df_input)    # 0 means buy, 1 means sell


# In[ ]:





# In[40]:


df_n_input=normalize_df(df_input)


# In[41]:


df_n_input.head()


# In[42]:


# df_num=df_input[~df_input.isin(df_cat)].dropna(axis=1)
# df=df_input.fillna(0)


# In[ ]:




