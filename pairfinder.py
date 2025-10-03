#!/usr/bin/env python
# coding: utf-8

# In[119]:


from pairmaker import select_pairs, calculate_metrics, plot_pairs
import numpy as np
import pandas as pd
import yfinance as yf


# In[132]:


ticker = ["KO", "PEP", "MSFT", "GOOG", "AMZN", "META"]
ticker = [
    "IEX", "CNMD", "GPRO", "MOH", "MIDD", "AIN",
    "ALGT", "BFS", "SAGE", "RS", "THG"]
ticker = [
    "AAPL", "ABBV", "ABT", "ACN", "ADBE", "AIG", "AMD", "AMGN", "AMT", "AMZN",
    "AVGO", "AXP", "BA", "BAC", "BK", "BKNG", "BLK", "BMY", "BRK-B", "C",
    "CAT", "CL", "CMCSA", "COF", "COP", "COST", "CRM", "CSCO", "CVS", "CVX",
    "DE", "DHR", "DIS", "DUK", "EMR", "FDX", "GD", "GE", "GILD", "GM",
    "GOOG", "GOOGL", "GS", "HD", "HON", "IBM", "INTC", "INTU", "ISRG", "JNJ"
]


# In[133]:


df = yf.download(ticker, start="2019-01-01", end="2025-01-01")['Close']
cumret = np.log(df).diff().cumsum()+1
cumret = cumret.dropna()


# In[134]:


cumret_train = cumret.loc[:"2022-12-31"]
cumret_test = cumret.loc["2023-01-01":]

train_form = cumret_train.loc[:'2021-01-01']
train_trade = cumret_train.loc['2021-01-02':]
test_form = cumret_test.loc[:'2023-12-31']
test_trade = cumret_test.loc['2024-01-01':]

pairs_train = select_pairs(train_form)
pairs_test = select_pairs(test_form)


# In[154]:


metrics_train_form = calculate_metrics(pairs_train.index, train_form, pairs_train)
metrics_train_trade = calculate_metrics(pairs_train.index, train_trade, pairs_train) # in trading period
metrics_test_form = calculate_metrics(pairs_test.index, test_form, pairs_test)
metrics_test_trade = calculate_metrics(pairs_test.index, test_trade, pairs_test)
metrics_test_form['Hedge ratio'] = pairs_test['Hedge ratio']


# In[136]:


data_train = metrics_train_form.copy()
data_train['Num zero-crossings trade'] = metrics_train_trade['Num zero-crossings']
data_test = metrics_test_form.copy()
data_test['Num zero-crossings trade'] = metrics_test_trade['Num zero-crossings']


# In[137]:


# convert data to numeric types
#data_train = data_train.apply(pd.to_numeric, errors='raise')
#data_test = data_test.apply(pd.to_numeric, errors='raise')


# In[138]:


X_train = data_train.values[:,:9]
X_test = data_test.values[:,:9]
y_train = data_train.values[:,9]
y_test = data_test.values[:,9]


# In[114]:


from sklearn.ensemble import RandomForestRegressor


# In[115]:


model = RandomForestRegressor()


# In[141]:


model.fit(X_train, y_train)
ypred = model.predict(X_test)
# Indices of top 3
top3_idx = np.argsort(ypred)[-3:][::-1]  # descending order
top3_values = ypred[top3_idx]

# Indices of bottom 3
bottom3_idx = np.argsort(ypred)[:3]  # ascending order
bottom3_values = ypred[bottom3_idx]


# In[145]:


ypred


# In[150]:


X_train.shape


# In[ ]:




