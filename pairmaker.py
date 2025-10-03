#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[9]:


def parse_pair(pair):
    '''
    parse pair string S1-S2
    return tickers S1, S2
    '''
    dp = pair.find('-')
    s1 = pair[:dp]
    s2 = pair[dp+1:]

    return s1,s2

def cadf_pvalue(s1, s2, cumret):
    '''
    perform CADF cointegration tests
    since it is sensitive to the order of stocks in the pair, perform both tests (s1-2 and s2-s1)
    return the smallest p-value of two tests
    '''
    from statsmodels.tsa.stattools import coint

    p1 = coint(cumret[s1], cumret[s2])[1]
    p2 = coint(cumret[s2], cumret[s1])[1]

    return min(p1,p2)

def calculate_halflife(spread):
    '''
    calculate half-life of mean reversion of the spread
    '''
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools.tools import add_constant

    ylag = spread.shift()
    deltay = spread - ylag
    ylag.dropna(inplace=True)
    deltay.dropna(inplace=True)

    res = OLS(deltay, add_constant(ylag)).fit()
    halflife = -np.log(2)/res.params[0]

    return halflife

def calculate_metrics(pairs, cumret, pairs_df):
    """
    Calculate metrics for pairs using data in cumret.
    Returns a dataframe of results, keeping all pairs and handling NaNs/errors gracefully.
    """
    import numpy as np
    import pandas as pd
    from hurst import compute_Hc
    from statsmodels.tsa.stattools import adfuller, coint
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools.tools import add_constant

    # Metrics to calculate
    cols = ['Distance', 'CADF p-value', 'ADF p-value', 'Spread SD', 'Pearson r',
            'Num zero-crossings', 'Hurst Exponent', 'Half-life of mean reversion',
            '% days within historical 2-SD band', 'Hedge ratio']

    results = pd.DataFrame(index=pairs, columns=cols)

    for pair in pairs:
        try:
            s1, s2 = parse_pair(pair)

            # Make sure both series exist in cumret
            if s1 not in cumret.columns or s2 not in cumret.columns:
                results.loc[pair] = np.nan
                continue

            hedge_ratio = pairs_df.loc[pair]['Hedge ratio']
            spread = cumret[s1] - hedge_ratio * cumret[s2]

            # Skip if spread has NaNs
            if spread.isna().any():
                results.loc[pair] = np.nan
                continue

            hist_mu = pairs_df.loc[pair]['Spread mean']
            hist_sd = pairs_df.loc[pair]['Spread SD']

            # Fill metrics
            try:
                cadf_p = coint(cumret[s1], cumret[s2])[1]
            except:
                cadf_p = np.nan
            results.loc[pair, 'CADF p-value'] = cadf_p

            try:
                adf_p = adfuller(spread)[1]
            except:
                adf_p = np.nan
            results.loc[pair, 'ADF p-value'] = adf_p

            results.loc[pair, 'Spread SD'] = hist_sd
            results.loc[pair, 'Pearson r'] = np.corrcoef(cumret[s1], cumret[s2])[0, 1]

            spread_nm = spread - hist_mu
            results.loc[pair, 'Distance'] = np.sqrt(np.sum(spread_nm**2))
            results.loc[pair, 'Num zero-crossings'] = ((spread_nm[1:].values * spread_nm[:-1].values) < 0).sum()

            try:
                results.loc[pair, 'Hurst Exponent'] = compute_Hc(spread)[0]
            except:
                results.loc[pair, 'Hurst Exponent'] = np.nan

            try:
                results.loc[pair, 'Half-life of mean reversion'] = calculate_halflife(spread)
            except:
                results.loc[pair, 'Half-life of mean reversion'] = np.nan

            results.loc[pair, '% days within historical 2-SD band'] = (abs(spread - hist_mu) < 2 * hist_sd).sum() / len(spread) * 100
            results.loc[pair, 'Hedge ratio'] = hedge_ratio

        except Exception as e:
            # If anything unexpected happens, fill row with NaNs
            results.loc[pair] = np.nan
            print(f"Warning: Failed to calculate metrics for {pair}: {e}")

    return results


def plot_pairs(pairs, cumret_train, cumret_test):
    '''
    plot cumulative returns of the spread for each pair in pairs
    '''

    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools.tools import add_constant

    for pair in pairs:
        s1,s2 = parse_pair(pair)
        res = OLS(cumret_train[s1], add_constant(cumret_train[s2])).fit()
        spread_train = cumret_train[s1] - res.params[s2]*cumret_train[s2]
        spread_test = cumret_test[s1] - res.params[s2]*cumret_test[s2]
        spread_mean = spread_train.mean() # historical mean
        spread_std = spread_train.std() # historical standard deviation

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18,4))
        fig.suptitle(f'Spread of {pair} pair', fontsize=16)
        ax1.plot(spread_train, label='spread')
        ax1.set_title('Formation period')
        ax1.axhline(y=spread_mean, color='g', linestyle='dotted', label='mean')
        ax1.axhline(y=spread_mean+2*spread_std, color='r', linestyle='dotted', label='2-SD band')
        ax1.axhline(y=spread_mean-2*spread_std, color='r', linestyle='dotted')
        ax1.legend()
        ax2.plot(spread_test, label='spread')
        ax2.set_title('Trading period')
        ax2.axhline(y=spread_mean, color='g', linestyle='dotted', label='mean')
        ax2.axhline(y=spread_mean+2*spread_std, color='r', linestyle='dotted', label='2-SD band')
        ax2.axhline(y=spread_mean-2*spread_std, color='r', linestyle='dotted')
        ax2.legend()

def select_pairs(train):
    '''
    select pairs using data from train dataframe
    return dataframe of selected pairs
    '''
    tested = []

    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools.tools import add_constant
    from hurst import compute_Hc
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.stattools import coint

    cols = ['Distance', 'Num zero-crossings', 'Pearson r', 'Spread mean', 
            'Spread SD', 'Hurst Exponent', 'Half-life of mean reversion', 'Hedge ratio']
    pairs = pd.DataFrame(columns=cols)

    for s1 in train.columns:
        for s2 in train.columns:
            if s1!=s2 and (f'{s1}-{s2}' not in tested):
                tested.append(f'{s1}-{s2}')
                cadf_p = coint(train[s1], train[s2])[1]
                if cadf_p<0.9 and (f'{s2}-{s1}' not in pairs.index): # stop if pair already added as s2-s1
                    res = OLS(train[s1], add_constant(train[s2])).fit()
                    hedge_ratio = res.params[s2]
                    if hedge_ratio > 0 : # hedge ratio should be posititve
                        spread = train[s1] - hedge_ratio*train[s2]
                        hurst = compute_Hc(spread)[0]
                        if hurst<0.5:
                            halflife = calculate_halflife(spread)
                            if halflife>1 and halflife<30:
                                # subtract the mean to calculate distances and num_crossings
                                spread_nm = spread - spread.mean() 
                                num_crossings = (spread_nm.values[1:] * spread_nm.values[:-1] < 0).sum()
                                if num_crossings>len(train.index)/252*12: 
                                    distance = np.sqrt(np.sum(spread_nm**2))
                                    pearson_r = np.corrcoef(train[s1], train[s2])[0][1]
                                    pairs.loc[f'{s1}-{s2}'] = [distance, num_crossings, pearson_r, spread.mean(),
                                                               spread.std(), hurst, halflife, hedge_ratio]

    return pairs


# In[ ]:




