# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
reviews = pd.read_csv("data/winemag-data-130k-v2.csv.zip", index_col=0)

#__________________________________________________________________________________________________________________
# This heatmap shows how much points on average a voter from a certain nationality gives to a variety of wine based 
# on the price class of the variety 
# _________________________________________________________________________________________________________________



#  Since the prices are heavily skewed (right skewness) we take the log to make more symmetric
reviews['log_price']=np.log(reviews['price'])

#   Make the plot appear in the browser
import plotly.io as pio
pio.renderers.default='browser'

#   Saving the four quartiles of the log_price 
log_prices_qrtls_dict={}
for i in range(1,5):
    log_prices_qrtls_dict['Q'+str(i)]=reviews['log_price'].quantile(i/4)
    
# Categorize price into Bronze(Min -> Q1), Silver(Q1 -> Q2), Golden(Q2 -> Q3), Platinum(Q3 -> Max)
def categorize_price(price):
    if price <= log_prices_qrtls_dict['Q1']:
        return 'Bronze'
    elif price <= log_prices_qrtls_dict['Q2']:
        return 'Silver'
    elif price <= log_prices_qrtls_dict['Q3']:
        return 'Golden'
    return 'Platinum'

# Add log_price_category column to the dataset by applying the categorize_price func
reviews['log_price_category']=reviews['log_price'].apply(categorize_price)

# Group by both the country and the category and getting the total price per group
df=reviews[['country','log_price_category','log_price','points']].sort_values(by='log_price')
df=df.groupby(['country','log_price_category']).mean()
df.reset_index(inplace=True)
df.sort_values(by=['country','log_price'],inplace=True)
df.columns=['country','log_price_category','mean_log_price','mean_points']


import plotly.graph_objects as go
fig = go.Figure(data=go.Heatmap(
        z=df['mean_points']
        ,x=df['log_price_category']
        ,y=df['country']
        ,colorscale='Jet'
        ))

fig.update_layout(
    title='VoterNationality_vs_priceCategory, mean points per Nationality and log_price_category <Heatmap>')

fig.show()

