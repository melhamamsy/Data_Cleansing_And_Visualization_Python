# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
reviews = pd.read_csv("data/winemag-data-130k-v2.csv.zip", index_col=0)

reviews_grp=reviews[['variety','country','price','points']].groupby(by='variety').agg({
    'country': "nunique",
    'price':np.mean,
    'points':np.mean
    }).reset_index()

#   We take a sample from our dataset, we set the seed to a fixed number to get consistent results
sample_size=500
reviews_sample=reviews_grp.dropna(axis = 0, how ='any')\
    .sample(sample_size,random_state=0).reset_index().drop(columns="index")

reviews_sample.columns=['variety','nationalities_count','mean_price','mean_points']

#  Getting rid of the nulls
prices=reviews_sample["mean_price"]
points=reviews_sample["mean_points"]
nat_counts=reviews_sample["nationalities_count"]

#   Visualizing the distribution of each variable before any transformation
plt.hist(points, bins = 20)
plt.title("points_before")
plt.show()

plt.hist(prices, bins = 20)
plt.title("prices_before")
plt.show()

plt.hist(nat_counts, bins = 10)
plt.title("nat_counts_before")
plt.show()

#  The prices is right skewed so we apply log func
log_prices=np.log(prices)

#   Visualizing the distribution of each variable after some transformations

#  Much better
plt.hist(log_prices, bins = 20)
plt.title("prices_after     Seems much better")
plt.show()





#   Make the plot appear in the browser
import plotly.io as pio
pio.renderers.default='browser'

import plotly.graph_objects as go

#   Axis variables,
# Diag here is to assign zero to z axis value at every point on y except for those at which x has a value
x,y,z = log_prices,nat_counts,np.diag(points)


fig = go.Figure( data=[go.Surface(z=z, x=x, y=y)] )
fig.update_layout(title='Bivariate-Gaussian-Distribution-Between-LogPrice-Points'
                  ,autosize=False\
                  ,width=750, height=750
                  ,margin=dict(l=65, r=50, b=65, t=90)
                  ,scene = dict(
                    xaxis_title='LOG_PRICES (x)',
                    yaxis_title='NATIONALITIES_COUNT (y)',
                    zaxis_title='POINTS (z)'))
fig.show()

