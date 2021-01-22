# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
reviews= pd.read_csv('data/winemag-data-130k-v2.csv.zip',index_col=0)

#____________________________________________________________________________________________________________________________
# This graph shows the relationship between the log of the average price of a certain winery and the expected points a voter  
# On average will give to this winery. 
#____________________________________________________________________________________________________________________________


#   Get the mean price of wine in a winery and mean points given to this winery for each winery of wine.
df=reviews[['winery','price','points']].groupby(by='winery').agg({
    'price':np.mean,
    'points':np.mean
    }).reset_index()

#   Rename the columns
df.columns=['winery','mean_price','mean_points']

#   Make the plot appear in the browser
import plotly.io as pio
pio.renderers.default='browser'


import plotly.graph_objects as go
from plotly.offline import iplot

#  Since the prices are heavily skewed (right skewness) we take the log to make more symmetric
X=np.log(df['mean_price'])
Y=df['mean_points']

#   Scatter plot between the two numeric variable mean_points and log_mean_price
#   There's a positive correlation between the log of the mean prices of a winery and the mean points given by critiques to this winery.
fig_scatter = go.Figure(
    data=go.Scatter(x=X, y=Y, mode='markers')
    ,layout= go.Layout(
        title=go.layout.Title(text="Mean_Points_(y-axis)_vs_Log_Mean_Price_(x-axis) for every winery <Scatter_Plot>")
        )
    )

#   Updating the xaxis and yaxis titles
fig_scatter.update_xaxes(title_text='LogMeanPrice')
fig_scatter.update_yaxes(title_text='MeanPoints')

#   Finally the plotting phase.
iplot(fig_scatter)


