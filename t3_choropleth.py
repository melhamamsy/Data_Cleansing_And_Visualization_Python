# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
reviews = pd.read_csv("data/winemag-data-130k-v2.csv.zip", index_col=0)

#___________________________________________________________________________________________________________
# This graph shows the intenisty by which each nationality in the dataset contributed in assessing the wines
#___________________________________________________________________________________________________________


#   Getting the mean points per each critique nationality 
df=reviews[['country','points']].groupby(by='country').sum().reset_index()

#   The total points per country is highly right skewed
import matplotlib.pyplot as plt
plt.hist(df['points'],10)
plt.show()

#   Transforming the total column to it's log to make it more symmetric
df['points']=np.log(df['points'])
                    
#   Renaming the columns
df.columns=['country','log_total_points']

#   Checking the effectiveness of the log
plt.hist(df['log_total_points'],10)
plt.show()

#   Make the plot appear in the browser
import plotly.io as pio
pio.renderers.default='browser'

import plotly.express as px 
fig = px.choropleth(df,  
                    locations="country"  
                    ,locationmode='country names'
                    ,color="log_total_points"  
                    ,hover_name="country" 
                    ) 
fig.update_layout(
    title_text = 'Log_Total_Points_per_Country <Choropleth>'
)
fig.show() 



