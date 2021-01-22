import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
reviews = pd.read_csv("data/winemag-data-130k-v2.csv.zip", index_col=0)

#  Getting rid of the nulls
prices=reviews[reviews['price'].notnull()]['price']
points=reviews[reviews['points'].notnull()]['points']

#   Here's how the prices and the points looks like 
plt.hist(prices, bins = 100)
plt.show()
# plt.hist(points, bins = 15)
# plt.show()

#  The prices are right skewed so we get the log of the prices values to make it more symmetric
log_prices=np.log(prices)

plt.hist(log_prices, bins = 50)
plt.show()


#   We take a sample from both sets, we set the seed to a fixed number to get consistent results
sample_size=500
log_prices=log_prices.sample(sample_size,random_state=0)
points=points.sample(sample_size,random_state=0)

#   Here's how the date looked like after the transformation 
# plt.hist(log_prices, bins = 20)
# plt.show()
# plt.hist(points, bins = 15)
# plt.show()

#   Make the plot appear in the browser
import plotly.io as pio
pio.renderers.default='browser'

#   Import the multivariate_normal fucntion
from scipy.stats import multivariate_normal
import plotly.graph_objects as go

#   Distribution Parameters to set
mu_x = log_prices.mean()
variance_x = log_prices.var()
mu_y = points.mean()
variance_y = points.var()


#   Getting the covariance matrix of the two vars
df = pd.DataFrame(list(zip(log_prices, points)), 
               columns =['log_prices', 'points']) 
cov=df.cov()


#   Create grid and multivariate normal
X, Y = np.meshgrid(log_prices,points)
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X; pos[:, :, 1] = Y
rv = multivariate_normal([mu_x, mu_y],cov)


fig = go.Figure( data=[go.Surface(z=rv.pdf(pos), x=X, y=Y)] )
fig.update_layout(title='Bivariate-Gaussian-Distribution-Between-LogPrice-Points', autosize=False,
                   width=750, height=750,
                  margin=dict(l=65, r=50, b=65, t=90))
fig.show()


