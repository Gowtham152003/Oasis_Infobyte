

import numpy as np
import pandas as pd
df=pd.read_csv("/content/drive/MyDrive/Unemployment in India.csv")
df.head()

df.head()

df.tail()

print(df.isnull().sum())

# Remove rows with missing values
df = df.dropna()

# Or fill missing values with mean
df = df.fillna(df.mean())

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df = df.dropna()

print(df.dtypes)

df.info()

df.shape

x = df['Region']
y=df[' Estimated Unemployment Rate (%)']
df2=df.iloc[:,3]
df2

import plotly.express as px
import matplotlib.pyplot as plt

fg = px.bar(df,x='Region',y=' Estimated Unemployment Rate (%)',color='Region',
            title='Unemploymeny Rate (State Wise) by Bar Graph',template='plotly')
fg.update_layout(xaxis={'categoryorder':'total descending'})
fg.show()

fg = px.box(df,x='Region',y=' Estimated Unemployment Rate (%)',color='Region',
            title='Unemploymeny Rate (Statewise) by Box Plot',template='plotly')
fg.update_layout(xaxis={'categoryorder':'total descending'})
fg.show()

fg = px.scatter(df,x='Region',y=' Estimated Unemployment Rate (%)',color='Region',
                title='Unemploymeny Rate (Statewise) by Scatter Plot',template='plotly')
fg.update_layout(xaxis={'categoryorder':'total descending'})
fg.show()

fg = px.histogram(df,x='Region',y=' Estimated Unemployment Rate (%)',color='Region',
                  title='Unemploymeny Rate (Statewise) by Histogram',template='plotly')
fg.update_layout(xaxis={'categoryorder':'total descending'})
fg.show()

print("Task 2 completed")

