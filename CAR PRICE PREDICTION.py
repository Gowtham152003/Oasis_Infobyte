# -*- coding: utf-8 -*-
"""Task 3 Car_Price_Prediction.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1pT3ohsqJMfdZMs0WH_QZ1h_55Y6ABrAM
"""

import pandas as pd
import datetime
import seaborn as sns
from sklearn.model_selection import train_test_split

df = pd.read_csv("/content/drive/MyDrive/car data.csv")

from google.colab import drive
drive.mount('/content/drive')

df.head()

df.tail()

print("Print number of Rows:", df.shape[0])
print("Print number of Columns:", df.shape[1])

df.info()

df.isnull().sum()

"""## Describing Data"""

df.describe()

"""## Data Preprocessing"""

df.head(1)

sns.boxplot(df['price'])

sorted(df["price"], reverse = True)

df = df[~(df["price"] >= 40000) & (df['price'] <= 45000)]
df.shape

df.head(1)

print(df['fueltype'].unique())
print(df['aspiration'].unique())
print(df['doornumber'].unique())
print(df['carbody'].unique())
print(df['drivewheel'].unique())
print(df['enginelocation'].unique())
print(df['enginetype'].unique())
print(df['cylindernumber'].unique())
print(df['fuelsystem'].unique())

fuelMap = {'gas': 0, 'diesel': 1}
aspMap = {'std': 0, 'turbo': 1}
doorMap = {'two': 2, 'four': 4}
bodyMap = {'convertible': 0, 'hatchback': 1, 'sedan': 2, 'wagon': 3, 'hardtop': 4}
wheelMap = {'rwd': 0, 'fwd': 1, '4wd': 2}
engineLocMap = {'front': 0, 'rear': 1}
engineTypeMap = {'dohc': 0, 'ohcv': 1, 'ohc': 2, 'l': 3, 'rotor': 4, 'ohcf': 5, 'dohcv': 6}
cylinderMap = {'four': 4, 'six': 6, 'five': 5, 'three': 3, 'twelve': 12, 'two': 2, 'eight': 8}
fuelSysMap = {'mpfi': 0, '2bbl': 1, 'mfi': 2, '1bbl': 3, 'spfi': 4, '4bbl': 5, 'idi': 6, 'spdi': 7}

df['fueltype'] = df['fueltype'].map(fuelMap)
df['fueltype'].unique()

df['aspiration'] = df['aspiration'].map(aspMap)
df['aspiration'].unique()

df['doornumber'] = df['doornumber'].map(doorMap)
df['doornumber'].unique()

df['carbody'] = df['carbody'].map(bodyMap)
df['carbody'].unique()

df['drivewheel'] = df['drivewheel'].map(wheelMap)
df['drivewheel'].unique()

df['enginelocation'] = df['enginelocation'].map(engineLocMap)
df['enginelocation'].unique()

df['enginetype'] = df['enginetype'].map(engineTypeMap)
df['enginetype'].unique()

df['cylindernumber'] = df['cylindernumber'].map(cylinderMap)
df['cylindernumber'].unique()

df['fuelsystem'] = df['fuelsystem'].map(fuelSysMap)
df['fuelsystem'].unique()

df.head()

X.head()

X.info()

Y

"""## Splitting Dataset into Training Set and Test Set"""

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.30, random_state = 42)

"""## Import Models"""

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

"""## Model Training"""

lr = LinearRegression()
lr.fit(X_train, Y_train)

rf = RandomForestRegressor()
rf.fit(X_train, Y_train)

gradBoost = GradientBoostingRegressor()
gradBoost.fit(X_train, Y_train)

xgb = XGBRegressor()
xgb.fit(X_train, Y_train)

"""## Prediction on Test Set"""

Y_pred1 = lr.predict(X_test)
Y_pred2 = rf.predict(X_test)
Y_pred3 = gradBoost.predict(X_test)
Y_pred4 = xgb.predict(X_test)

"""## Evaluating Algorithm"""

from sklearn import metrics

scoreLR = metrics.r2_score(Y_test, Y_pred1)
scoreRF = metrics.r2_score(Y_test, Y_pred2)
scoreGB = metrics.r2_score(Y_test, Y_pred3)
scoreXG = metrics.r2_score(Y_test, Y_pred4)

print(scoreLR, scoreRF, scoreGB, scoreXG)

r2Scores = pd.DataFrame({'Models': ['Linear Regressor', 'Random Forest Regressor', 'Gradient Boost Regressor', 'XGB Rehressor'],
              'R2 Score': [scoreLR, scoreRF, scoreGB, scoreXG]})

r2Scores

sns.barplot(x='R2 Score', y = 'Models', data=r2Scores)

"""## Saving the Model"""

gradBoost = GradientBoostingRegressor()
gbFinal = gradBoost.fit(X, Y)

import joblib

joblib.dump(gbFinal, 'carPricePredictor')

model = joblib.load('carPricePredictor')

"""## Predicting New Data"""

df.head()

symboling = int(input("Symboling: "))
fuelType = fuelMap[input("Fuel Type: ")]
aspiration = aspMap[input("Aspiration: ")]
doorNumber = doorMap[input("Door Number: ")]
carBody = bodyMap[input("Car Body: ")]
driveWheel = wheelMap[input("Drive Wheel: ")]
engineLocation = engineLocMap[input("Engine Location: ")]
wheelBase = float(input("Wheel Base: "))
carLength = float(input("Car Length: "))
carWidth = float(input("Car Width: "))
carHeight = float(input("Car Height: "))
curbWeight = int(input("Curb Weight: "))
engineType = engineTypeMap[input("Engine Type: ")]
cylinderNumber = int(input("Cylinder Number: "))
engineSize = int(input("Engine Size: "))
fuelSystem = fuelSysMap[input("Fuel System: ")]
boreRatio = float(input("Bore Ratio: "))
stroke = float(input("Stroke: "))
compressRatio = float(input("Compress Ratio: "))
horsePower = int(input("Horse Power: "))
peakRPM = int(input("Peak RPM: "))
cityMPG = int(input("City MPG: "))
highwayMPG = int(input("Highway MPG: "))

newData = pd.DataFrame({
    'symboling': [symboling],
    'fueltype': [fuelType],
    'aspiration': [aspiration],
    'doornumber': [doorNumber],
    'carbody': [carBody],
    'drivewheel': [driveWheel],
    'enginelocation': [engineLocation],
    'wheelbase': [wheelBase],
    'carlength': [carLength],
    'carwidth': [carWidth],
    'carheight': [carHeight],
    'curbweight': [curbWeight],
    'enginetype':  [engineType],
    'cylindernumber':  [cylinderNumber],
    'enginesize':  [engineSize],
    'fuelsystem':  [fuelSystem],
    'boreratio': [boreRatio],
    'stroke': [stroke],
    'compressionratio': [compressRatio],
    'horsepower':  [horsePower],
    'peakrpm':  [peakRPM],
    'citympg':  [cityMPG],
    'highwaympg': [highwayMPG]
})

print(model.predict(newData))