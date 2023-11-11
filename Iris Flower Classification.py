from google.colab import drive
drive.mount('/content/drive')
import pandas as pd
from sklearn.linear_model import LogisticRegression as lr
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("/content/drive/MyDrive/Iris.csv")

df.head()#first 5 dataset
df.tail()
df.info()
df['Species'].value_counts()#info
df.isnull().sum()#Checking for null values
duplicate_count = df.duplicated().sum()
print(duplicate_count)
sns.FacetGrid(df,hue='Species',height=4).map(plt.scatter,"SepalWidthCm" , "PetalLengthCm").add_legend()
corr = df.corr()
sns.heatmap(corr,annot=True,cmap='Blues') #Correlation matrix
flower_mapping= {'Iris-setosa': 0, 'Iris-versicolor' : 1, 'Iris-virginica' :2}
df['Species'] = df ['Species'].map (flower_mapping)
#pre-training
df.head()
x=df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm','PetalWidthCm']].values
y=df[['Species']].values

mod=lr()#Logistic regression
mod.fit(x,y)
expected = y
predicted = mod.predict(x)
predicted
print(metrics.classification_report(expected, predicted))#accuracy
print(metrics.confusion_matrix(expected, predicted))
print("TASK 1 COMPLETED")
