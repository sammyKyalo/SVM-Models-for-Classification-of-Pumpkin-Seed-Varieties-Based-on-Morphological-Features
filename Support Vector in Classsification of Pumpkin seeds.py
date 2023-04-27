import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

data = pd.read_csv("G:\Pumpkin_Seeds_Dataset.csv")

data.head()
data.info()

data.shape

data.size

data.count()

data["Class"].value_counts()

plt.scatter (data["Class"],data["Area"], color = 'red')
plt.show()

data.columns

x = data[['Area', 'Perimeter', 'Major_Axis_Length', 'Minor_Axis_Length','Convex_Area', 'Equiv_Diameter', 'Eccentricity', 'Solidity', 'Extent','Roundness', 'Aspect_Ration', 'Compactness']]

y = data['Class']

x[0:5]

# divide the data to train/test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state = 4)


#2000 rows used for training, 500 rows used for testing
x_train.shape
x_test.shape

y_train.shape
y_test.shape

#Modelling using SVC
from sklearn import svm
clasifier=svm.SVC(kernel='linear',gamma ='auto', C=2)

clasifier.fit(x_train, y_train)
y_predict = clasifier.predict(x_test)


#result Evaluation
from sklearn.metrics import classification_report
print (classification_report(y_test, y_predict))

















