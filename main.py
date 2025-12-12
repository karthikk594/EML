import numpy as np
import pandas as pd
import math
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
data = datasets.load_wine(as_frame=True)
X=data.data
Y=data.target
name=data.target_names
print(name)
df = pd.DataFrame(X,columns=data.feature_names)
df['wine class']=data.target
df['wine class']=df['wine class'].replace(to_replace=[0,1,2],value=['class0','class1','class2'])
print(df)
sns.pairplot(data=df,hue='wine class',palette='Set2')
plt.show()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=1)
y=math.sqrt(len(y_test))
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train,y_train)
predict1=knn.predict(x_test)
from sklearn import metrics
A1=metrics.accuracy_score(predict1)
from sklearn.preprocessing import StandardScaler
x_train=StandardScaler().fit_transform(x_train)
x_test=StandardScaler().fit_transform(x_test)
knn2 = KNeighborsClassifier(n_neighbors=int(y))
knn2.fit(x_train,y_train)
predict2 = knn2.predict(x_test)
A2 = metrics.accuracy_score(predict2)
print(A2)
