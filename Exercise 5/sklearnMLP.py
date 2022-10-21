
import pandas as pd
import numpy as np
df=pd.read_excel("D:\\MEPCO\\SEMESTER 5\\Machine Learning Essentials\\Raisin_Dataset.xlsx")
x=df.drop(["Class"],axis=1)
y=df["Class"]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.20)
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
from sklearn.neural_network import MLPClassifier
mlp=MLPClassifier(hidden_layer_sizes=(100),activation="relu",max_iter=1000)
mlp.fit(x_train,y_train)
predictions=mlp.predict(x_test)
from sklearn.metrics import confusion_matrix, accuracy_score
print(confusion_matrix(y_test,predictions))
print(accuracy_score(predictions, y_test))


