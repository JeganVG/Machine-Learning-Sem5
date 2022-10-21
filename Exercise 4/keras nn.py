#Artificial Neural Network Using Keras

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel("D:\\MEPCO\SEMESTER 5\\Machine Learning Essentials\\Raisin_Dataset.xlsx")
x = df.iloc[:,:-1]
y = df.iloc[:,-1]

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print(y)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.optimizers import SGD

nn = Sequential()
nn.add(Dense(12,activation = 'relu'))
#nn.add(Dropout(0.5))
nn.add(Dense(8,activation='relu'))
#nn.add(Dropout(0.5))
nn.add(Dense(1,activation = 'relu'))
#nn.add(Dropout(0.5))
#nn.add(Dense(9,activation='relu'))

# sgd = SGD(lr=0.01, decay=0.01, momentum=0.9, nesterov=True)

# nn.compile(loss='mean_squared_error',
#               optimizer=sgd)
nn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
history = nn.fit(x_train, y_train,
          epochs=100,
          batch_size=10)
res=nn.predict(x_test)
plt.plot(history.history['loss'])
plt.show()

y_pred = nn.predict(x_test)
print(y_pred)
y_pred = (y_pred > 0.5)
print(y_pred)

from sklearn.metrics import accuracy_score

acc = accuracy_score(y_test,y_pred)
print(acc)