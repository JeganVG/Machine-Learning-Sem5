
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel("D:\\MEPCO\\SEMESTER 5\\Machine Learning Essentials\\Raisin_Dataset.xlsx")

x=df.iloc[:,:-1]

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['Class']=le.fit_transform(df['Class'])
y=df.iloc[:,-1]
df.head()

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=4)

model=Sequential()
model.add(LSTM((1),batch_input_shape=(None,7,1),return_sequences=True))
model.add(LSTM((1),return_sequences=False))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

history=model.fit(x_train,y_train,epochs=10)

res=model.predict(x_test)
plt.plot(history.history['loss'])
plt.show()

print("Accuracy")
sum1 = sum(history.history['accuracy'])
len1 = len(history.history['accuracy'])
print(sum1/len1)





