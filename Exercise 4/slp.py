import neurolab as nl
import neurolab as nl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
df = pd.read_excel("D:\\MEPCO\\SEMESTER 5\\Machine Learning Essentials\\Raisin_Dataset.xlsx")
label_encoder = preprocessing.LabelEncoder()
df['Class']= label_encoder.fit_transform(df['Class'])
df['Class'] = df['Class'].astype('category')
x=df.iloc[:,:-1]
y=df.iloc[:,-1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)

feature_names=["Area","MajorAxisLength","MinorAxisLength","ConvexArea","Eccentricity","Extent","Perimeter"]
class_names = ["Kecimen","Besni"]
x_1=[]
for i in range(900):
    x_2 = []
    for f in feature_names:
        x_2.append(x[f].loc[x.index[i]])
    x_1.append(x_2)
    y_1=[]
    for i in range(900):
        y_3=[]
        y_3.append(y[i])
        y_1.append(y_3) 
print(y_1)


dim1_min,dim1_max=x['Area'].min(),x['Area'].max()
dim2_min,dim2_max=x['Perimeter'].min(),x['Perimeter'].max()
dim3_min,dim3_max=x['MajorAxisLength'].min(),x['MajorAxisLength'].max()
dim4_min,dim4_max=x['MinorAxisLength'].min(),x['MinorAxisLength'].max()
dim5_min,dim5_max=x['Eccentricity'].min(),x['Eccentricity'].max()
dim6_min,dim6_max=x['ConvexArea'].min(),x['ConvexArea'].max()
dim7_min,dim7_max=x['Extent'].min(),x['Extent'].max()
dim1=[dim1_min,dim1_max]
dim2=[dim2_min,dim2_max]
dim3=[dim3_min,dim3_max]
dim4=[dim4_min,dim4_max]
dim5=[dim5_min,dim5_max]
dim6=[dim6_min,dim6_max]
dim7=[dim7_min,dim7_max]
neural_net = nl.net.newp([dim1,dim2,dim3,dim4,dim5,dim6,dim7],cn=1)
error=neural_net.train(x_1,y_1,epochs=100,lr=0.2)

def accuracy(y_true,y_pred):
    correctly_predicted = 0  
    # iterating over every label and checking it with the true sample  
    for true_label, predicted in zip(y_true,y_pred):  
        if true_label == predicted:  
            correctly_predicted += 1  
    # computing the accuracy score  
    accuracy_score = correctly_predicted / len(y_true) 
    return accuracy_score
y_pred =[]
for i in x_1:
    y_pred.append(neural_net.sim([i])[0])
    accuracy(y,y_pred)



#and

a=[0,0,1,1]
b=[0,1,0,1]
out_a=[0,0,0,1]
output_a=[[0],[0],[0],[1]]
x_and=[[0,0],[0,1],[1,0],[1,1]]
dim1_min,dim1_max=min(a),max(a)
dim2_min,dim2_max=min(b),max(b)
dim1=[dim1_min,dim1_max]
dim2=[dim2_min,dim2_max]
neural_net1 = nl.net.newp([dim1,dim2],cn=1)
error2=neural_net1.train(x_and,output_a,epochs=100,lr=0.2)
y_pred2=[]
for i in x_and:
   y_pred2.append(neural_net1.sim([i])[0])
# print(y_pred2)
print("ACCURACY FOR AND PROBLEM :",accuracy(out_a,y_pred2))

plt.figure()
plt.title('AND OUTPUT')
print(error2)
plt.plot(error2)
plt.show()

#or
a2=[0,0,1,1]
b2=[0,1,0,1]
out_o=[0,1,1,1]
output_o=[[0],[1],[1],[1]]
x_or=[[0,0],[0,1],[1,0],[1,1]]
dim1_min,dim1_max=min(a),max(a)
dim2_min,dim2_max=min(b),max(b)
dim1=[dim1_min,dim1_max]
dim2=[dim2_min,dim2_max]
neural_net1 = nl.net.newp([dim1,dim2],cn=1)
error2=neural_net1.train(x_or,output_o,epochs=100,lr=0.2)
y_pred2=[]
for i in x_or:
   y_pred2.append(neural_net1.sim([i])[0])
# print(y_pred2)
print("ACCURACY FOR OR PROBLEM :",accuracy(out_o,y_pred2))


plt.figure()
plt.title('OR OUTPUT')
print(error2)
plt.plot(error2)
plt.show()

#xor
a=[0,0,1,1]
b=[0,1,0,1]
out_xor=[0,1,1,0]
output_xor=[[0],[1],[1],[0]]
x_xor=[[0,0],[0,1],[1,0],[1,1]]
dim1_min,dim1_max=min(a),max(a)
dim2_min,dim2_max=min(b),max(b)
dim1=[dim1_min,dim1_max]
dim2=[dim2_min,dim2_max]
neural_net1 = nl.net.newp([dim1,dim2],cn=1)
error2=neural_net1.train(x_xor,output_xor,epochs=100,lr=0.2)
y_pred3=[]
for i in x_xor:
   y_pred3.append(neural_net1.sim([i])[0])
# print(y_pred2)
print("ACCURACY FOR XOR PROBLEM :",accuracy(out_xor,y_pred2))
