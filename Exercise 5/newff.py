
import neurolab as nl
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
ps=pd.read_excel("D:\\MEPCO\\SEMESTER 5\\Machine Learning Essentials\\Raisin_Dataset.xlsx")
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
ps['Class']= label_encoder.fit_transform(ps['Class'])
print(ps)
ps=np.array(ps)
print(ps)
features = ps[:, :7]
print(ps[:, -1])
labels = ps[:, -1].reshape((ps.shape[0],1))
print("\n",labels)
print(ps[:, :1].min(),ps[:, :1].max())
dim1_min, dim1_max, dim2_min, dim2_max, dim3_min, dim3_max = ps[:, :1].min(), ps[:, :1].max(), ps[:, :2].min(), ps[:, :2].max(), ps[:, :3].min(), ps[:, :3].max()
dim4_min, dim4_max, dim5_min, dim5_max, dim6_min, dim6_max = ps[:, :4].min(), ps[:, :4].max(), ps[:, :5].min(), ps[:, :5].max(), ps[:, :6].min(), ps[:, :6].max()
dim7_min, dim7_max = ps[:, :7].min(), ps[:, :7].max()

dim1 = [dim1_min, dim1_max]
dim2 = [dim2_min, dim2_max]
dim3 = [dim3_min, dim3_max]
dim4 = [dim4_min, dim4_max]
dim5 = [dim5_min, dim5_max]
dim6 = [dim6_min, dim6_max]
dim7 = [dim7_min, dim7_max]
nn = nl.net.newff([dim1, dim2, dim3, dim4, dim5, dim6, dim7], [3,1])
error_progress = nn.train(features, labels,epochs=100, show=10, goal=0.02)
plt.plot(error_progress)
plt.xlabel('Number of Epochs')
plt.ylabel('Training Error')
plt.title('Training Error Progress')
plt.grid()
plt.show()


