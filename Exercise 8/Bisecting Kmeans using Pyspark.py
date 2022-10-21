from pyspark.sql import SparkSession
import pandas as pd

spark=SparkSession.builder.appName("Raisin_Dataset").getOrCreate()
df=spark.read.csv("D:\\MEPCO\\SEMESTER 5\\Machine Learning Essentials\\Raisin_Dataset.csv",header=True,inferSchema=True)
df.show(10)
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
df.columns
feat_cols=['Area','MajorAxisLength','MinorAxisLength','Eccentricity','ConvexArea','Extent','Perimeter']

assembler=VectorAssembler(inputCols=feat_cols,outputCol="Features")
final_df=assembler.transform(df)  

from pyspark.ml.clustering import BisectingKMeans
from pyspark.ml.evaluation import ClusteringEvaluator
eval=ClusteringEvaluator(predictionCol='prediction',featuresCol='Features',metricName='silhouette',distanceMeasure="squaredEuclidean")
scores=[]
for k in range(2,11):
  kmeans=BisectingKMeans(featuresCol='Features',k=k)
  kfit=kmeans.fit(final_df)
  output=kfit.transform(final_df)
  score=eval.evaluate(output)
  scores.append(score)
  print("k=",k," ",score)

import matplotlib.pyplot as plt
fig,ax=plt.subplots(1,1,figsize=(10,10))
ax.plot(range(2,11),scores)
ax.set_xlabel("K")
ax.set_ylabel("Accuracy")
plt.show()

import pandas as pd

data = pd.read_csv("D:\\MEPCO\\SEMESTER 5\\Machine Learning Essentials\\Raisin_Dataset.csv") 

from sklearn.preprocessing import LabelEncoder

le= LabelEncoder()
data.iloc[:,-1] = le.fit_transform(data.iloc[:,-1])

from scipy.cluster.hierarchy import dendrogram, linkage
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

Z = sch.linkage(data, method='average')
plt.title("Dendrogram")
plt.xlabel("Samples")
plt.ylabel("Euclidean distances")
dendrogram = sch.dendrogram(Z)
plt.show()
