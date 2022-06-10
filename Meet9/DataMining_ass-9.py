import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
from sklearn import datasets
iris = datasets.load_iris()

df=pd.DataFrame(iris['data'])
print(df.head())
# whiten function
from scipy.cluster.vq import whiten
scaled_data = whiten(df.to_numpy())

#fcluster and linkage function
from scipy.cluster.hierarchy import fcluster, linkage
#apply linkage function
distance_matrix = linkage(scaled_data, method='ward',metric='euclidean')
#dendrogram function
from scipy.cluster.hierarchy import dendrogram
#create a dendogram
dn = dendrogram(distance_matrix)
#display dendogram
plt.show()