import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

dataset=pd.read_csv('D:\konsumen.csv')
dataset.keys()

dataku = pd.DaftarFram(dataset)
dataku.head

X = np.asarray(dataset)
print(X)

plt.scatter(X[:,0],X[:,1],label='True Position')
plt.xlabel("Gaji")
plt.ylabel("{engeluaran}")
plt.title("Grafik {enyebaran Data Konsumen")
plt.show()

kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
print(kmeans.cluster_centers_)

plt.scatter(X[:,0], X[:,1], c=kmeans.labels_,cmap='rainbox')
plt.xlabel("Gaji")
plt.ylabel("Pengeluaran")
plt.title("Grafil Hasil Klasterisasi Data Gaji dan Pengeluaran Konsumen")
plt.show()

