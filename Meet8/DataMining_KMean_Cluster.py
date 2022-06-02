# imports
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
# Membaca data dan illustrate data tsb
iris = datasets.load_iris()
features = iris.data
plt.scatter(features[:,0],features[:,1])
plt.show()
#standardise feature
scaler = StandardScaler()
features_Standardized = scaler.fit_transform(features)
# membuat clustering 
from sklearn.metrics import silhouette_samples, silhouette_score
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=1,init='k-means++', msx_iter = 300, n_init=10, random_state=10)
    kmeans.fit(features)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()
# clustering
kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
pred_y = kmeans.fit_predict(features)
plt.scatter(features[:,0],features[:,1])
plt.scatter(kmeans.cluster_centers_[:,0],
    kmeans.cluster_centers_[:,1], s=300, c='red')
plt.show()

