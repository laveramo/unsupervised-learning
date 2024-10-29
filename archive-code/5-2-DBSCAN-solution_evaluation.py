import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.io import arff
from sklearn import cluster
from sklearn import metrics
from sklearn.preprocessing import StandardScaler


path = './artificial/'
name = "xclara.arff"
databrut = arff.loadarff(open(path + str(name), 'r'))
datanp = np.array([[x[0], x[1]] for x in databrut[0]])

# Standardiser les données
scaler = StandardScaler().fit(datanp)
data_scaled = scaler.transform(datanp)

# Appliquer DBSCAN
epsilon = 0.05
min_pts = 5
model = cluster.DBSCAN(eps=epsilon, min_samples=min_pts)


start_time = time.time()
model.fit(data_scaled)
end_time = time.time()

# Récupération des labels
labels = model.labels_

# Évaluation des métriques
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
silhouette_score = metrics.silhouette_score(data_scaled, labels)
davies_bouldin_score = metrics.davies_bouldin_score(data_scaled, labels)

# Affichage des résultats
print("-------Donnees apres clustering DBSCAN  - Epislon= "+str(epsilon)+" MinPts= "+str(min_pts))
print(f'Nombre de clusters: {n_clusters}')
print(f'Nombre de points de bruit: {n_noise}')
print(f'Silhouette Score: {silhouette_score}')
print(f'Davies-Bouldin Index: {davies_bouldin_score}')
print(f'Temps de calcul: {end_time - start_time:.4f} secondes')

# Visualisation des clusters
plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=labels, s=8)
plt.title("Résultats du clustering DBSCAN")
plt.show()
