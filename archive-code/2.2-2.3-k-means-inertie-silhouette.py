import os
os.environ["OMP_NUM_THREADS"] = '4'

import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from sklearn import cluster
from sklearn.metrics import silhouette_score

##################################################################
# Exemple :  k-Means Clustering

path = './artificial/'
name="2sp2glob.arff"

databrut = arff.loadarff(open(path+str(name), 'r'))
datanp = np.array([[x[0],x[1]] for x in databrut[0]])

print("---------------------------------------")
print("Affichage données initiales            "+ str(name))
f0 = datanp[:,0] # tous les élements de la première colonne
f1 = datanp[:,1] # tous les éléments de la deuxième colonne

#plt.figure(figsize=(6, 6))
plt.scatter(f0, f1, s=8)
plt.title("Donnees initiales : "+ str(name))
plt.show()

inerties = []
silhouette_scores = []
clusters = range(2,10)

# Run clustering method for a given number of clusters
print("------------------------------------------------------")
for k in clusters:
    print("Appel KMeans pour une valeur de k fixée")
    tps1 = time.time()
    model = cluster.KMeans(n_clusters=k, init='k-means++', n_init=1)
    model.fit(datanp)
    tps2 = time.time()
    labels = model.labels_
    # informations sur le clustering obtenu
    iteration = model.n_iter_
    inertie = model.inertia_
    centroids = model.cluster_centers_
    inerties.append(inertie)    
    silhouette_avg = silhouette_score(datanp, labels)
    silhouette_scores.append(silhouette_avg)

    print("nb clusters =",k,", nb iter =",iteration, ", silhouette score = ",silhouette_avg, ", inertie = ",inertie, ", runtime = ", round((tps2 - tps1)*1000,2),"ms")


plt.plot(clusters, inerties, "b:o")
plt.title("Inertie vs clusters")
plt.xlabel("# clusters")
plt.ylabel("Inertie")
plt.show()

plt.plot(clusters, silhouette_scores, "b:o")
plt.title("Silhouette value vs clusters")
plt.xlabel("# clusters")
plt.ylabel("Silhouette value")
plt.show()

# choice of the best solution based on silhouette score
solution = max(silhouette_scores)
final_clusters = clusters[silhouette_scores.index(solution)]
print("Best solution found with ", final_clusters, " clusters and silhouette score of ", solution)
#
model = cluster.KMeans(n_clusters=final_clusters, init='k-means++', n_init=1)
model.fit(datanp)
labels = model.labels_
# informations sur le clustering obtenu
iteration = model.n_iter_
centroids = model.cluster_centers_
plt.figure(figsize=(6, 6))
plt.scatter(f0, f1, c=labels, s=8)
plt.scatter(centroids[:, 0],centroids[:, 1], marker="x", s=50, linewidths=3, color="red")
plt.title("Données après clustering : "+ str(name) + " - Nb clusters ="+ str(final_clusters))
plt.show()