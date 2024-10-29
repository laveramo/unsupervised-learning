import os
os.environ["OMP_NUM_THREADS"] = '4'

import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from sklearn import cluster
from sklearn.metrics import silhouette_score, pairwise_distances_argmin_min

# Chargement des données
path = './artificial/'
name = "xclara.arff"
databrut = arff.loadarff(open(path+str(name), 'r'))
datanp = np.array([[x[0], x[1]] for x in databrut[0]])

# Affichage des données initiales
print("---------------------------------------")
print("Affichage données initiales            " + str(name))
f0 = datanp[:, 0]  # première colonne
f1 = datanp[:, 1]  # deuxième colonne
plt.scatter(f0, f1, s=8)
plt.title("Données initiales : " + str(name))
plt.show()

# Initialisation des listes pour les métriques
inerties = []
silhouette_scores = []
computation_times = []
clusters_range = range(2, 10)

# Variables pour la meilleure solution
best_k = None
best_silhouette = -1
best_inertia = None
best_runtime = None

# Boucle pour tester différents nombres de clusters
for k in clusters_range:
    print(f"\nClustering pour k={k}")
    tps1 = time.time()
    
    # Clustering agglomératif
    model = cluster.AgglomerativeClustering(linkage='average', n_clusters=k)
    labels = model.fit_predict(datanp)
    tps2 = time.time()
    
    # Calcul de l'inertie (somme des distances au centre du cluster)
    cluster_centers = [datanp[labels == i].mean(axis=0) for i in range(k)]
    inertia = sum([np.sum((datanp[labels == i] - cluster_centers[i]) ** 2) for i in range(k)])
    inerties.append(inertia)
    
    # Calcul de l'indice de silhouette
    silhouette_avg = silhouette_score(datanp, labels)
    silhouette_scores.append(silhouette_avg)

    # Temps de calcul pour cette exécution
    runtime = round((tps2 - tps1) * 1000, 2)  # en millisecondes
    computation_times.append(runtime)
    
    # Affichage des résultats pour chaque k
    print(f"nb clusters = {k}, silhouette score = {silhouette_avg:.3f}, inertie = {inertia:.3f}, runtime = {runtime} ms")
    
    # Mises à jour de la meilleure solution
    if silhouette_avg > best_silhouette:
        best_k = k
        best_silhouette = silhouette_avg
        best_inertia = inertia
        best_runtime = runtime

# Affichage de la meilleure solution trouvée
print("\n---------------------------------------")
print("Meilleure Solution:")
print(f"Nombre de clusters = {best_k}")
print(f"Score de silhouette = {best_silhouette:.3f}")
print(f"Inertie = {best_inertia:.3f}")
print(f"Temps de calcul = {best_runtime} ms")
print("---------------------------------------")

# Graphique Inertie vs Clusters
plt.plot(clusters_range, inerties, "b:o")
plt.title("Inertie vs Nombre de clusters")
plt.xlabel("Nombre de clusters")
plt.ylabel("Inertie")
plt.show()

# Graphique Silhouette vs Clusters
plt.plot(clusters_range, silhouette_scores, "b:o")
plt.title("Score de silhouette vs Nombre de clusters")
plt.xlabel("Nombre de clusters")
plt.ylabel("Score de silhouette")
plt.show()

# Graphe pour le temps de calcul
plt.plot(clusters_range, computation_times, "r:o")
plt.title("Temps de calcul vs Nombre de clusters")
plt.xlabel("Nombre de clusters")
plt.ylabel("Temps de calcul (ms)")
plt.show()
