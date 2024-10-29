import os
os.environ["OMP_NUM_THREADS"] = '4'

import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn import cluster
from sklearn.metrics import silhouette_score

# Chargement des données
path = './artificial/'
name = "xclara.arff"
databrut = arff.loadarff(open(path + str(name), 'r'))
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
distance_thresholds = np.linspace(2, 20, 10)  # valeurs de seuil de distance

# Clustering hiérarchique pour obtenir les liens
linked = linkage(datanp, method='average')

# Dendrogramme pour visualiser les clusters
plt.figure(figsize=(10, 7))
dendrogram(linked)
plt.title("Dendrogramme")
plt.xlabel("Points de données")
plt.ylabel("Distance")
plt.show()

# Boucle pour tester différents seuils de distance
for threshold in distance_thresholds:
    print(f"\nÉvaluation avec distance_threshold = {threshold}")
    tps1 = time.time()

    # Clustering agglomératif avec distance_threshold
    model = cluster.AgglomerativeClustering(linkage='average', n_clusters=None, distance_threshold=threshold)
    labels = model.fit_predict(datanp)
    tps2 = time.time()
    
    # Calcul de l'inertie (somme des distances au centre du cluster)
    unique_labels = np.unique(labels)
    cluster_centers = [datanp[labels == i].mean(axis=0) for i in unique_labels]
    inertia = sum([np.sum((datanp[labels == i] - cluster_centers[i]) ** 2) for i in range(len(cluster_centers))])
    inerties.append(inertia)

    # Calcul de l'indice de silhouette
    silhouette_avg = silhouette_score(datanp, labels) if len(cluster_centers) > 1 else 0
    silhouette_scores.append(silhouette_avg)

    # Temps de calcul pour cette exécution
    runtime = round((tps2 - tps1) * 1000, 2)  # en millisecondes
    computation_times.append(runtime)
    
    # Affichage des résultats
    print(f"nb clusters = {len(unique_labels)}, silhouette score = {silhouette_avg:.3f}, "
          f"inertie = {inertia:.3f}, runtime = {runtime} ms")

# Trouver le meilleur seuil de distance basé sur le meilleur score de silhouette et la plus basse inertie
best_index = max(range(len(distance_thresholds)), key=lambda i: (silhouette_scores[i], -inerties[i]))
best_threshold = distance_thresholds[best_index]
best_inertia = inerties[best_index]
best_silhouette = silhouette_scores[best_index]
best_runtime = computation_times[best_index]
best_n_clusters = len(np.unique(cluster.AgglomerativeClustering(linkage='average', n_clusters=None, distance_threshold=best_threshold).fit_predict(datanp)))

# Afficher les détails de la meilleure solution
print("\n---------------------------------------")
print("Meilleure solution trouvée")
print(f"Seuil de distance optimal : {best_threshold}")
print(f"Nombre de clusters : {best_n_clusters}")
print(f"Inertie : {best_inertia:.3f}")
print(f"Score de silhouette : {best_silhouette:.3f}")
print(f"Temps de calcul : {best_runtime} ms")
print("---------------------------------------")

# Graphique Inertie vs Seuil de distance
plt.plot(distance_thresholds, inerties, "b:o")
plt.title("Inertie vs Seuil de distance")
plt.xlabel("Seuil de distance")
plt.ylabel("Inertie")
plt.show()

# Graphique Silhouette vs Seuil de distance
plt.plot(distance_thresholds, silhouette_scores, "g:o")
plt.title("Score de silhouette vs Seuil de distance")
plt.xlabel("Seuil de distance")
plt.ylabel("Score de silhouette")
plt.show()

# Graphe pour le temps de calcul
plt.plot(distance_thresholds, computation_times, "r:o")
plt.title("Temps de calcul vs Seuil de distance")
plt.xlabel("Seuil de distance")
plt.ylabel("Temps de calcul (ms)")
plt.show()
