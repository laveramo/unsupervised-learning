from sklearn.neighbors import NearestNeighbors
from sklearn import cluster, metrics
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import arff
import time

# Charger les données
path = './artificial/'
name = "xclara.arff"
databrut = arff.loadarff(open(path + str(name), 'r'))
datanp = np.array([[x[0], x[1]] for x in databrut[0]])

# Normaliser les données pour DBSCAN
scaler = StandardScaler().fit(datanp)
data_scaled = scaler.transform(datanp)

# Calcul des distances moyennes aux k plus proches voisins pour déterminer le "coude"
k = 5
neigh = NearestNeighbors(n_neighbors=k)
neigh.fit(data_scaled)
distances, indices = neigh.kneighbors(data_scaled)

# Distance moyenne sur les k plus proches voisins, sans inclure le point d'origine
newDistances = np.asarray([np.average(distances[i][1:]) for i in range(distances.shape[0])])
distancetrie = np.sort(newDistances)

# Affichage de la courbe de distances pour repérer le coude
plt.title("Distances aux " + str(k) + " plus proches voisins")
plt.plot(distancetrie)
plt.xlabel("Index des points")
plt.ylabel("Distance moyenne")
plt.show()

# Détection automatique du coude de la courbe
gradients = np.diff(distancetrie)  # Calcul de la dérivée pour repérer les changements rapides
coude_index = np.argmax(gradients)  # Index du coude, là où la pente change le plus
eps_coude = distancetrie[coude_index]  # Valeur d'epsilon autour du coude

# Création d'une plage de valeurs d'epsilon autour de eps_coude avec une plus grande variation
eps_values = np.linspace(eps_coude * 0.5, eps_coude * 1.5, num=10)  # ±50% autour du coude
min_samples_values = [5, 10]  # Choix de quelques valeurs de min_samples

# Variables pour stocker les meilleurs paramètres et la meilleure évaluation
best_eps = None
best_min_samples = None
best_silhouette = -1  # Initialization à -1 
best_davies_bouldin = float('inf')  
best_n_clusters = 0
best_n_noise = 0
best_labels = None

print("Test des différentes valeurs de eps et min_samples :")

# Exécution de  DBSCAN pour chaque combinaison de epsilon et min_samples
for eps in eps_values:
    for min_samples in min_samples_values:
        # Appliquer DBSCAN
        model = cluster.DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(data_scaled)
        
        # Calcul des métriques uniquement si plus d'un cluster est formé
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        if n_clusters > 1:
            silhouette = metrics.silhouette_score(data_scaled, labels)
            davies_bouldin = metrics.davies_bouldin_score(data_scaled, labels)
            
            print(f"eps={eps:.4f}, min_samples={min_samples}, Clusters: {n_clusters}, Noises: {n_noise}, Silhouette: {silhouette:.4f}, Davies-Bouldin: {davies_bouldin:.4f}")
            
            # Mémorisation de la meilleure configuration
            if silhouette > best_silhouette and davies_bouldin < best_davies_bouldin:
                best_eps = eps
                best_min_samples = min_samples
                best_silhouette = silhouette
                best_davies_bouldin = davies_bouldin
                best_n_clusters = n_clusters
                best_n_noise = n_noise
                best_labels = labels

# Affichage des meilleurs résultats
print("\nMeilleure solution trouvée :")
print(f"eps = {best_eps}, min_samples = {best_min_samples}, Clusters: {best_n_clusters},Noises: {best_n_noise}, Silhouette: {best_silhouette:.4f}, Davies-Bouldin: {best_davies_bouldin:.4f}")

# Affichage des clusters de la meilleure solution
plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=best_labels, s=8)
plt.title(f"Clustering DBSCAN avec eps={best_eps} et min_samples={best_min_samples}")
plt.show()
