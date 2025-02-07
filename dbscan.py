import numpy as np
from sklearn.neighbors import NearestNeighbors  # type: ignore


# Usamos NearestNeighbors para la optimización de DBS

class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):  # standar
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None
        self.cs_indices = None
        self.components = None

    def _expand_cluster(self, X, neighborhoods, labels, index, cluster_id, core_samples):
        """
        Expande un cluster dado un punto núcleo inicial.
        """
        labels[index] = cluster_id
        queue = list(neighborhoods[index])

        while queue:
            point = queue.pop(0)
            if labels[point] == -1:  # Cambia puntos de ruido a borde
                labels[point] = cluster_id
            elif labels[point] != -1:
                continue

            labels[point] = cluster_id
            if core_samples[point]:  # Solo expandir desde puntos núcleo
                queue.extend(neighborhoods[point])

    # ------------------------------------------------------------------------------------------
    def fit(self, X, weight=None):
        # X data, weight = weight -\_(^.^)_/-
        neighbors = NearestNeighbors(radius=0.5, metric='euclidean') # Prueba con 'cosine'
        neighbors.fit(X)
        neighborhood = neighbors.radius_neighbors(X, return_distance=False)

        # Neighbors
        if weight is None:
            n_neighbors = np.array([len(neighbors) for neighbors in neighborhood])
        else:
            n_neighbors = np.array([np.sum(neighborhood[neighbors]) for neighbors in weight])

        # Noise -1 |-| determinar nucleos
        labels = np.full(X.shape[0], -1, dtype=int)
        core_samples = n_neighbors >= self.min_samples

        # Asignar clusters
        cluster_id = 0
        for i in range(X.shape[0]):
            if not core_samples[i] or labels[i] != -1:
                continue  # Omitir si no es un punto núcleo o ya etiquetado
            self._expand_cluster(X, neighborhood, labels, i, cluster_id, core_samples)
            cluster_id += 1

        # Guardar resultados
        self.core_sample_indices_ = np.where(core_samples)[0]
        self.labels_ = labels
        self.components_ = X[self.core_sample_indices_].copy()
        return self

    def predict(self, X_new):
        """
        Asigna etiquetas de cluster a nuevos puntos según su vecindad.

        Parámetros
        ----------
        X_new : array-like, forma (n_samples, n_features)
            Nuevas instancias para etiquetar.

        Retorna
        -------
        labels_new : array de etiquetas de cluster para X_new
        """
        labels_new = []
        for point in X_new:
            label = -1
            for cluster_id, core_point in enumerate(self.components_):
                if np.linalg.norm(point - core_point) <= self.eps:
                    label = self.labels_[self.core_sample_indices_[cluster_id]]
                    break
            labels_new.append(label)
        return np.array(labels_new)

    def fit_predict(self, X, weight=None):
        self.fit(X, weight)  # Llama a fit para realizar el clustering
        return self.labels_  # Retorna las etiquetas generadas

# Inpirado de https://scrunts23.medium.com/dbscan-algorithm-from-scratch-in-python-475b82e0571c
# y de https://github.com/choffstein/dbscan/blob/master/dbscan/dbscan.py
# y de la pagina 67 del pdf _Chapter 4.1_ Clustering.pdf del curso