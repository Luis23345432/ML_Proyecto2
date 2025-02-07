import numpy as np

class kmeans_pp:
    # Constructor -------------------------------------------------------------------------
    def __init__(self, data, k=3):
        self.data = data
        self.k = k
        self.assignment = [-1 for i in range(len(data))]
        self.history = []
    # Euclidean ---------------------------------------------------------------------------
    def euclidean_dist(self, x, y):
        return np.sqrt(np.sum((x - y) ** 2))
    # Kmeans++ algotihm -------------------------------------------------------------------
    def kmeans_plus_plus(self):
        centroids = self.data[np.random.choice(range(len(self.data)), size=self.k)]

        for i in range(1, self.k):
            min_sq_dist = [min([
                # de centroide a punto
                self.euclidean_dist(c, p) ** 2
                for c in centroids])
                for p in self.data]
            probability = min_sq_dist/sum(min_sq_dist)
            centroids = np.append(centroids, self.data[
                np.random.choice(range(len(self.data)), 
                size=1, p=probability)], 
                axis=0
            )
        return centroids
    # Si no esta asignado ------------------------------------------------------------------
    def unassigned(self, i):
        return self.assignment[i] == -1
    # Asignar cluster ----------------------------------------------------------------------    
    def assign(self, cent):
        for i in range(len(self.data)):
            for j in range(self.k):
                if self.unassigned(i):
                    self.assignment[i] = j
                    euclidean_dist = self.euclidean_dist(self.data[i], cent[j])
                else:
                    t_dist = self.euclidean_dist(self.data[i], cent[j])
                    if t_dist < euclidean_dist:
                        self.assignment[i] = j
                        euclidean_dist = t_dist
    # snap ------------------------------------------------------------------------------------
    def snap(self, centers):
        self.history.append(np.copy(centers))
    # Acompañando al kmeans++ se encarga de hacer los centroides ---------------------------
    def centers_maker(self):
        centers = []
        for j in range(self.k):
            cluster = np.array([self.data[k] for k in filter(lambda x : x >= 0, 
            [i if self.assignment[i] == j else -1 for i in range(len(self.data))])])
            centers.append(np.mean(cluster, axis=0))
        return np.array(centers)
    # Calculo entre centroides -------------------------------------------------------------
    def centers_diff(self, c1, c2):
        for i in range(self.k):
            if self.euclidean_dist(c1[i], c2[i]) != 0:
                return True
        return False
    # desasignar a todos -------------------------------------------------------------------
    def unasign_all(self):
        self.assignment = [-1 for _ in range(len(self.data))]
    # Entrenar al modelo -------------------------------------------------------------------
    def fit(self):
        centers = self.kmeans_plus_plus()
        self.assign(centers) #asisnar centros
        self.snap(centers)
        new_centers = self.centers_maker()
        while self.centers_diff(centers, new_centers):
            self.unasign_all()
            centers = new_centers
            self.assign(centers)
            self.snap(centers)
            new_centers = self.centers_maker()
    # Predict ------------------------------------------------------------------------------
    def predict(self, test_data):
        pred =[]
        for point in test_data:
            dist_min = float('inf')
            clusters_signados = -1
            for j in  range(self.k):
                dist = self.euclidean_dist(point, self.history[-1][j])
                if dist < dist_min:
                    dist_min = dist
                    assigned_cluster = j
            pred.append(assigned_cluster)
        return pred

#Inspirado de https://medium.com/@gallettilance/kmeans-from-scratch-24be6bee8021
#al modelo le falta predict implementar snap()