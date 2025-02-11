import numpy as np
import pandas as pd
from kmeans import kmeans_pp
from dbscan import DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score, mutual_info_score # type: ignore
from sklearn.model_selection import ParameterGrid
from sklearn.mixture import GaussianMixture

# 1 Paths
# 1.1 Umap paths
umap_path = "Reduction_data\\npy_umap.npy"
umapL_path = "Reduction_data\\npy_umapL.npy"
# 1.2 Tsne paths
tsne_path = "Reduction_data\\npy_tsne.npy"
tsneL_path = "Reduction_data\\npy_tsneL.npy"
# 1.3 Tests
test_umap = "Reduction_data\\npy_umap_test.npy"
test_tsne = "Reduction_data\\npy_tsne_test.npy"



umap_data = np.load(umap_path)  # Cargar la matriz completa
youtube_ids = umap_data[:, 0]  # Extraer IDs (strings)
umap_points = umap_data[:, 2:].astype(np.float32) # Extraer coordenadas numéricas y convertirlas a float
umap_l_points = np.load(umapL_path)
umap_l_points_pd = pd.DataFrame(umap_l_points)


# 2.2 Tsne
tsne_data = np.load(tsne_path)  # Cargar la matriz completa
youtube_ids_tsne = tsne_data[:, 0]  # Extraer IDs (strings)
tsne_points = tsne_data[:, 2:].astype(np.float32)  # Extraer coordenadas numéricas y convertirlas a float
tsne_l_points = np.load(tsneL_path)
tsne_l_points_pd = pd.DataFrame(tsne_l_points)

# 2.3 Tests
umap_test = np.load(test_umap)
youtube_ids = umap_test[:, 0]
umap_test_point = np.load(test_umap)[:, 1:].astype(np.float32)
tsne_test_point = np.load(test_tsne)[:, 1:].astype(np.float32)


# 1.4 Umap 3D paths
umap3d_path = "Reduction_data\\npy_umap_3d.npy"
umap3dL_path = "Reduction_data\\npy_umapL_3d.npy"

# 1.5 Tsne 3D paths
tsne3d_path = "Reduction_data\\npy_tsne_3d.npy"
tsne3dL_path = "Reduction_data\\npy_tsneL_3d.npy"

# 1.6 Test 3D
test_umap_3d = "Reduction_data\\npy_umap_test_3d.npy"
test_tsne_3d = "Reduction_data\\npy_tsne_test_3d.npy"


# 3D data
umap3d_data = np.load(umap3d_path)
youtube_ids_3d = umap3d_data[:, 0]
umap3d_points = umap3d_data[:, 2:].astype(np.float32)
umap3d_l_points = np.load(umap3dL_path)
umap3d_l_points_pd = pd.DataFrame(umap3d_l_points)

tsne3d_data = np.load(tsne3d_path)
youtube_ids_tsne_3d = tsne3d_data[:, 0]
tsne3d_points = tsne3d_data[:, 2:].astype(np.float32)
tsne3d_l_points = np.load(tsne3dL_path)
tsne3d_l_points_pd = pd.DataFrame(tsne3d_l_points)

# Test data for 3D
umap_test_point_3d = np.load(test_umap_3d)[:, 1:].astype(np.float32)
tsne_test_point_3d = np.load(test_tsne_3d)[:, 1:].astype(np.float32)
youtube_ids_test_umap_3d = np.load(test_umap_3d)[:, 0]
youtube_ids_test_tsne_3d = np.load(test_tsne_3d)[:, 0]


# Definir parámetros para la búsqueda en cuadrícula
"""param_grid = {
    "eps": np.arange(0.2, 1.2, 0.2),  # Variando eps de 0.2 a 1.0
    "min_samples": range(2, 6)  # min_samples de 2 a 5
}

best_score = -1
best_params = None
best_labels = None

# Búsqueda en cuadrícula manual
for params in ParameterGrid(param_grid):
    dbscan = DBSCAN(eps=params["eps"], min_samples=params["min_samples"])
    labels = dbscan.fit_predict(umap_points)

    # Validar que existan al menos 2 clusters distintos
    if len(set(labels)) > 1:
        score = silhouette_score(umap_points, labels)
        if score > best_score:
            best_score = score
            best_params = params
            best_labels = labels

print(f"Mejores parámetros DBSCAN: {best_params}, Silhouette Score: {best_score}")

# Evaluaciones con los mejores parámetros
labels_true = umap_l_points_pd[1].values
ri = adjusted_rand_score(labels_true=labels_true, labels_pred=best_labels)
mi = mutual_info_score(labels_true=labels_true, labels_pred=best_labels)

print(f"Rand Index: {ri}")
print(f"Mutual Information: {mi}")"""

############################### kmean++ y Umap ########################################
print("-------Resultados con Kmens++ y Umap-------")
# Kmean++ -----------------------------------------------------------
k_mU = kmeans_pp(umap_points, k=10)
k_mU.fit()

# Silhuete ----------------------------------------------------------
sil_grade = silhouette_score(X=umap_points, labels=k_mU.assignment)
print("Silhuete: " , sil_grade) # 10 veces, promedio:0,7 aprox

# Rand Index (RI) ---------------------------------------------------
labels_true = umap_l_points_pd[1].values
ri = adjusted_rand_score(labels_true=labels_true, labels_pred=k_mU.assignment)
print("Rand Index: ", ri) # 10 veces, promedio:0.8 aprox

# Mutual Information (MI) -------------------------------------------
mi = mutual_info_score(labels_true=labels_true, labels_pred=k_mU.assignment)
print("Mutual Information: ", mi)

# Predicciones en el conjunto de prueba ------------------------------------------------
umap_test_predictions = k_mU.predict(umap_test_point)
umap_test_predictions_df = pd.DataFrame({
    "youtube_id": youtube_ids,
    "label": umap_test_predictions
})
print("\nKmeans++ predictions for test data using UMAP:")
print(umap_test_predictions_df.head())

# Guardar predicciones de prueba en archivo CSV
umap_test_predictions_df.to_csv("submission/umap_test_predictions.csv", index=False)
print("Test predictions using UMAP saved to 'submission/umap_test_predictions.csv'.\n")

print()

############################### kmean++ y tsne ########################################
print("-------Resultados con Kmens++ y Tsne-------")
# Kmean++ -----------------------------------------------------------
k_mT = kmeans_pp(tsne_points, k=10)
k_mT.fit()

# Silhuete ----------------------------------------------------------
sil_grade = silhouette_score(X= tsne_points, labels=k_mT.assignment)
print("Silhuete: " , sil_grade) # 10 veces, promedio:0,6 aprox

# Rand Index (RI) ---------------------------------------------------
labels_true = tsne_l_points_pd[1].values
ri = adjusted_rand_score(labels_true=labels_true, labels_pred=k_mT.assignment)
print("Rand Index: ", ri) # 10 veces, promedio:0,8 aprox

# Mutual Information (MI) -------------------------------------------
mi = mutual_info_score(labels_true=labels_true, labels_pred=k_mT.assignment)
print("Mutual Information: ", mi)

# Predicciones en el conjunto de prueba ------------------------------------------------
tsne_test_predictions = k_mT.predict(tsne_test_point)
tsne_test_predictions_df = pd.DataFrame({
    "youtube_id": youtube_ids,
    "label": tsne_test_predictions
})
print("\nKmeans++ predictions for test data using t-SNE:")
print(tsne_test_predictions_df.head())

# Guardar predicciones de prueba en archivo CSV
tsne_test_predictions_df.to_csv("submission/tsne_test_predictions.csv", index=False)
print("Test predictions using t-SNE saved to 'submission/tsne_test_predictions.csv'.\n")


"""import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from joblib import Parallel, delayed

# Datos UMAP
X = umap_points
X_test = umap_test_point
youtube_ids = youtube_ids

# Rango de búsqueda de hiperparámetros
eps_values = np.arange(1.0, 20.0, 0.5)  # Ajustado para Manhattan
min_samples_values = range(2, 10)

# Lista para guardar resultados
results = []

# Función para evaluar DBSCAN con métricas mejoradas
def evaluate_dbscan(eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
    labels = dbscan.fit_predict(X)

    # Calcular % de ruido (-1s)
    noise_ratio = np.sum(labels == -1) / len(labels)  # Minimizar

    # Si DBSCAN asigna todo como ruido (-1), descartarlo
    unique_labels = np.unique(labels)
    if len(unique_labels) <= 1:
        return eps, min_samples, -1, -1, 100  # Penalización fuerte

    # Calcular métricas
    sil_score = silhouette_score(X, labels)
    dbi_score = davies_bouldin_score(X, labels)

    return eps, min_samples, sil_score, dbi_score, noise_ratio

# Ejecutar búsqueda en paralelo para optimizar velocidad
results = Parallel(n_jobs=-1)(
    delayed(evaluate_dbscan)(eps, min_samples) for eps in eps_values for min_samples in min_samples_values
)

# Convertir a DataFrame para ordenar y visualizar
df_results = pd.DataFrame(results, columns=["eps", "min_samples", "Silhouette", "DBI", "Noise %"])
df_results = df_results.sort_values(by=["Silhouette"], ascending=False)  # Ordenar por mejor Silhouette Score

# Imprimir todos los resultados
print("\n📌 Resultados completos de búsqueda de hiperparámetros:")
print(df_results)

# Seleccionar la mejor configuración
best_row = df_results.iloc[0]  # Primera fila con mejor Silhouette Score
best_eps, best_min_samples, best_sil, best_dbi, best_noise = best_row

print(f"\n📌 Mejor configuración encontrada:")
print(f"  - eps: {best_eps}")
print(f"  - min_samples: {best_min_samples}")
print(f"  - Silhouette Score: {best_sil:.4f}")
print(f"  - Davies-Bouldin Index: {best_dbi:.4f} (Menor es mejor)")
print(f"  - % de Ruido (-1s): {best_noise:.4%}")

# Entrenar DBSCAN con los mejores hiperparámetros
best_dbscan = DBSCAN(eps=best_eps, min_samples=int(best_min_samples), metric="euclidean")
umap_labels = best_dbscan.fit_predict(X)

# Predicciones en el conjunto de prueba
umap_test_predictions = best_dbscan.fit_predict(X_test)
umap_test_predictions_df = pd.DataFrame({
    "youtube_id": youtube_ids,
    "label": umap_test_predictions
})

# Guardar en CSV
umap_test_predictions_df.to_csv("submission/umap_dbscan_best.csv", index=False)
print("\n🚀 Test predictions saved to 'submission/umap_dbscan_best.csv'.")"""




############################### DBSCAN y Umap ########################################
print("-------Resultados con DBSCAN y Umap-------")
# DBSCAN -----------------------------------------------------------
dbscan_umap = DBSCAN(eps=0.7499999999999998, min_samples=2)  # Ajustar
umap_labels = dbscan_umap.fit_predict(umap_points)

# Silhouette --------------------------------------------------------
sil_grade = silhouette_score(X=umap_points, labels=umap_labels)
print("Silhouette: ", sil_grade)

# Rand Index (RI) ---------------------------------------------------
labels_true = umap_l_points_pd[1].values
ri = adjusted_rand_score(labels_true=labels_true, labels_pred=umap_labels)
print("Rand Index: ", ri)

# Mutual Information (MI) -------------------------------------------
mi = mutual_info_score(labels_true=labels_true, labels_pred=umap_labels)
print("Mutual Information: ", mi)

# Predicciones en el conjunto de prueba usando DBSCAN -----------------------
umap_test_predictions = dbscan_umap.fit_predict(umap_test_point)
umap_test_predictions_df = pd.DataFrame({
    "youtube_id": youtube_ids,
    "label": umap_test_predictions
})
print("\nDBSCAN predictions for test data using UMAP:")
print(umap_test_predictions_df.head())

umap_test_predictions_df.to_csv("submission/umap_dbscan_test_predictions.csv", index=False)
print("Test predictions using UMAP with DBSCAN saved to 'submission/umap_dbscan_test_predictions.csv'.\n")



############################### DBSCAN y tsne ########################################
print("-------Resultados con DBSCAN y Tsne-------")
# DBSCAN -----------------------------------------------------------
dbscan_tsne = DBSCAN(eps=0.5, min_samples=2)  # Ajustar eps y min_points según sea necesario
tsne_labels = dbscan_tsne.fit_predict(tsne_points)

# Silhouette --------------------------------------------------------
sil_grade = silhouette_score(X=tsne_points, labels=tsne_labels)
print("Silhouette: ", sil_grade)

# Rand Index (RI) ---------------------------------------------------
labels_true = tsne_l_points_pd[1].values
ri = adjusted_rand_score(labels_true=labels_true, labels_pred=tsne_labels)
print("Rand Index: ", ri)

# Mutual Information (MI) -------------------------------------------
mi = mutual_info_score(labels_true=labels_true, labels_pred=tsne_labels)
print("Mutual Information: ", mi)

# Guardar etiquetas en archivo CSV
tsne_test_predictions = dbscan_tsne.fit_predict(tsne_test_point)
tsne_dbscan_labels_df = pd.DataFrame({
    "youtube_id": youtube_ids,  # Ajustar según tus IDs
    "label": tsne_test_predictions
})
print("\nDBSCAN predictions for test data using TSNE:")
print(tsne_dbscan_labels_df.head())
# Guardar predicciones de prueba en archivo CSV
tsne_dbscan_labels_df.to_csv("submission/tsne_dbscan_test_predictions.csv", index=False)
print("Test predictions using t-SNE with DBSCAN saved to 'submission/tsne_dbscan_test_predictions.csv'.\n")


############################### GMM y Umap ########################################
print("-------Resultados con GMM y UMAP-------")
# GMM -----------------------------------------------------------
gmm_umap = GaussianMixture(n_components=10, random_state=42)
gmm_umap_labels = gmm_umap.fit_predict(umap_points)

# Silhouette --------------------------------------------------------
sil_grade = silhouette_score(X=umap_points, labels=gmm_umap_labels)
print("Silhouette: ", sil_grade)

# Rand Index (RI) ---------------------------------------------------
labels_true = umap_l_points_pd[1].values
ri = adjusted_rand_score(labels_true=labels_true, labels_pred=gmm_umap_labels)
print("Rand Index: ", ri)

# Mutual Information (MI) -------------------------------------------
mi = mutual_info_score(labels_true=labels_true, labels_pred=gmm_umap_labels)
print("Mutual Information: ", mi)

# Predicciones en el conjunto de prueba usando GMM -----------------------
umap_test_predictions = gmm_umap.predict(umap_test_point)
umap_test_predictions_df = pd.DataFrame({
    "youtube_id": youtube_ids,
    "label": umap_test_predictions
})
print("\nGMM predictions for test data using UMAP:")
print(umap_test_predictions_df.head())

umap_test_predictions_df.to_csv("submission/umap_gmm_test_predictions.csv", index=False)
print("Test predictions using UMAP with GMM saved to 'submission/umap_gmm_test_predictions.csv'.\n")


############################### GMM y Tsne ########################################
print("-------Resultados con GMM y TSNE-------")
# GMM -----------------------------------------------------------
gmm_tsne = GaussianMixture(n_components=10, random_state=42)
gmm_tsne_labels = gmm_tsne.fit_predict(tsne_points)

# Silhouette --------------------------------------------------------
sil_grade = silhouette_score(X=tsne_points, labels=gmm_tsne_labels)
print("Silhouette: ", sil_grade)

# Rand Index (RI) ---------------------------------------------------
labels_true = tsne_l_points_pd[1].values
ri = adjusted_rand_score(labels_true=labels_true, labels_pred=gmm_tsne_labels)
print("Rand Index: ", ri)

# Mutual Information (MI) -------------------------------------------
mi = mutual_info_score(labels_true=labels_true, labels_pred=gmm_tsne_labels)
print("Mutual Information: ", mi)

# Predicciones en el conjunto de prueba usando GMM -----------------------
tsne_test_predictions = gmm_tsne.predict(tsne_test_point)
tsne_test_predictions_df = pd.DataFrame({
    "youtube_id": youtube_ids,
    "label": tsne_test_predictions
})
print("\nGMM predictions for test data using TSNE:")
print(tsne_test_predictions_df.head())

tsne_test_predictions_df.to_csv("submission/tsne_gmm_test_predictions.csv", index=False)
print("Test predictions using TSNE with GMM saved to 'submission/tsne_gmm_test_predictions.csv'.\n")


############################### kmean++ y Umap 3D ########################################
print("-------Resultados con Kmens++ y Umap 3D-------")
# Kmean++ -----------------------------------------------------------
k_mU_3d = kmeans_pp(umap3d_points, k=10)
k_mU_3d.fit()

# Silhouette ----------------------------------------------------------
sil_grade_3d = silhouette_score(X=umap3d_points, labels=k_mU_3d.assignment)
print("Silhouette (3D): ", sil_grade_3d)

# Rand Index (RI) ---------------------------------------------------
labels_true_3d = umap3d_l_points_pd[1].values
ri_3d = adjusted_rand_score(labels_true=labels_true_3d, labels_pred=k_mU_3d.assignment)
print("Rand Index (3D): ", ri_3d)

# Mutual Information (MI) -------------------------------------------
mi_3d = mutual_info_score(labels_true=labels_true_3d, labels_pred=k_mU_3d.assignment)
print("Mutual Information (3D): ", mi_3d)

# Predicciones en el conjunto de prueba ------------------------------------------------
umap_test_predictions_3d = k_mU_3d.predict(umap_test_point_3d)
print(len(youtube_ids_test_umap_3d))
print(len(umap_test_predictions_3d))

umap_test_predictions_df_3d = pd.DataFrame({
    "youtube_id": youtube_ids_test_umap_3d,
    "label": umap_test_predictions_3d
})
print("\nKmeans++ predictions for test data using UMAP 3D:")
print(umap_test_predictions_df_3d.head())

umap_test_predictions_df_3d.to_csv("submission/umap3d_test_predictions.csv", index=False)
print("Test predictions using UMAP 3D saved to 'submission/umap3d_test_predictions.csv'.\n")


############################### K-means++ y t-SNE 3D ########################################
print("-------Resultados con K-means++ y t-SNE 3D-------")
# K-means++ -----------------------------------------------------------
k_mT_3d = kmeans_pp(tsne3d_points, k=10)
k_mT_3d.fit()

# Silhouette ----------------------------------------------------------
sil_grade_3d = silhouette_score(X=tsne3d_points, labels=k_mT_3d.assignment)
print("Silhouette (3D): ", sil_grade_3d)

# Rand Index (RI) ---------------------------------------------------
labels_true_3d = tsne3d_l_points_pd[1].values
ri_3d = adjusted_rand_score(labels_true=labels_true_3d, labels_pred=k_mT_3d.assignment)
print("Rand Index (3D): ", ri_3d)

# Mutual Information (MI) -------------------------------------------
mi_3d = mutual_info_score(labels_true=labels_true_3d, labels_pred=k_mT_3d.assignment)
print("Mutual Information (3D): ", mi_3d)

# Predicciones en el conjunto de prueba ------------------------------------------------
tsne_test_predictions_3d = k_mT_3d.predict(tsne_test_point_3d)
print(len(youtube_ids_test_tsne_3d))
print(len(umap_test_predictions_3d))

tsne_test_predictions_df_3d = pd.DataFrame({
    "youtube_id": youtube_ids_test_tsne_3d,
    "label": tsne_test_predictions_3d
})
print("\nK-means++ predictions for test data using t-SNE 3D:")
print(tsne_test_predictions_df_3d.head())

tsne_test_predictions_df_3d.to_csv("submission/tsne3d_kmeans_test_predictions.csv", index=False)
print("Test predictions using t-SNE 3D with K-means++ saved to 'submission/tsne3d_kmeans_test_predictions.csv'.\n")


############################### DBSCAN y Tsne 3D ########################################
print("-------Resultados con DBSCAN y Tsne 3D-------")
# DBSCAN -----------------------------------------------------------
dbscan_tsne_3d = DBSCAN(eps=0.5, min_samples=2)
tsne_labels_3d = dbscan_tsne_3d.fit_predict(tsne3d_points)

# Silhouette --------------------------------------------------------
sil_grade_3d = silhouette_score(X=tsne3d_points, labels=tsne_labels_3d)
print("Silhouette (3D): ", sil_grade_3d)

# Rand Index (RI) ---------------------------------------------------
labels_true_3d = tsne3d_l_points_pd[1].values
ri_3d = adjusted_rand_score(labels_true=labels_true_3d, labels_pred=tsne_labels_3d)
print("Rand Index (3D): ", ri_3d)

# Mutual Information (MI) -------------------------------------------
mi_3d = mutual_info_score(labels_true=labels_true_3d, labels_pred=tsne_labels_3d)
print("Mutual Information (3D): ", mi_3d)

# Guardar etiquetas en archivo CSV
tsne_test_predictions_3d = dbscan_tsne_3d.fit_predict(tsne_test_point_3d)
tsne_dbscan_labels_df_3d = pd.DataFrame({
    "youtube_id": youtube_ids_test_tsne_3d,
    "label": tsne_test_predictions_3d
})
print("\nDBSCAN predictions for test data using TSNE 3D:")
print(tsne_dbscan_labels_df_3d.head())

tsne_dbscan_labels_df_3d.to_csv("submission/tsne3d_dbscan_test_predictions.csv", index=False)
print("Test predictions using TSNE 3D with DBSCAN saved to 'submission/tsne3d_dbscan_test_predictions.csv'.\n")


############################### DBSCAN y UMAP 3D ########################################
print("-------Resultados con DBSCAN y UMAP 3D-------")
# DBSCAN -----------------------------------------------------------
dbscan_umap_3d = DBSCAN(eps=0.5, min_samples=2)
umap_labels_3d = dbscan_umap_3d.fit_predict(umap3d_points)

# Silhouette --------------------------------------------------------
sil_grade_3d = silhouette_score(X=umap3d_points, labels=umap_labels_3d)
print("Silhouette (3D): ", sil_grade_3d)

# Rand Index (RI) ---------------------------------------------------
labels_true_3d = umap3d_l_points_pd[1].values
ri_3d = adjusted_rand_score(labels_true=labels_true_3d, labels_pred=umap_labels_3d)
print("Rand Index (3D): ", ri_3d)

# Mutual Information (MI) -------------------------------------------
mi_3d = mutual_info_score(labels_true=labels_true_3d, labels_pred=umap_labels_3d)
print("Mutual Information (3D): ", mi_3d)

# Predicciones en el conjunto de prueba ------------------------------------------------
umap_test_predictions_3d = dbscan_umap_3d.fit_predict(umap_test_point_3d)
umap_dbscan_labels_df_3d = pd.DataFrame({
    "youtube_id": youtube_ids_test_umap_3d,
    "label": umap_test_predictions_3d
})
print("\nDBSCAN predictions for test data using UMAP 3D:")
print(umap_dbscan_labels_df_3d.head())

umap_dbscan_labels_df_3d.to_csv("submission/umap3d_dbscan_test_predictions.csv", index=False)
print("Test predictions using UMAP 3D with DBSCAN saved to 'submission/umap3d_dbscan_test_predictions.csv'.\n")




############################### GMM y Umap 3D ########################################
print("-------Resultados con GMM y UMAP 3D-------")
# GMM -----------------------------------------------------------
gmm_umap_3d = GaussianMixture(n_components=10, random_state=42)
gmm_umap_labels_3d = gmm_umap_3d.fit_predict(umap3d_points)

# Silhouette --------------------------------------------------------
sil_grade_3d = silhouette_score(X=umap3d_points, labels=gmm_umap_labels_3d)
print("Silhouette (3D): ", sil_grade_3d)

# Rand Index (RI) ---------------------------------------------------
labels_true_3d = umap3d_l_points_pd[1].values
ri_3d = adjusted_rand_score(labels_true=labels_true_3d, labels_pred=gmm_umap_labels_3d)
print("Rand Index (3D): ", ri_3d)

# Mutual Information (MI) -------------------------------------------
mi_3d = mutual_info_score(labels_true=labels_true_3d, labels_pred=gmm_umap_labels_3d)
print("Mutual Information (3D): ", mi_3d)

# Predicciones en el conjunto de prueba usando GMM -----------------------
umap_test_predictions_3d = gmm_umap_3d.predict(umap_test_point_3d)
umap_test_predictions_df_3d = pd.DataFrame({
    "youtube_id": youtube_ids_test_umap_3d,
    "label": umap_test_predictions_3d
})
print("\nGMM predictions for test data using UMAP 3D:")
print(umap_test_predictions_df_3d.head())

umap_test_predictions_df_3d.to_csv("submission/umap3d_gmm_test_predictions.csv", index=False)
print("Test predictions using UMAP 3D with GMM saved to 'submission/umap3d_gmm_test_predictions.csv'.\n")


############################### GMM y t-SNE 3D ########################################
print("-------Resultados con GMM y t-SNE 3D-------")
# GMM -----------------------------------------------------------
gmm_tsne_3d = GaussianMixture(n_components=10, random_state=42)
gmm_tsne_labels_3d = gmm_tsne_3d.fit_predict(tsne3d_points)

# Silhouette --------------------------------------------------------
sil_grade_3d = silhouette_score(X=tsne3d_points, labels=gmm_tsne_labels_3d)
print("Silhouette (3D): ", sil_grade_3d)

# Rand Index (RI) ---------------------------------------------------
labels_true_3d = tsne3d_l_points_pd[1].values
ri_3d = adjusted_rand_score(labels_true=labels_true_3d, labels_pred=gmm_tsne_labels_3d)
print("Rand Index (3D): ", ri_3d)

# Mutual Information (MI) -------------------------------------------
mi_3d = mutual_info_score(labels_true=labels_true_3d, labels_pred=gmm_tsne_labels_3d)
print("Mutual Information (3D): ", mi_3d)

# Predicciones en el conjunto de prueba usando GMM -----------------------
tsne_test_predictions_3d = gmm_tsne_3d.predict(tsne_test_point_3d)
tsne_test_predictions_df_3d = pd.DataFrame({
    "youtube_id": youtube_ids_test_tsne_3d,
    "label": tsne_test_predictions_3d
})
print("\nGMM predictions for test data using t-SNE 3D:")
print(tsne_test_predictions_df_3d.head())

tsne_test_predictions_df_3d.to_csv("submission/tsne3d_gmm_test_predictions.csv", index=False)
print("Test predictions using t-SNE 3D with GMM saved to 'submission/tsne3d_gmm_test_predictions.csv'.\n")

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def plot_clustering(data, labels, title, is_3d=False):
    """
    Función para graficar los resultados del clustering.

    Parameters:
        data (numpy array): Puntos reducidos en 2D o 3D.
        labels (numpy array): Etiquetas del clustering.
        title (str): Título del gráfico.
        is_3d (bool): Indica si la visualización es en 3D.
    """
    unique_labels = np.unique(labels)
    cmap = plt.colormaps.get_cmap("tab10")  # Forma correcta en Matplotlib 3.7+

    if is_3d:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(data[mask, 0], data[mask, 1], data[mask, 2], label=f"Cluster {label}",
                       alpha=0.6, edgecolors='k', c=[cmap(i % 10)] * np.sum(mask))  # Se corrige 'c'
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_zlabel("Component 3")
    else:
        plt.figure(figsize=(8, 6))
        for i, label in enumerate(unique_labels):
            mask = labels == label
            plt.scatter(data[mask, 0], data[mask, 1], label=f"Cluster {label}",
                        alpha=0.6, edgecolors='k', c=[cmap(i % 10)] * np.sum(mask))  # Se corrige 'c'
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")

    plt.title(title)
    plt.legend()
    plt.show()



# Aplicar la función de visualización a todos los métodos de clustering

# K-means++ con UMAP y t-SNE
plot_clustering(umap_points, k_mU.assignment, "K-means++ con UMAP")
plot_clustering(tsne_points, k_mT.assignment, "K-means++ con t-SNE")

# DBSCAN con UMAP y t-SNE
plot_clustering(umap_points, umap_labels, "DBSCAN con UMAP")
plot_clustering(tsne_points, tsne_labels, "DBSCAN con t-SNE")

# GMM con UMAP y t-SNE
plot_clustering(umap_points, gmm_umap_labels, "GMM con UMAP")
plot_clustering(tsne_points, gmm_tsne_labels, "GMM con t-SNE")

# Visualización en 3D
plot_clustering(umap3d_points, k_mU_3d.assignment, "K-means++ con UMAP 3D", is_3d=True)
plot_clustering(tsne3d_points, k_mT_3d.assignment, "K-means++ con t-SNE 3D", is_3d=True)
plot_clustering(umap3d_points, umap_labels_3d, "DBSCAN con UMAP 3D", is_3d=True)
plot_clustering(tsne3d_points, tsne_labels_3d, "DBSCAN con t-SNE 3D", is_3d=True)
plot_clustering(umap3d_points, gmm_umap_labels_3d, "GMM con UMAP 3D", is_3d=True)
plot_clustering(tsne3d_points, gmm_tsne_labels_3d, "GMM con t-SNE 3D", is_3d=True)
