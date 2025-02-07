import numpy as np
import pandas as pd
import glob
from sklearn.preprocessing import StandardScaler  # type: ignore
from sklearn.manifold import TSNE  # type: ignore
import umap
import pickle
import os


def process_features(features_path, csv_path=None, scaler=None, reducer_2d=None, reducer_3d=None, apply_umap=True,
                     apply_tsne=True):
    features_files = glob.glob(f"{features_path}/*.npy")
    show_f = [np.load(npy_file) for npy_file in features_files]
    features_med, youtube_ids, labels = [], [], []

    for i, feature in enumerate(show_f):
        if feature.size == 0:
            continue
        mean_row = np.mean(feature, axis=0)
        features_med.append(mean_row)
        youtube_id = features_files[i].split("\\")[-1].split("_")[0]
        youtube_ids.append(youtube_id)

        if csv_path:
            csv_df = pd.read_csv(csv_path)
            label_row = csv_df[csv_df['youtube_id'] == youtube_id]
            label = label_row['label'].values[0] if not label_row.empty else None
            labels.append(label)

    features_array = np.array(features_med)

    if scaler is None:
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_array.astype(float))
    else:
        features_scaled = scaler.transform(features_array.astype(float))

    umap_data_2d, tsne_data_2d, umap_data_3d, tsne_data_3d = None, None, None, None

    if apply_umap:
        if reducer_2d is None:
            n_neighbors = min(10, features_scaled.shape[0] - 1)
            reducer_2d = umap.UMAP(n_neighbors=n_neighbors, n_components=2)
            umap_data_2d = reducer_2d.fit_transform(features_scaled)
        else:
            umap_data_2d = reducer_2d.transform(features_scaled)

        if reducer_3d is None:
            reducer_3d = umap.UMAP(n_neighbors=n_neighbors, n_components=3)
            umap_data_3d = reducer_3d.fit_transform(features_scaled)
        else:
            umap_data_3d = reducer_3d.transform(features_scaled)

    if apply_tsne:
        tsne_2d = TSNE(n_components=2, random_state=42)
        tsne_data_2d = tsne_2d.fit_transform(features_scaled)

        tsne_3d = TSNE(n_components=3, random_state=42)
        tsne_data_3d = tsne_3d.fit_transform(features_scaled)

    youtube_ids_array = np.array(youtube_ids).reshape(-1, 1)
    labels_array = np.array(labels).reshape(-1, 1) if labels else None

    if labels_array is not None:
        return (
            np.hstack((youtube_ids_array, labels_array, umap_data_2d)),
            np.hstack((youtube_ids_array, labels_array, tsne_data_2d)),
            np.hstack((youtube_ids_array, labels_array, umap_data_3d)),
            np.hstack((youtube_ids_array, labels_array, tsne_data_3d)),
            np.hstack((youtube_ids_array, umap_data_2d)),
            np.hstack((youtube_ids_array, tsne_data_2d)),
            np.hstack((youtube_ids_array, umap_data_3d)),
            np.hstack((youtube_ids_array, tsne_data_3d)),
            scaler, reducer_2d, reducer_3d
        )
    else:
        return (
            np.hstack((youtube_ids_array, umap_data_2d)),
            np.hstack((youtube_ids_array, tsne_data_2d)),
            np.hstack((youtube_ids_array, umap_data_3d)),
            np.hstack((youtube_ids_array, tsne_data_3d)),
            scaler, reducer_2d, reducer_3d
        )

# Crear carpeta para almacenar los archivos
output_folder = "Reduction_data"
os.makedirs(output_folder, exist_ok=True)


# Procesar y guardar datos de entrenamiento
umap_train, tsne_train, umap_train_3d, tsne_train_3d, umap_train_L, tsne_train_L, umap_train_L_3d, tsne_train_L_3d, scaler, reducer_2d, reducer_3d = process_features(
    features_path="Feature_extraction\\train\\r21d\\r2plus1d_18_16_kinetics",
    csv_path="Data\\train_subset_10.csv"
)
np.save(os.path.join(output_folder, "npy_umap.npy"), umap_train)
np.save(os.path.join(output_folder, "npy_tsne.npy"), tsne_train)
np.save(os.path.join(output_folder, "npy_umap_3d.npy"), umap_train_3d)
np.save(os.path.join(output_folder, "npy_tsne_3d.npy"), tsne_train_3d)
np.save(os.path.join(output_folder, "npy_umapL.npy"), umap_train_L)
np.save(os.path.join(output_folder, "npy_tsneL.npy"), tsne_train_L)
np.save(os.path.join(output_folder, "npy_umapL_3d.npy"), umap_train_L_3d)
np.save(os.path.join(output_folder, "npy_tsneL_3d.npy"), tsne_train_L_3d)

# Guardar el scaler y el reducer de UMAP
with open(os.path.join(output_folder, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)
with open(os.path.join(output_folder, "umap_reducer_2d.pkl"), "wb") as f:
    pickle.dump(reducer_2d, f)
with open(os.path.join(output_folder, "umap_reducer_3d.pkl"), "wb") as f:
    pickle.dump(reducer_3d, f)

umap_test, tsne_test, umap_test_3d, tsne_test_3d, scaler, reducer_2d, reducer_3d = process_features(
    features_path="Feature_extraction\\test\\r21d\\r2plus1d_18_16_kinetics",
    scaler=scaler,
    reducer_2d=reducer_2d,
    reducer_3d=reducer_3d
)
np.save(os.path.join(output_folder, "npy_umap_test.npy"), umap_test)
np.save(os.path.join(output_folder, "npy_tsne_test.npy"), tsne_test)
np.save(os.path.join(output_folder, "npy_umap_test_3d.npy"), umap_test_3d)
np.save(os.path.join(output_folder, "npy_tsne_test_3d.npy"), tsne_test_3d)

