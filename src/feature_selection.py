import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

# Yükle
features = np.load('../data/face_features.npy')
labels = np.load('../data/face_labels.npy')

# --- Sabit (constant) özellikleri kaldır ---
variances = features.var(axis=0)
nonconstant_idx = np.where(variances > 0)[0]
features = features[:, nonconstant_idx]
print(f'Sabit özellikler kaldırıldı. Yeni shape: {features.shape}')

# --- PCA ile boyut indirgeme ---
pca_components = min(200, features.shape[1])  # Özellik sayısından fazla bileşen seçilmesin
d = features.shape[1]
pca = PCA(n_components=pca_components, random_state=42)
features_pca = pca.fit_transform(features)
np.save('../data/face_features_pca.npy', features_pca)
print(f'PCA sonrası shape: {features_pca.shape}')

# --- SelectKBest ile en iyi K özelliği seçme ---
k_best = min(200, d)  # Özellik sayısından fazla seçilmesin
selector = SelectKBest(score_func=f_classif, k=k_best)
features_kbest = selector.fit_transform(features, labels)
np.save('../data/face_features_kbest.npy', features_kbest)
print(f'SelectKBest sonrası shape: {features_kbest.shape}')
