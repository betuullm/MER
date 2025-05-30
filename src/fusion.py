import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from scipy.stats import mode

# --- Verileri yükle ---
X_eeg = np.load('../data/eeg_signals.npy')
X_face = np.load('../data/face_images.npy')
y = np.load('../data/eeg_labels.npy')

etiket = 0  # 0: valence, 1: arousal
if y.ndim == 2:
    y = y[:, etiket]
print('Sınıf dağılımı:', np.bincount(y))

# --- Özellikleri ölçekle ve PCA uygula ---
scaler_eeg = StandardScaler()
X_eeg = scaler_eeg.fit_transform(X_eeg)
pca_eeg = PCA(n_components=40, random_state=42)
X_eeg = pca_eeg.fit_transform(X_eeg)

scaler_face = StandardScaler()
X_face = scaler_face.fit_transform(X_face)
pca_face = PCA(n_components=40, random_state=42)
X_face = pca_face.fit_transform(X_face)

# --- GEÇ FÜZYON (Late Fusion) ---
print('\n==== LATE FUSION (Karar Seviyesi) ====')
# Ayrı ayrı eğitilmiş modellerin test tahminleri
X_train_eeg, X_test_eeg, y_train_eeg, y_test_eeg = train_test_split(X_eeg, y, test_size=0.2, random_state=44, stratify=y)
X_train_face, X_test_face, y_train_face, y_test_face = train_test_split(X_face, y, test_size=0.2, random_state=44, stratify=y)

smote_eeg = SMOTE(random_state=42)
X_train_eeg_smote, y_train_eeg_smote = smote_eeg.fit_resample(X_train_eeg, y_train_eeg)
rf_eeg = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=10, random_state=44, class_weight="balanced")
rf_eeg.fit(X_train_eeg_smote, y_train_eeg_smote)
proba_eeg = rf_eeg.predict_proba(X_test_eeg)

def get_face_model():
    smote_face = SMOTE(random_state=42)
    X_train_face_smote, y_train_face_smote = smote_face.fit_resample(X_train_face, y_train_face)
    rf_face = RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_leaf=15, random_state=44, class_weight="balanced", min_impurity_decrease=0.002)
    rf_face.fit(X_train_face_smote, y_train_face_smote)
    return rf_face

rf_face = get_face_model()
proba_face = rf_face.predict_proba(X_test_face)

eeg_weight = 0.2
face_weight = 0.8
proba_fused = eeg_weight * proba_eeg + face_weight * proba_face
late_fusion_preds = np.argmax(proba_fused, axis=1)
late_fusion_acc = accuracy_score(y_test_eeg, late_fusion_preds)
print(f'Late Fusion Test accuracy: {late_fusion_acc:.3f}')
print('Late Fusion Classification Report:')
print(classification_report(y_test_eeg, late_fusion_preds))