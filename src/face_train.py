import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import cross_val_score


# --- Verileri yükle ---
X_face = np.load('../data/face_features.npy')  # (720, n_face_features)
y = np.load('../data/face_labels.npy')  # (720, 2)
info = np.load('../data/face_info.npy', allow_pickle=True)
# Sabit öznitelikleri kaldır
vt = VarianceThreshold(threshold=0.0)
X_face = vt.fit_transform(X_face)

# --- Öznitelik seçimi (SelectKBest) ---
K = 300  # Seçilecek öznitelik sayısı (isteğe göre değiştirilebilir)
selector = SelectKBest(score_func=f_classif, k=K)
X_face = selector.fit_transform(X_face, y)
print(f'Seçilen öznitelik sayısı: {X_face.shape[1]}')

# --- Hangi etikete göre sınıflandırma yapılacak? ---
print('Sınıf dağılımı:', np.bincount(y))

scaler_face = StandardScaler()
X_face = scaler_face.fit_transform(X_face)
pca_face = PCA(n_components=40, random_state=42)
X_face = pca_face.fit_transform(X_face)

# --- FACE PIPELINE (RandomForest) ---
print('\n==== FACE PIPELINE (RandomForest) ====')
X_train_face, X_test_face, y_train_face, y_test_face, info_train_face, info_test_face = train_test_split(
    X_face, y, info, test_size=0.2, random_state=44, stratify=y)
# SMOTE kaldırıldı, doğrudan orijinal eğitim seti kullanılacak
print(f'Face eğitim seti boyutu: {X_train_face.shape}')
print(f'Face eğitim seti sınıf dağılımı: {np.bincount(y_train_face)}')

# RandomForestClassifier yerine CalibratedClassifierCV ile sarmala
rf_face = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=10, min_samples_split=20, max_features='sqrt', random_state=43, class_weight="balanced")
calibrated_rf_face = CalibratedClassifierCV(rf_face, method='isotonic', cv=3)
calibrated_rf_face.fit(X_train_face, y_train_face)
train_preds_face = calibrated_rf_face.predict(X_train_face)
test_preds_face = calibrated_rf_face.predict(X_test_face)
train_acc_face = accuracy_score(y_train_face, train_preds_face)
test_acc_face = accuracy_score(y_test_face, test_preds_face)
print(f'Face RandomForest - Train accuracy: {train_acc_face:.3f}')
print(f'Face RandomForest - Test accuracy: {test_acc_face:.3f}')
print('Face Classification Report:')
print(classification_report(y_test_face, test_preds_face))

# 10-fold cross-validation ile doğruluk
cv_scores = cross_val_score(calibrated_rf_face, X_train_face, y_train_face, cv=10, scoring='accuracy')
print(f'10-fold CV doğruluk skorları: {cv_scores}')
print(f'10-fold CV ortalama doğruluk: {cv_scores.mean():.3f}')

# Test tahminlerini ve gerçek etiketleri kaydet (füzyon için)
np.save('../data/test_preds_face.npy', test_preds_face)
np.save('../data/y_test_face.npy', y_test_face)
# Olasılık bazlı füzyon için kalibre edilmiş olasılıkları kaydet
test_proba_face = calibrated_rf_face.predict_proba(X_test_face)
np.save('../data/test_proba_face.npy', test_proba_face)

# Eğitim doğruluğunu kaydet (füzyon için)
np.save('../data/train_acc_face.npy', train_acc_face)

# Info dosyasını yükle
info = np.load('../data/face_info.npy', allow_pickle=True)

# train_test_split'e info'yu da ekle
X_train_face, X_test_face, y_train_face, y_test_face, info_train_face, info_test_face = train_test_split(
    X_face, y, info, test_size=0.2, random_state=44, stratify=y)

# Test info'yu kaydet (sıra kontrolü için)
np.save('../data/face_info_test.npy', info_test_face)