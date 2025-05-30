import os
import cv2
import numpy as np
import scipy.io
from sklearn.preprocessing import StandardScaler
import torch
import torchvision.transforms as transforms
import torchvision.models as models

FACE_ROOT = '../DEAP_Face_Images'
LABEL_ROOT = '../DEAP_Signals/Labels'
OUTPUT_PATH = '../data/face_features.npy'
LABELS_PATH = '../data/face_labels.npy'
INFO_PATH = '../data/face_info.npy'

IMG_SIZE = (224, 224)

labels = []
info = []
combined_features = []

# Pretrained VGG16 (VGG-Face alternatifi) yükle ve feature extractor olarak ayarla
vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
vgg.eval()
vgg = torch.nn.Sequential(*list(vgg.children())[:-1])  # Son fc katmanını çıkar

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

for subject in sorted(os.listdir(FACE_ROOT)):
    subject_path = os.path.join(FACE_ROOT, subject, 'kullanilacak_goruntuler')
    if not os.path.isdir(subject_path):
        continue
    label_file = os.path.join(LABEL_ROOT, f'{subject}_label.mat')
    if not os.path.exists(label_file):
        continue
    label_mat = scipy.io.loadmat(label_file)
    label_key = [k for k in label_mat.keys() if not k.startswith('__')][0]
    label_arr = label_mat[label_key]
    for class_folder in sorted(os.listdir(subject_path), key=lambda x: int(x)):
        class_path = os.path.join(subject_path, class_folder)
        if not os.path.isdir(class_path):
            continue
        bmp_files = sorted([f for f in os.listdir(class_path) if f.endswith('.bmp')])
        if bmp_files:
            img_file = bmp_files[-1]  # Son görüntüyü al
            img_path = os.path.join(class_path, img_file)
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Sadece VGG16 feature
            input_tensor = transform(img_rgb).unsqueeze(0)
            with torch.no_grad():
                vgg_feat = vgg(input_tensor).squeeze().flatten().numpy()
            combined_features.append(vgg_feat)
            class_idx = int(class_folder) - 1
            valence = 1 if label_arr[class_idx, 0] >= 5 else 0
            labels.append(valence)
            info.append((subject, int(class_folder)))

X = np.array(combined_features)
labels = np.array(labels)
scaler = StandardScaler()
X = scaler.fit_transform(X)

np.save(OUTPUT_PATH, X)
np.save(LABELS_PATH, labels)
np.save(INFO_PATH, np.array(info))
print(f'VGG öznitelik shape: {X.shape}, Etiketler shape: {labels.shape}')
print(f'Etiketlerde valence=1 sayısı: {np.sum(labels == 1)}, valence=0 sayısı: {np.sum(labels == 0)}')
print(f'Info örnek sayısı: {len(info)}')