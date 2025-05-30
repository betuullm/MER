import os
import numpy as np
import scipy.io
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler
import scipy.signal
import scipy.stats

# Ana EEG sinyal klasörü
EEG_ROOT = '../DEAP_Signals/Signals'
LABEL_ROOT = '../DEAP_Signals/Labels'
OUTPUT_PATH = '../data/eeg_features.npy'
LABELS_PATH = '../data/eeg_labels.npy'

# Band power için frekans aralıkları (DEAP: 128 Hz, 8064 örnek, 63 sn)
BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}
FS = 128  # Sampling frequency

features = []
labels = []

for subject_file in sorted(os.listdir(EEG_ROOT)):
    if subject_file.endswith('.mat'):
        subject_id = subject_file.split('_')[0]  # s01, s02, ...
        data_path = os.path.join(EEG_ROOT, subject_file)
        label_path = os.path.join(LABEL_ROOT, f'{subject_id}_label.mat')
        mat = scipy.io.loadmat(data_path)
        label_mat = scipy.io.loadmat(label_path)
        X = mat[list(mat.keys())[-1]]  # (40, 8064, kanal)
        y = label_mat[list(label_mat.keys())[-1]]  # (40, 4)
        valence = (y[:, 0] >= 5).astype(int)
        for i, trial in enumerate(X):  # trial: (8064, kanal)
            trial_feats = []
            alpha_de = None
            F3_idx, F4_idx = 7, 8  # DEAP kanal sırası
            channel_stats = []  # Her kanal için istatistiksel özellikler
            hjorth_params = []  # Her kanal için Hjorth parametreleri
            band_power_filtered = []  # Her kanal için bandpass filtreli band power
            for ch in range(trial.shape[1]):
                signal = trial[:, ch]
                # --- Bandpass filtreli band power ---
                band_power_filt = []
                for band, (low, high) in BANDS.items():
                    b, a = scipy.signal.butter(4, [low/(FS/2), high/(FS/2)], btype='band')
                    filtered = scipy.signal.filtfilt(b, a, signal)
                    f, Pxx = welch(filtered, fs=FS, nperseg=FS*2)
                    idx = np.logical_and(f >= low, f <= high)
                    band_power = np.trapz(Pxx[idx], f[idx])
                    band_power_filt.append(band_power)
                band_power_filtered.extend(band_power_filt)
                # --- Mevcut band power (Welch doğrudan) ---
                f, Pxx = welch(signal, fs=FS, nperseg=FS*2)
                band_powers = []
                for band, (low, high) in BANDS.items():
                    idx = np.logical_and(f >= low, f <= high)
                    band_power = np.trapz(Pxx[idx], f[idx])
                    band_powers.append(band_power)
                trial_feats.extend(band_powers)
                # PSD (ortalama güç spektrumu)
                psd_mean = np.mean(Pxx)
                trial_feats.append(psd_mean)
                # Frontal asimetri için alpha bandı gücü
                if ch == F3_idx or ch == F4_idx:
                    if alpha_de is None:
                        alpha_de = {}
                    alpha_de[ch] = band_powers[2]  # alpha bandı index=2
                # --- Hjorth parametreleri ---
                activity = np.var(signal)
                diff1 = np.diff(signal)
                diff2 = np.diff(diff1)
                mobility = np.sqrt(np.var(diff1) / activity) if activity > 0 else 0
                complexity = (np.sqrt(np.var(diff2) / np.var(diff1)) / mobility) if mobility > 0 and np.var(diff1) > 0 else 0
                hjorth_params.extend([activity, mobility, complexity])
                # --- İstatistiksel özellikler ---
                mean = np.mean(signal)
                std = np.std(signal)
                amplitude = np.max(signal) - np.min(signal)
                skewness = scipy.stats.skew(signal)
                kurt = scipy.stats.kurtosis(signal)
                median = np.median(signal)
                iqr = np.percentile(signal, 75) - np.percentile(signal, 25)
                perc95 = np.percentile(signal, 95)
                channel_stats.extend([mean, std, amplitude, skewness, kurt, median, iqr, perc95])
            trial_feats.extend(band_power_filtered)
            trial_feats.extend(hjorth_params)
            trial_feats.extend(channel_stats)
            # --- Kanal çiftleri korelasyonları ---
            # Örnek: F3-F4, T7-T8, P7-P8 (DEAP kanal sırası: 7-8, 21-22, 25-26)
            channel_pairs = [(7,8), (21,22), (25,26)]
            for ch1, ch2 in channel_pairs:
                if ch1 < trial.shape[1] and ch2 < trial.shape[1]:
                    corr = np.corrcoef(trial[:,ch1], trial[:,ch2])[0,1]
                else:
                    corr = 0
                trial_feats.append(corr)
            # Frontal asimetri (alpha): F4_alpha - F3_alpha
            if alpha_de is not None and F3_idx in alpha_de and F4_idx in alpha_de:
                fa = alpha_de[F4_idx] - alpha_de[F3_idx]
            else:
                fa = 0
            trial_feats.append(fa)
            features.append(trial_feats)
            labels.append(valence[i])

features = np.array(features)
labels = np.array(labels)

# Öznitelik ölçekleme (standartlaştırma)
scaler = StandardScaler()
features = scaler.fit_transform(features)

np.save(OUTPUT_PATH, features)
np.save(LABELS_PATH, labels)
print(f'Öznitelik shape: {features.shape}, Etiketler shape: {labels.shape}')

features = []
labels = []
info = []

for subject_file in sorted(os.listdir(EEG_ROOT)):
    if subject_file.endswith('.mat'):
        subject_id = subject_file.split('_')[0]  # s01, s02, ...
        data_path = os.path.join(EEG_ROOT, subject_file)
        label_path = os.path.join(LABEL_ROOT, f'{subject_id}_label.mat')
        mat = scipy.io.loadmat(data_path)
        label_mat = scipy.io.loadmat(label_path)
        X = mat[list(mat.keys())[-1]]  # (40, 8064, kanal)
        y = label_mat[list(label_mat.keys())[-1]]  # (40, 4)
        valence = (y[:, 0] >= 5).astype(int)
        for trial_idx, trial in enumerate(X):  # trial: (8064, kanal)
            trial_feats = []
            alpha_de = None
            F3_idx, F4_idx = 7, 8  # DEAP kanal sırası
            channel_stats = []  # Her kanal için istatistiksel özellikler
            hjorth_params = []  # Her kanal için Hjorth parametreleri
            band_power_filtered = []  # Her kanal için bandpass filtreli band power
            for ch in range(trial.shape[1]):
                signal = trial[:, ch]
                # --- Bandpass filtreli band power ---
                band_power_filt = []
                for band, (low, high) in BANDS.items():
                    b, a = scipy.signal.butter(4, [low/(FS/2), high/(FS/2)], btype='band')
                    filtered = scipy.signal.filtfilt(b, a, signal)
                    f, Pxx = welch(filtered, fs=FS, nperseg=FS*2)
                    idx = np.logical_and(f >= low, f <= high)
                    band_power = np.trapz(Pxx[idx], f[idx])
                    band_power_filt.append(band_power)
                band_power_filtered.extend(band_power_filt)
                # --- Mevcut band power (Welch doğrudan) ---
                f, Pxx = welch(signal, fs=FS, nperseg=FS*2)
                band_powers = []
                for band, (low, high) in BANDS.items():
                    idx = np.logical_and(f >= low, f <= high)
                    band_power = np.trapz(Pxx[idx], f[idx])
                    band_powers.append(band_power)
                trial_feats.extend(band_powers)
                # PSD (ortalama güç spektrumu)
                psd_mean = np.mean(Pxx)
                trial_feats.append(psd_mean)
                # Frontal asimetri için alpha bandı gücü
                if ch == F3_idx or ch == F4_idx:
                    if alpha_de is None:
                        alpha_de = {}
                    alpha_de[ch] = band_powers[2]  # alpha bandı index=2
                # --- Hjorth parametreleri ---
                activity = np.var(signal)
                diff1 = np.diff(signal)
                diff2 = np.diff(diff1)
                mobility = np.sqrt(np.var(diff1) / activity) if activity > 0 else 0
                complexity = (np.sqrt(np.var(diff2) / np.var(diff1)) / mobility) if mobility > 0 and np.var(diff1) > 0 else 0
                hjorth_params.extend([activity, mobility, complexity])
                # --- İstatistiksel özellikler ---
                mean = np.mean(signal)
                std = np.std(signal)
                amplitude = np.max(signal) - np.min(signal)
                skewness = scipy.stats.skew(signal)
                kurt = scipy.stats.kurtosis(signal)
                median = np.median(signal)
                iqr = np.percentile(signal, 75) - np.percentile(signal, 25)
                perc95 = np.percentile(signal, 95)
                channel_stats.extend([mean, std, amplitude, skewness, kurt, median, iqr, perc95])
            trial_feats.extend(band_power_filtered)
            trial_feats.extend(hjorth_params)
            trial_feats.extend(channel_stats)
            # --- Kanal çiftleri korelasyonları ---
            # Örnek: F3-F4, T7-T8, P7-P8 (DEAP kanal sırası: 7-8, 21-22, 25-26)
            channel_pairs = [(7,8), (21,22), (25,26)]
            for ch1, ch2 in channel_pairs:
                if ch1 < trial.shape[1] and ch2 < trial.shape[1]:
                    corr = np.corrcoef(trial[:,ch1], trial[:,ch2])[0,1]
                else:
                    corr = 0
                trial_feats.append(corr)
            # Frontal asimetri (alpha): F4_alpha - F3_alpha
            if alpha_de is not None and F3_idx in alpha_de and F4_idx in alpha_de:
                fa = alpha_de[F4_idx] - alpha_de[F3_idx]
            else:
                fa = 0
            trial_feats.append(fa)
            features.append(trial_feats)
            labels.append(valence[trial_idx])
            info.append((subject_id, trial_idx + 1))  # (subject, trial)

np.save('../data/eeg_info.npy', np.array(info))
