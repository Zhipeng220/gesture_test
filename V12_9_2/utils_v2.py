# utils_v2.py
import os
import sys
import time
import logging
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d


# ... (Lookahead, setup_logger, set_plot_style, time_warping, time_scaling are unchanged) ...

class Lookahead(Optimizer):
    def __init__(self, optimizer, k=5, alpha=0.5):
        self.optimizer = optimizer;
        self.k = k;
        self.alpha = alpha;
        self.param_groups = self.optimizer.param_groups;
        self.state = defaultdict(dict)
        for group in self.param_groups: group["lookahead_step"] = 0

    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        for group in self.param_groups:
            if group["lookahead_step"] % self.k == 0:
                for p in group["params"]:
                    param_state = self.state[p]
                    if "slow_buffer" not in param_state: param_state["slow_buffer"] = torch.clone(p.data).detach()
                    slow = param_state["slow_buffer"];
                    slow.add_(p.data - slow, alpha=self.alpha);
                    p.data.copy_(slow)
            group["lookahead_step"] += 1
        return loss

    def zero_grad(self, set_to_none: bool = False):
        self.optimizer.zero_grad(set_to_none=set_to_none)


def setup_logger(log_dir="."):
    log_filepath = os.path.join(log_dir, f"log_{time.strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_filepath, encoding='utf-8'),
                                  logging.StreamHandler(sys.stdout)])
    logging.info(f"Logging to: {log_filepath}")


def set_plot_style():
    sns.set_theme(style="whitegrid");
    plt.rcParams['axes.unicode_minus'] = False


def time_warping(sequence: np.ndarray, sigma=0.25, num_knots=4):
    T, N, D = sequence.shape
    if T < 2: return sequence
    time_points = np.linspace(0, T - 1, T);
    knot_time_points = np.linspace(0, T - 1, num_knots)
    random_offsets = np.random.normal(loc=0.0, scale=sigma, size=(num_knots,)) * (T - 1)
    warp_values = knot_time_points + random_offsets
    spline = interp1d(knot_time_points, np.sort(warp_values), kind='cubic', fill_value="extrapolate")
    warped_time_points = np.clip(spline(time_points), 0, T - 1)
    warped_sequence = np.zeros_like(sequence)
    for i in range(N):
        for j in range(D):
            interp_func = interp1d(time_points, sequence[:, i, j], kind='linear', fill_value="extrapolate")
            warped_sequence[:, i, j] = interp_func(warped_time_points)
    return warped_sequence


def time_scaling(sequence: np.ndarray, scale_factor: float) -> np.ndarray:
    if scale_factor == 1.0:
        return sequence
    T, N, D = sequence.shape
    if T < 2: return sequence
    original_time = np.linspace(0, T - 1, T)
    new_time = np.linspace(0, (T - 1) / scale_factor, T)
    resampled_sequence = np.zeros((T, N, D))
    for i in range(N):
        for j in range(D):
            interp_func = interp1d(original_time, sequence[:, i, j], kind='linear', fill_value="extrapolate")
            resampled_sequence[:, i, j] = interp_func(new_time)
    return resampled_sequence


# MODIFIED: The mixup function is fixed to handle both data and skeletal features simultaneously
def mixup_data(x, y, alpha=1.0, use_skeletal=False, skeletal_features=None):
    """Applies mixup augmentation to the main data and optionally to skeletal features."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    mixed_skeletal = skeletal_features
    if use_skeletal and skeletal_features is not None:
        mixed_skeletal = lam * skeletal_features + (1 - lam) * skeletal_features[index, :]

    return mixed_x, mixed_skeletal, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam, **kwargs):
    return lam * criterion(pred, y_a, **kwargs) + (1 - lam) * criterion(pred, y_b, **kwargs)


# ... (apply_temporal_occlusion, apply_spatial_dropout, mirror_skeleton, base_augmentations, extract_skeletal_features, etc. are unchanged) ...
def apply_temporal_occlusion(x, prob=0.5, occlusion_ratio=0.3):
    if np.random.rand() < prob and x.shape[0] > 0:
        seq_len = x.shape[0];
        num_occluded_frames = int(seq_len * occlusion_ratio)
        if num_occluded_frames > 0:
            start_frame = np.random.randint(0, seq_len - num_occluded_frames + 1)
            x[start_frame:start_frame + num_occluded_frames] = 0
    return x


def apply_spatial_dropout(x, prob=0.5, dropout_ratio=0.3):
    if np.random.rand() < prob and x.shape[1] > 0:
        num_keypoints = x.shape[1];
        num_dropped_keypoints = int(num_keypoints * dropout_ratio)
        if num_dropped_keypoints > 0:
            dropped_indices = np.random.choice(num_keypoints, num_dropped_keypoints, replace=False)
            x[:, dropped_indices, :] = 0
    return x


def mirror_skeleton(sequence: torch.Tensor, axis='x'):
    mirrored_seq = sequence.clone()
    if axis == 'x':
        mirrored_seq[..., 0] = -mirrored_seq[..., 0]
    elif axis == 'y':
        mirrored_seq[..., 1] = -mirrored_seq[..., 1]
    return mirrored_seq


def base_augmentations(landmarks, rotation_angle=15.0, scale_range=0.15):
    landmarks = torch.from_numpy(time_warping(landmarks.numpy())).float()
    pos_data = landmarks[..., :3];
    center = torch.mean(pos_data, dim=1, keepdim=True)
    angle = np.random.uniform(-rotation_angle, rotation_angle) * (np.pi / 180.0)
    cos, sin = np.cos(angle), np.sin(angle)
    R = torch.eye(3, dtype=torch.float32, device=landmarks.device);
    R[0, 0] = cos;
    R[0, 1] = -sin;
    R[1, 0] = sin;
    R[1, 1] = cos
    rotated_pos = torch.einsum('tnc,cr->tnr', pos_data - center, R) + center
    if landmarks.shape[-1] > 3:
        landmarks = torch.cat([rotated_pos, landmarks[..., 3:]], dim=-1)
    else:
        landmarks = rotated_pos
    scaled = landmarks * np.random.uniform(1 - scale_range, 1 + scale_range)
    return scaled


def extract_skeletal_features(sequence: np.ndarray):
    T, N, D = sequence.shape
    if T < 1 or N != 21 or D < 3: return torch.zeros((T, 30), dtype=torch.float32)
    bone_connections = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11),
                        (11, 12), (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]
    angle_connections = [(1, 2, 3), (2, 3, 4), (5, 6, 7), (6, 7, 8), (9, 10, 11), (10, 11, 12), (13, 14, 15),
                         (14, 15, 16), (17, 18, 19), (18, 19, 20)]
    bone_lengths = np.linalg.norm(
        sequence[:, [p[1] for p in bone_connections], :3] - sequence[:, [p[0] for p in bone_connections], :3], axis=-1)
    angles = []
    for p0, p1, p2 in angle_connections:
        v1, v2 = sequence[:, p0, :3] - sequence[:, p1, :3], sequence[:, p2, :3] - sequence[:, p1, :3]
        cos_theta = np.sum(v1 * v2, axis=-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-8)
        angles.append(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    angles = np.stack(angles, axis=1)
    return torch.from_numpy(np.concatenate([bone_lengths, angles], axis=1).astype(np.float32))


def apply_spatial_noise(landmarks, noise_level=0.005):
    """Adds Gaussian noise to the keypoint positions."""
    noise = torch.randn_like(landmarks[..., :3]) * noise_level
    landmarks_with_noise = landmarks.clone()
    landmarks_with_noise[..., :3] += noise
    return landmarks_with_noise


def apply_temporal_jitter(landmarks, drop_rate=0.1):
    """Randomly drops a percentage of frames and pads back to original length."""
    if np.random.rand() < 0.5:
        return landmarks

    original_len = landmarks.shape[0]
    if original_len == 0:
        return landmarks

    keep_prob = 1.0 - drop_rate
    mask = torch.rand(original_len) < keep_prob
    jittered = landmarks[mask, ...]

    current_len = jittered.shape[0]
    if current_len == 0:
        return torch.zeros_like(landmarks)

    if current_len < original_len:
        pad_len = original_len - current_len
        padding = jittered[-1:, ...].repeat(pad_len, 1, 1)
        jittered = torch.cat([jittered, padding], dim=0)

    return jittered[:original_len]


def master_augmentations(landmarks):
    augmented_landmarks = base_augmentations(landmarks)

    if np.random.rand() < 0.5:
        augmented_landmarks = mirror_skeleton(augmented_landmarks)

    augmented_landmarks = apply_spatial_noise(augmented_landmarks)
    augmented_landmarks = apply_temporal_jitter(augmented_landmarks)

    np_landmarks = augmented_landmarks.numpy()
    np_landmarks = apply_temporal_occlusion(np_landmarks, prob=0.5, occlusion_ratio=0.3)
    np_landmarks = apply_spatial_dropout(np_landmarks, prob=0.5, dropout_ratio=0.2)
    return torch.from_numpy(np_landmarks)


class StandardGestureDataset(Dataset):
    # ... (class is unchanged)
    def __init__(self, sequences, labels, seq_len=32, augment=False, use_skeletal_features=False,
                 skeletal_patch_size=4, is_pretraining=False):
        self.sequences, self.labels, self.seq_len, self.augment = sequences, labels, seq_len, augment
        self.use_skeletal_features, self.skeletal_patch_size = use_skeletal_features, skeletal_patch_size
        self.is_pretraining = is_pretraining

    def __len__(self):
        return len(self.labels)

    def _process_sequence(self, seq_raw):
        """Helper function to handle padding, truncation, and augmentation."""
        L = len(seq_raw)
        mask = torch.zeros(self.seq_len, dtype=torch.bool)
        if L >= self.seq_len:
            start = np.random.randint(0, L - self.seq_len + 1) if self.augment and L > self.seq_len else 0
            seq = seq_raw[start:start + self.seq_len]
        else:
            mask[L:] = True
            seq = F.pad(seq_raw, (0, 0, 0, 0, 0, self.seq_len - L))

        if self.augment:
            seq = master_augmentations(seq.clone())

        if self.use_skeletal_features:
            skeletal_seq = extract_skeletal_features(seq.numpy())
            if skeletal_seq.shape[0] < self.seq_len:
                skeletal_seq = F.pad(skeletal_seq, (0, 0, 0, self.seq_len - skeletal_seq.shape[0]))
            return seq, skeletal_seq, mask

        return seq, mask

    def __getitem__(self, idx):
        seq_raw, label = self.sequences[idx].clone(), self.labels[idx]

        processed_data = self._process_sequence(seq_raw)

        if self.use_skeletal_features:
            seq, skeletal_seq, mask = processed_data
            return seq, skeletal_seq, torch.tensor(label, dtype=torch.long), mask
        else:
            seq, mask = processed_data
            return seq, torch.tensor(label, dtype=torch.long), mask


class AdvancedGestureDataset(StandardGestureDataset):
    # ... (__getitem__ is unchanged)
    def __getitem__(self, idx):
        if self.is_pretraining:
            seq_raw = self.sequences[idx].clone()
            view1_data = self._process_sequence(seq_raw)
            view2_data = self._process_sequence(seq_raw)
            return view1_data, view2_data
        else:
            return super().__getitem__(idx)

    @staticmethod
    def collate_fn(batch, mixup_alpha=0.4):
        # ... (pre-training logic is unchanged)
        if not batch:
            return None

        if isinstance(batch[0][0], tuple):
            view1_batch, view2_batch = zip(*batch)

            use_skeletal = len(view1_batch[0]) == 3
            if use_skeletal:
                seq1, skel1, mask1 = zip(*view1_batch)
                seq2, skel2, mask2 = zip(*view2_batch)
                return torch.stack(seq1), torch.stack(skel1), torch.stack(mask1), \
                    torch.stack(seq2), torch.stack(skel2), torch.stack(mask2)
            else:
                seq1, mask1 = zip(*view1_batch)
                seq2, mask2 = zip(*view2_batch)
                return torch.stack(seq1), torch.stack(mask1), torch.stack(seq2), torch.stack(mask2)

        use_skeletal = len(batch[0]) == 4
        if use_skeletal:
            sequences, skeletal_seqs, labels, masks = zip(*batch)
            skeletal_seqs = torch.stack(skeletal_seqs)
        else:
            sequences, labels, masks = zip(*batch)
            skeletal_seqs = None

        sequences, labels, masks = torch.stack(sequences), torch.tensor(labels, dtype=torch.long), torch.stack(masks)

        # MODIFIED: Call mixup_data only once to get consistent mixing
        if sequences.size(0) > 1 and np.random.rand() < 0.5 and mixup_alpha > 0:
            mixed_seqs, mixed_skeletal, y_a, y_b, lam = mixup_data(
                sequences, labels, alpha=mixup_alpha,
                use_skeletal=use_skeletal, skeletal_features=skeletal_seqs
            )

            if use_skeletal:
                return mixed_seqs, mixed_skeletal, y_a, y_b, torch.tensor(lam, dtype=torch.float32), masks
            else:
                return mixed_seqs, None, y_a, y_b, torch.tensor(lam, dtype=torch.float32), masks

        # Return format must be consistent
        if use_skeletal:
            return sequences, skeletal_seqs, labels, labels, torch.tensor(1.0, dtype=torch.float32), masks
        else:
            return sequences, None, labels, labels, torch.tensor(1.0, dtype=torch.float32), masks


# ... (normalize_sequence, calculate_kinematics, get_processed_sequence, _process_dataframe_to_sequences, load_and_preprocess_sequential_data, log_to_file, plot_training_history, and plot_confusion_matrices are unchanged)
def normalize_sequence(sequence: np.ndarray):
    if sequence.shape[0] < 1: return sequence
    centered = sequence - sequence[0:1, 0:1, :]
    ref_indices = [(0, 1), (0, 5), (0, 9), (0, 13), (0, 17), (5, 6), (6, 7), (7, 8)]
    all_lengths = [np.linalg.norm(centered[t, end, :3] - centered[t, start, :3]) for t in range(sequence.shape[0]) for
                   start, end in ref_indices if end < sequence.shape[1]]
    valid_lengths = [l for l in all_lengths if l > 1e-6]
    if not valid_lengths: return centered
    scale = np.median(valid_lengths)
    return centered / scale if scale > 1e-6 else centered


def calculate_kinematics(pos_seq: np.ndarray):
    if pos_seq.shape[0] < 2: return np.concatenate([pos_seq, np.zeros_like(pos_seq), np.zeros_like(pos_seq)], axis=-1)
    vel_seq, acc_seq = np.gradient(pos_seq, axis=0), np.gradient(np.gradient(pos_seq, axis=0), axis=0)
    return np.concatenate([pos_seq, vel_seq, acc_seq], axis=-1)


def get_processed_sequence(pos_3d: np.ndarray) -> torch.Tensor:
    if pos_3d.shape[0] < 1:
        return torch.zeros((0, 21, 9), dtype=torch.float32)
    normalized_pos = normalize_sequence(pos_3d)
    kinematic_seq = calculate_kinematics(normalized_pos)
    if kinematic_seq.shape[1] != 21 or kinematic_seq.shape[2] != 9:
        return torch.zeros((0, 21, 9), dtype=torch.float32)
    return torch.from_numpy(kinematic_seq.astype(np.float32))


def _process_dataframe_to_sequences(df, le):
    coord_cols = [c for c in df.columns if c.startswith(('x_', 'y_'))]
    df['label_encoded'] = le.transform(df['gesture_id'])
    df['sid'] = df.apply(lambda r: f"{r['gesture_id']}_{str(r['subject_id'])}_{r['essai_id']}", axis=1)
    sequences, labels, groups = [], [], []
    for _, group in df.groupby('sid'):
        coords = group[coord_cols].values.reshape(len(group), -1, 2)[:, :21, :]
        if coords.shape[1] != 21: continue
        pos_3d = np.pad(coords, ((0, 0), (0, 0), (0, 1)))
        processed_seq = get_processed_sequence(pos_3d)
        if processed_seq.shape[0] > 0:
            sequences.append(processed_seq)
            labels.append(group['label_encoded'].iloc[0])
            groups.append(str(group['subject_id'].iloc[0]))
    logging.info(f"Processed {len(sequences)} valid sequences.")
    return sequences, labels, groups


def load_and_preprocess_sequential_data(train_csv_path, test_csv_path, **kwargs):
    logging.info("--- Loading and Combining Data from All Sources ---")
    train_df = pd.read_csv(train_csv_path, header=0, low_memory=False).fillna(0.0);
    test_df = pd.read_csv(test_csv_path, header=0, low_memory=False).fillna(0.0)
    combined_df = pd.concat([train_df, test_df], ignore_index=True);
    logging.info(f"Combined data shape: {combined_df.shape}")
    le = LabelEncoder().fit(combined_df['gesture_id']);
    num_classes = len(le.classes_);
    logging.info(f"Found {num_classes} unique classes.")
    all_sequences, all_labels, all_groups = _process_dataframe_to_sequences(combined_df, le)
    class_weights = compute_class_weight('balanced', classes=np.unique(all_labels), y=np.array(all_labels));
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    logging.info("Calculated class weights for Focal Loss.")
    logging.info("Creating a hold-out test set based on subject IDs...")
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_val_idx, test_idx = next(splitter.split(all_sequences, all_labels, all_groups))
    X_train_val = [all_sequences[i] for i in train_val_idx];
    y_train_val = [all_labels[i] for i in train_val_idx];
    groups_train_val = [all_groups[i] for i in train_val_idx]
    X_test = [all_sequences[i] for i in test_idx];
    y_test = [all_labels[i] for i in test_idx]
    accepted_keys = ['seq_len', 'use_skeletal_features', 'skeletal_patch_size']
    dataset_kwargs = {key: kwargs[key] for key in accepted_keys if key in kwargs}
    test_ds = StandardGestureDataset(X_test, y_test, **dataset_kwargs, augment=False)
    logging.info(f"Data split: {len(X_train_val)} for Train/Val (CV), {len(test_ds)} for Final Test.")
    return (X_train_val, y_train_val, groups_train_val), test_ds, le, class_weights


def log_to_file(filepath, header, content):
    with open(filepath, 'a', encoding='utf-8') as f: f.write(
        f"\n{'=' * 80}\n{header.center(80)}\n{'=' * 80}\n{content}\n")


def plot_training_history(history, name, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(18, 6));
    epochs = range(1, len(history.get('val_acc', [])) + 1)
    if not epochs: return
    axes[0].plot(epochs, [np.mean(history['train_loss'][max(0, i - 5):i + 1]) for i in range(len(epochs))], 'bo-',
                 label='Training Loss (Smoothed)')
    axes[0].plot(epochs, history['val_loss'], 'go-', label='Validation Loss')
    axes[0].set_title(f'Model Loss - {name}');
    axes[0].set_xlabel('Epoch');
    axes[0].set_ylabel('Loss');
    axes[0].legend()
    if history.get('train_acc'): axes[1].plot(epochs, history['train_acc'], 'ro-', label='Training Accuracy')
    axes[1].plot(epochs, history['val_acc'], 'go-', label='Validation Accuracy');
    axes[1].plot(epochs, history['val_f1'], 'yo-', label='Validation F1 Score')
    axes[1].set_title(f'Model Metrics - {name}');
    axes[1].set_xlabel('Epoch');
    axes[1].set_ylabel('Metric');
    axes[1].legend()
    plt.tight_layout();
    plt.savefig(save_path);
    plt.close()


def plot_confusion_matrices(model, test_ds, le, device, save_path, use_arcface=False, batch_size=64):
    model.eval()
    all_preds, all_labels = [], []

    safe_batch_size = min(batch_size, 64)
    test_loader = DataLoader(test_ds, batch_size=safe_batch_size, shuffle=False)

    with torch.no_grad():
        for batch in test_loader:
            use_skeletal = len(batch) == 4
            if use_skeletal:
                seq, skeletal_features, labels, mask = [d.to(device) for d in batch]
            else:
                seq, labels, mask = [d.to(device) for d in batch]
                skeletal_features = None
            outputs = model(seq, skeletal_features=skeletal_features, key_padding_mask=mask,
                            labels=labels.to(device) if use_arcface else None)
            all_preds.append(outputs['logits'].argmax(1).cpu())
            all_labels.append(labels.cpu())
    all_preds, all_labels = torch.cat(all_preds), torch.cat(all_labels)
    target_names_str = [str(cls) for cls in le.classes_]
    cm = confusion_matrix(all_labels, all_preds, labels=range(len(le.classes_)))
    plt.figure(figsize=(14, 11));
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names_str, yticklabels=target_names_str)
    plt.title('Confusion Matrix', fontsize=16);
    plt.ylabel('True Label');
    plt.xlabel('Predicted Label')
    plt.tight_layout();
    plt.savefig(save_path, dpi=300);
    plt.close()
    logging.info(f"Confusion matrix plot saved to: {save_path}")
    return classification_report(all_labels, all_preds, target_names=target_names_str, zero_division=0)