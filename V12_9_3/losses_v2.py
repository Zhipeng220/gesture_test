# losses_v2.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List
import math


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)

        if isinstance(self.alpha, torch.Tensor):
            alpha_t = self.alpha.to(targets.device)[targets]
        else:
            alpha_t = self.alpha

        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class GraphSparsityLoss(nn.Module):
    """Encourages the learned graph attention to be sparse."""

    def __init__(self):
        super().__init__()

    def forward(self, attention_matrices: List[torch.Tensor]) -> torch.Tensor:
        if not attention_matrices or attention_matrices[0] is None:
            return torch.tensor(0.0, device=attention_matrices[0].device if attention_matrices else 'cpu')

        total_loss = 0.0
        num_matrices = 0
        for attn_matrix in attention_matrices:
            if attn_matrix is not None:
                total_loss += torch.mean(torch.abs(attn_matrix))
                num_matrices += 1

        return total_loss / num_matrices if num_matrices > 0 else torch.tensor(0.0)


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch_size = features.shape[0] // 2
        z1 = features[:batch_size]
        z2 = features[batch_size:]
        similarity_matrix = F.cosine_similarity(z1.unsqueeze(1), z2.unsqueeze(0), dim=2) / self.temperature
        labels = torch.arange(batch_size, device=features.device)
        return self.criterion(similarity_matrix, labels)


# NEW: Tier 3-H - Knowledge Distillation Loss
class DistillationLoss(nn.Module):
    """
    Implements Knowledge Distillation. This loss combines a standard student loss
    (like CrossEntropy or FocalLoss) with a distillation loss that encourages the
    student to match the softened outputs of a teacher model.
    """

    def __init__(self, base_criterion, teacher_model, alpha=0.3, temperature=2.0):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        self.alpha = alpha
        self.temperature = temperature
        self.kl_div_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_inputs, student_skeletal, student_mask, student_logits, labels):
        # 1. Calculate the standard supervised loss using the base criterion (e.g., FocalLoss)
        base_loss = self.base_criterion(student_logits, labels)

        # 2. Get the teacher's predictions (in no_grad mode)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                student_inputs,
                skeletal_features=student_skeletal,
                key_padding_mask=student_mask
            )
            teacher_logits = teacher_outputs['logits']

        # 3. Calculate the distillation loss (KL Divergence between softened logits)
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_predictions = F.log_softmax(student_logits / self.temperature, dim=-1)

        # The (temperature**2) scaling factor is a standard practice in distillation
        # to ensure the gradients from the soft targets are on a similar scale to the hard targets.
        distillation_loss = self.kl_div_loss(soft_predictions, soft_targets) * (self.temperature ** 2)

        # 4. Return the weighted average of the two losses
        return self.alpha * base_loss + (1.0 - self.alpha) * distillation_loss


def get_default_anatomical_config() -> Dict[str, Any]:
    config = {
        'loss_weights': {'angle': 1.0, 'length': 1.0, 'planarity': 1.0},
        'angle_limits': {'mcp': (-0.52, 1.66), 'pip': (0.0, 1.92), 'dip': (-0.17, 1.57)},
        'thumb_angle_limits': {'mcp': (0.0, 0.96), 'ip': (-0.26, 1.48)},
        'length_ratios': {'medial_proximal': (0.55, 0.75), 'distal_proximal': (0.35, 0.55)},
        'thumb_length_ratios': {'distal_proximal': (0.60, 0.85)},
        'planarity_confidence_threshold': 0.3,
        'loss_type': 'squared_hinge',
        'device': 'cpu',
    }
    return config


class AnatomicalLoss(nn.Module):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.config = get_default_anatomical_config()
        if config is not None:
            self.config.update(config)
        self.device = self.config.get('device', 'cpu')
        self.loss_weights = self.config.get('loss_weights', {'angle': 1.0, 'length': 1.0, 'planarity': 1.0})
        self.finger_indices = {
            'thumb': [1, 2, 3, 4], 'index': [5, 6, 7, 8], 'middle': [9, 10, 11, 12],
            'ring': [13, 14, 15, 16], 'pinky': [17, 18, 19, 20]
        }
        self.palm_indices = [0, 1, 5, 9, 13, 17]
        self.joint_to_angle_limit_key = {
            5: 'mcp', 6: 'pip', 7: 'dip', 9: 'mcp', 10: 'pip', 11: 'dip',
            13: 'mcp', 14: 'pip', 15: 'dip', 17: 'mcp', 18: 'pip', 19: 'dip',
            2: 'mcp', 3: 'ip',
        }

    def _apply_penalty(self, value: torch.Tensor) -> torch.Tensor:
        if self.config.get('loss_type', 'squared_hinge') == 'squared_hinge':
            return torch.relu(value) ** 2
        return torch.relu(value)

    def _calculate_angle_loss(self, hand_kpts: torch.Tensor, confidences: Optional[torch.Tensor]) -> torch.Tensor:
        loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        confidences = confidences if confidences is not None else torch.ones(hand_kpts.shape[0], 21, device=self.device)
        for finger, indices in self.finger_indices.items():
            limits = self.config['thumb_angle_limits'] if finger == 'thumb' else self.config['angle_limits']
            for i in range(len(indices) - 2):
                p0_idx, p1_idx, p2_idx = indices[i], indices[i + 1], indices[i + 2]
                p0, p1, p2 = hand_kpts[:, p0_idx], hand_kpts[:, p1_idx], hand_kpts[:, p2_idx]
                v1, v2 = p0 - p1, p2 - p1
                v1_norm = F.normalize(v1, p=2, dim=-1)
                v2_norm = F.normalize(v2, p=2, dim=-1)
                cos_theta = torch.sum(v1_norm * v2_norm, dim=1)
                cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
                angle = torch.acos(cos_theta)
                limit_key = self.joint_to_angle_limit_key.get(p1_idx)
                if limit_key and limit_key in limits:
                    min_angle, max_angle = limits[limit_key]
                    angle_loss = self._apply_penalty(angle - max_angle) + self._apply_penalty(min_angle - angle)
                    conf = torch.min(
                        torch.stack([confidences[:, p0_idx], confidences[:, p1_idx], confidences[:, p2_idx]]),
                        dim=0).values
                    loss += torch.mean(conf * angle_loss)
        return loss

    def _calculate_length_loss(self, hand_kpts: torch.Tensor, confidences: Optional[torch.Tensor]) -> torch.Tensor:
        loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        confidences = confidences if confidences is not None else torch.ones(hand_kpts.shape[0], 21, device=self.device)
        for finger, indices in self.finger_indices.items():
            if len(indices) < 4: continue
            ref_len = torch.norm(hand_kpts[:, indices[1]] - hand_kpts[:, indices[0]], dim=1) + 1e-6
            if finger == 'thumb':
                min_r, max_r = self.config['thumb_length_ratios']['distal_proximal']
                vec = hand_kpts[:, indices[3]] - hand_kpts[:, indices[2]]
                ratio = torch.norm(vec, dim=1) / ref_len
                ratio = torch.clamp(ratio, max=1000.0)
                length_loss = self._apply_penalty(ratio - max_r) + self._apply_penalty(min_r - ratio)
                conf = torch.min(confidences[:, [indices[1], indices[2], indices[3]]], dim=1).values
                loss += torch.mean(conf * length_loss)
            else:
                min_r_med, max_r_med = self.config['length_ratios']['medial_proximal']
                min_r_dist, max_r_dist = self.config['length_ratios']['distal_proximal']
                vec1 = hand_kpts[:, indices[2]] - hand_kpts[:, indices[1]]
                ratio1 = torch.norm(vec1, dim=1) / ref_len
                vec2 = hand_kpts[:, indices[3]] - hand_kpts[:, indices[2]]
                ratio2 = torch.norm(vec2, dim=1) / ref_len
                ratio1 = torch.clamp(ratio1, max=1000.0)
                ratio2 = torch.clamp(ratio2, max=1000.0)
                loss1 = self._apply_penalty(ratio1 - max_r_med) + self._apply_penalty(min_r_med - ratio1)
                loss2 = self._apply_penalty(ratio2 - max_r_dist) + self._apply_penalty(min_r_dist - ratio2)
                conf = torch.min(confidences[:, indices[:4]], dim=1).values
                loss += torch.mean(conf * (loss1 + loss2))
        return loss

    def _calculate_planarity_loss(self, hand_kpts: torch.Tensor, confidences: Optional[torch.Tensor]) -> torch.Tensor:
        # This function remains but is not called from forward() to ensure stability.
        confidences = confidences if confidences is not None else torch.ones(hand_kpts.shape[0], 21, device=self.device)
        palm_kpts = hand_kpts[:, self.palm_indices, :]
        palm_conf = confidences[:, self.palm_indices]
        valid_mask = torch.mean(palm_conf, dim=1) > self.config['planarity_confidence_threshold']
        if not torch.any(valid_mask):
            return torch.tensor(0.0, device=self.device)
        palm_kpts, palm_conf = palm_kpts[valid_mask], palm_conf[valid_mask]
        centroid = torch.sum(palm_kpts * palm_conf.unsqueeze(-1), dim=1, keepdim=True) / (
                torch.sum(palm_conf, dim=1, keepdim=True).unsqueeze(-1) + 1e-8)
        centered = palm_kpts - centroid
        try:
            C = torch.bmm((centered * palm_conf.unsqueeze(-1)).transpose(1, 2), centered)
            _, _, V = torch.linalg.svd(C.to(torch.float32))
            normal = V[:, :, -1]
        except torch.linalg.LinAlgError:
            return torch.tensor(0.0, device=self.device)
        dists_sq = torch.sum(centered * normal.unsqueeze(1), dim=2) ** 2
        return torch.mean(palm_conf * dists_sq)

    def forward(self, input_batch: torch.Tensor, disable_losses: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        disable_losses = disable_losses if disable_losses is not None else []
        if input_batch.dim() == 4:
            B, T, N, D = input_batch.shape
            flat_batch = input_batch.view(B * T, N, D)
        else:
            flat_batch = input_batch
        keypoints, confidences = (flat_batch[..., :3], flat_batch[..., 3]) if flat_batch.shape[-1] > 3 else (
            flat_batch, None)
        valid_mask = torch.sum(torch.abs(keypoints), dim=(1, 2)) > 1e-6
        if not torch.any(valid_mask):
            return {'total': torch.tensor(0.0, device=self.device, dtype=torch.float32),'angle': torch.tensor(0.0, device=self.device, dtype=torch.float32),'length': torch.tensor(0.0, device=self.device, dtype=torch.float32),'planarity': torch.tensor(0.0, device=self.device, dtype=torch.float32),}
        valid_kpts = keypoints[valid_mask]
        valid_confs = confidences[valid_mask] if confidences is not None else None
        loss_dict = {}
        if 'angle' not in disable_losses:
            loss_dict['angle'] = self._calculate_angle_loss(valid_kpts, valid_confs)
        if 'length' not in disable_losses:
            loss_dict['length'] = self._calculate_length_loss(valid_kpts, valid_confs)
        # Planarity loss remains disabled for stability.
        # if 'planarity' not in disable_losses:
        #     loss_dict['planarity'] = self._calculate_planarity_loss(valid_kpts, valid_confs)
        total_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        for loss_name, loss_val in loss_dict.items():
            total_loss += self.loss_weights.get(loss_name, 1.0) * loss_val
        loss_dict['total'] = total_loss
        loss_dict.setdefault('planarity', torch.tensor(0.0, device=self.device, dtype=torch.float32))
        return loss_dict