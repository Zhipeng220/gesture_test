# train_v2.py
import os
import sys
import logging
from collections import defaultdict
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.amp import GradScaler, autocast
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
import numpy as np
from tqdm import tqdm
import copy
from functools import partial
from torch.nn import CrossEntropyLoss

from model_v2 import MODEL_REGISTRY
from utils_v2 import (load_and_preprocess_sequential_data, log_to_file,
                      plot_training_history, plot_confusion_matrices,
                      mixup_criterion, Lookahead, StandardGestureDataset, AdvancedGestureDataset,
                      mirror_skeleton, extract_skeletal_features, time_scaling)
# MODIFIED: Import GraphSparsityLoss
from losses_v2 import FocalLoss, AnatomicalLoss, InfoNCELoss, DistillationLoss, GraphSparsityLoss


class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        self.decay = decay
        self.source_model = model
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    def update(self):
        with torch.no_grad():
            for name, param in self.source_model.state_dict().items():
                if name in self.ema_model.state_dict():
                    ema_param = self.ema_model.state_dict()[name]
                    if ema_param.shape == param.shape:
                        if param.requires_grad:
                            ema_param.copy_(self.decay * ema_param + (1.0 - self.decay) * param)
                        else:
                            ema_param.copy_(param)

    def get_model_for_saving(self):
        return self.ema_model.state_dict()


class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0, path='checkpoint.pt', monitor='val_metric',
                 trace_func=logging.info):
        self.patience, self.verbose, self.delta, self.path, self.monitor, self.trace_func = patience, verbose, delta, path, monitor, trace_func
        self.counter, self.best_score, self.early_stop = 0, None, False

    def __call__(self, val_metric, model_state):
        score = val_metric
        if self.best_score is None or score > self.best_score + self.delta:
            if self.verbose: self.trace_func(
                f'Validation metric improved ({self.best_score or -np.inf:.6f} --> {val_metric:.6f}). Saving model...')
            torch.save(model_state, self.path)
            self.best_score, self.counter = score, 0
        else:
            self.counter += 1
            if self.verbose: self.trace_func(f'EarlyStopping counter: {self.counter} / {self.patience}')
            if self.counter >= self.patience: self.early_stop = True


def create_optimizer_with_warmup(model, config):
    if config.get('optimizer_type', 'adamw_custom') == 'adamw_custom':
        lr = config.get('learning_rate', 1e-4)
        wd = config.get('weight_decay', 1e-2)
        classifier_mult = config.get('classifier_lr_mult', 1.0)
        decay_params, no_decay_params, classifier_params = [], [], []
        for name, param in model.named_parameters():
            if not param.requires_grad: continue
            if 'classifier' in name:
                classifier_params.append(param)
            elif len(param.shape) == 1 or name.endswith(".bias") or 'norm' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        param_groups = [
            {'params': decay_params, 'weight_decay': wd, 'lr': lr},
            {'params': no_decay_params, 'weight_decay': 0.0, 'lr': lr},
            {'params': classifier_params, 'weight_decay': wd, 'lr': lr * classifier_mult}
        ]
        optimizer = AdamW(param_groups, lr=lr, betas=(0.9, 0.98), eps=1e-8)

        warmup_epochs = config.get('warmup_epochs', 0)
        warmup_scheduler = None
        if warmup_epochs > 0:
            def warmup_lambda(current_epoch):
                return float(current_epoch + 1) / float(max(1, warmup_epochs)) if current_epoch < warmup_epochs else 1.0

            warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)

        t_max_value = config.get('epochs', 1) - warmup_epochs
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max(1, t_max_value), eta_min=1e-7)

        return optimizer, warmup_scheduler, cosine_scheduler
    else:
        logging.info("Using a simple AdamW optimizer without schedulers for ablation.")
        optimizer = AdamW(model.parameters(), lr=config.get('learning_rate', 1e-4))
        return optimizer, None, None


def train_one_experiment(experiment_name, config, datasets, device):
    logging.info(f"\n{'=' * 25} Running Experiment: {experiment_name} {'=' * 25}")
    train_ds, val_ds, le, class_weights = datasets
    model_params = config.get('model_params', {})

    ModelClass = MODEL_REGISTRY.get(model_params.get('model_class', 'DHSGNet_V4'))
    model = ModelClass(num_classes=len(le.classes_), **model_params).to(device)

    if 'pretrained_backbone_path' in model_params:
        logging.info(f"Loading pre-trained backbone from: {model_params['pretrained_backbone_path']}")
        model.load_state_dict(torch.load(model_params['pretrained_backbone_path'], map_location=device), strict=False)

    for module in model.modules():
        if isinstance(module, torch.nn.LSTM):
            module.flatten_parameters()

    ema_model = ModelEMA(model)

    if config.get('loss_function') == 'focal':
        base_crit = FocalLoss(alpha=class_weights.to(device), gamma=config.get('focal_gamma', 2.0))
    else:
        base_crit = CrossEntropyLoss(label_smoothing=config.get('label_smoothing', 0.0),
                                     weight=class_weights.to(device))

    criterion = base_crit
    if config.get('use_knowledge_distillation', False):
        logging.info("Knowledge Distillation is enabled.")
        teacher_model_paths = config.get('teacher_models', [])
        if not teacher_model_paths:
            raise ValueError("Knowledge distillation is enabled, but no teacher models were provided in config.")

        teacher_model = ModelClass(num_classes=len(le.classes_), **model_params).to(device)
        teacher_model.load_state_dict(torch.load(teacher_model_paths[0], map_location=device))
        teacher_model.eval()
        logging.info(f"Loaded teacher model from: {teacher_model_paths[0]}")

        criterion = DistillationLoss(
            base_criterion=base_crit, teacher_model=teacher_model,
            alpha=config.get('distillation_alpha', 0.3), temperature=config.get('distillation_temp', 2.0)
        )

    dataset_keys = ['seq_len', 'use_skeletal_features', 'skeletal_patch_size', 'is_pretraining']
    dataset_kwargs = {key: config[key] for key in dataset_keys if key in config}

    current_train_ds = AdvancedGestureDataset(**train_ds, augment=True, **dataset_kwargs)
    current_val_ds = StandardGestureDataset(**val_ds, augment=False, **dataset_kwargs)

    collate_with_mixup = partial(AdvancedGestureDataset.collate_fn, mixup_alpha=config.get('mixup_alpha', 0.0))

    train_loader = DataLoader(current_train_ds, batch_size=config['batch_size'], shuffle=True,
                              num_workers=min(os.cpu_count(), 8), pin_memory=True,
                              collate_fn=collate_with_mixup)
    val_loader = DataLoader(current_val_ds, batch_size=config['batch_size'] * 2, shuffle=False,
                            num_workers=min(os.cpu_count(), 8))

    optimizer, warmup_scheduler, cosine_scheduler = create_optimizer_with_warmup(model, config)
    scaler = GradScaler(enabled=device.type == 'cuda')
    model_save_path = f"results/best_model_{experiment_name}.pth"
    history = defaultdict(list)
    early_stopping = EarlyStopping(patience=config.get('early_stopping_patience', 40), verbose=True,
                                   path=model_save_path)

    # NEW: Instantiate sparsity loss criterion if needed
    sparsity_criterion = GraphSparsityLoss()

    for epoch in range(config['epochs']):
        model.train()
        epoch_losses = defaultdict(float)
        for data in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=scaler.is_enabled()):
                inputs, skeletal, y_a, y_b, lam, mask = [d.to(device) for d in data]
                outputs = model(inputs, skeletal_features=skeletal, key_padding_mask=mask, labels=y_a)

                if config.get('use_knowledge_distillation', False):
                    class_loss = criterion(inputs, skeletal, mask, outputs['logits'], y_a)
                else:
                    class_loss = mixup_criterion(criterion, outputs['logits'], y_a, y_b, lam)

                total_loss = class_loss
                epoch_losses['class_loss'] += class_loss.item()

                if config.get('anatomical_loss_weight', 0.0) > 0:
                    anatomical_loss = config['anatomical_loss_weight'] * \
                                      AnatomicalLoss(config={'device': device})(outputs['anatomical_prediction'])[
                                          'total']
                    total_loss += anatomical_loss
                    epoch_losses['anatomical_loss'] += anatomical_loss.item()

                # --- NEW: Sparsity Loss Calculation ---
                sparsity_weight = config.get('sparsity_loss_weight', 0.0)
                if model_params.get('use_dynamic_graph', False) and sparsity_weight > 0:
                    spatial_graphs = outputs.get('spatial_attention_weights')
                    if spatial_graphs:
                        sparsity_loss = sparsity_weight * sparsity_criterion(spatial_graphs)
                        total_loss += sparsity_loss
                        epoch_losses['sparsity_loss'] += sparsity_loss.item()
                # --- End of New Block ---

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            ema_model.update()
            epoch_losses['total'] += total_loss.item()

        if warmup_scheduler and epoch < config.get('warmup_epochs', 0):
            warmup_scheduler.step()
        elif cosine_scheduler:
            cosine_scheduler.step()

        model.eval()
        val_loss, all_val_preds, all_val_labels = 0.0, [], []
        with torch.no_grad():
            for batch in val_loader:
                seq, skeletal_seq, labels, mask = [d.to(device) for d in batch]
                outputs = ema_model.ema_model(seq, skeletal_features=skeletal_seq, key_padding_mask=mask, labels=labels)
                val_loss += base_crit(outputs['logits'], labels).item()
                all_val_preds.append(outputs['logits'].argmax(1).cpu())
                all_val_labels.append(labels.cpu())

        val_acc = accuracy_score(torch.cat(all_val_labels), torch.cat(all_val_preds))
        val_f1 = f1_score(torch.cat(all_val_labels), torch.cat(all_val_preds), average='macro', zero_division=0)

        # MODIFIED: Corrected history update logic to append instead of overwrite
        history['train_loss'].append(epoch_losses['total'] / len(train_loader))
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)

        logging.info(
            f"Epoch {epoch + 1}/{config['epochs']} | LR: {optimizer.param_groups[0]['lr']:.2e} | Train Loss: {history['train_loss'][-1]:.4f} | Val Loss: {history['val_loss'][-1]:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        early_stopping(val_f1, ema_model.get_model_for_saving())
        if early_stopping.early_stop:
            logging.info("Early stopping triggered.")
            break

    logging.info(f"Loading best performing model from: {model_save_path}")
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    return model, history


def pretrain_one_experiment(experiment_name, config, datasets, device):
    logging.info(f"\n{'=' * 25} Running Pre-training: {experiment_name} {'=' * 25}")
    train_ds, _, le, _ = datasets
    model_params = config.get('model_params', {})
    dataset_keys = ['seq_len', 'use_skeletal_features', 'skeletal_patch_size']
    dataset_kwargs = {key: config[key] for key in dataset_keys if key in config}

    pretrain_dataset = AdvancedGestureDataset(**train_ds, augment=True, is_pretraining=True, **dataset_kwargs)
    pretrain_collate_fn = partial(AdvancedGestureDataset.collate_fn, mixup_alpha=0.0)
    pretrain_loader = DataLoader(
        pretrain_dataset, batch_size=config['batch_size'], shuffle=True,
        num_workers=min(os.cpu_count(), 8), collate_fn=pretrain_collate_fn
    )

    ModelClass = MODEL_REGISTRY.get(model_params.get('model_class', 'DHSGNet_V4'))
    model = ModelClass(num_classes=len(le.classes_), **model_params).to(device)

    criterion = InfoNCELoss(temperature=config.get('contrastive_temperature', 0.07))
    optimizer, _, cosine_scheduler = create_optimizer_with_warmup(model, config)
    model_save_path = f"results/pretrained_backbone_{experiment_name}.pth"
    scaler = GradScaler(enabled=device.type == 'cuda')

    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        for data_batch in tqdm(pretrain_loader, desc=f"Pre-training Epoch {epoch + 1}"):
            view1, skel1, mask1, view2, skel2, mask2 = [d.to(device) for d in data_batch]
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=scaler.is_enabled()):
                features1 = model(view1, skeletal_features=skel1, key_padding_mask=mask1)['projected_features']
                features2 = model(view2, skeletal_features=skel2, key_padding_mask=mask2)['projected_features']
                features = torch.cat([features1, features2], dim=0)
                loss = criterion(features)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        if cosine_scheduler:
            cosine_scheduler.step()

        avg_loss = total_loss / len(pretrain_loader)
        logging.info(f"Pre-training Epoch {epoch + 1}/{config['epochs']} | Loss: {avg_loss:.4f}")

    logging.info(f"Saving pre-trained backbone to {model_save_path}")
    torch.save(model.state_dict(), model_save_path)
    return model


def run_all_experiments(experiments, base_config, common_params, file_paths, device):
    (X_train_val, y_train_val, groups_train_val), test_ds, le, class_weights = load_and_preprocess_sequential_data(
        train_csv_path=file_paths["train_csv_path"], test_csv_path=file_paths["test_csv_path"],
        **common_params
    )
    results_dir = os.path.dirname(file_paths.get("summary_log_path", "results/summary.txt"))
    os.makedirs(results_dir, exist_ok=True)
    summary_log_path = file_paths["summary_log_path"]
    with open(summary_log_path, 'w', encoding='utf-8') as f:
        f.write("Experiment Results Summary\n" + "=" * 80 + "\n")

    for name, exp_config in sorted(experiments.items()):
        full_config = {**base_config, **exp_config, **common_params}
        mode = full_config.get('mode', 'supervised')

        if mode == 'pretrain':
            train_data = {'sequences': X_train_val, 'labels': y_train_val}
            pretrain_one_experiment(name, full_config, (train_data, None, le, None), device)
            continue

        logging.info("Creating a single train/validation split based on subject IDs...")
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
        train_idx, val_idx = next(splitter.split(X_train_val, y_train_val, groups=groups_train_val))

        y_train_val_np = np.array(y_train_val)
        train_data = {"sequences": [X_train_val[i] for i in train_idx], "labels": y_train_val_np[train_idx].tolist()}
        val_data = {"sequences": [X_train_val[i] for i in val_idx], "labels": y_train_val_np[val_idx].tolist()}
        datasets_single_split = (train_data, val_data, le, class_weights)

        final_model, history = train_one_experiment(name, full_config, datasets_single_split, device)
        plot_training_history(history, f"{name}_single_run", f"{results_dir}/training_history_{name}.png")

        logging.info(f"\n{'=' * 25} Starting Full Evaluation of Final Model: {name} (Single Model with TTA) {'=' * 25}")
        final_model.eval()

        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
        all_preds, all_labels = [], []
        seq_len = full_config['model_params']['seq_len']

        use_tta = full_config.get('use_tta', True)
        desc = "Evaluating with Single Model & TTA" if use_tta else "Evaluating with TTA DISABLED"

        if use_tta:
            scale_factors = [0.95, 1.0, 1.05];
            mirrors = [False, True]
        else:
            scale_factors = [1.0];
            mirrors = [False]

        with torch.no_grad():
            for data_batch in tqdm(test_loader, desc=desc):
                seq_raw_np, label = data_batch[0].squeeze(0).numpy(), data_batch[1].item()
                all_labels.append(label)
                tta_logits = []
                for scale_factor in scale_factors:
                    scaled_seq = torch.from_numpy(time_scaling(seq_raw_np, scale_factor)).float()
                    for mirror in mirrors:
                        mirrored_seq = mirror_skeleton(scaled_seq) if mirror else scaled_seq
                        total_len = mirrored_seq.shape[0]
                        if use_tta:
                            start_offsets = [-4, -2, 0, 2, 4] if total_len > seq_len + 8 else [0]
                        else:
                            start_offsets = [0]
                        for offset in start_offsets:
                            start_idx, end_idx = max(0, offset), min(total_len, seq_len + offset)
                            if end_idx - start_idx < seq_len:
                                seq_window_unpadded = mirrored_seq[start_idx:end_idx]
                                pad_len = seq_len - len(seq_window_unpadded)
                                seq_window = F.pad(seq_window_unpadded, (0, 0, 0, 0, 0, pad_len), "constant", 0)
                            else:
                                seq_window = mirrored_seq[start_idx:end_idx]
                            seq_window = seq_window.unsqueeze(0).to(device)
                            mask = torch.zeros(1, seq_len, dtype=torch.bool).to(device)
                            skeletal_window = extract_skeletal_features(seq_window.squeeze(0).cpu().numpy()).unsqueeze(
                                0).to(device)
                            outputs = final_model(seq_window, skeletal_features=skeletal_window, key_padding_mask=mask,
                                                  labels=None)
                            tta_logits.append(outputs['logits'].cpu())
                final_logits = torch.stack(tta_logits).mean(dim=0)
                all_preds.append(final_logits.argmax(1).item())

        final_acc = accuracy_score(all_labels, all_preds)
        f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        header = f"Final Model Performance Summary - {name}";
        content = f"  - Final Test Set Accuracy (Single Model + TTA): {final_acc:.4%}\n  - Final Test Set F1-Score (Macro, Single Model + TTA): {f1_macro:.4%}"
        log_to_file(summary_log_path, header, content);
        logging.info("\n" + header + "\n" + content)
        plot_confusion_matrices(final_model, test_ds, le, device, f"{results_dir}/confusion_matrix_{name}.png",
                                use_arcface=full_config['model_params'].get('use_arcface', False))

    logging.info("All experiments have completed successfully.")