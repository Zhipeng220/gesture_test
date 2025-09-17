# config_v2.py
import json


def get_experiments():
    """
    Defines and returns configurations for all experiments.
    """
    base_config = {
        'use_skeletal_features': True,
    }

    v15_model_params = {
        'model_class': 'DHSGNet_V4',
        'seq_len': 128,
        'num_layers': 8, 'num_heads': 16, 'embed_dim': 512, 'input_dim': 9,
        'projection_head_dims': [512, 256, 128],
        'num_gnn_layers': 3,
        'gnn_type': 'HGCN',
        'skeletal_patch_size': 4,
        # NEW: Add default parameters for the dynamic GNN feature
        'use_dynamic_graph': False,  # Disabled by default
        'top_k': 5,  # Default K for the graph learner
    }

    experiments = {
        # --- V16 and V17 Experiments from your V12_7 code ---
        "12_Exp_V16_Optimized": {
            'batch_size': 8, 'epochs': 200,
            'optimizer_type': 'adamw_custom', 'learning_rate': 5e-5,
            'warmup_epochs': 10,
            'classifier_lr_mult': 2.0, 'scheduler_type': 'cosine',
            'loss_function': 'focal', 'focal_gamma': 1.8, 'label_smoothing': 0.05, 'weight_decay': 1e-2,
            'anatomical_loss_weight': 0.2, 'mixup_alpha': 0.4,
            'early_stopping_patience': 40,
            'model_params': {
                'seq_len': 96, 'num_gnn_layers': 2, 'dropout': 0.35, 'stochastic_depth_rate': 0.2, 'use_arcface': False,
            }
        },
        "13_Exp_V17_Tier1_Improved": {
            'batch_size': 8, 'epochs': 200,
            'optimizer_type': 'adamw_custom', 'learning_rate': 5e-5,
            'warmup_epochs': 10,
            'classifier_lr_mult': 2.0, 'scheduler_type': 'cosine',
            'loss_function': 'focal', 'focal_gamma': 1.8, 'label_smoothing': 0.05, 'weight_decay': 1e-2,
            'anatomical_loss_weight': 0.2, 'mixup_alpha': 0.4,
            'early_stopping_patience': 40,
            'model_params': {
                'gnn_residual': True, 'dropout': 0.25, 'attention_dropout': 0.1, 'stochastic_depth_rate': 0.15,
                'use_arcface': False,
            }
        },
        "14_Exp_V18_Tier2_Advanced": {
            'batch_size': 8, 'epochs': 200,
            'optimizer_type': 'adamw_custom', 'learning_rate': 5e-5,
            'warmup_epochs': 10,
            'classifier_lr_mult': 2.0, 'scheduler_type': 'cosine',
            'loss_function': 'focal', 'focal_gamma': 1.8, 'label_smoothing': 0.05, 'weight_decay': 1e-2,
            'anatomical_loss_weight': 0.2, 'mixup_alpha': 0.4,
            'early_stopping_patience': 40,
            'model_params': {
                'gnn_residual': True, 'dropout': 0.25, 'attention_dropout': 0.1, 'stochastic_depth_rate': 0.15,
                'use_temporal_conv': True, 'tcn_kernel_size': 3, 'dilations': [1, 2, 4],
                'use_cross_modal_attention': True, 'adaptive_depth': True, 'layer_selection_threshold': 0.1,
                'use_arcface': False,
            }
        },

        # --- NEW: V19 Experiment to test Dynamic GNN based on the best V17 model ---
        "19_Exp_V17_With_DynamicGNN": {
            'batch_size': 8, 'epochs': 200,
            'optimizer_type': 'adamw_custom', 'learning_rate': 5e-5,
            'warmup_epochs': 10,
            'classifier_lr_mult': 2.0, 'scheduler_type': 'cosine',
            'loss_function': 'focal', 'focal_gamma': 1.8, 'label_smoothing': 0.05, 'weight_decay': 1e-2,
            'anatomical_loss_weight': 0.2, 'mixup_alpha': 0.4,
            'early_stopping_patience': 40,
            'model_params': {
                'gnn_residual': True, 'dropout': 0.25, 'attention_dropout': 0.1, 'stochastic_depth_rate': 0.15,
                'use_arcface': False,
                'use_dynamic_graph': True, # Enable the new feature
                'top_k': 5,
            }
        },
    }

    for name, exp_config in experiments.items():
        exp_model_params = exp_config.get('model_params', {})
        final_params = {**v15_model_params, **exp_model_params}
        exp_config['model_params'] = final_params

    return experiments, base_config, v15_model_params


def get_file_paths():
    """Dynamically loads all file and directory paths from an external JSON file."""
    try:
        with open('config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError("Error: `config.json` configuration file not found.")
    paths = config.get("file_paths", {})
    paths["log_dir"] = config.get("log_dir", "logs")
    paths["summary_log_path"] = config.get("summary_log_path", "results/summary.txt")
    if not paths.get("train_csv_path") or not paths.get("test_csv_path"): raise ValueError(
        "Error: `config.json` must contain `train_csv_path` and `test_csv_path`.")
    return paths