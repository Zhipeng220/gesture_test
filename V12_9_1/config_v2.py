# config_v2.py

import json
from copy import deepcopy
from easydict import EasyDict

# --- Base Configuration ---
# 定义一个所有新实验共享的基础配置，避免重复
cfg = EasyDict()
cfg.base = {
    # 训练参数
    'batch_size': 8,
    'epochs': 200,
    'optimizer_type': 'adamw_custom',
    # --- MODIFICATION START ---
    # 降低学习率以增加训练稳定性，防止权重值变得过大
    'learning_rate': 3e-5,  # Original value was 5e-5
    # --- MODIFICATION END ---
    'warmup_epochs': 10,
    'classifier_lr_mult': 2.0,
    'scheduler_type': 'cosine',
    'loss_function': 'focal',
    'focal_gamma': 1.8,
    'label_smoothing': 0.05,
    'weight_decay': 1e-2,
    'anatomical_loss_weight': 0.2,
    'mixup_alpha': 0.4,
    'early_stopping_patience': 40,
    # 路线一的核心参数：混合图 + 稀疏度损失
    'sparsity_loss_weight': 0.1,  # 默认权重，可在具体实验中覆盖
    # 模型参数
    'model_params': {
        'model_class': 'DHSGNet_V4',
        'seq_len': 128,
        'num_layers': 8,
        'num_heads': 16,
        'embed_dim': 512,
        'input_dim': 9,
        'projection_head_dims': [512, 256, 128],
        'num_gnn_layers': 3,
        'gnn_type': 'HGCN',
        'skeletal_patch_size': 4,
        'gnn_residual': True,
        'dropout': 0.25,
        'attention_dropout': 0.1,
        'stochastic_depth_rate': 0.15,
        'use_arcface': False,
        # 路线一的核心开关
        'use_dynamic_graph': True,
        'use_hybrid_graph': True,
        'top_k': 5,
    }
}

# --- New Experiments for Route One ---
experiments = {}

# 实验 24: 优化稀疏度权重 (较低)
# 目的: 验证较弱的稀疏约束是否能让图学到更丰富的关系，从而提升性能。
exp24 = deepcopy(cfg.base)
exp24['sparsity_loss_weight'] = 0.05
experiments['24_Exp_V17_Hybrid_Sparsity005'] = exp24

# 实验 25: 优化稀疏度权重 (较高)
# 目的: 验证更强的稀疏约束是否能进一步减少噪声，从而提升性能。
exp25 = deepcopy(cfg.base)
exp25['sparsity_loss_weight'] = 0.2
experiments['25_Exp_V17_Hybrid_Sparsity02'] = exp25

# 实验 26: 将混合图机制应用于V18架构
# 目的: 结合最强的动态图策略和最强的模型架构，探索性能上限。
exp26 = deepcopy(cfg.base)
# V18架构的特定参数 (来自您之前的 14_Exp_V18_Tier2_Advanced 配置)
exp26['model_params'].update({
    'use_temporal_conv': True,
    'tcn_kernel_size': 3,
    'dilations': [1, 2, 4],
    'use_cross_modal_attention': True,
    'adaptive_depth': True,
    'layer_selection_threshold': 0.1,
})
experiments['26_Exp_V18_With_HybridGNN'] = exp26


def get_experiments():
    """
    返回为路线一新定义的实验配置。
    """
    # 从旧配置中提取 base_config 和 common_params 以保持兼容性
    # 这里的 base_config 和 common_params 实际上在新的 `run_all_experiments` 中不再起主要作用
    # 但为了最小化对 train_v2.py 的改动，我们保留它们。
    base_config = {'use_skeletal_features': True}
    common_params = cfg.base['model_params']
    return experiments, base_config, common_params


def get_file_paths():
    """从外部 JSON 文件动态加载所有文件和目录路径。"""
    try:
        with open('config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError("错误: 未找到 `config.json` 配置文件。")
    paths = config.get("file_paths", {})
    paths["log_dir"] = config.get("log_dir", "logs")
    paths["summary_log_path"] = config.get("summary_log_path", "results/summary.txt")
    if not paths.get("train_csv_path") or not paths.get("test_csv_path"):
        raise ValueError("错误: `config.json` 必须包含 `train_csv_path` 和 `test_csv_path`。")
    return paths
