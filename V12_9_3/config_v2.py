# config_v2.py
import json
from copy import deepcopy
from easydict import EasyDict


cfg = EasyDict()
cfg.base = {
    # 训练参数
    'batch_size': 8,
    'epochs': 200,
    'optimizer_type': 'adamw_custom',
    'learning_rate': 5e-5,
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
    # 核心参数：以最优实验 Exp 25 的配置为基准
    'sparsity_loss_weight': 0.2,
    # 模型参数 (V17 架构)
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
        'use_dynamic_graph': True,
        'top_k': 5,
        'use_hybrid_graph': True,
    }
}

# --- New Experiments for Next Phase ---
experiments = {}

# 阶段一：核心优化 (实验 27)
# 目的: 在最优稀疏度权重下，测试更强大的V18架构
exp27 = deepcopy(cfg.base)
# 更新模型参数以匹配V18架构
exp27['model_params'].update({
    'use_temporal_conv': True,
    'tcn_kernel_size': 3,
    'dilations': [1, 2, 4],
    'use_cross_modal_attention': True,
    'adaptive_depth': True,
    'layer_selection_threshold': 0.1,
})
# 稀疏度权重已经是0.2 (继承自base)
experiments['27_Exp_V18_Hybrid_Sparsity02'] = exp27


# 阶段二：探索性微调 (实验 28, 29)
# 实验 28: 探索更高的稀疏度权重
exp28 = deepcopy(cfg.base) # 基于V17架构
exp28['sparsity_loss_weight'] = 0.3
experiments['28_Exp_V17_Hybrid_Sparsity03'] = exp28

# 实验 29: 探索更少的图连接数 (top_k)
exp29 = deepcopy(cfg.base) # 基于V17架构
exp29['model_params']['top_k'] = 4
experiments['29_Exp_V17_Hybrid_TopK_4'] = exp29


def get_experiments():
    """返回为下一阶段新定义的实验配置。"""
    # 为了保持与旧版 train_v2.py 的兼容性，我们保留这些返回值
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