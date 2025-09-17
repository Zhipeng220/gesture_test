import os
import sys
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from typing import Dict, List, Any
import seaborn as sns

# 从同一项目目录导入必要的模块
from model_v2 import MODEL_REGISTRY
from utils_v2 import load_and_preprocess_sequential_data, setup_logger, set_plot_style
from config_v2 import get_experiments, get_file_paths

# --- 常量定义 ---
JOINT_NAMES = [
    'Wrist', 'Thumb_CMC', 'Thumb_MCP', 'Thumb_IP', 'Thumb_TIP',
    'Index_MCP', 'Index_PIP', 'Index_DIP', 'Index_TIP',
    'Middle_MCP', 'Middle_PIP', 'Middle_DIP', 'Middle_TIP',
    'Ring_MCP', 'Ring_PIP', 'Ring_DIP', 'Ring_TIP',
    'Pinky_MCP', 'Pinky_PIP', 'Pinky_DIP', 'Pinky_TIP'
]


# --- 绘图与模型加载函数 (无需修改) ---

def load_models(model_paths: List[str], exp_names: List[str], num_classes: int, device: torch.device) -> Dict[
    str, torch.nn.Module]:
    """从文件路径加载多个训练好的模型。"""
    if len(model_paths) != len(exp_names):
        raise ValueError("模型路径的数量必须与实验名称的数量相匹配。")

    experiments, base_config, _ = get_experiments()
    loaded_models = {}

    print("正在加载模型...")
    for model_path, exp_name in zip(model_paths, exp_names):
        if exp_name not in experiments:
            print(f"警告: 在config_v2.py中未找到实验 '{exp_name}' 的配置，跳过。")
            continue

        full_config = {**base_config, **experiments[exp_name]}
        model_params = full_config.get('model_params', {})
        ModelClass = MODEL_REGISTRY.get(model_params.get('model_class', 'DHSGNet_V4'))

        model = ModelClass(num_classes=num_classes, **model_params).to(device)

        try:
            model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        except RuntimeError as e:
            print(f"无法加载模型 '{exp_name}' 从 {model_path}。错误: {e}")
            continue

        model.eval()
        loaded_models[exp_name] = model
        print(f"- 已加载 '{exp_name}' 从 {model_path}")

    return loaded_models


def run_inference(models: Dict[str, torch.nn.Module], sample_seq: torch.Tensor, device: torch.device) -> Dict[str, Any]:
    """在多个模型上对单个样本进行推理。"""
    results = {}
    sample_seq_dev = sample_seq.unsqueeze(0).to(device)
    with torch.no_grad():
        for name, model in models.items():
            results[name] = model(sample_seq_dev)
    return results


# In visualize.py

def plot_comparative_attention(results: Dict[str, Any], layer_idx: int, frame_idx: int, output_path: str):
    """Creates a side-by-side comparison of spatial attention maps from multiple models."""
    if not results: return
    num_models = len(results)
    # Determine a dynamic, safe frame index for visualization
    safe_frame_idx = -1

    fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 5), sharey=True)
    if num_models == 1: axes = [axes]

    # First, find a valid attention map to determine a safe frame index
    for name, output in results.items():
        attn_weights = output.get('spatial_attention_weights')
        if attn_weights and layer_idx < len(attn_weights):
            # Get the shape of the attention tensor for one sample: (num_frames, num_heads, joints, joints)
            num_frames = attn_weights[layer_idx].shape[1]
            # Select the middle frame dynamically
            safe_frame_idx = num_frames // 2
            fig.suptitle(f'Spatial Attention Comparison (Layer {layer_idx}, Frame {safe_frame_idx})', fontsize=16)
            break

    if safe_frame_idx == -1:
        print("Could not find any attention weights to visualize.")
        plt.close()
        return

    # Now, generate the plots using the safe frame index
    for ax, (name, output) in zip(axes, results.items()):
        attn_weights = output.get('spatial_attention_weights')
        if not attn_weights or layer_idx >= len(attn_weights):
            ax.set_title(f"{name}\n(No attention data)")
            continue

        # Use the dynamically calculated safe_frame_idx
        attn_map = attn_weights[layer_idx].cpu().numpy()[0, safe_frame_idx]
        attn_map_avg = np.mean(attn_map, axis=0)

        sns.heatmap(attn_map_avg, xticklabels=JOINT_NAMES, yticklabels=JOINT_NAMES, cmap="viridis", ax=ax, cbar=False)
        ax.set_title(name)
        ax.tick_params(axis='x', rotation=90)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path)
    plt.close()
    print(f"比较注意力图已保存至: {output_path}")

def plot_prediction_probabilities(results: Dict[str, Any], le, ground_truth_idx: int, top_k: int, output_path: str):
    """创建多个模型预测概率的并排比较图。"""
    if not results: return
    num_models = len(results)
    fig, axes = plt.subplots(num_models, 1, figsize=(10, 4 * num_models), sharex=True)
    if num_models == 1: axes = [axes]

    ground_truth_label = le.inverse_transform([ground_truth_idx])[0]
    fig.suptitle(f'Top-{top_k} Prediction Probabilities (True Label: {ground_truth_label})', fontsize=16)

    for ax, (name, output) in zip(axes, results.items()):
        logits = output['logits'][0]
        probabilities = torch.softmax(logits, dim=0).cpu().numpy()

        top_indices = np.argsort(probabilities)[-top_k:]
        top_probs = probabilities[top_indices]
        top_labels = le.inverse_transform(top_indices)

        colors = ['green' if idx == ground_truth_idx else 'skyblue' for idx in top_indices]
        bars = ax.barh(top_labels, top_probs, color=colors)
        ax.set_title(name)
        ax.set_xlabel('Probability')
        ax.set_xlim(0, 1)
        ax.bar_label(bars, fmt='%.3f')

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path)
    plt.close()
    print(f"预测概率图已保存至: {output_path}")


# --- 主执行函数 (融合后的逻辑) ---

def main():
    """
    主执行函数 - 自动为所有在config中定义且已训练的模型生成一个横向对比可视化报告。
    """
    # 1. 配置对比任务的通用参数
    OUTPUT_DIR = "visualizations/full_comparison"
    CASE_STUDY_SAMPLE_IDX = 25

    # 2. 初始化
    print("\n" + "=" * 55)
    print(">>> 手势识别模型全自动横向对比流程 <<<")
    print("=" * 55 + "\n")

    try:
        # In visualize.py

        def plot_comparative_attention(results: Dict[str, Any], layer_idx: int, frame_idx: int, output_path: str):
            """Creates a side-by-side comparison of spatial attention maps from multiple models."""
            if not results: return
            num_models = len(results)
            # Determine a dynamic, safe frame index for visualization
            safe_frame_idx = -1

            fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 5), sharey=True)
            if num_models == 1: axes = [axes]

            # First, find a valid attention map to determine a safe frame index
            for name, output in results.items():
                attn_weights = output.get('spatial_attention_weights')
                if attn_weights and layer_idx < len(attn_weights):
                    # Get the shape of the attention tensor for one sample: (num_frames, num_heads, joints, joints)
                    num_frames = attn_weights[layer_idx].shape[1]
                    # Select the middle frame dynamically
                    safe_frame_idx = num_frames // 2
                    fig.suptitle(f'Spatial Attention Comparison (Layer {layer_idx}, Frame {safe_frame_idx})',
                                 fontsize=16)
                    break

            if safe_frame_idx == -1:
                print("Could not find any attention weights to visualize.")
                plt.close()
                return

            # Now, generate the plots using the safe frame index
            for ax, (name, output) in zip(axes, results.items()):
                attn_weights = output.get('spatial_attention_weights')
                if not attn_weights or layer_idx >= len(attn_weights):
                    ax.set_title(f"{name}\n(No attention data)")
                    continue

                # Use the dynamically calculated safe_frame_idx
                attn_map = attn_weights[layer_idx].cpu().numpy()[0, safe_frame_idx]
                attn_map_avg = np.mean(attn_map, axis=0)

                sns.heatmap(attn_map_avg, xticklabels=JOINT_NAMES, yticklabels=JOINT_NAMES, cmap="viridis", ax=ax,
                            cbar=False)
                ax.set_title(name)
                ax.tick_params(axis='x', rotation=90)

            fig.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(output_path)
            plt.close()
            print(f"比较注意力图已保存至: {output_path}")
    except Exception as e:
        logging.error(f"发生未捕获的严重错误: {e}", exc_info=True)
        sys.exit(1)


# --- 脚本入口 ---
if __name__ == "__main__":
    main()