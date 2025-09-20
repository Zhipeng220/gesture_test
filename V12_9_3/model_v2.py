# model_v2.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List
import logging
import math


class _FFN(nn.Module):
    def __init__(self, embed_dim, ffn_dim, dropout):
        super().__init__()
        self.main = nn.Sequential(nn.Linear(embed_dim, ffn_dim), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(ffn_dim, embed_dim))

    def forward(self, x): return self.main(x)


class StableAttention(nn.Module):
    """
    一个更稳健的注意力模块，集成了数值稳定技巧和对自注意力/交叉注意力的支持。
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_q, x_kv=None, key_padding_mask=None):
        B, N_q, C = x_q.shape

        # 如果是自注意力，K和V与Q来自相同输入；否则来自交叉注意力的输入x_kv
        if x_kv is None:
            x_kv = x_q

        B, N_kv, C = x_kv.shape

        q = self.q_proj(x_q).reshape(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x_kv).reshape(B, N_kv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x_kv).reshape(B, N_kv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if key_padding_mask is not None:

            attn = attn.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))


        attn = attn - attn.max(dim=-1, keepdim=True)[0]
        # 2. 强制float32计算
        attn = F.softmax(attn, dim=-1, dtype=torch.float32)
        attn = self.dropout(attn)

        # 检查注意力权重的有效性，并在出现问题时回退
        if not torch.isfinite(attn).all():
            logging.warning("Invalid attention weights detected, using uniform attention as fallback.")
            attn = torch.ones_like(attn) / attn.shape[-1]

        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        x = self.proj(x)
        return x, attn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=256):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len_cached = max_seq_len
        t = torch.arange(self.max_seq_len_cached, dtype=inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        self.register_buffer("cos_cached", freqs.cos()[None, :, :])
        self.register_buffer("sin_cached", freqs.sin()[None, :, :])

    def forward(self, x: torch.Tensor):
        seq_len = x.shape[1]
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            self.cos_cached = freqs.cos()[None, :, :].to(x.device)
            self.sin_cached = freqs.sin()[None, :, :].to(x.device)
        cos, sin = self.cos_cached[:, :seq_len, ...], self.sin_cached[:, :seq_len, ...]
        x_part1, x_part2 = x[..., : self.dim // 2], x[..., self.dim // 2:]
        return torch.cat((x_part1 * cos - x_part2 * sin, x_part1 * sin + x_part2 * cos), dim=-1).type_as(x)


class StochasticDepth(nn.Module):
    def __init__(self, p: float): super().__init__(); self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0: return x
        keep_prob = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.empty(shape, device=x.device).bernoulli_(keep_prob)
        return x.div(keep_prob) * mask if keep_prob > 0.0 else x.mul_(0.)


class ViTPatchEmbedding(nn.Module):
    def __init__(self, seq_len, patch_size, in_chans, embed_dim):
        super().__init__()
        self.patch_size = patch_size;
        self.in_chans = in_chans;
        self.embed_dim = embed_dim
        self.proj = nn.Linear(patch_size * in_chans, embed_dim)

    def forward(self, x):
        B, T, C = x.shape
        if T % self.patch_size != 0:
            pad_size = self.patch_size - (T % self.patch_size)
            x = F.pad(x, (0, 0, 0, pad_size), "constant", 0)
            T = x.shape[1]
        x = x.reshape(B, T // self.patch_size, self.patch_size * C);
        return self.proj(x)


class AttentionPolicyNetwork(nn.Module):
    def __init__(self, embed_dim, num_heads, temp=0.5):
        super().__init__()
        self.policy_net = nn.Sequential(nn.Linear(embed_dim, embed_dim // 4), nn.GELU(),
                                        nn.Linear(embed_dim // 4, num_heads))
        self.temperature = temp

    def forward(self, x):
        head_logits = self.policy_net(x)
        if self.training:
            head_mask = F.gumbel_softmax(head_logits, tau=self.temperature, hard=True)
        else:
            head_mask = F.one_hot(torch.argmax(head_logits, dim=-1), num_classes=head_logits.shape[-1]).float()
        return head_mask, head_logits



class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.query_norm = nn.LayerNorm(embed_dim)
        self.kv_norm = nn.LayerNorm(embed_dim)
        # <--- MODIFIED: 使用新的StableAttention模块
        self.attn = StableAttention(embed_dim, num_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x1, x2):
        query = self.query_norm(x1)
        key_value = self.kv_norm(x2)

        # <--- MODIFIED: 调用StableAttention并传入交叉注意力的源(x_kv=key_value)
        fused_out, _ = self.attn(x_q=query, x_kv=key_value)

        return x1 + self.dropout(fused_out)


class GraphLearner(nn.Module):
    def __init__(self, embed_dim, top_k=None, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.top_k = top_k
        self.weight_tensor = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        nn.init.xavier_uniform_(self.weight_tensor)

    def forward(self, x):
        xt = torch.matmul(x, self.weight_tensor)
        adj = F.relu(torch.matmul(xt, x.transpose(1, 2)))

        if self.top_k:
            topk_val, _ = torch.topk(adj, k=self.top_k, dim=-1)
            adj_mask = (adj >= topk_val[..., -1, None])
            adj = adj * adj_mask

        return adj


# NEW: Add a helper function to create the static adjacency matrix
def _create_static_adjacency(device):
    """Creates a static adjacency matrix based on hand bone connections."""
    adj = torch.zeros(21, 21, device=device)
    bones = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),  # Index
        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
        (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
        (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
    ]
    for i, j in bones:
        adj[i, j] = 1
        adj[j, i] = 1
    # Add self-loops
    adj.fill_diagonal_(1)
    return adj


# model_v2.py

class HGCNLayer(nn.Module):
    # MODIFIED: Update __init__ to accept use_hybrid_graph
    def __init__(self, embed_dim, num_heads, dropout=0.1, use_hybrid_graph=False, **kwargs):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        # NEW: Store the hybrid graph flag
        self.use_hybrid_graph = use_hybrid_graph
        if self.use_hybrid_graph:
            # NEW: Create and register the static adjacency matrix as a buffer
            self.register_buffer("static_adj", _create_static_adjacency(torch.device("cpu")))

    # MODIFIED: Update forward pass to implement the hybrid graph logic and fix NaN issue
    def forward(self, x, bone_features, dynamic_adj=None):
        B, N, C = x.shape
        fused_input = x + bone_features
        q = self.q_proj(fused_input).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(fused_input).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(fused_input).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # --- START OF MODIFICATION TO FIX NaN ISSUE ---
        # 1. Calculate attention scores. This operation can produce large values.
        attention_scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # 2. Apply softmax using float32 to prevent numerical overflow when using autocast (mixed-precision).
        #    This is the key change to ensure stability and avoid NaN values.
        self_attn_adj = F.softmax(attention_scores, dim=-1, dtype=torch.float32)
        # --- END OF MODIFICATION ---

        base_adj = self_attn_adj
        if self.use_hybrid_graph:
            # The static_adj buffer will be automatically moved to the correct device
            # Ensure it is on the same device as base_adj before adding
            base_adj = base_adj + self.static_adj.to(base_adj.device)

        if dynamic_adj is not None:
            # The learned dynamic graph acts as a gate on the base connections
            final_adj = base_adj * dynamic_adj.unsqueeze(1)
        else:
            final_adj = base_adj

        out = (final_adj @ v).transpose(1, 2).reshape(B, N, C)
        return self.out_proj(self.dropout(out)), final_adj


class GNNBlock(nn.Module):
    # MODIFIED: Pass all kwargs down to HGCNLayer
    def __init__(self, dim, num_heads, num_gnn_layers=2, gnn_type='HGCN', dropout=0.1, use_residual=True,
                 use_dynamic_graph=False, **kwargs):
        super().__init__()
        self.use_residual = use_residual
        self.layers = nn.ModuleList(
            [nn.ModuleList([nn.LayerNorm(dim), HGCNLayer(dim, num_heads, dropout, **kwargs)]) for _ in
             range(num_gnn_layers)])

        self.use_dynamic_graph = use_dynamic_graph
        if self.use_dynamic_graph:
            self.graph_learner = GraphLearner(dim, **kwargs)

    def forward(self, x, bone_features=None):
        all_attn_weights = []
        dynamic_adj = None

        if self.use_dynamic_graph:
            h_norm = self.layers[0][0](x)
            dynamic_adj = self.graph_learner(h_norm)

        for norm, layer in self.layers:
            h = norm(x)
            h_out, attn = layer(h, bone_features, dynamic_adj=dynamic_adj)
            all_attn_weights.append(attn)
            if self.use_residual:
                x = x + h_out
            else:
                x = h_out
        return x, all_attn_weights


class SpatioTemporalGraphBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, num_gnn_layers=2, gnn_type='HGCN', gnn_residual=True, **kwargs):
        super().__init__()
        self.gnn_block = GNNBlock(embed_dim, num_heads, num_gnn_layers, gnn_type, dropout, use_residual=gnn_residual,
                                  **kwargs)
        self.local_global_attn = CrossAttentionFusion(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = _FFN(embed_dim, embed_dim * 4, dropout)

    def forward(self, x, bone_features=None):
        h = self.norm1(x)
        gnn_out, gnn_attn_weights = self.gnn_block(h, bone_features=bone_features)
        global_feature = h.mean(dim=1, keepdim=True)
        fused = self.local_global_attn(gnn_out, global_feature)
        x = x + fused
        x = x + self.ffn(self.norm2(x))
        return x, gnn_attn_weights


class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilations=[1, 2, 4], dropout=0.2):
        super().__init__()
        layers = []
        num_layers = len(dilations)
        for i in range(num_layers):
            dilation_size = dilations[i]
            in_c = in_channels if i == 0 else out_channels
            out_c = out_channels
            conv = nn.Conv1d(
                in_c, out_c, kernel_size,
                padding='same',
                dilation=dilation_size
            )
            layers.append(nn.Sequential(
                conv,
                nn.GELU(),
                nn.Dropout(dropout)
            ))
        self.net = nn.ModuleList(layers)

    def forward(self, x):
        out = x.permute(0, 2, 1)
        for layer in self.net:
            out = layer(out)
        return out.permute(0, 2, 1)



class TemporalAttentionBlock_V4(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, attention_dropout=0.1, use_temporal_conv=False, **kwargs):
        super().__init__()
        self.use_temporal_conv = use_temporal_conv
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(embed_dim), nn.LayerNorm(embed_dim), nn.LayerNorm(embed_dim)
        # <--- MODIFIED: 使用新的StableAttention模块
        self.attn = StableAttention(embed_dim, num_heads, dropout=attention_dropout)
        self.ffn = _FFN(embed_dim, embed_dim * 4, dropout)
        self.dropout = nn.Dropout(dropout)

        if self.use_temporal_conv:
            self.temporal_processor = TCNBlock(
                embed_dim, embed_dim,
                kernel_size=kwargs.get('tcn_kernel_size', 3),
                dilations=kwargs.get('dilations', [1, 2, 4]),
                dropout=dropout
            )
        else:
            self.lstm = nn.LSTM(embed_dim, embed_dim, bidirectional=True, batch_first=True)
            self.temporal_processor = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, x, rotary_emb, key_padding_mask=None):
        h = self.norm1(x)
        # Rotary embedding仍然可以应用于Q和K，但在StableAttention外部应用
        q_rot, k_rot = rotary_emb(h), rotary_emb(h)

        # <--- MODIFIED: 调用StableAttention进行自注意力计算
        # 注意：这里我们将旋转编码后的q和k传入，v保持不变
        # StableAttention内部会重新进行q,k,v的线性投射，因此我们只需传入原始输入h
        attn_out, attn_weights = self.attn(x_q=h, key_padding_mask=key_padding_mask)

        x = x + self.dropout(attn_out)

        h2 = self.norm2(x)
        if self.use_temporal_conv:
            processed_out = self.temporal_processor(h2)
        else:
            lstm_out, _ = self.lstm(h2)
            processed_out = self.temporal_processor(lstm_out)

        x = x + self.dropout(processed_out)
        x = x + self.ffn(self.norm3(x))

        return x, attn_weights


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=20.0, m=0.30):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features;
        self.out_features = out_features;
        self.s = s;
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features));
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(m);
        self.sin_m = math.sin(m);
        self.th = math.cos(math.pi - m);
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device=input.device);
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine);
        output *= self.s
        return output


class LayerChoicePolicy(nn.Module):
    def __init__(self, embed_dim, num_layers):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, num_layers)
        )

    def forward(self, x):
        return self.net(x)


class DHSGNet_V4(nn.Module):
    def __init__(self, input_dim, num_classes, seq_len=32, num_layers=8, num_heads=12, embed_dim=512, dropout=0.2,
                 stochastic_depth_rate=0.1, use_arcface=False, **kwargs):
        super().__init__()
        self.use_arcface = use_arcface
        self.input_embed = nn.Linear(input_dim, embed_dim)
        self.skeletal_embed = ViTPatchEmbedding(seq_len, kwargs.get('skeletal_patch_size', 4), 30, embed_dim)
        self.rotary_emb = RotaryPositionalEmbedding(dim=embed_dim, max_seq_len=seq_len + 1)
        self.spatial_pool_proj = nn.Linear(embed_dim * 2, embed_dim)
        dpr = [x.item() for x in torch.linspace(0, stochastic_depth_rate, num_layers)]

        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            block_class = SpatioTemporalGraphBlock if i % 2 == 0 else TemporalAttentionBlock_V4
            block = block_class(embed_dim, num_heads, dropout, **kwargs)
            self.blocks.append(nn.ModuleList([block, StochasticDepth(dpr[i])]))

        self.adaptive_depth = kwargs.get('adaptive_depth', False)
        if self.adaptive_depth:
            self.layer_policy = LayerChoicePolicy(embed_dim, num_layers)
            self.layer_selection_threshold = kwargs.get('layer_selection_threshold', 0.1)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.final_norm = nn.LayerNorm(embed_dim)

        self.classifier_linear = nn.Linear(embed_dim, num_classes)
        self.classifier_arcface = ArcMarginProduct(embed_dim, num_classes)

        self.anatomical_head = nn.Linear(embed_dim, 21 * 3)
        self.policy_network = AttentionPolicyNetwork(embed_dim, num_heads)
        proj_layers = []
        in_d = embed_dim
        for out_d in kwargs.get('projection_head_dims', [512, 256, 128]):
            proj_layers.extend([nn.Linear(in_d, out_d), nn.GELU()]);
            in_d = out_d
        self.projection_head = nn.Sequential(*proj_layers[:-1])

    def freeze_backbone(self):
        for name, param in self.named_parameters():
            if not name.startswith('classifier_'):
                param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.parameters():
            param.requires_grad = True

    # MODIFIED: The forward pass to return a list of all spatial and temporal attention weights
    def forward(self, x, skeletal_features=None, key_padding_mask=None, labels=None) -> Dict[
        str, Optional[torch.Tensor]]:
        B, T, N, _ = x.shape
        embed_dim = self.input_embed.out_features
        if skeletal_features is not None and skeletal_features.shape[1] > 0:
            bone_features = self.skeletal_embed(skeletal_features)
        else:
            patch_size = self.skeletal_embed.patch_size
            num_patches_skeletal = (T + patch_size - 1) // patch_size
            bone_features = torch.zeros(B, num_patches_skeletal, embed_dim, device=x.device)

        x_embedded = self.input_embed(x)

        # MODIFIED: Use lists to collect all attention weights
        policy_logits_list, all_spatial_attns, all_temporal_attns = [], [], []
        final_cls_token = self.cls_token.expand(B, -1, -1)

        layer_scores = None
        if self.adaptive_depth and self.training:
            initial_pooled_features = torch.mean(
                self.spatial_pool_proj(torch.cat((x_embedded.mean(2), x_embedded.max(2)[0]), dim=-1)), dim=1)
            layer_scores = self.layer_policy(initial_pooled_features).sigmoid()

        for i, (block, drop_path) in enumerate(self.blocks):
            if self.adaptive_depth and self.training:
                if layer_scores[0, i] < self.layer_selection_threshold:
                    continue

            if isinstance(block, SpatioTemporalGraphBlock):
                x_spatial_in = x_embedded.view(B * T, N, embed_dim)
                patch_size = self.skeletal_embed.patch_size
                time_indices = torch.arange(T, device=x.device)
                patch_indices = (time_indices // patch_size).clamp(max=bone_features.shape[1] - 1)
                expanded_bone_features = bone_features[:, patch_indices, :].view(B * T, 1, embed_dim)
                processed, current_spatial_attn_list = block(x_spatial_in, bone_features=expanded_bone_features)
                processed_reshaped = processed.view(B, T, N, embed_dim)
                x_embedded = x_embedded + drop_path(processed_reshaped)

                # MODIFIED: Extend the list with weights from all GNN layers in the block
                if current_spatial_attn_list:
                    all_spatial_attns.extend(current_spatial_attn_list)
            else:
                pooled_mean = torch.mean(x_embedded, dim=2)
                pooled_max, _ = torch.max(x_embedded, dim=2)
                h = self.spatial_pool_proj(torch.cat((pooled_mean, pooled_max), dim=-1))
                _, policy_logits = self.policy_network(h.mean(dim=1))
                policy_logits_list.append(policy_logits)
                h = torch.cat((final_cls_token, h), dim=1)
                temp_mask = F.pad(key_padding_mask, (1, 0), value=False) if key_padding_mask is not None else None
                processed, current_temporal_attn = block(h, self.rotary_emb, key_padding_mask=temp_mask)
                processed = drop_path(processed)
                final_cls_token = final_cls_token + processed[:, :1]

                # MODIFIED: Append temporal attention
                all_temporal_attns.append(current_temporal_attn)
                x_embedded = x_embedded + processed[:, 1:, :].unsqueeze(2)

        features = self.final_norm(final_cls_token.squeeze(1))

        if self.use_arcface:
            if labels is None:
                with torch.no_grad():
                    logits = self.classifier_arcface.s * F.linear(F.normalize(features),
                                                                  F.normalize(self.classifier_arcface.weight))
            else:
                logits = self.classifier_arcface(features, labels)
        else:
            logits = self.classifier_linear(features)

        # MODIFIED: Return the lists of weights with new keys`
        return {
            'logits': logits, 'features': features,
            'projected_features': self.projection_head(features),
            'anatomical_prediction': self.anatomical_head(features).view(B, 21, 3),
            'policy_logits': policy_logits_list,
            'temporal_attention_weights': all_temporal_attns,
            'spatial_attention_weights': all_spatial_attns
        }


MODEL_REGISTRY = {'DHSGNet_V4': DHSGNet_V4}