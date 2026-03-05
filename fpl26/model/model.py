from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import dgl

from models.gin import GIN
from models.feature_extraction import FeatureExtractionModule
from models.feature_sharing import FeatureSharingModule
from models.ensemble_prediction import EnsemblePredictionModule

# helpers
def graph_readout_mean(g: dgl.DGLGraph, node_feat: torch.Tensor) -> torch.Tensor:
    """
    Compute a graph-level embedding from node embeddings.
    Default: mean pooling.

    Args:
      g: batched DGLGraph
      node_feat: (N_total, D)

    Returns:
      graph_feat: (B, D)
    """
    with g.local_scope():
        g.ndata["h"] = node_feat
        hg = dgl.mean_nodes(g, "h")
    return hg



class TopCircuitSeqModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.T = cfg.num_tasks

        # -------------------------
        # Encoders: 电路图 + 架构图 两个 GIN
        # -------------------------
        self.gin_circ = GIN(
            input_dim=cfg.gin_circ_in_dim,
            output_dim=cfg.gin_circ_hidden_dim,
            num_layers=cfg.gin_circ_layers,
        )
        self.gin_arch = GIN(
            input_dim=cfg.gin_arch_in_dim,
            output_dim=cfg.gin_arch_hidden_dim,
            num_layers=cfg.gin_arch_layers,
        )

        # -------------------------
        # direct concat, no projection
        # -------------------------
        self.fusion_in = cfg.gin_circ_hidden_dim + cfg.gin_arch_hidden_dim
        self.D = self.fusion_in
        self.fuse = nn.Identity()

        # -------------------------
        # Per-task Feature Extraction
        # -------------------------
        self.feature_extractors = nn.ModuleList([
            FeatureExtractionModule(input_dim=self.D, num_heads=cfg.fe_num_heads, num_layers=cfg.fe_num_layers)
            for _ in range(self.T)
        ])

        # -------------------------
        # Feature Sharing across tasks
        # Input/Output: (B, T, D)
        # -------------------------
        self.feature_sharing = FeatureSharingModule(
            dim=self.D,
            num_tasks=self.T,
            num_heads=cfg.fs_num_heads,
            num_layers=cfg.fs_num_layers,
            mlp_hidden_dim=cfg.fs_mlp_hidden_dim,
            dropout=cfg.fs_dropout,
            attn_dropout=cfg.fs_attn_dropout,
        )

        # -------------------------
        # Per-task prediction heads (ensemble)
        # -------------------------
        self.ensembles = nn.ModuleList([
            EnsemblePredictionModule(
                input_dim=self.D,
                output_dim=1,
                num_classes=cfg.ens_num_classes,
                num_layers=cfg.ens_num_layers,
                hidden_dim=cfg.ens_hidden_dim,
            )
            for _ in range(self.T)
        ])

    def _graph_pool(self, g: dgl.DGLGraph, node_emb: torch.Tensor) -> torch.Tensor:
        if self.cfg.graph_pool == "mean":
            return graph_readout_mean(g, node_emb)
        raise ValueError(f"Unknown graph_pool={self.cfg.graph_pool}")

    def forward(
        self,
        g_circ: dgl.DGLGraph,
        g_arch: dgl.DGLGraph,
        target_bins: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            g_circ: batched DGLGraph，对应电路 AAG
            g_arch: batched DGLGraph，对应架构 XML
            target_bins: (B, T) 每个样本每个任务的分桶索引，训练时提供用于 teacher forcing。
                        如果为 None，则使用 classifier 的 argmax 选择分支（推理模式）。

        Returns:
            values: (B, T, out_dim) 回归输出
            logits: (B, T, C) 分类 logits，如果 num_classes == 1 则为 None
        """
        # --------- Circuit graph encoder ---------
        if "nf" not in g_circ.ndata:
            raise KeyError("Circuit graph missing node feature 'nf'. Please set g_circ.ndata['nf'].")
        h_circ = g_circ.ndata["nf"].to(torch.float32)
        node_emb_circ = self.gin_circ(g_circ, h_circ)
        g_emb_circ = self._graph_pool(g_circ, node_emb_circ)  # (B, gin_circ_hidden_dim)

        # --------- Arch graph encoder ---------
        if "nf" not in g_arch.ndata:
            raise KeyError("Arch graph missing node feature 'nf'. Please set g_arch.ndata['nf'].")
        h_arch = g_arch.ndata["nf"].to(torch.float32)
        node_emb_arch = self.gin_arch(g_arch, h_arch)
        g_emb_arch = self._graph_pool(g_arch, node_emb_arch)  # (B, gin_arch_hidden_dim)

        # --------- Direct concat fusion ---------
        base = torch.cat([g_emb_circ, g_emb_arch], dim=-1)  # (B, D)
        base = self.fuse(base)

        B = base.shape[0]
        T = self.T
        D = base.shape[-1]

        # --------- Build per-task features (复制给每个任务分支) ---------
        # (B, T, D)
        F_task = base.unsqueeze(1).expand(B, T, D).contiguous()

        # --------- Per-task feature extraction (每任务不同提取器) ---------
        F_out = []
        for i in range(T):
            Fi = F_task[:, i, :]                   # (B, D)
            Fi = self.feature_extractors[i](Fi)    # (B, D) 任务i专属
            F_out.append(Fi)
        F_task = torch.stack(F_out, dim=1)         # (B, T, D)

        # --------- Feature sharing across tasks ---------
        F_task = self.feature_sharing(F_task)      # (B, T, D)

        # --------- Per-task ensemble prediction ---------
        values = []
        logits_list = []
        has_logits = False

        for i in range(T):
            Fi = F_task[:, i, :]                   # (B, D)
            # 获取当前任务的 target_bin（如果提供）
            target_bin_i = target_bins[:, i] if target_bins is not None else None
            yi, li = self.ensembles[i](Fi, target_bin=target_bin_i)  # yi: (B, out_dim), li: (B, C) or None
            values.append(yi)
            if li is not None:
                logits_list.append(li)
                has_logits = True

        values = torch.stack(values, dim=1)         # (B, T, out_dim)

        # 如果所有 ensemble 都是 num_classes == 1，则 logits 为 None
        if has_logits and len(logits_list) == T:
            logits = torch.stack(logits_list, dim=1)  # (B, T, C)
        else:
            logits = None

        return values, logits


class TopCircuitSeqModelCfg:
    def __init__(
        self,
        num_tasks,
        gin_circ_in_dim,
        gin_circ_hidden_dim,
        gin_circ_layers,
        gin_arch_in_dim,
        gin_arch_hidden_dim,
        gin_arch_layers,
        fe_num_heads,
        fe_num_layers,
        fs_num_heads,
        fs_num_layers,
        ens_num_classes,
        ens_num_layers,
        ens_hidden_dim,
        fs_dropout=0.0,
        fs_attn_dropout=0.0,
        ens_dropout=0.0,
        ens_attn_dropout=0.0,
        graph_pool="mean",
    ):
        self.num_tasks = num_tasks

        self.gin_circ_in_dim = gin_circ_in_dim
        self.gin_circ_hidden_dim = gin_circ_hidden_dim
        self.gin_circ_layers = gin_circ_layers

        self.gin_arch_in_dim = gin_arch_in_dim
        self.gin_arch_hidden_dim = gin_arch_hidden_dim
        self.gin_arch_layers = gin_arch_layers

        self.task_dim = gin_circ_hidden_dim + gin_arch_hidden_dim

        self.fe_num_heads = fe_num_heads
        self.fe_input_dim = self.task_dim
        self.fe_num_layers = fe_num_layers
        
        self.fs_num_heads = fs_num_heads
        self.fs_num_layers = fs_num_layers
        self.fs_mlp_hidden_dim = self.task_dim

        self.ens_num_classes = ens_num_classes
        self.ens_num_layers = ens_num_layers
        self.ens_hidden_dim = ens_hidden_dim

        self.fs_dropout = fs_dropout
        self.fs_attn_dropout = fs_attn_dropout
        self.ens_dropout = ens_dropout
        self.ens_attn_dropout = ens_attn_dropout
        self.graph_pool = graph_pool
        
# -----------------------------
# Quick test (optional)
# -----------------------------
if __name__ == "__main__":
    # dummy test (requires a real DGLGraph with ndata['nf'])
    cfg = TopCircuitSeqModelCfg(
        num_tasks=2,
        gin_circ_in_dim=7,
        gin_circ_hidden_dim=128,
        gin_circ_layers=2,
        gin_arch_in_dim=16,
        gin_arch_hidden_dim=128,
        gin_arch_layers=2,
        fe_num_heads=4,
        fe_num_layers=2,
        fs_num_heads=4,
        fs_num_layers=2,
        ens_num_classes=1,
        ens_num_layers=3,
        ens_hidden_dim=256,
    )
    model = TopCircuitSeqModel(cfg)
    print(model)