import os
import sys
from typing import Dict, Any, Optional, Tuple

import torch
import dgl


# ----------------------------------------------------
# 动态添加 data/arch_parse_ref 到 sys.path，方便直接复用解析代码
# ----------------------------------------------------
_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.abspath(os.path.join(_CUR_DIR, ".."))
_ARCH_PARSE_DIR = os.path.join(_PROJ_ROOT, "data", "arch_parse_ref")

if _ARCH_PARSE_DIR not in sys.path:
    sys.path.append(_ARCH_PARSE_DIR)

from extract_xml import process_xml_to_data  # type: ignore


def _default_arch_xml_path() -> Optional[str]:
    """
    返回一个默认的架构 XML，用于推断特征维度。
    这里使用用户提供的 vib_new_546.xml。
    """
    path = os.path.join(_PROJ_ROOT, "data", "archs", "vib_new_546.xml")
    return path if os.path.exists(path) else None


def _infer_arch_node_feat_dim() -> int:
    """
    通过实际解析一个架构 XML（vib_new_546.xml）来自动推断节点特征维度。
    这样当你在 arch_parse_ref 中增加/减少特征时，这里的维度会自动同步。
    """
    xml_path = _default_arch_xml_path()
    if xml_path is None:
        raise FileNotFoundError(
            "无法找到默认架构 XML（data/archs/vib_new_546.xml），"
            "请确认文件存在，或者在 arch_preprocessing.py 中手动指定一条可用的架构路径。"
        )

    # label 与 node_labels 在我们这里并不用作监督，只是为了调用接口，给默认值即可
    data = process_xml_to_data(xml_path, label=0.0, node_labels={})
    if not hasattr(data, "x"):
        raise AttributeError("process_xml_to_data 返回结果中缺少属性 x。")
    return int(data.x.shape[1])


# 对外暴露：架构图中节点特征的维度
ARCH_NODE_FEAT_DIM: int = _infer_arch_node_feat_dim()


def xml_to_dgl_graph(
    xml_path: str,
    *,
    label: float = 0.0,
    node_labels: Optional[Dict[str, float]] = None,
) -> Tuple[dgl.DGLGraph, Dict[str, Any]]:
    """
    使用 data/arch_parse_ref 中的解析逻辑，将架构 XML 转成 DGLGraph。

    返回：
      - g: DGLGraph，节点特征保存在 g.ndata["nf"]，维度为 ARCH_NODE_FEAT_DIM
      - meta: 目前仅包含 global_features（从 other_attrs 中取出），后续可按需扩展
    """
    if node_labels is None:
        node_labels = {}

    data = process_xml_to_data(xml_path, label=label, node_labels=node_labels)

    x = torch.as_tensor(data.x, dtype=torch.float32)  # (N, F)
    edge_index = torch.as_tensor(data.edge_index, dtype=torch.long)  # (2, E)

    if edge_index.dim() != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index 形状非法: {tuple(edge_index.shape)}, 期望为 (2, E)")

    src = edge_index[0]
    dst = edge_index[1]

    num_nodes = x.shape[0]
    g = dgl.graph((src, dst), num_nodes=int(num_nodes))

    # 与 AAG 图保持一致，使用同一个键名 "nf" 存储节点特征
    g.ndata["nf"] = x

    # 全局特征（Data.other_attrs）保存在 meta 中，后续如需融合可以从 meta 里取
    other_attrs = getattr(data, "other_attrs", None)
    meta: Dict[str, Any] = {
        "other_attrs": other_attrs,
        "num_nodes": int(num_nodes),
        "feat_dim": int(x.shape[1]),
    }

    return g, meta

