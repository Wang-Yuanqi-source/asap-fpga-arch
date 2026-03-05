import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import dgl
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dataset_loader import CircuitSeqDataset, collate_circuit_seq
from split_dataset import split_dataset
from model import TopCircuitSeqModel, TopCircuitSeqModelCfg
from label_normalizer import LabelNormalizer, compute_metrics_original_space
from quantile_bins import QuantileBinManager
from arch_preprocessing import ARCH_NODE_FEAT_DIM


# -----------------------------
# utils
# -----------------------------
def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _extract_pred(out):
    """model forward could return Tensor or (values, logits, ...)"""
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


def _extract_labels_from_batch(batch: Dict[str, Any], label_names: List[str], device: str) -> torch.Tensor:
    """
    从 batch 中提取独立 label，并拼成 (B, T) 的 tensor。
    batch 里应有 batch["area"], batch["delay"] 这种 (B,) 张量。
    """
    ys = []
    for lb in label_names:
        if lb not in batch:
            raise KeyError(f"Label '{lb}' not found in batch. Available keys: {list(batch.keys())}")
        y = batch[lb]
        if not torch.is_tensor(y):
            y = torch.tensor(y)
        ys.append(y.to(device).to(torch.float32))  # (B,)
    return torch.stack(ys, dim=1)  # (B, T)


def mse_loss(pred_norm: torch.Tensor, y_norm: torch.Tensor) -> torch.Tensor:
    """
    pred_norm: (B,T) or (B,T,1)
    y_norm   : (B,T)
    """
    if pred_norm.ndim == 3 and pred_norm.shape[-1] == 1:
        pred_norm = pred_norm.squeeze(-1)
    if pred_norm.ndim != 2:
        raise ValueError(f"Unexpected pred shape: {tuple(pred_norm.shape)}")
    return ((pred_norm - y_norm) ** 2).mean()


def _split_batch_to_micro_batches(
    batch: Dict[str, Any], label_names: List[str], device: str
) -> List[Dict[str, Any]]:
    """
    将 batch 拆成多个 micro-batch（每个样本一个），用于 OOM 时梯度累积。
    """
    bg_circ = batch["g"]
    bg_arch = batch["arch_g"]

    graphs_circ = dgl.unbatch(bg_circ)
    graphs_arch = dgl.unbatch(bg_arch)

    B = len(graphs_circ)
    micro_batches = []
    for i in range(B):
        mb = {
            "g": graphs_circ[i],
            "arch_g": graphs_arch[i],
        }
        for lb in label_names:
            mb[lb] = batch[lb][i : i + 1].to(device)
        micro_batches.append(mb)
    return micro_batches


def _is_oom_error(e: Exception) -> bool:
    """判断是否为 CUDA OOM 错误"""
    oom_strs = ("out of memory", "CUDA out of memory", "cudaErrorOutOfMemory")
    err_str = str(e).lower()
    return any(s.lower() in err_str for s in oom_strs)


def combined_loss(
    values: torch.Tensor, 
    logits: Optional[torch.Tensor], 
    y_regression: torch.Tensor, 
    y_classification: Optional[torch.Tensor],
    num_classes: int,
    alpha: float = 1.0
) -> torch.Tensor:
    """
    Combined loss: MSE + Cross-Entropy Loss

    Args:
        values: (B, T, out_dim) - regression values
        logits: (B, T, C) - classification logits, None if classification is not used
        y_regression: (B, T) - 回归真值（用于 MSE）
        y_classification: (B, T) - 分类真值（分桶索引，用于 CE），如果为 None 则不计算 CE
        num_classes: Number of classes for classification
        alpha: 分类损失权重，默认为 1.0

    Returns:
        Total loss combining MSE and classification loss
    """
    values = values.squeeze(-1)  # (B, T)
    
    # MSE loss (regression task)
    mse = ((values - y_regression) ** 2).mean()

    # If logits is None or y_classification is None, only return MSE loss
    if logits is None or y_classification is None:
        return mse

    # CrossEntropy loss (classification task)
    # y_classification 应该是由 QuantileBinManager 生成的分桶索引
    ce_loss = nn.CrossEntropyLoss()(
        logits.view(-1, num_classes), 
        y_classification.view(-1).long()
    )  # (B*T, C) vs (B*T,)

    # Combine both losses
    total_loss = mse + alpha * ce_loss
    return total_loss


@torch.no_grad()
def compute_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    label_names: List[str],
    normalizer: Optional[LabelNormalizer] = None,
    eps: float = 1e-8,
    is_ddp: bool = False,
    use_oom_fallback: bool = True,
) -> Dict[str, Any]:
    """
    指标在 ORIGINAL space 计算：
      - 如果 normalizer != None：认为 model 输出的是“normalized space”的预测
        => 先 denormalize 再算 R²/MAPE/MAE/MSE
      - 否则：直接用 model 输出当 original
    """
    model.eval()
    preds = []
    trues = []

    for batch in loader:
        try:
            g = batch["g"].to(device)
            seq = batch["seq"].to(device)
            seq_len = batch["seq_len"].to(device)
            y = _extract_labels_from_batch(batch, label_names, device)
            out = model(g, seq, seq_len)
            p = _extract_pred(out)
        except RuntimeError as e:
            if use_oom_fallback and _is_oom_error(e):
                torch.cuda.empty_cache()
                micro_batches = _split_batch_to_micro_batches(batch, label_names, device)
                for mb in micro_batches:
                    g = mb["g"].to(device)
                    out = model(g, mb["seq"], mb["seq_len"])
                    p = _extract_pred(out)
                    if p.ndim == 3 and p.shape[-1] == 1:
                        p = p.squeeze(-1)
                    if normalizer is not None:
                        p = normalizer.denormalize(p)
                    y = _extract_labels_from_batch(mb, label_names, device)
                    preds.append(p.detach())
                    trues.append(y.detach())
                continue
            raise

        if p.ndim == 3 and p.shape[-1] == 1:
            p = p.squeeze(-1)
        if p.ndim != 2:
            raise ValueError(f"Unexpected pred shape: {tuple(p.shape)}")
        if normalizer is not None:
            p = normalizer.denormalize(p)
        preds.append(p.detach())
        trues.append(y.detach())

    pred = torch.cat(preds, dim=0)
    true = torch.cat(trues, dim=0)

    if is_ddp and dist.is_initialized() and dist.get_world_size() > 1:
        world_size = dist.get_world_size()
        local_size = pred.shape[0]
        size_tensor = torch.tensor([local_size], device=pred.device, dtype=torch.long)
        size_list = [torch.zeros_like(size_tensor) for _ in range(world_size)]
        dist.all_gather(size_list, size_tensor)
        max_size = max(s.item() for s in size_list)
        if local_size < max_size:
            pad_pred = torch.zeros(max_size - local_size, pred.shape[1], device=pred.device, dtype=pred.dtype)
            pad_true = torch.zeros(max_size - local_size, true.shape[1], device=true.device, dtype=true.dtype)
            pred = torch.cat([pred, pad_pred], dim=0)
            true = torch.cat([true, pad_true], dim=0)
        pred_list = [torch.zeros_like(pred) for _ in range(world_size)]
        true_list = [torch.zeros_like(true) for _ in range(world_size)]
        dist.all_gather(pred_list, pred)
        dist.all_gather(true_list, true)
        preds_unpad, trues_unpad = [], []
        for i, sz in enumerate(size_list):
            s = sz.item()
            preds_unpad.append(pred_list[i][:s])
            trues_unpad.append(true_list[i][:s])
        pred = torch.cat(preds_unpad, dim=0)
        true = torch.cat(trues_unpad, dim=0)

    pred = pred.cpu()
    true = true.cpu()
    return compute_metrics_original_space(pred, true, label_names, eps)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    device: str,
    grad_clip: float,
    label_names: List[str],
    num_classes: int,
    normalizer: Optional[LabelNormalizer] = None,
    bin_manager: Optional[QuantileBinManager] = None,
    ce_alpha: float = 1.0,
    use_oom_fallback: bool = True,
    is_ddp: bool = False,
    accum_steps: int = 1,
) -> float:
    """
    训练一个 epoch，使用 MSE 和分类损失之和。
    支持 OOM 时自动拆分为 micro-batch 梯度累积（动态 batch size）。
    支持 accum_steps 梯度累积以降低显存。
    
    Args:
        bin_manager: 分位数管理器，用于生成分类标签。如果为 None 则不使用分类损失。
        ce_alpha: 分类损失权重，默认为 1.0
    """
    model.train()
    total = 0.0
    n = 0
    accum_cnt = 0

    for batch in loader:
        bs = batch["seq"].shape[0]
        y = _extract_labels_from_batch(batch, label_names, device)  # (B,T)
        y_for_loss = normalizer.normalize(y) if normalizer is not None else y  # (B,T)
        
        # 生成分类标签（分桶索引）
        if bin_manager is not None and bin_manager.is_fitted:
            y_classification = bin_manager.get_bin_indices(y)  # (B, T)
        else:
            y_classification = None

        if accum_cnt == 0:
            opt.zero_grad(set_to_none=True)

        try:
            g = batch["g"].to(device)
            seq = batch["seq"].to(device)
            seq_len = batch["seq_len"].to(device)
            # 传入 target_bins 用于 teacher forcing
            out = model(g, seq, seq_len, target_bins=y_classification)
            values, logits = out
            loss = combined_loss(
                values, logits, y_for_loss, y_classification,
                num_classes=num_classes, alpha=ce_alpha
            )
            loss = loss / accum_steps
            loss.backward()
            total += float(loss.detach().cpu()) * bs * accum_steps
            n += bs
            accum_cnt += 1
            if accum_cnt >= accum_steps:
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
                accum_cnt = 0

        except RuntimeError as e:
            if use_oom_fallback and _is_oom_error(e):
                torch.cuda.empty_cache()
                if is_ddp:
                    if dist.get_rank() == 0:
                        print(f"  [OOM] batch_size={bs} -> fallback to micro-batches (size=1)", flush=True)
                else:
                    print(f"  [OOM] batch_size={bs} -> fallback to micro-batches (size=1)", flush=True)
                micro_batches = _split_batch_to_micro_batches(batch, label_names, device)
                opt.zero_grad(set_to_none=True)
                for mb in micro_batches:
                    mb_y = _extract_labels_from_batch(mb, label_names, device)
                    mb_y_for_loss = normalizer.normalize(mb_y) if normalizer is not None else mb_y
                    # 生成分类标签
                    if bin_manager is not None and bin_manager.is_fitted:
                        mb_y_classification = bin_manager.get_bin_indices(mb_y)
                    else:
                        mb_y_classification = None
                    g_mb = mb["g"].to(device)
                    out = model(g_mb, mb["seq"], mb["seq_len"], target_bins=mb_y_classification)
                    values, logits = out
                    loss = combined_loss(
                        values, logits, mb_y_for_loss, mb_y_classification,
                        num_classes=num_classes, alpha=ce_alpha
                    )
                    loss.backward()
                    total += float(loss.detach().cpu())
                    n += 1
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
                accum_cnt = 0
            else:
                raise

    if accum_cnt > 0:
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

    return total / max(1, n)



# -----------------------------
# cfg
# -----------------------------
def setup_ddp() -> tuple:
    """初始化 DDP，返回 (rank, local_rank, world_size, device).
    使用 torchrun 启动时自动启用多卡。
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = f"cuda:{local_rank}"
        return rank, local_rank, world_size, device
    # 单卡/CPU 模式
    return 0, 0, 1, None


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


@dataclass
class TrainCfg:
    seed: int = 0
    split_mode: str = "within_circuit"

    batch_size: int = 2  # 大图显存紧张时可调小，配合 accum_steps 保持有效 batch
    accum_steps: int = 1  # 梯度累积步数，有效 batch = batch_size * accum_steps
    num_workers: int = 1
    use_ddp: bool = True  # 多卡时自动使用 DDP
    use_oom_fallback: bool = True  # OOM 时自动拆分为 micro-batch

    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 5000
    grad_clip: float = 1.0

    # early stopping: average R2 of train & val
    patience: int = 20
    r2_min_delta: float = 1e-6  # improvement threshold

    # label normalization
    use_label_normalization: bool = True
    
    # quantile bin manager (分位数分桶)
    use_quantile_bins: bool = True  # 是否使用分位数分桶
    num_bins: int = 8  # 分桶数量（分支数量）
    ce_alpha: float = 1.0  # 分类损失权重

    # data paths
    csv_path: str = "/home/yfdai/asap/data/vtr_abcd_epfl_iscas.csv"
    circuit_dir: str = "/home/yfdai/asap/data/aag"
    seq_dir: str = "/home/yfdai/asap/data/seq/"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: str = "./ckpt"
    save_name: str = "best_r2_aag.pt"
    normalizer_name: str = "label_normalizer_area_delay.json"
    bin_manager_name: str = "quantile_bins.json"


def build_dataset(
    labels: List[str],
    csv_path: str,
    circuit_dir: str,
    seq_dir: str,
    verbose: bool = True,
) -> CircuitSeqDataset:
    """
    只加载你关心的 labels（area/delay）。
    verbose=False 时跳过 preload 日志（DDP 下仅 rank 0 打印）。
    """
    kwargs: Dict[str, Any] = dict(
        csv_path=csv_path,
        circuit_dir=circuit_dir,
        seq_dir=seq_dir,
        use_header=True,
        check_paths=False,
        preload_graphs=True,
        labels=labels,
    )
    # 兼容旧版 dataset_loader（无 verbose 参数）
    import inspect
    if "verbose" in inspect.signature(CircuitSeqDataset.__init__).parameters:
        kwargs["verbose"] = verbose
    ds = CircuitSeqDataset(**kwargs)
    return ds


def main():
    cfg = TrainCfg()
    set_seed(cfg.seed)

    # 缓解 CUDA 显存碎片，可在环境变量中设置 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ and torch.cuda.is_available():
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

    # -----------------------------
    # DDP 初始化（由 torchrun 启动时自动启用）
    # -----------------------------
    rank, local_rank, world_size, ddp_device = setup_ddp()
    is_ddp = world_size > 1
    device = ddp_device if is_ddp else (cfg.device if torch.cuda.is_available() else "cpu")

    if is_ddp and rank == 0:
        print(f"[DDP] world_size={world_size}, using NCCL backend", flush=True)

    # -----------------------------
    # Only two tasks
    # -----------------------------
    task_names = ["area", "delay"]
    num_tasks = len(task_names)

    # -----------------------------
    # 动态生成文件名（包含 csv 名称和预测指标）
    # -----------------------------
    csv_basename = os.path.splitext(os.path.basename(cfg.csv_path))[0]  # 去掉路径和 .csv
    tasks_str = "_".join(task_names)  # e.g., "area_delay"
    name_suffix = f"{csv_basename}_{tasks_str}"  # e.g., "vtr_abcd_epfl_iscas_area_delay"
    
    save_name = f"best_r2_{name_suffix}.pt"
    normalizer_name = f"label_normalizer_{name_suffix}.json"
    bin_manager_name = f"quantile_bins_{name_suffix}.json"

    # -----------------------------
    # Dataset + Split (within_circuit)
    # -----------------------------
    ds = build_dataset(
        labels=task_names,
        csv_path=cfg.csv_path,
        circuit_dir=cfg.circuit_dir,
        seq_dir=cfg.seq_dir,
        verbose=(rank == 0),
    )
    if "gid" not in ds.df.columns:
        raise ValueError("ds.df must contain a 'gid' column to use split_dataset().")

    train_ds, val_ds, test_ds = split_dataset(
        ds,
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
        mode="within_circuit",
        seed=cfg.seed,
        min_per_split_per_gid=0,
    )

    # -----------------------------
    # Label Normalizer (fit on train only, rank 0 only for DDP)
    # -----------------------------
    normalizer: Optional[LabelNormalizer] = None
    if rank == 0:
        os.makedirs(cfg.save_dir, exist_ok=True)

    if cfg.use_label_normalization:
        if rank == 0:
            print("\n[INFO] Computing label statistics from train_ds ONLY ...", flush=True)
        normalizer = LabelNormalizer(labels=task_names)
        normalizer.compute_stats(train_ds)
        if rank == 0:
            normalizer.print_stats()
            normalizer_path = os.path.join(cfg.save_dir, normalizer_name)
            normalizer.save(normalizer_path)
        if is_ddp:
            dist.barrier()
        if is_ddp and rank != 0:
            normalizer_path = os.path.join(cfg.save_dir, normalizer_name)
            normalizer = LabelNormalizer.load(normalizer_path)
    if not cfg.use_label_normalization and rank == 0:
        print("\n[INFO] Label normalization is DISABLED.", flush=True)

    # -----------------------------
    # DataLoaders（DDP 使用 DistributedSampler）
    # -----------------------------
    train_sampler = (
        DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        if is_ddp
        else None
    )
    val_sampler = (
        DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
        if is_ddp
        else None
    )
    test_sampler = (
        DistributedSampler(test_ds, num_replicas=world_size, rank=rank, shuffle=False)
        if is_ddp
        else None
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_circuit_seq,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_circuit_seq,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_circuit_seq,
    )

    # -----------------------------
    # Quantile Bin Manager (fit on train only)
    # -----------------------------
    bin_manager: Optional[QuantileBinManager] = None
    if cfg.use_quantile_bins:
        if rank == 0:
            print("\n[INFO] Computing quantile bin boundaries from train_ds ONLY ...", flush=True)
        
        # 从训练集提取所有标签
        # Sample 对象的标签存储在 .y 字典中
        import numpy as np
        train_labels = []
        for i in range(len(train_ds)):
            sample = train_ds[i]
            label_row = [sample.y[lb].item() if torch.is_tensor(sample.y[lb]) else float(sample.y[lb]) for lb in task_names]
            train_labels.append(label_row)
        train_labels = np.array(train_labels)  # (N, T)
        
        bin_manager = QuantileBinManager(num_tasks=num_tasks, num_bins=cfg.num_bins)
        bin_manager.fit(train_labels)
        
        if rank == 0:
            stats = bin_manager.get_bin_statistics(train_labels)
            print(f"[INFO] Quantile bins computed: num_bins={cfg.num_bins}")
            for t, name in enumerate(task_names):
                task_stats = stats[f"task_{t}"]
                boundaries = task_stats.get("boundaries", [])
                print(f"  {name}: boundaries={[f'{b:.4f}' for b in boundaries]}")
                for b in range(cfg.num_bins):
                    bin_info = task_stats.get(f"bin_{b}", {})
                    count = bin_info.get("count", 0)
                    pct = bin_info.get("percentage", 0)
                    print(f"    bin_{b}: count={count} ({pct:.1f}%)")
            
            # 保存分位数边界
            bin_manager_path = os.path.join(cfg.save_dir, bin_manager_name)
            bin_manager.save(bin_manager_path)
            print(f"[INFO] Quantile bin manager saved to {bin_manager_path}")
        
        if is_ddp:
            dist.barrier()
        if is_ddp and rank != 0:
            bin_manager_path = os.path.join(cfg.save_dir, bin_manager_name)
            bin_manager = QuantileBinManager.load(bin_manager_path)
    
    if not cfg.use_quantile_bins and rank == 0:
        print("\n[INFO] Quantile bin classification is DISABLED.", flush=True)

    # -----------------------------
    # Model
    # -----------------------------
    # 当使用分位数分桶时，ens_num_classes 使用 num_bins；否则为 1（单分支）
    ens_num_classes = cfg.num_bins if cfg.use_quantile_bins else 1
    
    model_cfg = TopCircuitSeqModelCfg(
        num_tasks=num_tasks,
        gin_in_dim=8,
        gin_hidden_dim=128,
        gin_layers=2,
        seq_in_dim=16,
        seq_hidden_dim=128,
        seq_layers=2,
        fe_num_heads=4,
        fe_num_layers=2,
        fs_num_heads=4,
        fs_num_layers=2,
        ens_num_classes=ens_num_classes,
        ens_num_layers=3,
        ens_hidden_dim=256,
    )
    model = TopCircuitSeqModel(model_cfg).to(device)
    if is_ddp and cfg.use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # -----------------------------
    # Early stopping
    # -----------------------------
    best_score = -1e18
    best_epoch = -1
    bad_count = 0
    best_path = os.path.join(cfg.save_dir, save_name)

    if rank == 0:
        print("\n" + "=" * 70)
        print(f"[INFO] split_mode=within_circuit")
        print(f"[INFO] tasks={task_names}")
        print(f"[INFO] max_epochs={cfg.max_epochs}, early_stop_patience={cfg.patience} (avg R2)")
        print(f"[INFO] use_label_normalization={cfg.use_label_normalization}")
        print(f"[INFO] use_quantile_bins={cfg.use_quantile_bins}, num_bins={cfg.num_bins}, ce_alpha={cfg.ce_alpha}")
        print(f"[INFO] DDP={is_ddp} world_size={world_size}")
        print(f"[INFO] batch_size={cfg.batch_size} accum_steps={cfg.accum_steps} (eff_batch={cfg.batch_size * cfg.accum_steps})")
        print(f"[INFO] use_oom_fallback={cfg.use_oom_fallback} (dynamic batch size)")
        print("=" * 70 + "\n")

    for epoch in range(1, cfg.max_epochs + 1):
        if is_ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        tr_loss = train_one_epoch(
            model,
            train_loader,
            opt,
            device,
            cfg.grad_clip,
            num_classes=model_cfg.ens_num_classes,
            label_names=task_names,
            normalizer=normalizer,
            bin_manager=bin_manager,
            ce_alpha=cfg.ce_alpha,
            use_oom_fallback=cfg.use_oom_fallback,
            is_ddp=is_ddp,
            accum_steps=cfg.accum_steps,
        )

        tr = compute_metrics(
            model, train_loader, device, task_names, normalizer=normalizer,
            is_ddp=is_ddp, use_oom_fallback=cfg.use_oom_fallback,
        )
        va = compute_metrics(
            model, val_loader, device, task_names, normalizer=normalizer,
            is_ddp=is_ddp, use_oom_fallback=cfg.use_oom_fallback,
        )
        avg_r2 = 0.5 * (tr["r2_mean"] + va["r2_mean"])

        def fmt(m: Dict[str, Any]) -> str:
            r2s = " ".join([f"{task_names[i]}={m['r2_per_task'][i]:.4f}" for i in range(num_tasks)])
            mapes = " ".join([f"{task_names[i]}={m['mape_per_task'][i]*100:.2f}%" for i in range(num_tasks)])
            return (
                f"mse={m['mse']:.6f} R2_mean={m['r2_mean']:.4f} | {r2s} "
                f"MAPE_mean={m['mape_mean']*100:.2f}% | {mapes}"
            )

        if rank == 0:
            print(f"[Epoch {epoch:04d}] train_loss(norm)={tr_loss:.6f}  avgR2(train+val)/2={avg_r2:.6f}")
            print(f"  TRAIN {fmt(tr)}")
            print(f"  VAL   {fmt(va)}")

        if avg_r2 > best_score + cfg.r2_min_delta:
            best_score = avg_r2
            best_epoch = epoch
            bad_count = 0

            if rank == 0:
                state_dict = model.module.state_dict() if is_ddp else model.state_dict()
                opt_state = opt.state_dict()
                torch.save(
                    {
                        "model": state_dict,
                        "opt": opt_state,
                        "epoch": epoch,
                        "best_score_avg_r2": best_score,
                        "task_names": task_names,
                        "csv_path": cfg.csv_path,
                        "csv_name": os.path.basename(cfg.csv_path),
                        "train_metrics": {
                            "r2_mean": tr["r2_mean"],
                            "r2_per_task": {name: tr["r2_per_task"][i] for i, name in enumerate(task_names)},
                            "mape_mean": tr["mape_mean"],
                            "mape_per_task": {name: tr["mape_per_task"][i] for i, name in enumerate(task_names)},
                            "mse": tr["mse"],
                        },
                        "val_metrics": {
                            "r2_mean": va["r2_mean"],
                            "r2_per_task": {name: va["r2_per_task"][i] for i, name in enumerate(task_names)},
                            "mape_mean": va["mape_mean"],
                            "mape_per_task": {name: va["mape_per_task"][i] for i, name in enumerate(task_names)},
                            "mse": va["mse"],
                        },
                        "use_normalization": cfg.use_label_normalization,
                        "use_quantile_bins": cfg.use_quantile_bins,
                        "num_bins": cfg.num_bins,
                        "bin_manager_path": os.path.join(cfg.save_dir, bin_manager_name) if cfg.use_quantile_bins else None,
                    },
                    best_path,
                )
                print(f"  -> save best: {best_path} (best_avgR2={best_score:.6f})")
        else:
            bad_count += 1
            if bad_count >= cfg.patience:
                if rank == 0:
                    print(f"\n[EARLY STOP] No avgR2 improvement for {cfg.patience} epochs.")
                    print(f"Best epoch={best_epoch}, best_avgR2={best_score:.6f}")
                break

    # -----------------------------
    # Load best and evaluate on test
    # -----------------------------
    if rank == 0:
        print("\n" + "=" * 70)
        print("[FINAL] Loading best checkpoint and evaluating on TEST ...")
    ckpt = torch.load(best_path, map_location=device)
    load_model = model.module if is_ddp else model
    load_model.load_state_dict(ckpt["model"])

    te = compute_metrics(
        model, test_loader, device, task_names, normalizer=normalizer,
        is_ddp=is_ddp, use_oom_fallback=cfg.use_oom_fallback,
    )

    if rank == 0:
        print(f"[BEST] epoch={ckpt['epoch']}  best_avgR2={ckpt['best_score_avg_r2']:.6f}")
        print(f"[TEST] mse={te['mse']:.6f}  R2_mean={te['r2_mean']:.4f}  MAPE_mean={te['mape_mean']*100:.2f}%")
        for i, name in enumerate(task_names):
            print(f"  {name}: R2={te['r2_per_task'][i]:.4f}  MAPE={te['mape_per_task'][i]*100:.2f}%  MAE={te['mae_per_task'][i]:.6f}")
        print("=" * 70)

    cleanup_ddp()


if __name__ == "__main__":
    main()
