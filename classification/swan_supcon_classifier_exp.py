
"""
SWAN + Supervised Contrastive Learning for Solar Flare Time-Series Classification

What this version does
----------------------
1. Uses SWAN as the backbone encoder.
2. Treats SWAN as an encoder that returns embeddings, not class logits.
3. Uses a projection head on top of the encoder for Phase 1 supervised contrastive learning.
4. Uses grouped contrastive labels:
       non-flare : FQ / B / C  -> 0
       flare     : M / X       -> 1
5. Freezes the encoder in Phase 2 and trains a linear classifier on top.
6. Uses missing-value aware inputs:
       x : standardized values with NaNs filled by 0
       m : observation mask
       d : time-since-last-observed delta

Expected data file fields
-------------------------
partition1_grouped.npz / partition2_grouped.npz must contain:
    features    : (N, T, F)
    flare_type  : (N,)

Notes
-----
- AR region and timestamp conditioning are not used here.
- ForecastHead and other unused experimental blocks were omitted for clarity.
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, WeightedRandomSampler

from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns


try:
    from utils.logging_utils import build_logger
    logger = build_logger("classify", "dl_proj")
except Exception:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("dl_proj")



device = torch.device(
    "mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else
    "cpu"
)
use_pin_memory = device.type == "cuda"
logger.info(f"[Device] {device} | pin_memory={use_pin_memory}")



def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



SUBTYPE_TO_BINARY = {
    "F": 0,
    "FQ": 0,
    "Q": 0,
    "B": 0,
    "C": 0,
    "M": 1,
    "X": 1,
}


def normalize_subtype(value) -> str:
    """Map raw subtype strings into a canonical small vocabulary."""
    s = str(value).strip().upper()
    aliases = {
        "F": "FQ",
        "FQ": "FQ",
        "Q": "FQ",
        "B": "B",
        "C": "C",
        "M": "M",
        "X": "X",
    }
    return aliases.get(s, s)


def unpack_binary_and_subtype(label) -> Tuple[int | None, str | None]:
    """
    Expected raw label format: subtype only.

    Returns
    -------
    binary : int or None
        0 for non-flare, 1 for flare
    subtype : str or None
        Canonical subtype string
    """
    subtype = normalize_subtype(label)

    if subtype in {"M", "X"}:
        binary = 1
    elif subtype in {"FQ", "B", "C"}:
        binary = 0
    else:
        binary = None

    return binary, subtype


def compute_keep_mask(labels_raw: np.ndarray) -> np.ndarray:
    """Keep only the subtype vocabulary used in this experiment."""
    return np.array(
        [normalize_subtype(lbl) in {"FQ", "B", "C", "M", "X"} for lbl in labels_raw],
        dtype=bool,
    )


def prepare_labels(labels_raw: np.ndarray, split_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build:
        y_binary    : used for classification
        y_contrast  : used for grouped SupCon

    Grouping for contrastive loss:
        0 -> non-flare group (FQ, B, C)
        1 -> flare group     (M, X)
    """
    y_binary: List[int] = []
    y_contrast: List[int] = []
    kept_subtypes: List[str] = []

    for raw_label in labels_raw:
        binary, subtype = unpack_binary_and_subtype(raw_label)
        if subtype not in {"FQ", "B", "C", "M", "X"}:
            continue
        y_binary.append(binary)
        y_contrast.append(binary)
        kept_subtypes.append(subtype)

    y_binary_np = np.asarray(y_binary, dtype=np.int64)
    y_contrast_np = np.asarray(y_contrast, dtype=np.int64)
    kept_subtypes_np = np.asarray(kept_subtypes)

    uniq, cnt = np.unique(kept_subtypes_np, return_counts=True)
    subtype_stats = {str(k): int(v) for k, v in zip(uniq, cnt)}
    logger.info(f"[{split_name}] subtype counts: {subtype_stats}")
    logger.info(f"[{split_name}] binary counts : {np.bincount(y_binary_np)}")

    return y_binary_np, y_contrast_np



def build_mask(X: np.ndarray) -> np.ndarray:
    """1 where observed, 0 where missing."""
    return (~np.isnan(X)).astype(np.float32)


def build_delta(mask: np.ndarray) -> np.ndarray:
    """
    Delta[t, f] = time since last observation for feature f.

    If previous timestep was observed, delta resets to 1.
    Otherwise it increases by 1.
    """
    N, T, F = mask.shape
    delta = np.zeros((N, T, F), dtype=np.float32)

    for n in range(N):
        for f in range(F):
            for t in range(1, T):
                if mask[n, t - 1, f] == 1:
                    delta[n, t, f] = 1.0
                else:
                    delta[n, t, f] = delta[n, t - 1, f] + 1.0
    return delta


def fill_missing_with_zero(X_std: np.ndarray) -> np.ndarray:
    """Replace NaNs with 0 after standardization."""
    return np.nan_to_num(X_std, nan=0.0).astype(np.float32)
    # med = np.nanmedian(X_std, axis=(0, 1))  # shape (F,)
    # X_filled = np.where(np.isnan(X_std), med, X_std).astype(np.float32)
    # return X_filled


def fit_feature_scaler_observed_only(X_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean/std per feature using only observed entries.
    """
    _, _, F = X_train.shape
    means = np.zeros(F, dtype=np.float32)
    stds = np.ones(F, dtype=np.float32)

    for f in range(F):
        vals = X_train[:, :, f].reshape(-1)
        vals = vals[~np.isnan(vals)]
        if vals.size > 0:
            means[f] = float(vals.mean())
            std = float(vals.std())
            stds[f] = std if std > 1e-8 else 1.0

    return means, stds


def apply_standardization(X: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    """
    Standardize observed values only; keep NaNs as NaNs.
    """
    X_std = X.copy().astype(np.float32)
    for f in range(X.shape[-1]):
        observed = ~np.isnan(X_std[:, :, f])
        X_std[:, :, f][observed] = (X_std[:, :, f][observed] - means[f]) / stds[f]
    return X_std



class SWANDataset(Dataset):
    def __init__(self, x, m, d, y, y_contrast):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.m = torch.tensor(m, dtype=torch.float32)
        self.d = torch.tensor(d, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.y_contrast = torch.tensor(y_contrast, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "x": self.x[idx],
            "m": self.m[idx],
            "d": self.d[idx],
            "y": self.y[idx],
            "y_contrast": self.y_contrast[idx],
        }


class TAPE(nn.Module):
    """
    Learnable absolute temporal embedding.
    """
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, T: int) -> torch.Tensor:
        return self.pe[:, :T, :]


class RelativePositionBias(nn.Module):
    """
    Head-specific relative position bias generated by a small MLP.
    """
    def __init__(self, num_heads: int, hidden_dim: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_heads),
        )

    def forward(self, T: int, device: torch.device) -> torch.Tensor:
        pos = torch.arange(T, device=device)
        rel = pos[:, None] - pos[None, :]            # [T, T]
        rel = rel.float().unsqueeze(-1)              # [T, T, 1]
        rel = torch.sign(rel) * torch.log1p(torch.abs(rel))
        bias = self.mlp(rel)                         # [T, T, H]
        return bias.permute(2, 0, 1).contiguous()   # [H, T, T]


class MultiHeadSelfAttentionRPE(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.rpe = RelativePositionBias(num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.num_heads, self.d_head).transpose(1, 2)  # [B,H,T,D]
        k = k.view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)  # [B,H,T,T]
        scores = scores + self.rpe(T, x.device).unsqueeze(0)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # [B,H,T,D]
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)


# class EncoderBlock(nn.Module):
#     def __init__(self, d_model: int, num_heads: int, mlp_ratio: int = 4, dropout: float = 0.1):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(d_model)
#         self.attn = MultiHeadSelfAttentionRPE(d_model, num_heads, dropout)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.ffn = nn.Sequential(
#             nn.Linear(d_model, d_model * mlp_ratio),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(d_model * mlp_ratio, d_model),
#             nn.Dropout(dropout),
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = x + self.attn(self.norm1(x))
#         x = x + self.ffn(self.norm2(x))
#         return x



class GCTAFBlock(nn.Module):
    def __init__(self, d_model, num_heads, n_global=2, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.n_global = n_global

        # learnable constant global tokens
        self.global_tokens = nn.Parameter(torch.empty(1, n_global, d_model))
        nn.init.trunc_normal_(self.global_tokens, std=0.02)

        # global <- sequence
        self.cross_attn_global = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_g1 = nn.LayerNorm(d_model)
        self.ffn_g = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * mlp_ratio, d_model),
            nn.Dropout(dropout),
        )
        self.norm_g2 = nn.LayerNorm(d_model)
        self.norm1 = nn.LayerNorm(d_model)

        # sequence <- global
        self.cross_attn_fusion = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn = MultiHeadSelfAttentionRPE(d_model, num_heads, dropout)
        self.norm_s1 = nn.LayerNorm(d_model)
        self.ffn_s = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * mlp_ratio, d_model),
            nn.Dropout(dropout),
        )
        self.norm_s2 = nn.LayerNorm(d_model)

    def forward(self, seq):
        """
        seq: (B, T, D)
        returns: (B, T, D)
        """
        B, T, D = seq.shape

        # repeat the same learnable tokens for each batch item
        g = self.global_tokens.expand(B, -1, -1)   # (B, G, D)

        # Step 1: global tokens attend to sequence
        g_att, _ = self.cross_attn_global(
            query=g, key=seq, value=seq
        )
        g_summary = g_att.mean(dim=1, keepdim=True)   # (B, 1, D)
        # seq = self.norm_g1(seq + g_summary)
        seq = seq + g_summary.expand(-1, T, -1)

        # Step 2: sequence attends back to refined global tokens
        # seq = self.norm_s1(seq + s_att)
        # seq = self.norm_s2(seq + self.ffn_s(seq))

        seq = self.attn(self.norm1(seq))


        return seq


class SWANEncoder(nn.Module):
    """
    SWAN encoder that returns pooled embeddings for downstream heads.

    Inputs
    ------
    x : [B, T, F]  standardized values with NaNs filled by 0
    m : [B, T, F]  observation mask
    d : [B, T, F]  time-since-last-observed delta
    """
    def __init__(
        self,
        n_features: int,
        seq_len: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.value_proj = nn.Linear(n_features, d_model)
        self.mask_proj = nn.Linear(n_features, d_model)
        self.delta_proj = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        self.tape = TAPE(seq_len, d_model)
        self.blocks = nn.ModuleList([
            GCTAFBlock(d_model, n_heads, dropout=dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.pool = nn.Linear(d_model, 1)

    def encode_sequence(self, x: torch.Tensor, m: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        h = (
            self.value_proj(x)
            + self.mask_proj(m)
            + self.delta_proj(torch.log1p(d))
            + self.tape(x.shape[1])
        )

        for blk in self.blocks:
            h = blk(h)

        return self.norm(h)  # [B, T, D]

    def encode(self, x: torch.Tensor, m: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        h = self.encode_sequence(x, m, d)
        attn = torch.softmax(self.pool(h), dim=1)   # [B, T, 1]
        pooled = torch.sum(attn * h, dim=1)         # [B, D]
        return pooled

    def forward(self, x: torch.Tensor, m: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        return self.encode(x, m, d)



class ProjectionHead(nn.Module):
    """MLP projection head for contrastive learning."""
    def __init__(self, embed_dim: int = 128, proj_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, proj_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(h), dim=1)


class SupConSWANModel(nn.Module):
    """
    SWAN encoder + projection head for supervised contrastive learning.
    """
    def __init__(self, encoder: SWANEncoder, embed_dim: int = 128, proj_dim: int = 128):
        super().__init__()
        self.encoder = encoder
        self.projector = ProjectionHead(embed_dim, proj_dim)

    def forward(self, x: torch.Tensor, m: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        h = self.encoder.encode(x, m, d)  # [B, D]
        return self.projector(h)          # [B, proj_dim]


class LinearClassifier(nn.Module):
    """Linear head trained on top of a frozen encoder."""
    def __init__(self, embed_dim: int = 128, num_classes: int = 2):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc(h)



class SupConLoss(nn.Module):
    """
    Single-view supervised contrastive loss.

    Positives are other samples in the batch with the same grouped contrast label:
        0 -> non-flare group (FQ/B/C)
        1 -> flare group     (M/X)
    """
    def __init__(self, temperature: float = 0.15):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        features: torch.Tensor,    # [B, n_views, D]
        labels: torch.Tensor,      # [B]
        temperature: float | None = None,
    ) -> torch.Tensor:
        tau = temperature if temperature is not None else self.temperature
        dev = features.device
        B, n_views, dim = features.shape

        contrast = features.reshape(B * n_views, dim)

        labels_exp = labels.repeat_interleave(n_views)
        pos_mask = torch.eq(
            labels_exp.unsqueeze(1),
            labels_exp.unsqueeze(0),
        ).float().to(dev)

        N = B * n_views
        eye = torch.eye(N, device=dev)
        logits_mask = 1.0 - eye
        pos_mask = pos_mask * logits_mask

        dot = torch.matmul(contrast, contrast.T) / tau
        dot = dot - dot.max(dim=1, keepdim=True).values.detach()

        exp_dot = torch.exp(dot) * logits_mask
        log_prob = dot - torch.log(exp_dot.sum(dim=1, keepdim=True) + 1e-9)

        n_pos = pos_mask.sum(dim=1)
        loss_vec = -(pos_mask * log_prob).sum(dim=1) / (n_pos + 1e-9)

        valid = n_pos > 0
        return loss_vec[valid].mean() if valid.any() else loss_vec.mean()


def tss_hss(cm: np.ndarray) -> Tuple[float, float]:
    """Compute TSS and HSS from a 2x2 confusion matrix."""
    tn, fp, fn, tp = cm.ravel().astype(float)

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    tss = sens + spec - 1.0

    denom_hss = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    hss = 2.0 * (tp * tn - fp * fn) / denom_hss if denom_hss > 0 else 0.0

    return float(tss), float(hss)


@torch.no_grad()
def evaluate(
    encoder: nn.Module,
    classifier: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> Dict[str, object]:
    encoder.eval()
    classifier.eval()

    all_probs, all_preds, all_true = [], [], []
    total_loss, n = 0.0, 0

    for batch in loader:
        xb = batch["x"].to(device)
        m = batch["m"].to(device)
        d = batch["d"].to(device)
        yb = batch["y"].to(device)

        h = encoder.encode(xb, m, d)
        logits = classifier(h)
        loss = loss_fn(logits, yb)

        probs = F.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(dim=1)

        total_loss += loss.item() * xb.size(0)
        n += xb.size(0)
        all_probs.append(probs.cpu())
        all_preds.append(preds.cpu())
        all_true.append(yb.cpu())

    probs = torch.cat(all_probs).numpy()
    preds = torch.cat(all_preds).numpy()
    true = torch.cat(all_true).numpy()

    cm = confusion_matrix(true, preds)
    tss, hss = tss_hss(cm)

    return {
        "loss": total_loss / n,
        "f1_macro": f1_score(true, preds, average="macro", zero_division=0),
        "f1_minor": f1_score(true, preds, pos_label=1, average="binary", zero_division=0),
        "roc_auc": roc_auc_score(true, probs),
        "pr_auc": average_precision_score(true, probs),
        "tss": tss,
        "hss": hss,
        "cm": cm,
        "report": classification_report(true, preds, zero_division=0),
    }


def plot_confusion_matrix(cm: np.ndarray, title: str = "Confusion Matrix", save_path: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"],
        ax=ax,
        annot_kws={"size": 14},
    )
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(history: Dict[str, list], save_path: str | None = None) -> None:
    fig = plt.figure(figsize=(20, 4))
    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.35)

    ax0 = fig.add_subplot(gs[0])
    ax0.plot(history["supcon_loss"], lw=2)
    ax0.set_title("Phase 1 — SupCon Loss", fontweight="bold")
    ax0.set_xlabel("Epoch")
    ax0.set_ylabel("Loss")
    ax0.grid(True, alpha=0.3)

    ax1 = fig.add_subplot(gs[1])
    ax1.plot(history["clf_train_loss"], label="Train", lw=2)
    ax1.plot(history["clf_val_loss"], label="Val", lw=2)
    ax1.set_title("Phase 2 — Cross-Entropy Loss", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[2])
    ax2.plot(history["val_f1_minor"], label="F1 (class 1)", lw=2)
    ax2.plot(history["val_roc_auc"], label="ROC-AUC", lw=2)
    ax2.plot(history["val_pr_auc"], label="PR-AUC", lw=2)
    ax2.set_title("AUC / F1 Metrics", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[3])
    ax3.plot(history["val_tss"], label="TSS", lw=2)
    ax3.plot(history["val_hss"], label="HSS", lw=2)
    ax3.axhline(0, color="gray", lw=1, ls="--", alpha=0.6)
    ax3.set_title("Skill Scores (TSS / HSS)", fontweight="bold")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Score")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)



def train_supcon_epoch(
    model: SupConSWANModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: SupConLoss,
    device: torch.device,
    temperature: float | None = None,
) -> Tuple[float, float]:
    """
    Single-view supervised contrastive training.

    Each sample is encoded once.
    Positives are samples in the same batch sharing the same grouped contrast label.
    """
    model.train()
    total_loss, total_gnorm, n = 0.0, 0.0, 0

    for batch in loader:
        xb = batch["x"].to(device)
        m = batch["m"].to(device)
        d = batch["d"].to(device)
        contrast_y = batch["y_contrast"].to(device)

        z = model(xb, m, d)            # [B, proj_dim]
        features = z.unsqueeze(1)      # [B, 1, proj_dim]
        loss = criterion(features, contrast_y, temperature=temperature)

        optimizer.zero_grad()
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        total_gnorm += float(gnorm.item()) * xb.size(0)
        n += xb.size(0)

    return total_loss / n, total_gnorm / n


def train_clf_epoch(
    encoder: SWANEncoder,
    classifier: LinearClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    """
    Train a linear classifier on top of a frozen encoder.
    """
    encoder.eval()
    classifier.train()
    total_loss, n = 0.0, 0

    for batch in loader:
        xb = batch["x"].to(device)
        m = batch["m"].to(device)
        d = batch["d"].to(device)
        yb = batch["y"].to(device)

        with torch.no_grad():
            h = encoder.encode(xb, m, d)  # [B, D]

        logits = classifier(h)
        loss = loss_fn(logits, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        n += xb.size(0)

    return total_loss / n


def summarize_runs(run_metrics: List[Dict[str, float]]) -> Dict[str, Dict[str, object]]:
    metric_keys = ["loss", "f1_macro", "f1_minor", "roc_auc", "pr_auc", "tss", "hss"]
    summary = {}
    for key in metric_keys:
        vals = np.array([m[key] for m in run_metrics], dtype=np.float64)
        summary[key] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=0)),
            "values": vals,
        }
    return summary


def format_mean_std(mean: float, std: float) -> str:
    return f"{mean:.4f} ± {std:.4f}"



def run_single_experiment(seed: int = 42) -> Dict[str, object]:
    set_seed(seed)

    data_train = np.load("LLM_TS/partition1_grouped.npz", allow_pickle=True)
    data_test = np.load("LLM_TS/partition2_grouped.npz", allow_pickle=True)

    X_train_raw = data_train["features"]           # [N, T, F]
    y_train_raw = data_train["flare_type"]         # [N]

    X_test_raw = data_test["features"]
    y_test_raw = data_test["flare_type"]

    # target_classes = ['F','B', 'M', 'X']
        

    # # 2. Create a mask: True for rows we keep, False for rows we discard
    # # This works if labels_train is a NumPy array of strings
    # mask = np.isin(y_train_raw, target_classes)
    # X_train_raw = X_train_raw[mask]
    # y_train_raw = y_train_raw[mask]
    # logger.info(f'the unique labels in train {np.unique(y_train_raw)}')

    # Keep only the canonical subtype set, and keep all arrays aligned
    keep_train = compute_keep_mask(y_train_raw)
    keep_test = compute_keep_mask(y_test_raw)

    X_train_raw = X_train_raw[keep_train]
    y_train_raw = y_train_raw[keep_train]

    X_test_raw = X_test_raw[keep_test]
    y_test_raw = y_test_raw[keep_test]

    y_train, y_train_contrast = prepare_labels(y_train_raw, split_name="train")
    y_test, y_test_contrast = prepare_labels(y_test_raw, split_name="test")

    # Standardize observed entries only, then build missing-aware tensors
    means, stds = fit_feature_scaler_observed_only(X_train_raw)

    X_train_std = apply_standardization(X_train_raw, means, stds)
    X_test_std = apply_standardization(X_test_raw, means, stds)

    M_train = build_mask(X_train_std)
    M_test = build_mask(X_test_std)

    D_train = build_delta(M_train)
    D_test = build_delta(M_test)

    X_train_fill = fill_missing_with_zero(X_train_std)
    X_test_fill = fill_missing_with_zero(X_test_std)

    logger.info(
        f"[Data] Train: {X_train_fill.shape} | "
        f"Test: {X_test_fill.shape} | "
        f"Binary class counts: {np.bincount(y_train)}"
    )

    train_ds = SWANDataset(X_train_fill, M_train, D_train, y_train, y_train_contrast)
    test_ds = SWANDataset(X_test_fill, M_test, D_test, y_test, y_test_contrast)

    # Balanced sampler for training
    num_classes = int(y_train.max()) + 1
    class_counts = np.bincount(y_train, minlength=num_classes)
    min_count = class_counts.min()
    class_w      = np.where(class_counts > 0, 1.0 / class_counts, 0.0)
    sample_w     = class_w[y_train]

    indices = []
    for c in range(num_classes):
        class_idx = np.where(y_train == c)[0]
        selected = np.random.choice(class_idx, min_count, replace=False)
        indices.extend(selected)

    indices = np.array(indices)
    np.random.shuffle(indices)
    sampler = SubsetRandomSampler(indices)
    # sampler = WeightedRandomSampler(
    #     weights=torch.as_tensor(sample_w, dtype=torch.double),
    #     num_samples=len(sample_w),
    #     replacement=True,
    # )

    BATCH_SIZE = 64
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,
        pin_memory=use_pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=use_pin_memory,
    )

    # Weighted CE for classifier phase
    class_w = np.where(class_counts > 0, 1.0 / class_counts, 0.0)
    lw = class_w.copy()
    nonzero = class_counts > 0
    lw[nonzero] *= nonzero.sum() / lw[nonzero].sum()
    clf_loss_fn = nn.CrossEntropyLoss(
        weight=torch.as_tensor(lw, dtype=torch.float32, device=device)
    )

    # Build models
    EMBED_DIM = 128
    PROJ_DIM = 128
    SEQ_LEN = X_train_fill.shape[1]
    N_FEATURES = X_train_fill.shape[2]

    encoder = SWANEncoder(
        n_features=N_FEATURES,
        seq_len=SEQ_LEN,
        d_model=EMBED_DIM,
        n_heads=4,
        n_layers=3,
        dropout=0.1,
    ).to(device)

    supcon_model = SupConSWANModel(
        encoder=encoder,
        embed_dim=EMBED_DIM,
        proj_dim=PROJ_DIM,
    ).to(device)

    classifier = LinearClassifier(embed_dim=EMBED_DIM, num_classes=2).to(device)

    # Phase 1: SupCon pretraining
    SUPCON_EPOCHS = 70
    WARMUP_EPOCHS = 5
    BASE_LR = 3e-4

    supcon_crit = SupConLoss(temperature=0.15).to(device)
    supcon_opt = torch.optim.AdamW(
        supcon_model.parameters(), lr=BASE_LR, weight_decay=1e-4
    )

    def lr_lambda(epoch: int) -> float:
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        progress = (epoch - WARMUP_EPOCHS) / max(1, SUPCON_EPOCHS - WARMUP_EPOCHS)
        return max(0.01, 0.5 * (1.0 + np.cos(np.pi * progress)))

    supcon_sched = torch.optim.lr_scheduler.LambdaLR(supcon_opt, lr_lambda)

    def get_temperature(epoch: int) -> float:
        if epoch < WARMUP_EPOCHS:
            return 0.5 - (0.5 - 0.15) * (epoch / WARMUP_EPOCHS)
        return 0.15

    history = dict(
        supcon_loss=[],
        clf_train_loss=[],
        clf_val_loss=[],
        val_f1_minor=[],
        val_roc_auc=[],
        val_pr_auc=[],
        val_tss=[],
        val_hss=[],
    )

    logger.info("\n" + "─" * 68)
    logger.info("  PHASE 1 | SWAN + single-view supervised contrastive learning")
    logger.info("          | positives: non-flare(FQ/B/C) together, flare(M/X) together")
    logger.info("─" * 68)

    for epoch in range(1, SUPCON_EPOCHS + 1):
        tau = get_temperature(epoch - 1)
        loss, gnorm = train_supcon_epoch(
            supcon_model, train_loader, supcon_opt, supcon_crit, device, temperature=tau
        )
        supcon_sched.step()
        history["supcon_loss"].append(loss)

        if epoch % 5 == 0 or epoch == 1:
            warmup_tag = " [warmup]" if epoch <= WARMUP_EPOCHS else ""
            logger.info(
                f"  Epoch {epoch:3d}/{SUPCON_EPOCHS} | "
                f"SupCon: {loss:.4f} | τ={tau:.3f} | "
                f"GradNorm: {gnorm:.2f} | LR: {supcon_sched.get_last_lr()[0]:.2e}{warmup_tag}"
            )

    # Phase 2: frozen encoder + linear classifier
    encoder = supcon_model.encoder
    CLF_EPOCHS = 40
    clf_opt = torch.optim.Adam(classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    clf_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        clf_opt, T_max=CLF_EPOCHS, eta_min=1e-5
    )

    best_tss = -1.0
    best_state = None

    logger.info("\n" + "─" * 60)
    logger.info("  PHASE 2 | linear classifier on frozen SWAN encoder")
    logger.info("─" * 60)

    for epoch in range(1, CLF_EPOCHS + 1):
        tr_loss = train_clf_epoch(encoder, classifier, train_loader, clf_opt, clf_loss_fn, device)
        clf_sched.step()

        m = evaluate(encoder, classifier, test_loader, clf_loss_fn, device)
        history["clf_train_loss"].append(tr_loss)
        history["clf_val_loss"].append(m["loss"])
        history["val_f1_minor"].append(m["f1_minor"])
        history["val_roc_auc"].append(m["roc_auc"])
        history["val_pr_auc"].append(m["pr_auc"])
        history["val_tss"].append(m["tss"])
        history["val_hss"].append(m["hss"])

        if m["tss"] > best_tss:
            best_tss = m["tss"]
            best_state = {
                "encoder": {k: v.cpu().clone() for k, v in encoder.state_dict().items()},
                "classifier": {k: v.cpu().clone() for k, v in classifier.state_dict().items()},
            }

        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                f"  Epoch {epoch:3d}/{CLF_EPOCHS} | "
                f"Train: {tr_loss:.4f} | Val: {m['loss']:.4f} | "
                f"F1(1): {m['f1_minor']:.4f} | ROC: {m['roc_auc']:.4f} | "
                f"PR: {m['pr_auc']:.4f} | TSS: {m['tss']:.4f} | HSS: {m['hss']:.4f}"
            )

    if best_state is not None:
        encoder.load_state_dict(best_state["encoder"])
        classifier.load_state_dict(best_state["classifier"])

    encoder.to(device)
    classifier.to(device)

    final = evaluate(encoder, classifier, test_loader, clf_loss_fn, device)

    logger.info("\n" + "═" * 60)
    logger.info("  FINAL TEST RESULTS")
    logger.info("═" * 60)
    logger.info(f"  F1  (macro)   : {final['f1_macro']:.4f}")
    logger.info(f"  F1  (class 1) : {final['f1_minor']:.4f}")
    logger.info(f"  ROC-AUC       : {final['roc_auc']:.4f}")
    logger.info(f"  PR-AUC        : {final['pr_auc']:.4f}")
    logger.info(f"  TSS           : {final['tss']:.4f}")
    logger.info(f"  HSS           : {final['hss']:.4f}")
    logger.info("\n" + final["report"])

    plot_confusion_matrix(final["cm"], title="Test Set Confusion Matrix", save_path="confusion_matrix.png")
    plot_training_curves(history, save_path="training_curves.png")

    torch.save(
        {
            "encoder": encoder.state_dict(),
            "classifier": classifier.state_dict(),
            "means": means,
            "stds": stds,
            "embed_dim": EMBED_DIM,
            "seq_len": SEQ_LEN,
            "n_features": N_FEATURES,
            "backbone": "SWAN",
        },
        "supcon_swan_best.pt",
    )
    logger.info("[Saved] supcon_swan_best.pt")
    logger.info("[Saved] confusion_matrix.png")
    logger.info("[Saved] training_curves.png")

    return final


def main() -> Dict[str, Dict[str, object]]:
    N_RUNS = 3
    seeds = [42 + i for i in range(N_RUNS)]
    all_results = []

    logger.info("\n" + "=" * 68)
    logger.info(f"  REPEATED RUNS SUMMARY | mean ± std over {N_RUNS} runs")
    logger.info(f"  Seeds: {seeds}")
    logger.info("=" * 68)

    for run_idx, seed in enumerate(seeds, start=1):
        logger.info("\n" + "-" * 68)
        logger.info(f"  RUN {run_idx}/{N_RUNS} | seed={seed}")
        logger.info("-" * 68)
        final = run_single_experiment(seed=seed)
        all_results.append(final)

        logger.info(
            "  Run result | "
            f"F1(macro): {final['f1_macro']:.4f} | "
            f"F1(class 1): {final['f1_minor']:.4f} | "
            f"ROC-AUC: {final['roc_auc']:.4f} | "
            f"PR-AUC: {final['pr_auc']:.4f} | "
            f"TSS: {final['tss']:.4f} | "
            f"HSS: {final['hss']:.4f}"
        )

    summary = summarize_runs(all_results)

    logger.info("\n" + "═" * 60)
    logger.info("  FINAL AGGREGATED RESULTS (mean ± std)")
    logger.info("═" * 60)
    logger.info(f"  Loss          : {format_mean_std(summary['loss']['mean'], summary['loss']['std'])}")
    logger.info(f"  F1  (macro)   : {format_mean_std(summary['f1_macro']['mean'], summary['f1_macro']['std'])}")
    logger.info(f"  F1  (class 1) : {format_mean_std(summary['f1_minor']['mean'], summary['f1_minor']['std'])}")
    logger.info(f"  ROC-AUC       : {format_mean_std(summary['roc_auc']['mean'], summary['roc_auc']['std'])}")
    logger.info(f"  PR-AUC        : {format_mean_std(summary['pr_auc']['mean'], summary['pr_auc']['std'])}")
    logger.info(f"  TSS           : {format_mean_std(summary['tss']['mean'], summary['tss']['std'])}")
    logger.info(f"  HSS           : {format_mean_std(summary['hss']['mean'], summary['hss']['std'])}")

    return summary


if __name__ == "__main__":
    main()