import torch
import torch.nn as nn

from yolox.models.ioumemory_bank import IoUMemoryBank


class FFN(nn.Module):
    """Feed-Forward Network as used in Transformer blocks."""
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class CrossAttentionLocal(nn.Module):
    """
    Multi-Head Cross Attention operating purely on raw feature representations.
    - Query: current_feats (B, N_curr, C)
    - Key & Value: memory_feats (B, N_mem, C)
    """
    def __init__(self, dim, num_heads=8, bias=False, attn_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=bias)
        self.k_proj = nn.Linear(dim, dim, bias=bias)
        self.v_proj = nn.Linear(dim, dim, bias=bias)

        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, current_feats: torch.Tensor, memory_feats: torch.Tensor) -> torch.Tensor:
        B, N_curr, C = current_feats.shape


        N_mem = memory_feats.shape[1]

        # Multi-Head Q, K, V Projections
        q = self.q_proj(current_feats).reshape(B, N_curr, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(memory_feats).reshape(B, N_mem, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(memory_feats).reshape(B, N_mem, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Compute Scaled Dot-Product Attention: (Q @ K^T) * scale
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N_curr, N_mem)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Aggregate Memory Values
        out = (attn @ v).transpose(1, 2).reshape(B, N_curr, C)
        return out


class MemoryTransformerBlock(nn.Module):
    """
    Single Cross-Attention Transformer Block for raw features.
    """
    def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=False,
                 dropout=0., attn_drop=0., drop_path=0., use_ffn=True):
        super().__init__()
        self.use_ffn = use_ffn

        self.norm1 = nn.LayerNorm(dim)
        self.attn = CrossAttentionLocal(
            dim=dim,
            num_heads=num_heads,
            bias=qkv_bias,
            attn_drop=attn_drop
        )

        if self.use_ffn:
            self.norm2 = nn.LayerNorm(dim)
            self.mlp = FFN(dim, int(dim * mlp_ratio), dropout=dropout)

        self.dropout = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, current_feats: torch.Tensor, memory_feats: torch.Tensor = None) -> torch.Tensor:

        if memory_feats is None:
            memory_feats = current_feats

        # Pre-LN Cross-Attention Sub-layer
        attn_out = self.attn(self.norm1(current_feats), memory_feats)
        x = current_feats + self.dropout(attn_out)

        # Feed-Forward Sub-layer
        if self.use_ffn:
            x = x + self.dropout(self.mlp(self.norm2(x)))

        return x


class MemoryCrossAggregator(nn.Module):
    """
    Stack of MemoryTransformerBlock layers, mirroring the LocalAggregation
    wrapper pattern in post_trans.py. Intended as the per-FPN-level (P3 / P4 / P5)
    aggregator reading from that level's memory bank.
    """
    def __init__(self, dim, num_heads=8, blocks=1, mlp_ratio=4.,
                 qkv_bias=False, dropout=0., attn_drop=0., drop_path=0., memory_length=4800, key_length=480, updating_policy="random", diverse_threshold=0.5,
                 diversity_sim_score_thresh=0.5):
        super().__init__()
        self.blocks = blocks
        self.layers = nn.ModuleList([
            MemoryTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                dropout=dropout,
                attn_drop=attn_drop,
                drop_path=drop_path
            ) for _ in range(blocks)
        ])
        self.memory_bank = IoUMemoryBank(max_length=memory_length, key_length=key_length, updating_policy=updating_policy, diversity_iou_thresh=diverse_threshold,
                                          diversity_sim_score_thresh=diversity_sim_score_thresh)
    def reset_memory_bank(self):
        #logger.info("reset_memory_bank")
        self.memory_bank.reset()

    def update_memory_bank(self, x, x_info):
        #logger.info("update_memory_bank")
        self.memory_bank.update(x, x_info)

    def pre_update_candidate_memory_bank(self, x, x_info):
        #logger.info("pre_update_candidate_memory_bank")
        self.memory_bank.pre_update_candidate_memory(x, x_info)

    def init_memory_bank(self, x, x_info):
        #logger.info("init_memory_bank")
        #logger.info("x.shape: {}".format(x.shape))
        self.memory_bank.init_memory(x, x_info)

    def post_init_memory_bank(self, result):
        self.memory_bank.post_init_memory(result)

    def post_update_memory_bank(self, result):
        self.memory_bank.post_update_memory(result)

    def post_update_candidate_memory_bank(self, result):
        self.memory_bank.post_update_candidate_memory(result)

    def forward(self, x_cur: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_cur:    Current frame raw features  (N_curr, C)
        Returns:
            Refined current raw features (N_curr, C)
        """
        memory_feats = self.memory_bank.sample()
        if len(memory_feats) != 0:
            x = x_cur.unsqueeze(0)
            # self.memory_bank.feat is a plain tensor attribute, not a registered
            # buffer, so module-wide dtype/device casts (e.g. model.half() before
            # eval) never touch it; it can lag behind x_cur's dtype/device.
            memory_feats = memory_feats.unsqueeze(0).to(dtype=x_cur.dtype, device=x_cur.device)

            for layer in self.layers:
                x = layer(x, memory_feats)
            x_cur = x.squeeze(0)

        return x_cur