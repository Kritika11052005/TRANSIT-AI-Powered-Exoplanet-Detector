"""
Model architecture — must exactly match NB-02 Cell 6.
Any drift here vs. the trained checkpoint will break state_dict loading.
"""
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=25, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1), :])


class CNNBranch(nn.Module):
    def __init__(self, channels, kernels, dropout):
        super().__init__()
        blocks = []
        in_ch = 1
        for out_ch, k in zip(channels, kernels):
            blocks += [
                nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(out_ch), nn.GELU(), nn.MaxPool1d(2),
                nn.Dropout(dropout * 0.4),
            ]
            in_ch = out_ch
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)


class MLPBranch(nn.Module):
    def __init__(self, in_dim, hidden_dims, dropout):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dropout * 0.4)]
            d = h
        self.net = nn.Sequential(*layers)
        self.out_dim = d

    def forward(self, x):
        return self.net(x)


class AttentionTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dim_ff, dropout):
        super().__init__()
        self.attn  = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads,
                                            dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.attn(
            x_norm, x_norm, x_norm, need_weights=True, average_attn_weights=True
        )
        x = x + self.drop(attn_out)
        x = x + self.drop(self.ff(self.norm2(x)))
        return x, attn_weights


class TransitClassifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        chs  = cfg["CNN_CHANNELS"]
        ks   = cfg["CNN_KERNELS"]
        d    = cfg["TF_D_MODEL"]
        drop = cfg["DROPOUT"]
        self._seq_len = cfg["N_BINS"] // (2 ** len(chs))

        self.cnn     = CNNBranch(chs, ks, drop)
        self.pos_enc = PositionalEncoding(d, max_len=self._seq_len, dropout=drop * 0.2)
        self.attn_block = AttentionTransformerBlock(
            d_model=d, n_heads=cfg["TF_N_HEADS"], dim_ff=cfg["TF_DIM_FF"], dropout=drop * 0.3
        )
        self.mlp = MLPBranch(cfg["N_SCALAR"], cfg["MLP_HIDDEN"], drop)
        mlp_out  = cfg["MLP_HIDDEN"][-1]
        fusion_in = d + mlp_out

        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, cfg["FUSION_HIDDEN"]), nn.GELU(), nn.Dropout(drop),
            nn.Linear(cfg["FUSION_HIDDEN"], cfg["N_CLASSES"]),
        )
        self.reg_head = nn.Sequential(
            nn.Linear(fusion_in, 32), nn.GELU(), nn.Dropout(drop * 0.5),
            nn.Linear(32, cfg["N_REG"]),
        )

    def forward(self, profile, scalar):
        cnn_out = self.cnn(profile)
        seq     = cnn_out.permute(0, 2, 1)
        seq     = self.pos_enc(seq)
        seq, attn_weights = self.attn_block(seq)
        profile_feat = seq.mean(dim=1)
        scalar_feat  = self.mlp(scalar)
        fused   = torch.cat([profile_feat, scalar_feat], dim=1)
        logits  = self.fusion(fused)
        reg_out = self.reg_head(fused)
        return logits, reg_out, attn_weights