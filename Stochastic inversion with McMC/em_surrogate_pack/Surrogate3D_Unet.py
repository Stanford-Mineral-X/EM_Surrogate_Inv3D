import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


# Define NN
# Building blocks
class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_p=0.2, norm="instance"):
        super().__init__()
        Norm = nn.InstanceNorm3d if norm == "instance" else nn.BatchNorm3d
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            Norm(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout_p) if dropout_p > 0 else nn.Identity(),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            Norm(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout_p) if dropout_p > 0 else nn.Identity(),
        )

    def forward(self, x):
        return self.net(x)

class Downsample3D(nn.Module):
    """Learnable downsample: halves D,H,W via stride-2 conv (MPS-friendly)."""
    def __init__(self, in_ch, out_ch, norm="instance"):
        super().__init__()
        Norm = nn.InstanceNorm3d if norm == "instance" else nn.BatchNorm3d
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)
        self.bn   = Norm(out_ch)
        self.act  = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

def _interp1d_along(x, out_len: int, dim: int):
    """
    Upsample along a single spatial dim using 1D linear interpolate (MPS-safe).
    x: (B, C, D, H, W)
    dim: 2 (D) or 3 (H) or 4 (W)
    """
    assert dim in (2, 3, 4)
    B, C, D, H, W = x.shape
    if dim == 2:
        x = x.permute(0,1,3,4,2).contiguous().view(B*C*H*W, 1, D)  # (BCHW,1,D)
        x = F.interpolate(x, size=out_len, mode="linear", align_corners=False)
        x = x.view(B, C, H, W, out_len).permute(0,1,4,2,3).contiguous()
    elif dim == 3:
        x = x.permute(0,1,2,4,3).contiguous().view(B*C*D*W, 1, H)  # (BCD*W,1,H)
        x = F.interpolate(x, size=out_len, mode="linear", align_corners=False)
        x = x.view(B, C, D, W, out_len).permute(0,1,2,4,3).contiguous()
    else:  # dim == 4
        x = x.view(B*C*D*H, 1, W)                                  # (BCD*H,1,W)
        x = F.interpolate(x, size=out_len, mode="linear", align_corners=False)
        x = x.view(B, C, D, H, out_len)
    return x

class UpSample3D_Separable(nn.Module):
    """
    MPS-safe upsampling: do 1D linear interpolate along D, then H, then W,
    then a 1x1x1 conv to set channels.
    """
    def __init__(self, in_ch, out_ch, norm="instance"):
        super().__init__()
        Norm = nn.InstanceNorm3d if norm == "instance" else nn.BatchNorm3d
        self.proj = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False),
            Norm(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, target_size):
        # target_size is a tuple (D_t, H_t, W_t)
        Dt, Ht, Wt = target_size
        if x.shape[2] != Dt:
            x = _interp1d_along(x, Dt, dim=2)
        if x.shape[3] != Ht:
            x = _interp1d_along(x, Ht, dim=3)
        if x.shape[4] != Wt:
            x = _interp1d_along(x, Wt, dim=4)
        return self.proj(x)


def sample_along_x_linear(y_1d: torch.Tensor, x_coords_norm, N_x: int) -> torch.Tensor:
    """
    y_1d: (B, C, Dx)  — features along x after (y,z) squeeze
    x_coords_norm: (N_x,) or (B,N_x), numpy or torch, normalized to [-1,1]
    returns: (B, C, N_x)
    """
    B, C, Dx = y_1d.shape

    # accept numpy or torch, move/cast to match y_1d
    if isinstance(x_coords_norm, np.ndarray):
        x_coords = torch.as_tensor(x_coords_norm, device=y_1d.device, dtype=y_1d.dtype)
    elif not torch.is_tensor(x_coords_norm):
        x_coords = torch.tensor(x_coords_norm, device=y_1d.device, dtype=y_1d.dtype)
    else:
        x_coords = x_coords_norm.to(device=y_1d.device, dtype=y_1d.dtype)

    # normalize [-1,1] -> index space [0, Dx-1]
    x_idx = 0.5 * (x_coords + 1) * (Dx - 1)   # (N_x,) or (B,N_x)

    # floor/ceil and weights
    i0 = torch.floor(x_idx).clamp(0, Dx - 1)
    i1 = (i0 + 1).clamp(0, Dx - 1)
    w1 = (x_idx - i0).clamp(0, 1)
    w0 = 1 - w1

    # shape to (B,1,N_x)
    if x_idx.dim() == 1:
        i0 = i0[None, :].expand(B, -1)
        i1 = i1[None, :].expand(B, -1)
        w0 = w0[None, :].expand(B, -1)
        w1 = w1[None, :].expand(B, -1)

    # long indices for gather
    i0_long = i0.long()  # (B,N_x)
    i1_long = i1.long()

    # gather expects index shape (B,C,N_x)
    i0_g = i0_long[:, None, :].expand(B, C, -1)
    i1_g = i1_long[:, None, :].expand(B, C, -1)

    # (B,C,N_x) from (B,C,Dx)
    y0 = torch.gather(y_1d, dim=2, index=i0_g)
    y1 = torch.gather(y_1d, dim=2, index=i1_g)

    # blend
    w0 = w0[:, None, :]  # (B,1,N_x)
    w1 = w1[:, None, :]
    y_lin = w0 * y0 + w1 * y1  # (B,C,N_x)
    return y_lin


def center_crop_to(x, ref):
    """Center-crop x spatial dims (D,H,W) to ref's."""
    _, _, D, H, W = x.shape
    _, _, Dr, Hr, Wr = ref.shape
    d0 = max((D - Dr) // 2, 0)
    h0 = max((H - Hr) // 2, 0)
    w0 = max((W - Wr) // 2, 0)
    return x[:, :, d0:d0+Dr, h0:h0+Hr, w0:w0+Wr]

    

# ------------ Main model (no pooling ops) ------------
class UNet3D_Profile(nn.Module):
    """
    Input:  (B, 1, 45, 20, 45)  ln-conductivity
    Output: (B, n_times, n_locs)    (12 time channels, 19 x-positions)
    """
    def __init__(
        self,
        in_channels=1,
        n_times=12,
        base_ch=32,
        depth=4,
        dropout_p=0.2,
        norm="instance",
        learn_mix=False,      # False = depthwise (per-channel) squeeze over (y,z)
        use_grid_x=False,     # False = x resample by interpolate to 19; True = grid_sample (optional)
        N_x=19
    ):
        super().__init__()
        self.depth = depth
        self.n_times = n_times
        self.learn_mix = learn_mix
        self.use_grid_x = use_grid_x
        self.N_x = N_x

        # Encoder stages and learnable downsamples
        chans = [base_ch * (2 ** i) for i in range(depth)]  # e.g., [32, 64, 128, 256]
        self.enc = nn.ModuleList()
        self.down = nn.ModuleList()

        # enc[0]: 1 -> chans[0]
        self.enc.append(ConvBlock3D(in_channels, chans[0], dropout_p=dropout_p, norm=norm))

        # enc[1..]: chans[i] -> chans[i]
        for i in range(1, depth):
            self.enc.append(ConvBlock3D(chans[i], chans[i], dropout_p=dropout_p, norm=norm))
        
        # down[i]: chans[i] -> chans[i+1]  for i = 0 .. depth-2
        for i in range(depth - 1):
            self.down.append(Downsample3D(chans[i], chans[i+1], norm=norm))

        # Bottleneck
        self.bottleneck = ConvBlock3D(chans[-1], chans[-1] * 2, dropout_p=dropout_p, norm=norm)

        # Decoder (transpose-conv upsampling)
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        ch = chans[-1] * 2
        for c in reversed(chans):
            self.up.append(UpSample3D_Separable(ch, c, norm=norm))         # <-- HERE
            self.dec.append(ConvBlock3D(c * 2, c, dropout_p=dropout_p, norm=norm))
            ch = c

        # Harmonize channel count at the end of decoder
        self.final_feat = nn.Conv3d(chans[0], chans[0], kernel_size=1)

        # Learned squeeze over (y,z); initialize lazily to match Hy,Wz
        #self.squeeze_yz = None  # Conv3d(C->C, kernel=(1,Hy,Wz), groups=C or 1)
        C = chans[0]                                # channels right before squeeze
        groups = 1 if self.learn_mix else C
        self.squeeze_yz = nn.Conv3d(
            in_channels=C, out_channels=C,
            kernel_size=(1, 20, 45),  # H=20, W=45
            stride=1, padding=0, bias=True, groups=groups
        )

        # Head to map features → n_times along x
        self.time_head = nn.Sequential(
            nn.Conv1d(chans[0], chans[0], kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p) if dropout_p > 0 else nn.Identity(),
            nn.Conv1d(chans[0], n_times, kernel_size=1),
        )

    def forward(self, x, x_coords_norm=None):
        # x: (B, 1, 45, 20, 45)
        B = x.size(0)

        # ----- Encoder -----
        skips = []
        y = x
        for i in range(self.depth):
            y = self.enc[i](y)          # (B, chans[i], ...)
            skips.append(y)
            if i < self.depth - 1:
                y = self.down[i](y)     # (B, chans[i+1], ...)...)

        # ----- Bottleneck -----
        y = self.bottleneck(y)

        # ----- Decoder -----
        for i in range(self.depth):
            skip = skips[-(i + 1)]
            y = self.up[i](y, target_size=skip.shape[2:])   # MPS-safe upsample
            y = torch.cat([y, skip], dim=1)
            y = self.dec[i](y)

        # Bring back to the exact input spatial size (45,20,45)
        y = self.final_feat(y)
        Dt, Ht, Wt = x.shape[2:]
        if y.shape[2] != Dt:
            y = _interp1d_along(y, Dt, dim=2)
        if y.shape[3] != Ht:
            y = _interp1d_along(y, Ht, dim=3)
        if y.shape[4] != Wt:
            y = _interp1d_along(y, Wt, dim=4)

        # ----- Learned squeeze over (y,z) → (B, C, 45, 1, 1) -----
        _, C, Dx, Hy, Wz = y.shape
        if (self.squeeze_yz is None or
            self.squeeze_yz.in_channels != C or
            self.squeeze_yz.out_channels != C or
            self.squeeze_yz.weight.shape[2:] != (1, Hy, Wz)):
            groups = 1 if self.learn_mix else C  # groups=C = depthwise per-channel squeeze
            self.squeeze_yz = nn.Conv3d(
                in_channels=C, out_channels=C,
                kernel_size=(1, Hy, Wz), stride=1, padding=0,
                bias=True, groups=groups
            ).to(y.device)
        y = self.squeeze_yz(y)                   # (B, C, Dx, 1, 1)
        y = y.squeeze(-1).squeeze(-1)            # (B, C, Dx)

        # ----- Choose 19 x-positions -----
        if self.use_grid_x:
            # Optional path (irregular stations). Uses grid_sample on a (B,C,Dx,1,1) tensor.
            # grid_sample(3D) is usually OK on MPS, but if you ever hit an op gap,
            # prefer the interpolate path below + custom 1D linear gather.
            if x_coords_norm is None:
                raise ValueError("use_grid_x=True requires x_coords_norm in [-1,1] of shape (N_x,) or (B,N_x).")
            y3 = y.unsqueeze(-1).unsqueeze(-1)  # (B, C, Dx, 1, 1)
            if x_coords_norm.dim() == 1:
                xg = x_coords_norm.view(1, self.N_x, 1, 1).expand(B, -1, -1, -1)
            else:
                xg = x_coords_norm.view(B, self.N_x, 1, 1)
            yg = torch.zeros_like(xg)
            zg = torch.zeros_like(xg)
            grid = torch.stack((xg, yg, zg), dim=-1)  # (B, N_x, 1, 1, 3)
            y = sample_along_x_linear(y, x_coords_norm, self.N_x)  # -> (B,C,19)
        else:
            # MPS-safe: 1D interpolate to exactly 19 samples along x
            y = F.interpolate(y, size=self.N_x, mode="linear", align_corners=False)  # (B, C, 19)

        # ----- Map features → time channels -----
        out = self.time_head(y)  # (B, n_times, 19)
        return out


