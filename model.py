# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder

# ------------------------------------------------------------------- helpers
def sample_gumbel(shape, eps=1e-20, device=None):
    u = torch.rand(shape, device=device)
    return -torch.log(-torch.log(u + eps) + eps)

def gumbel_topk_mask(logits_flat, k, tau=1.0, hard=False):
    g = sample_gumbel(logits_flat.shape, device=logits_flat.device)
    p = F.softmax((logits_flat + g) / tau, dim=1)
    m_soft = k * p
    if not hard:
        return m_soft, None
    topk_idx = m_soft.topk(k, dim=1).indices
    m_hard = torch.zeros_like(m_soft).scatter_(1, topk_idx, 1.0)
    return m_soft, m_hard

# ------------------------------------------------------------------- ASPP
class ASPPModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), self.relu)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels), self.relu)

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_channels), self.relu)

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_channels), self.relu)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), self.relu)

        self.out_proj = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), self.relu)

    def forward(self, x):
        size = x.size()[2:]
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        out4 = self.conv4(x)

        out5 = self.global_pool(x)
        out5 = self.conv5(out5)
        out5 = F.interpolate(out5, size=size, mode='bilinear', align_corners=True)

        out = torch.cat([out1, out2, out3, out4, out5], dim=1)
        return self.out_proj(out)

# ------------------------------------------------------------------- network
class PathLossNet(nn.Module):
    """UNet completion + optional selector head (Gumbel-Top-k)."""
    def __init__(self,
                 in_channels: int = 4,
                 use_selector: bool = True,
                 k_frac: float    = 0.005,
                 tau: float       = 0.7,
                 extra_channels: int = 2,             # mask + sparse
                 encoder_name: str  = "resnet18",
                 decoder_channels   = (256,128,64,32,16)):
        super().__init__()
        self.use_selector = use_selector
        self.k_frac = k_frac
        self.tau    = tau

        # -------- encoder --------------------------------------------------
        self.encoder = get_encoder(encoder_name, in_channels=in_channels, weights=None)
        enc_ch = list(self.encoder.out_channels)

        # -------- ASPP on bottleneck --------------------------------------
        self.aspp = ASPPModule(enc_ch[-1], enc_ch[-1])

        # -------- selector head -------------------------------------------
        if use_selector:
            self.selector_head = nn.Conv2d(enc_ch[-1], 1, 1)

        # -------- completion decoder --------------------------------------
        enc_ch[0] += extra_channels
        self.decoder = UnetDecoder(
            encoder_channels = enc_ch,
            decoder_channels = decoder_channels,
            n_blocks         = len(decoder_channels),
            use_batchnorm    = True)
        self.seg_head = nn.Conv2d(decoder_channels[-1], 1, 1)

    # ------------------------------------------------------------------- fwd
    def forward(self,
                x,                  # (B,C,H,W)
                y_full     = None,
                ext_mask   = None,
                ext_sparse = None,
                tau        = None,
                hard: bool = True):
        B, _, H, W = x.shape
        tau = tau or self.tau

        # -------- encoder + ASPP
        feats = self.encoder(x)
        feats[-1] = self.aspp(feats[-1])     # enrich bottleneck

        # -------- selector branch
        if self.use_selector:
            logits = self.selector_head(F.interpolate(feats[-1],
                                                      size=(H, W),
                                                      mode='bilinear',
                                                      align_corners=False))
            logits_flat = logits.flatten(2).transpose(1, 2)
            k = max(1, int(self.k_frac * H * W))
            m_soft, m_hard = gumbel_topk_mask(logits_flat, k, tau, hard)
            m_soft = m_soft.view(B, 1, H, W)
            m_hard = m_hard.view(B, 1, H, W) if m_hard is not None else None

            if self.training:
                if y_full is None:
                    raise ValueError("y_full required when training with selector.")
                y_sparse = (y_full * m_hard) if hard else (y_full * m_soft)
            else:
                y_sparse = torch.zeros_like(m_soft)
        else:
            if ext_mask is None or ext_sparse is None:
                raise ValueError("Provide ext_mask & ext_sparse when selector disabled.")
            m_soft, y_sparse = ext_mask, ext_sparse
            logits = m_hard = None

        # -------- inject mask + sparse into first skip
        feats[0] = torch.cat([feats[0], m_soft, y_sparse], dim=1)

        # -------- decode & predict
        dec  = self.decoder(*feats)
        pred = self.seg_head(dec).squeeze(1)

        return {
            "pred": pred,
            "logits": logits,
            "mask_soft": m_soft,
            "mask_hard": m_hard,
            "sparse": y_sparse
        }
