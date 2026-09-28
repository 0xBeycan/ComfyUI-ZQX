"""Low-frequency initial-noise composition from a reference latent.

Method (heuristic built on FreeInit's frequency split, arXiv:2312.07537):
FreeInit re-initialises video noise as  LPF(z_T_noised) + HPF(fresh noise)
with a Gaussian / Butterworth / ideal low-pass filter in normalised frequency.
Here the low band comes from a *real photo latent* instead of a previous
sample, so the model gets a composition / lighting prior (where the big dark
and bright masses are) without copying any detail.

Let eps ~ N(0, I) be the sampler's noise and r the reference latent,
standardised per (batch, channel) to zero mean / unit std and resized to the
noise resolution.  With F = FFT2 over (H, W), H(f) in [0,1] the low-pass
filter and alpha in [0,1] the strength, per frequency bin f:

    b(f)   = sqrt(alpha) * H(f)          (reference weight)
    a(f)   = sqrt(1 - alpha * H(f)^2)    (noise weight;  a^2 + b^2 = 1 in every bin)
    c      = sqrt( sum H^2 |F(eps)|^2 / sum H^2 |F(r)|^2 )        (per batch, channel)
    F(out) = a F(eps) + b c F(r),        out = Re IFFT2(F(out))

* Where H = 0 (high band) the output spectrum is exactly the noise's.  With an
  ideal filter and alpha = 1 the low band is exactly c * LPF(r); with alpha < 1
  it is sqrt(1-alpha) LPF(eps) + sqrt(alpha) c LPF(r).
* a^2 + b^2 = 1 per bin plus the H^2-weighted energy match c keep the expected
  energy equal to the noise's, i.e. unit variance, also for soft (Gaussian /
  Butterworth) filters.  (FreeInit's plain LPF(z) + (1 - LPF)(eps) complement
  loses the 2 H (1 - H) cross energy and ends up below unit variance.)
* alpha = 0 returns the noise tensor unchanged (short-circuit, bitwise identical).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def normalized_radius(h: int, w: int, device=None) -> torch.Tensor:
    """D(f) = sqrt((2 f_y)^2 + (2 f_x)^2), f in cycles/sample (unshifted FFT layout).

    Equals FreeInit's d = sqrt((2h/H - 1)^2 + (2w/W - 1)^2) on the fftshifted grid;
    D = 1 is the Nyquist frequency along one axis.
    """
    fy = torch.fft.fftfreq(h, device=device, dtype=torch.float64)
    fx = torch.fft.fftfreq(w, device=device, dtype=torch.float64)
    return torch.sqrt((2 * fy)[:, None] ** 2 + (2 * fx)[None, :] ** 2)


def lowpass_filter(h: int, w: int, cutoff: float, kind: str = "gaussian", order: int = 4,
                   device=None) -> torch.Tensor:
    """Low-pass filter H(f) on the unshifted FFT grid, float64, shape (h, w).

    gaussian:    exp(-D^2 / (2 d0^2))            (FreeInit gaussian_low_pass_filter)
    butterworth: 1 / (1 + (D^2 / d0^2)^n)        (FreeInit butterworth_low_pass_filter)
    ideal:       1[D <= d0]
    """
    if cutoff <= 0:
        raise ValueError("cutoff must be > 0")
    d = normalized_radius(h, w, device=device)
    if kind == "gaussian":
        return torch.exp(-(d ** 2) / (2 * cutoff ** 2))
    if kind == "butterworth":
        if order < 1:
            raise ValueError("butterworth order must be >= 1")
        return 1.0 / (1.0 + (d ** 2 / cutoff ** 2) ** order)
    if kind == "ideal":
        return (d <= cutoff).to(torch.float64)
    raise ValueError(f"unknown filter kind {kind!r}")


def standardize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per (batch, channel) zero-mean / unit-std over spatial dims."""
    dims = tuple(range(2, x.ndim))
    mu = x.mean(dim=dims, keepdim=True)
    sd = x.std(dim=dims, keepdim=True, unbiased=False)
    return (x - mu) / (sd + eps)


def prepare_reference(ref: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Resize (bilinear, antialiased when downscaling) a (B, C, H', W') latent to (h, w) and standardise."""
    if ref.ndim != 4:
        raise ValueError(f"reference latent must be 4-D (B, C, H, W), got {tuple(ref.shape)}")
    r = ref.to(torch.float64)
    if r.shape[-2:] != (h, w):
        r = F.interpolate(r, size=(h, w), mode="bilinear", align_corners=False,
                          antialias=(h < r.shape[-2] or w < r.shape[-1]))
    return standardize(r)


def lowfreq_mix(noise: torch.Tensor, ref: torch.Tensor, strength: float, cutoff: float,
                kind: str = "gaussian", order: int = 4) -> torch.Tensor:
    """See module docstring.  noise: (B, C, H, W) (or (B, C, T, H, W) with T handled per frame)."""
    if not (0.0 <= strength <= 1.0) or not math.isfinite(strength):
        raise ValueError(f"strength must be in [0, 1], got {strength}")
    if strength == 0.0:
        return noise
    squeeze_t = False
    if noise.ndim == 5:
        b, c, t, h, w = noise.shape
        n4 = noise.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        squeeze_t = True
    elif noise.ndim == 4:
        n4 = noise
        b, c, h, w = noise.shape
    else:
        raise ValueError(f"noise must be 4-D or 5-D, got {tuple(noise.shape)}")
    if ref.ndim == 5:
        if ref.shape[2] != 1:
            raise ValueError("5-D reference latents must have a single frame")
        ref = ref[:, :, 0]
    if ref.shape[1] != c:
        raise ValueError(f"reference latent has {ref.shape[1]} channels, noise has {c}")
    r = prepare_reference(ref, h, w).to(n4.device)
    if r.shape[0] == 1:
        r = r.expand(n4.shape[0], -1, -1, -1)
    elif r.shape[0] != n4.shape[0]:
        raise ValueError(f"reference batch {r.shape[0]} must be 1 or equal to noise batch {n4.shape[0]}")

    hfilt = lowpass_filter(h, w, cutoff, kind, order, device=n4.device)
    fe = torch.fft.fft2(n4.to(torch.float64))
    fr = torch.fft.fft2(r)
    e_noise = (hfilt * fe).abs().pow(2).sum(dim=(-2, -1), keepdim=True)
    e_ref = (hfilt * fr).abs().pow(2).sum(dim=(-2, -1), keepdim=True)
    if torch.any(e_ref <= 0):
        raise ValueError("reference latent has no energy in the low-frequency band (constant image?)")
    cscale = torch.sqrt(e_noise / e_ref)
    alpha = float(strength)
    w_ref = math.sqrt(alpha) * hfilt
    w_noise = torch.sqrt(torch.clamp(1.0 - alpha * hfilt ** 2, min=0.0))
    f_out = w_noise * fe + w_ref * cscale * fr
    out = torch.fft.ifft2(f_out).real
    if squeeze_t:
        out = out.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
    return out.to(noise.dtype)
